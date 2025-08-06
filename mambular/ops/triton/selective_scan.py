import math

import numpy as np
import torch
import torch.nn as nn

from .kernel import mamba_tt  # type: ignore
from .triton_utils import reduce, reshape_inputs

try:
    import triton
    import triton.language as tl

except ImportError:
    triton = None
    tl = None
if triton is None or tl is None:
    raise ImportError("Triton is not installed. Please install Triton to use this module.")


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, c, delta, d=None):
        if x.dim() != 4:
            if x.dim() == 3 and a.dim() == 2:
                # reshape
                a, b, c, delta, x, _ = reshape_inputs(a, b, c, delta, x, None, rev=False)
            else:
                # throw error
                raise ValueError(f"Expected input with 3 or 4 dimensions, got {x.dim()}D tensor.")

        Ba, _, D, L = x.shape
        _, N, _, _ = b.shape

        # staging bounds are based on gut feeling. Might need to be adjusted
        # use async instruction copy. May lead to less comparibility with older hardware
        if N * D > 1024:
            num_stages = 1
        else:
            num_stages = 2

        BLOCKSIZE = 16

        BLOCKS = math.ceil(L / BLOCKSIZE)
        if BLOCKS % 2 != 0:
            BLOCKS = BLOCKS + 1
            padded = True

        if D >= 32:
            D_step = 2
        else:
            D_step = D

        # make placeholders
        dx, db, dc, ddelta = None, None, None, None
        # (torch.zeros_like(b).float().cuda() for b in [x, b, c, delta])
        da = None
        # torch.zeros(Ba, N, D, BLOCKS).float().cuda()
        y = torch.ones(Ba, 1, D, L, dtype=torch.float16, device="cuda")
        h = torch.zeros(2, 2, Ba, N, D, BLOCKS, dtype=torch.float16, device="cuda")
        dh, dy = None, None

        mamba_tt[(BLOCKS, Ba)](
            x,
            dx,
            a,
            da,
            b,
            db,
            c,
            dc,
            delta,
            ddelta,
            h[0],
            dh,
            y,
            dy,
            h[0],
            dh,
            back=0,
            step=1,
            L=L,
            K=BLOCKSIZE,
            D_step=D_step,
            D=D,
            N=N,
            num_stages=num_stages,
        )
        reduce(h, False, Ba * N * D)
        # reduce(dh, True, Ba * N * D)
        mamba_tt[(BLOCKS, Ba)](
            x,
            dx,
            a,
            da,
            b,
            db,
            c,
            dc,
            delta,
            ddelta,
            h[1],
            dh,
            y,
            dy,
            h[1],
            dh,
            back=0,
            step=2,
            L=L,
            K=BLOCKSIZE,
            D_step=D_step,
            D=D,
            N=N,
            num_stages=num_stages,
        )
        ctx.save_for_backward(a, b, c, delta, x, d)

        _, _, _, _, x, y = reshape_inputs(a, b, c, delta, x, y, rev=True)

        if d is not None:
            y = y + d * x
        return y  # , h, dh

    def backward(ctx, grad_output_y):
        # if grad_output_y is None:
        #    raise ValueError("grad_output_y is None!")
        a, b, c, delta, x, d = ctx.saved_tensors

        Ba, _, D, L = x.shape
        _, N, _, _ = b.shape

        if N * D > 1024:
            num_stages = 1
        else:
            num_stages = 2

        # we need to hardwire BLOCKSIZE since we need to know the number of Blocks
        BLOCKSIZE = 16
        BLOCKS = math.ceil(L / BLOCKSIZE)
        if BLOCKS % 2 != 0:
            BLOCKS = BLOCKS + 1
            padded = True

        if D >= 32:
            D_step = 2
        else:
            D_step = D

        dx, db, dc, ddelta = (torch.zeros_like(b).half().cuda() for b in [x, b, c, delta])
        da = torch.zeros(Ba, N, D, BLOCKS, dtype=torch.float16, device="cuda")
        dy = torch.ones(Ba, 1, D, L, dtype=torch.float16, device="cuda")
        y = None
        h, dh = (torch.zeros(2, 2, Ba, N, D, BLOCKS, dtype=torch.float16, device="cuda") for _ in range(2))
        # assert BLOCKS == SEQLEN // K
        mamba_tt[(BLOCKS, Ba)](
            x,
            dx,
            a,
            da,
            b,
            db,
            c,
            dc,
            delta,
            ddelta,
            h[0],
            dh[0],
            y,
            dy,
            h[0],
            dh[0],
            back=1,
            step=1,
            L=L,
            K=BLOCKSIZE,
            D_step=D_step,
            D=D,
            N=N,
            num_stages=num_stages,
        )
        reduce(h, False, Ba * N * D)
        reduce(dh, True, Ba * N * D)
        mamba_tt[(BLOCKS, Ba)](
            x,
            dx,
            a,
            da,
            b,
            db,
            c,
            dc,
            delta,
            ddelta,
            h[1],
            dh[1],
            y,
            dy,
            h[1],
            dh[1],
            back=1,
            step=2,
            L=L,
            K=BLOCKSIZE,
            D_step=D_step,
            D=D,
            N=N,
            num_stages=num_stages,
        )

        grad_output_y = grad_output_y.permute(0, 2, 1).unsqueeze(1).contiguous()  #  Ba,1,D,L

        if d is not None:
            dd = x
            dd = dd * grad_output_y  # Ba,1,D,L*Ba,1,D,L
            dd = dd.squeeze(1).sum((0, 2), keepdim=False)  # .contiguous()  # Ba,1,D,L -> D
        else:
            dd = None
        dx = dx * grad_output_y  # Ba,1,D,L * Ba,1,D,L -> Ba,1,D,L
        dx = dx.squeeze(1).permute(0, 2, 1)  # .contiguous()  # Ba,1,D,L -> Ba,L,D

        da = da.sum(dim=-1)
        grad_output_y_reduced = grad_output_y.sum(dim=-1)
        grad_output_y_reduced = grad_output_y_reduced.expand(-1, da.shape[1], -1)
        da = da * grad_output_y_reduced  # Ba, N, D
        da = da.sum(dim=0, keepdim=False).permute(1, 0)  # .contiguous()  # Ba,N,D -> D,N

        db = db * grad_output_y  # Ba,N,1,L * Ba,1,D,L -> Ba,N,D,L we need Ba,L,N
        db = db.sum(-2, keepdim=False).permute(0, 2, 1)  # .contiguous()  # Ba,N,D,L -> Ba,L,N

        dc = dc * grad_output_y  # Ba,N,1,L * Ba,1,D,L -> Ba,N,D,L we need Ba,L,N
        dc = dc.sum(-2, keepdim=False).permute(0, 2, 1)  # .contiguous()  # Ba,N,D,L -> Ba,L,N

        ddelta = ddelta * grad_output_y  # Ba,1,D,L * Ba,1,D,L -> Ba,1,D,L we need Ba,L,D
        ddelta = ddelta.squeeze(1).permute(0, 2, 1)  # .contiguous()

        return dx, da, db, dc, ddelta, dd
