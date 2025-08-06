import torch

try:
    import triton
    import triton.language as tl

except ImportError:
    triton = None
    tl = None
if triton is None or tl is None:
    raise ImportError("Triton is not installed. Please install Triton to use this Mamba-Triton.")

from mambular.ops.triton.triton_utils import discretize_back, discretize_tt, roll, ssm_scan

# Mamba-Triton kernel inspired by https://github.com/srush/annotated-mamba


@triton.jit
def mamba_tt(
    X,
    dX,
    A,
    dA,
    B,
    dB,
    C,
    dC,
    Delta,
    dDelta,
    H_0,
    dH_0,
    Y,
    dY,
    H,
    dH,
    back: tl.constexpr,
    step: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    D_step: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
):  # maybe include meta parameter optimization?
    # Setup
    pid = tl.program_id(0)  # program id axis 0 = Which chunk are we in?
    bid = tl.program_id(1)  # Which batch instance are we in?
    kid = pid * K  # thread offset per block
    nH = tl.num_programs(0)  # num chunks
    Ba = tl.num_programs(1)  # Batchsize

    Ks = tl.arange(0, K)[None, None, :]  # 1 x 1 x K
    Ns = tl.arange(0, N)[:, None, None]  # N x 1 x 1
    Ds = tl.arange(0, D_step)[None, :, None]  # 1 x D x 1

    # init pointers (without batchdim-> less overhead, Batch instance is 1 and stride is 1)
    Nx1xK = bid * N * L + Ns * L + (Ks + kid)  # (N,K) -> that would be Ba,N,L in parallel
    DxK = bid * D * L + Ds * L + Ks + kid  # (D,K) -> Ba,D,L
    NxDx1 = bid * N * D + Ns * D + Ds
    NxDx1_H = bid * N * D * nH + Ns * D * nH + Ds * nH + pid  # chunkwise state h
    h_off = Ba * N * D * nH  # size of one state in h

    # masking of memory access
    mask = (kid + Ks) < L

    # Load forward
    b = tl.load(B + Nx1xK, mask=mask, other=0.0)
    c = tl.load(C + Nx1xK, mask=mask, other=0.0)

    # init grad outputs
    if back == 1:
        db_out = tl.zeros_like(b)
        dc_out = tl.zeros_like(c)

    for did in range(0, D // D_step):
        a = tl.load(A + NxDx1)  # no masking needed.
        # Load forward
        delta = tl.load(Delta + DxK, mask=mask, other=0.0)
        x = tl.load(X + DxK, mask=mask, other=0.0)
        a_, b_ = discretize_tt(a, b, delta)

        if step == 2:
            h2_0 = tl.load(H_0 + 1 * h_off + NxDx1_H) * (Ks == 0)  # load state in first thread of chunk
        else:
            h2_0 = tl.zeros_like(a_)
        # Compute Forward
        h1, h2 = ssm_scan(a_, b_ * x, h2_0, dim=2)

        if step == 1:
            tl.store(H + 0 * h_off + NxDx1_H + 0 * Ks, h1, Ks == K - 1)  # sram to hbm
            tl.store(H + 1 * h_off + NxDx1_H + 0 * Ks, h2, Ks == K - 1)
        if step == 2 and back == 0:
            y = tl.sum(c * h2, 0, 1)  # Optional: integrate skip connection here for performance
            tl.store(Y + DxK, y, mask=mask)  # store output

        # #Compute backward (fused bwd-fwd design since we use recomputation)
        if back == 1:
            # Load Backward
            dy = tl.load(dY + DxK, mask=mask, other=0.0)
            dh2_0 = tl.load(dH_0 + 1 * h_off + NxDx1_H) * (Ks == K - 1)
            delta_shift = tl.load(Delta + DxK + 1, (Ks + kid) < L - 1, 0)  # see Trambular Paper for details
            a_s, _ = discretize_tt(a, b, delta_shift)
            dh1, dh = ssm_scan(a_s, c * dy, dh2_0, rev=1, dim=2)
            if step == 1:
                tl.store(dH + 0 * h_off + NxDx1_H + 0 * Ks, dh1, Ks == 0)
                tl.store(dH + 1 * h_off + NxDx1_H + 0 * Ks, dh, Ks == 0)

        if back == 1 and step == 2:
            dc = tl.sum(h2 * dy, 1, 1)  # N x K
            rh2 = roll(h2, 2)
            rh2 = h2_0 * (Ks == 0) + rh2 * (Ks > 0)
            da, db, ddelta = discretize_back(a, b, delta, dh * rh2, dh * x)
            # Save (sums keep_dims=1): can be optimized further.
            tl.store(dX + DxK, tl.sum(b_ * dh, 0, 1), mask=mask)
            tl.store(dA + NxDx1_H, tl.sum(da, 2, 1))  # note: no L dimension, since we sum over it
            tl.store(dDelta + DxK, tl.sum(ddelta, 0, 1), mask=mask)
            db_out = db_out + tl.sum(db, 1, 1)
            dc_out = dc_out + dc
        Ds = Ds + D_step

    if back == 1 and step == 2:
        tl.store(dB + Nx1xK, db_out, mask=mask)
        tl.store(dC + Nx1xK, dc_out, mask=mask)
