import torch

try:
    import triton
    import triton.language as tl

except ImportError:
    triton = None
    tl = None
if triton is None or tl is None:
    raise ImportError("Triton is not installed. Please install Triton to use this module.")


@triton.jit
def first_order_op(fl, xl, fr, xr):
    f = fr * fl
    x = fr * xl + xr
    return f, x


def reshape_inputs(a, b, c, delta, x, y=None, rev=False):
    if y is None and rev is True:
        # throw error
        raise ValueError("y = None and rev == True simmultaniously not possible")

    if rev:
        Ba, _, D, L = x.shape
        _, N, _, _ = b.shape

        # a: (Ba, N, D, 1) -> (D, N)
        a = a[:, :, :, 0].permute(2, 1, 0).mean(dim=2).contiguous()  # average over batch
        # b, c: (Ba, N, 1, L) -> (Ba, L, N)
        b = b.squeeze(2).permute(0, 2, 1).contiguous()
        c = c.squeeze(2).permute(0, 2, 1).contiguous()

        # delta, x,y: (Ba, 1, D, L) -> (Ba, L, D)
        delta = delta.squeeze(1).permute(0, 2, 1).contiguous()
        x = x.squeeze(1).permute(0, 2, 1).contiguous()
        if y is not None:
            y = y.squeeze(1).permute(0, 2, 1).contiguous()

    else:
        Ba, L, D = x.shape
        _, _, N = b.shape

        x = x.permute(0, 2, 1).unsqueeze(1).contiguous().cuda()
        y = x  # (Ba, 1, D, L)
        delta = delta.permute(0, 2, 1).unsqueeze(1).contiguous().cuda()  # (Ba, 1, D, L)
        a = a.T.unsqueeze(0).unsqueeze(-1).expand(Ba, -1, -1, 1).contiguous().cuda()  # (Ba, N, D, 1)
        b = b.permute(0, 2, 1).unsqueeze(2).contiguous().cuda()  # (Ba, N, 1, L)
        c = c.permute(0, 2, 1).unsqueeze(2).contiguous().cuda()  # (Ba, N, 1, L)

    return a, b, c, delta, x, y


# utilities for blockwise upsweep
# See https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
@triton.jit
def rol(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur


@triton.jit
def roll(y, dim, rev=0):
    _, rh2, _ = tl.associative_scan((1 + 0 * y, 0.0 * y, y), dim, rol, reverse=rev)
    return rh2


@triton.jit
def ssm_scan(h1, h2, h2_0, rev: tl.constexpr = 0, dim: tl.constexpr = 0):
    # Apply initial
    n1, n2 = first_order_op(tl.zeros_like(h1) + 1.0, h2_0, h1, h2)
    # Scan
    h1, h2 = tl.associative_scan((n1, n2), dim, first_order_op, reverse=rev)
    return h1, h2


@triton.jit
def discretize_tt(a, b, delta):
    da = delta * a
    a_ = tl.exp(da)
    b_ = b * delta  # approximation of b_
    return a_, b_


@triton.jit
def discretize_back(a, b, d, da_, db_):
    da = d * a
    a_ = tl.exp(da)
    da_da = d * a_
    da_ddelta = a * a_

    db_db = d
    db_ddelta = b

    return da_ * da_da, db_ * db_db, da_ * da_ddelta + db_ * db_ddelta


@triton.jit
def simple_ssm_tt(X, A, B, C, Y, K: tl.constexpr):  # simple associative scan
    Ks = tl.arange(0, K)
    bid = tl.program_id(0)
    kid = bid * K
    x = tl.load(X + Ks + kid)
    a, b, c = ssm_load(Ks + kid, A, B, C)

    # Compute
    h1, h2 = tl.associative_scan((a, b * x), 0, first_order_op)
    y = c * h2
    # Save
    tl.store(Y + Ks + kid, y)


# upsweep on HBM (its basically the upsweep of the blelloch scan as described in Trambular)
def reduce(v, rev, batch=1):
    if rev:
        v[0, :] = v[0].flip(-1)
    o = torch.ones_like(v[0, 0])
    simple_ssm_tt[(batch,)](v[0, 1], v[0, 0], o, o, v[1, 1], K=v.shape[-1])
    v[..., -1] = 0.0
    v[:] = torch.roll(v, 1)
    if rev:
        v[1, :] = v[1].flip(-1)


@triton.jit
def ssm_load(Ks, A, B, C):
    "Helper for loading"
    a = tl.load(A + Ks)
    b = tl.load(B + Ks)
    c = tl.load(C + Ks)
    return a, b, c
