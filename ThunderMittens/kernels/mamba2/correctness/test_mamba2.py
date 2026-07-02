"""Correctness test for Mamba-2 / SSD forward.

Y_t = sum_{j<=t} (C_t . B_j) * exp(cumlog_t - cumlog_j) * X_j, cumlog = cumsum(log a).
Reference: ((C@Bᵀ) ⊙ exp(cumlog_i - cumlog_j) ⊙ tril) @ X. Validated on relative error.
NOTE the oracle masks the decay EXPONENT (with -inf) before exp: exp(cl_i - cl_j) overflows to
inf in the upper triangle once N*|log a| exceeds ~88, and inf * 0 = NaN would poison the
reference (the kernels never form the upper triangle, so they are unaffected).
Run from kernels/:  python -m pytest mamba2/correctness/test_mamba2.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import (mamba2, mamba2_bwd, mamba2_chunked, mamba2_bwd_chunked, mamba2_dcl_to_da,
                ssd_decode)

# The quadratic kernel handles D in {64,128} at any N%8; the chunked linear-time pipeline
# (64x64 state quadrants) handles both D for N%64==0, auto-routed at the measured thresholds
# (N>=2048 for D=64, N>=4096 for D=128). The last two shapes exercise the auto-routed chunked
# path through the public API; the forced-route tests below cover it cheaply at small N.
SHAPES = [(1, 2, 64, 64), (2, 2, 128, 64), (1, 1, 256, 64),
          (1, 2, 64, 128), (2, 1, 128, 128),
          (1, 1, 2048, 64), (1, 1, 4096, 128)]


def _fwd_ref(C, Bm, X, cumlog, N):
    """Quadratic fp32 reference with the exponent masked BEFORE exp (NaN-safe at any N)."""
    scores = mx.matmul(C.astype(mx.float32), mx.swapaxes(Bm.astype(mx.float32), -1, -2))
    expo = cumlog[..., :, None] - cumlog[..., None, :]
    causal = mx.arange(N)[None, :] <= mx.arange(N)[:, None]
    expo = mx.where(causal, expo, mx.full(expo.shape, -mx.inf))
    return mx.matmul(scores * mx.exp(expo), X.astype(mx.float32))


def _fwd_inputs(shape, seed=0):
    B, H, N, D = shape
    mx.random.seed(seed)
    C = mx.random.normal((B, H, N, D)).astype(mx.bfloat16) * 0.5
    Bm = mx.random.normal((B, H, N, D)).astype(mx.bfloat16) * 0.5
    X = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    a = mx.sigmoid(mx.random.normal((B, H, N))) * 0.5 + 0.5      # decay a in (0.5, 1)
    cumlog = mx.cumsum(mx.log(a), axis=-1).astype(mx.float32)
    return C, Bm, X, cumlog


@pytest.mark.parametrize("shape", SHAPES)
def test_mamba2(shape):
    B, H, N, D = shape
    C, Bm, X, cumlog = _fwd_inputs(shape)
    got = mamba2(C, Bm, X, cumlog)
    exp = _fwd_ref(C, Bm, X, cumlog, N)
    mx.eval(got, exp)
    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp)).item()
    scale = mx.max(mx.abs(exp)).item() + 1e-9
    assert diff / scale < 0.03, f"relative diff {diff/scale}"


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 2, 192, 64),
                                   (1, 2, 128, 128), (1, 1, 256, 128)])
def test_mamba2_chunked_forced(shape):
    """The forced chunked route at small N (below the auto thresholds), both head dims."""
    B, H, N, D = shape
    C, Bm, X, cumlog = _fwd_inputs(shape, seed=N + D)
    got = mamba2_chunked(C, Bm, X, cumlog)
    exp = _fwd_ref(C, Bm, X, cumlog, N)
    mx.eval(got, exp)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp)).item()
    scale = mx.max(mx.abs(exp)).item() + 1e-9
    assert diff / scale < 0.03, f"relative diff {diff/scale}"


def _bwd_oracle(C, Bn, X, cl, dY):
    """Torch-autograd quadratic reference. The decay exponent is masked (-inf) BEFORE exp —
    tril() after the fact zeroes forward infs, but their backward turns 0 * inf into NaN."""
    import torch
    Ct = torch.tensor(C, requires_grad=True)
    Bt = torch.tensor(Bn, requires_grad=True)
    Xt = torch.tensor(X, requires_grad=True)
    clt = torch.tensor(cl, requires_grad=True)
    N = C.shape[2]
    G = torch.einsum("bhid,bhjd->bhij", Ct, Bt)
    expo = clt[:, :, :, None] - clt[:, :, None, :]
    causal = torch.arange(N)[None, :] <= torch.arange(N)[:, None]
    expo = expo.masked_fill(~causal, float("-inf"))
    Y = torch.einsum("bhij,bhjd->bhid", G * torch.exp(expo), Xt)
    Y.backward(torch.tensor(dY))
    return Ct.grad.numpy(), Bt.grad.numpy(), Xt.grad.numpy(), clt.grad.numpy()


def _bwd_inputs(shape, seed):
    rng = np.random.default_rng(seed)
    Bh, H, N, D = shape
    C = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    Bn = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    X = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    a = rng.uniform(0.9, 1.0, (Bh, H, N)).astype(np.float32)
    cl = np.cumsum(np.log(a), axis=2).astype(np.float32)
    dY = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    return C, Bn, X, cl, dY


def _rel(g, ref):
    return np.abs(np.array(g.astype(mx.float32)) - ref).max() / (np.abs(ref).max() + 1e-6)


@pytest.mark.parametrize("shape", [(2, 2, 64, 64), (1, 2, 128, 64), (2, 2, 64, 128),
                                   (1, 1, 256, 128),
                                   # auto-routed chunked (N >= threshold, N%64==0):
                                   (1, 1, 2048, 64)])
def test_mamba2_bwd(shape):
    torch = pytest.importorskip("torch")  # noqa: F841
    Bh, H, N, D = shape
    C, Bn, X, cl, dY = _bwd_inputs(shape, Bh + H + N + D)
    rC, rB, rX, rcl = _bwd_oracle(C, Bn, X, cl, dY)
    dC, dB, dX, dcl = mamba2_bwd(mx.array(C).astype(mx.bfloat16), mx.array(Bn).astype(mx.bfloat16),
                                 mx.array(X).astype(mx.bfloat16), mx.array(cl),
                                 mx.array(dY).astype(mx.bfloat16))
    mx.eval(dC, dB, dX, dcl)
    assert _rel(dC, rC) < 0.06
    assert _rel(dB, rB) < 0.06
    assert _rel(dX, rX) < 0.06
    assert _rel(dcl, rcl) < 0.08


@pytest.mark.parametrize("shape", [(1, 1, 128, 64), (2, 2, 192, 64),
                                   (1, 2, 128, 128), (1, 1, 256, 128)])
def test_mamba2_bwd_chunked_forced(shape):
    """The forced chunked linear-time backward at small N, both head dims, vs autograd."""
    torch = pytest.importorskip("torch")  # noqa: F841
    Bh, H, N, D = shape
    C, Bn, X, cl, dY = _bwd_inputs(shape, 10 + N + D)
    rC, rB, rX, rcl = _bwd_oracle(C, Bn, X, cl, dY)
    dC, dB, dX, dcl = mamba2_bwd_chunked(
        mx.array(C).astype(mx.bfloat16), mx.array(Bn).astype(mx.bfloat16),
        mx.array(X).astype(mx.bfloat16), mx.array(cl), mx.array(dY).astype(mx.bfloat16))
    mx.eval(dC, dB, dX, dcl)
    assert _rel(dC, rC) < 0.06
    assert _rel(dB, rB) < 0.06
    assert _rel(dX, rX) < 0.06
    assert _rel(dcl, rcl) < 0.08


@pytest.mark.parametrize("shape", [(1, 1, 128, 64), (2, 2, 192, 64), (1, 2, 256, 64),
                                   (1, 1, 128, 128)])
def test_mamba2_bwd_chunked_matches_quadratic(shape):
    """The chunked linear-time backward must agree bit-for-bit-ish with the O(N^2) quadratic route
    on the same input (forced routes on both sides)."""
    Bh, H, N, D = shape
    C, Bn, X, cl, dY = _bwd_inputs(shape, 100 + N)
    args = (mx.array(C).astype(mx.bfloat16), mx.array(Bn).astype(mx.bfloat16),
            mx.array(X).astype(mx.bfloat16), mx.array(cl), mx.array(dY).astype(mx.bfloat16))
    ch = mamba2_bwd_chunked(*args)                   # chunked, forced
    qd = mamba2_bwd(*args, force_quadratic=True)     # quadratic, forced
    mx.eval(*ch, *qd)
    for g, h in zip(ch, qd):
        g = np.array(g.astype(mx.float32)); h = np.array(h.astype(mx.float32))
        assert np.abs(g - h).max() / (np.abs(h).max() + 1e-6) < 0.02


def test_mamba2_bwd_all_ones_decay():
    """Degenerate decay: a ≡ 1 so cumlog ≡ 0 and L ≡ 1 (S = tril(C·Bᵀ), no exponential taper), with
    C = B = e0 so C_i·B_j ≡ 1 and S ≡ tril(1). M = dSt ∘ S is then the raw lower-triangular dY·Xᵀ,
    so dcl = rowsum(M) − colsum(M) stresses the col_sum + row/col-vec store paths directly (the
    less-trodden substrate route) with no decay masking. Both routes must match the oracle."""
    torch = pytest.importorskip("torch")
    Bh, H, N, D = 1, 1, 128, 64
    rng = np.random.default_rng(7)
    # C = B = e0-like so that C_i . B_j == 1 for all i,j (first coord 1, rest 0).
    C = np.zeros((Bh, H, N, D), np.float32); C[..., 0] = 1.0
    Bn = np.zeros((Bh, H, N, D), np.float32); Bn[..., 0] = 1.0
    X = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    cl = np.zeros((Bh, H, N), np.float32)             # a == 1 -> cumlog == 0 -> L ≡ 1
    dY = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    Ct = torch.tensor(C, requires_grad=True); Bt = torch.tensor(Bn, requires_grad=True)
    Xt = torch.tensor(X, requires_grad=True); clt = torch.tensor(cl, requires_grad=True)
    G = torch.einsum("bhid,bhjd->bhij", Ct, Bt)
    Ld = torch.exp(clt[:, :, :, None] - clt[:, :, None, :])
    Y = torch.einsum("bhij,bhjd->bhid", torch.tril(G * Ld), Xt)
    Y.backward(torch.tensor(dY))
    ref = clt.grad.numpy()
    args = (mx.array(C).astype(mx.bfloat16), mx.array(Bn).astype(mx.bfloat16),
            mx.array(X).astype(mx.bfloat16), mx.array(cl), mx.array(dY).astype(mx.bfloat16))
    for route in (mamba2_bwd_chunked(*args), mamba2_bwd(*args)):
        dcl = route[3]
        mx.eval(dcl)
        dcl = np.array(dcl.astype(mx.float32))
        assert np.abs(dcl - ref).max() / (np.abs(ref).max() + 1e-6) < 0.02, \
            f"dcl rel err {np.abs(dcl - ref).max() / (np.abs(ref).max() + 1e-6)}"


def test_mamba2_dcl_to_da():
    torch = pytest.importorskip("torch")
    Bh, H, N, D = 1, 1, 64, 64
    rng = np.random.default_rng(9)
    C = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    Bn = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    X = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    a = rng.uniform(0.9, 1.0, (Bh, H, N)).astype(np.float32)
    dY = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    at = torch.tensor(a, requires_grad=True)
    cl_t = torch.cumsum(torch.log(at), dim=2)
    G = torch.einsum("bhid,bhjd->bhij", torch.tensor(C), torch.tensor(Bn))
    L = torch.exp(cl_t[:, :, :, None] - cl_t[:, :, None, :])
    Y = torch.einsum("bhij,bhjd->bhid", torch.tril(G * L), torch.tensor(X))
    Y.backward(torch.tensor(dY))
    cl = cl_t.detach().numpy().astype(np.float32)
    _, _, _, dcl = mamba2_bwd(mx.array(C).astype(mx.bfloat16), mx.array(Bn).astype(mx.bfloat16),
                              mx.array(X).astype(mx.bfloat16), mx.array(cl),
                              mx.array(dY).astype(mx.bfloat16))
    da = mamba2_dcl_to_da(dcl, mx.array(a))
    mx.eval(da)
    assert np.abs(np.array(da) - at.grad.numpy()).max() / (np.abs(at.grad.numpy()).max() + 1e-6) < 0.1


@pytest.mark.parametrize("D", [64, 128])
def test_ssd_decode(D):
    """T iterated decode steps == the fp32 recurrence oracle (state and readout)."""
    B, H, T = 2, 2, 5
    rng = np.random.default_rng(3 + D)
    S = np.zeros((B, H, D, D), np.float32)
    Smx = mx.array(S)
    for t in range(T):
        alpha = rng.uniform(0.9, 1.0, (B, H)).astype(np.float32)
        x = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
        k = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
        q = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
        y, Smx = ssd_decode(Smx, mx.array(alpha), mx.array(x), mx.array(k), mx.array(q))
        mx.eval(y, Smx)
        S = alpha[..., None, None] * S + np.einsum("bhp,bhn->bhpn", x, k)
        ref = np.einsum("bhpn,bhn->bhp", S, q)
        scale = np.abs(ref).max() + 1e-6
        assert np.abs(np.array(y) - ref).max() / scale < 1e-4, f"step {t}"
        assert np.abs(np.array(Smx) - S).max() / (np.abs(S).max() + 1e-6) < 1e-4, f"step {t}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_mamba2(shp)
        print("ok", shp)
