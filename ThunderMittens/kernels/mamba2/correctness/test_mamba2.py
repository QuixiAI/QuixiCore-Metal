"""Correctness test for Mamba-2 / SSD forward.

Y_t = sum_{j<=t} (C_t . B_j) * exp(cumlog_t - cumlog_j) * X_j, cumlog = cumsum(log a).
Reference: ((C@Bᵀ) ⊙ exp(cumlog_i - cumlog_j) ⊙ tril) @ X. Validated on relative error.
Run from kernels/:  python -m pytest mamba2/correctness/test_mamba2.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import mamba2, mamba2_bwd, mamba2_dcl_to_da

# forward covers D=64 and D=128; the quadratic kernel handles both, chunked route is D=64 only.
SHAPES = [(1, 2, 64, 64), (2, 2, 128, 64), (1, 1, 256, 64),
          (1, 2, 64, 128), (2, 1, 128, 128)]


@pytest.mark.parametrize("shape", SHAPES)
def test_mamba2(shape):
    B, H, N, D = shape
    mx.random.seed(0)
    C = mx.random.normal((B, H, N, D)).astype(mx.bfloat16) * 0.5
    Bm = mx.random.normal((B, H, N, D)).astype(mx.bfloat16) * 0.5
    X = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    a = mx.sigmoid(mx.random.normal((B, H, N))) * 0.5 + 0.5      # decay a in (0.5, 1)
    cumlog = mx.cumsum(mx.log(a), axis=-1).astype(mx.float32)
    got = mamba2(C, Bm, X, cumlog)
    scores = mx.matmul(C.astype(mx.float32), mx.swapaxes(Bm.astype(mx.float32), -1, -2))
    decay = mx.exp(cumlog[..., :, None] - cumlog[..., None, :])
    mask = (mx.arange(N)[None, :] <= mx.arange(N)[:, None]).astype(mx.float32)
    exp = mx.matmul(scores * decay * mask, X.astype(mx.float32))
    mx.eval(got, exp)
    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp)).item()
    scale = mx.max(mx.abs(exp)).item() + 1e-9
    assert diff / scale < 0.03, f"relative diff {diff/scale}"


@pytest.mark.parametrize("shape", [(2, 2, 64, 64), (1, 2, 128, 64), (2, 2, 64, 128),
                                   (1, 1, 256, 128),
                                   # chunked linear-time route (D=64, N%64==0, N>=128):
                                   (1, 1, 128, 64), (2, 2, 192, 64), (1, 2, 256, 64)])
def test_mamba2_bwd(shape):
    torch = pytest.importorskip("torch")
    Bh, H, N, D = shape
    rng = np.random.default_rng(Bh + H + N + D)
    C = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    Bn = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    X = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    a = rng.uniform(0.9, 1.0, (Bh, H, N)).astype(np.float32)
    cl = np.cumsum(np.log(a), axis=2).astype(np.float32)
    dY = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)

    Ct = torch.tensor(C, requires_grad=True)
    Bt = torch.tensor(Bn, requires_grad=True)
    Xt = torch.tensor(X, requires_grad=True)
    clt = torch.tensor(cl, requires_grad=True)
    G = torch.einsum("bhid,bhjd->bhij", Ct, Bt)
    L = torch.exp(clt[:, :, :, None] - clt[:, :, None, :])
    S = torch.tril(G * L)
    Y = torch.einsum("bhij,bhjd->bhid", S, Xt)
    Y.backward(torch.tensor(dY))

    dC, dB, dX, dcl = mamba2_bwd(mx.array(C).astype(mx.bfloat16), mx.array(Bn).astype(mx.bfloat16),
                                 mx.array(X).astype(mx.bfloat16), mx.array(cl),
                                 mx.array(dY).astype(mx.bfloat16))
    mx.eval(dC, dB, dX, dcl)

    def rel(g, ref):
        return np.abs(np.array(g.astype(mx.float32)) - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel(dC, Ct.grad.numpy()) < 0.06
    assert rel(dB, Bt.grad.numpy()) < 0.06
    assert rel(dX, Xt.grad.numpy()) < 0.06
    assert rel(dcl, clt.grad.numpy()) < 0.08


@pytest.mark.parametrize("shape", [(1, 1, 128, 64), (2, 2, 192, 64), (1, 2, 256, 64)])
def test_mamba2_bwd_chunked_matches_quadratic(shape):
    """The chunked linear-time backward must agree bit-for-bit-ish with the O(N^2) quadratic route
    on the same D=64/N%64==0 input (force_quadratic pins the fallback)."""
    Bh, H, N, D = shape
    rng = np.random.default_rng(100 + N)
    C = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    Bn = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    X = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    a = rng.uniform(0.9, 1.0, (Bh, H, N)).astype(np.float32)
    cl = np.cumsum(np.log(a), axis=2).astype(np.float32)
    dY = (0.3 * rng.standard_normal((Bh, H, N, D))).astype(np.float32)
    args = (mx.array(C).astype(mx.bfloat16), mx.array(Bn).astype(mx.bfloat16),
            mx.array(X).astype(mx.bfloat16), mx.array(cl), mx.array(dY).astype(mx.bfloat16))
    ch = mamba2_bwd(*args)                          # chunked (auto)
    qd = mamba2_bwd(*args, force_quadratic=True)     # quadratic fallback
    mx.eval(*ch, *qd)
    for g, h in zip(ch, qd):
        g = np.array(g.astype(mx.float32)); h = np.array(h.astype(mx.float32))
        assert np.abs(g - h).max() / (np.abs(h).max() + 1e-6) < 0.02


def test_mamba2_bwd_all_ones_decay():
    """Degenerate decay: a ≡ 1 so cumlog ≡ 0 and L ≡ 1 (S = tril(C·Bᵀ), no exponential taper), with
    C = B = e0 so C_i·B_j ≡ 1 and S ≡ tril(1). M = dSt ∘ S is then the raw lower-triangular dY·Xᵀ,
    so dcl = rowsum(M) − colsum(M) stresses the col_sum + row/col-vec store paths directly (the
    less-trodden substrate route) with no decay masking. Must still match the autograd oracle."""
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
    _, _, _, dcl = mamba2_bwd(mx.array(C).astype(mx.bfloat16), mx.array(Bn).astype(mx.bfloat16),
                              mx.array(X).astype(mx.bfloat16), mx.array(cl),
                              mx.array(dY).astype(mx.bfloat16))
    mx.eval(dcl)
    ref = clt.grad.numpy()
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


if __name__ == "__main__":
    for shp in SHAPES:
        test_mamba2(shp)
        print("ok", shp)
