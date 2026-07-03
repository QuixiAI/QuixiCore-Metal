"""Correctness test for the ThunderMittens RMSNorm Metal kernel (oracle mx.fast.rms_norm).

Run from kernels/:  python -m pytest rms_norm/correctness/test_rms_norm.py -v
"""

import mlx.core as mx
import pytest

from tk import rms_norm


def ref_rms_norm(x, w, eps):
    xf = x.astype(mx.float32)
    ms = (xf * xf).mean(axis=-1, keepdims=True)
    return (xf * mx.rsqrt(ms + eps) * w.astype(mx.float32)).astype(mx.bfloat16)


SHAPES = [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)]


@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_matches_mlx(shape):
    eps = 1e-5
    D = shape[-1]
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)
    got = rms_norm(x, w, eps=eps)
    exp_mlx = mx.fast.rms_norm(x, w, eps)
    exp_ref = ref_rms_norm(x, w, eps)
    mx.eval(got, exp_mlx, exp_ref)
    assert got.shape == x.shape and got.dtype == mx.bfloat16
    assert mx.allclose(got, exp_mlx, atol=2e-2, rtol=2e-2), \
        f"vs mlx: {mx.max(mx.abs(got.astype(mx.float32)-exp_mlx.astype(mx.float32))).item()}"
    assert mx.allclose(got, exp_ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    for shp in SHAPES:
        test_rms_norm_matches_mlx(shp)
        print("ok", shp)


import numpy as np
from tk import rms_norm_backward


@pytest.mark.parametrize("R,D", [(8, 256), (4, 512), (16, 64)])
def test_rms_norm_backward(R, D):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(R + D)
    x = (0.5 * rng.standard_normal((R, D))).astype(np.float32)
    w = (0.5 * rng.standard_normal((D,))).astype(np.float32)
    dy = (0.3 * rng.standard_normal((R, D))).astype(np.float32)
    eps = 1e-5
    xt = torch.tensor(x, requires_grad=True); wt = torch.tensor(w, requires_grad=True)
    y = xt * torch.rsqrt((xt ** 2).mean(-1, keepdim=True) + eps) * wt
    y.backward(torch.tensor(dy))
    dx, dw = rms_norm_backward(mx.array(x), mx.array(w), mx.array(dy), eps)
    mx.eval(dx, dw)

    def rel(g, ref):
        return np.abs(np.array(g) - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel(dx, xt.grad.numpy()) < 1e-4
    assert rel(dw, wt.grad.numpy()) < 1e-4
