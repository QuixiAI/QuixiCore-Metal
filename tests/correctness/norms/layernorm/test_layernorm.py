"""Correctness test for the ThunderMittens LayerNorm Metal kernel.

Oracle: mx.fast.layer_norm (shipped with MLX) plus an explicit fp32 reference.
Run from the kernels/ directory:

    python -m pytest layernorm/correctness/test_layernorm.py -v
"""

import mlx.core as mx
import pytest

from tk import layernorm


def ref_layernorm(x, w, b, eps):
    """Explicit numpy-style fp32 oracle."""
    xf = x.astype(mx.float32)
    mu = xf.mean(axis=-1, keepdims=True)
    var = ((xf - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (xf - mu) * mx.rsqrt(var + eps) * w.astype(mx.float32) + b.astype(mx.float32)
    return y.astype(mx.bfloat16)


SHAPES = [
    (2, 128, 1024),
    (4, 64, 512),
    (1, 256, 768),
    (8, 256),  # 2D input, D=256
]


@pytest.mark.parametrize("shape", SHAPES)
def test_layernorm_matches_mlx(shape):
    eps = 1e-5
    D = shape[-1]
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    w = mx.random.normal((D,)).astype(mx.bfloat16)
    b = mx.random.normal((D,)).astype(mx.bfloat16)

    got = layernorm(x, w, b, eps=eps)
    exp_mlx = mx.fast.layer_norm(x, w, b, eps)
    exp_ref = ref_layernorm(x, w, b, eps)
    mx.eval(got, exp_mlx, exp_ref)

    assert got.shape == x.shape
    assert got.dtype == mx.bfloat16
    # bf16 storage of inputs/outputs => ~2e-2 tolerance; compute is fp32 in-kernel.
    assert mx.allclose(got, exp_mlx, atol=2e-2, rtol=2e-2), \
        f"max diff vs mlx: {mx.max(mx.abs(got.astype(mx.float32) - exp_mlx.astype(mx.float32))).item()}"
    assert mx.allclose(got, exp_ref, atol=2e-2, rtol=2e-2), \
        f"max diff vs ref: {mx.max(mx.abs(got.astype(mx.float32) - exp_ref.astype(mx.float32))).item()}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_layernorm_matches_mlx(shp)
        print(f"ok {shp}")


import numpy as np
from tk import layernorm_backward


@pytest.mark.parametrize("R,D", [(8, 256), (4, 512), (16, 64)])
def test_layernorm_backward(R, D):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(R + D + 1)
    x = (0.5 * rng.standard_normal((R, D))).astype(np.float32)
    w = (0.5 * rng.standard_normal((D,))).astype(np.float32)
    b = (0.3 * rng.standard_normal((D,))).astype(np.float32)
    dy = (0.3 * rng.standard_normal((R, D))).astype(np.float32)
    eps = 1e-5
    xt = torch.tensor(x, requires_grad=True); wt = torch.tensor(w, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    y = torch.nn.functional.layer_norm(xt, (D,), wt, bt, eps)
    y.backward(torch.tensor(dy))
    dx, dw, db = layernorm_backward(mx.array(x), mx.array(w), mx.array(dy), eps)
    mx.eval(dx, dw, db)

    def rel(g, ref):
        return np.abs(np.array(g) - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel(dx, xt.grad.numpy()) < 1e-4
    assert rel(dw, wt.grad.numpy()) < 1e-4
    assert rel(db, bt.grad.numpy()) < 1e-4
