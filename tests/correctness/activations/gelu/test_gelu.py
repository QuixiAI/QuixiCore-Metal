"""Correctness test for the ThunderMittens GELU kernel (oracle mx.nn.gelu_approx).

Run from kernels/:  python -m pytest gelu/correctness/test_gelu.py -v
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from tk import gelu

SHAPES = [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)]


@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_matches_mlx(shape):
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    got = gelu(x)
    exp = nn.gelu_approx(x.astype(mx.float32)).astype(mx.bfloat16)
    mx.eval(got, exp)
    assert got.shape == x.shape and got.dtype == mx.bfloat16
    assert mx.allclose(got, exp, atol=2e-2, rtol=2e-2), \
        f"max diff: {mx.max(mx.abs(got.astype(mx.float32)-exp.astype(mx.float32))).item()}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_gelu_matches_mlx(shp)
        print("ok", shp)


import numpy as np
from tk import gelu_backward


@pytest.mark.parametrize("shape", [(6, 128), (4, 512), (3, 5, 64)])
def test_gelu_backward(shape):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(hash(shape) % 1000)
    x = (1.5 * rng.standard_normal(shape)).astype(np.float32)
    dy = (0.4 * rng.standard_normal(shape)).astype(np.float32)
    xt = torch.tensor(x, requires_grad=True)
    torch.nn.functional.gelu(xt, approximate="tanh").backward(torch.tensor(dy))
    dx = gelu_backward(mx.array(x), mx.array(dy))
    mx.eval(dx)
    assert np.abs(np.array(dx) - xt.grad.numpy()).max() / (np.abs(xt.grad.numpy()).max() + 1e-9) < 1e-4
