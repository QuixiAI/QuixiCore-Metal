"""Correctness test for hedgehog feature-map linear attention.

out = phi(Q) @ (phi(K)^T @ V), phi(x) = exp(x - rowmax(x)). Validated on relative error.
Run from kernels/:  python -m pytest hedgehog/correctness/test_hedgehog.py -v
"""

import mlx.core as mx
import pytest

from tk import hedgehog

SHAPES = [(1, 2, 128, 64), (2, 4, 256, 64), (1, 1, 512, 64)]


def _phi(x):
    xf = x.astype(mx.float32)
    return mx.exp(xf - mx.max(xf, axis=-1, keepdims=True))


@pytest.mark.parametrize("use_kernel", [True, False], ids=["kernel", "routed"])
@pytest.mark.parametrize("shape", SHAPES)
def test_hedgehog(shape, use_kernel):
    B, H, N, D = shape
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, H, N, D)).astype(mx.bfloat16)
    got = hedgehog(q, k, v, use_kernel=use_kernel)
    kv = mx.matmul(mx.swapaxes(_phi(k), -1, -2), v.astype(mx.float32))
    exp = mx.matmul(_phi(q), kv)
    mx.eval(got, exp)
    assert got.shape == (B, H, N, D)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp)).item()
    scale = mx.max(mx.abs(exp)).item() + 1e-9
    assert diff / scale < 0.03, f"relative diff {diff/scale}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_hedgehog(shp)
        print("ok", shp)
