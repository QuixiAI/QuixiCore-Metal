"""Correctness test for the multi-simdgroup threadgroup-staged GEMM (oracle x @ y).

Run from kernels/:  python -m pytest gemm_staged/correctness/test_gemm_staged.py -v
"""

import mlx.core as mx
import pytest

from tk import gemm_staged

SHAPES = [(32, 16, 32), (64, 32, 64), (128, 64, 128), (256, 128, 256)]


@pytest.mark.parametrize("dtype,atol", [(mx.float32, 1e-3), (mx.bfloat16, 0.5)])
@pytest.mark.parametrize("shape", SHAPES)
def test_gemm_staged(shape, dtype, atol):
    N, K, M = shape
    mx.random.seed(0)
    x = mx.random.uniform(shape=(N, K)).astype(dtype)
    y = mx.random.uniform(shape=(K, M)).astype(dtype)
    got = gemm_staged(x, y)
    exp = (x.astype(mx.float32) @ y.astype(mx.float32)).astype(dtype)
    mx.eval(got, exp)
    assert got.shape == (N, M)
    assert mx.allclose(got, exp, atol=atol, rtol=atol), \
        f"max diff: {mx.max(mx.abs(got.astype(mx.float32)-exp.astype(mx.float32))).item()}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_gemm_staged(shp, mx.float32, 1e-3)
        print("ok", shp)
