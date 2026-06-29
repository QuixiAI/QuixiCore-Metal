"""Correctness test for the ThunderMittens matmul_custom Metal kernel.

The compiled kernel uses a fixed <4,2,4> block tiling (32x16x32 element blocks),
so shapes must satisfy N%32==0, M%32==0, K%16==0. Oracle: mx.matmul (x @ y).
Run from the kernels/ directory:

    python -m pytest matmul_custom/correctness/test_matmul.py -v
"""

import mlx.core as mx
import pytest

from tk import matmul_custom

# (N, K, M)
SHAPES = [(32, 16, 32), (64, 32, 64), (128, 64, 128), (256, 128, 256)]


@pytest.mark.parametrize("shape", SHAPES)
def test_matmul_f32(shape):
    N, K, M = shape
    mx.random.seed(0)
    x = mx.random.uniform(shape=(N, K)).astype(mx.float32)
    y = mx.random.uniform(shape=(K, M)).astype(mx.float32)
    got = matmul_custom(x, y)
    exp = x @ y
    mx.eval(got, exp)
    assert got.shape == (N, M)
    assert mx.allclose(got, exp, atol=1e-3, rtol=1e-4), \
        f"max diff: {mx.max(mx.abs(got - exp)).item()}"


@pytest.mark.parametrize("shape", SHAPES)
def test_matmul_bf16(shape):
    N, K, M = shape
    mx.random.seed(0)
    x = mx.random.uniform(shape=(N, K)).astype(mx.bfloat16)
    y = mx.random.uniform(shape=(K, M)).astype(mx.bfloat16)
    got = matmul_custom(x, y)
    # Kernel accumulates in fp32 then stores bf16; reference in fp32 then cast.
    exp = (x.astype(mx.float32) @ y.astype(mx.float32)).astype(mx.bfloat16)
    mx.eval(got, exp)
    assert got.shape == (N, M)
    diff = mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32))).item()
    # bf16 rounding of magnitude-~K/4 entries; generous absolute tolerance.
    assert mx.allclose(got, exp, atol=0.3, rtol=5e-2), f"max diff: {diff}"


# Arbitrary (non-tile-multiple) shapes: tk.matmul_custom zero-pads to tile multiples.
ARB_SHAPES = [(40, 20, 48), (100, 50, 70), (1, 1, 1), (33, 17, 65)]


@pytest.mark.parametrize("shape", ARB_SHAPES)
def test_matmul_arbitrary_shapes(shape):
    N, K, M = shape
    mx.random.seed(0)
    x = mx.random.uniform(shape=(N, K)).astype(mx.float32)
    y = mx.random.uniform(shape=(K, M)).astype(mx.float32)
    got = matmul_custom(x, y)
    exp = x @ y
    mx.eval(got, exp)
    assert got.shape == (N, M)
    assert mx.allclose(got, exp, atol=1e-3, rtol=1e-4), \
        f"max diff: {mx.max(mx.abs(got - exp)).item()}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_matmul_f32(shp)
        test_matmul_bf16(shp)
        print(f"ok {shp}")
