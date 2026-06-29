"""Correctness test for the ThunderMittens add_rt Metal kernel.

add_rt is the simplest end-to-end smoke test: elementwise out = x + y over 8x8
register tiles (one threadgroup per tile), so shapes must be multiples of 8.
Run from the kernels/ directory:

    python -m pytest add_rt/correctness/test_add.py -v
"""

import mlx.core as mx
import pytest

from tk import add_rt

SHAPES = [(8, 8), (32, 32), (64, 128), (128, 64)]
DTYPES = [mx.float32, mx.float16, mx.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_add_rt(shape, dtype):
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(dtype)
    y = mx.random.normal(shape).astype(dtype)
    got = add_rt(x, y)
    exp = x + y
    mx.eval(got, exp)
    assert got.shape == shape
    assert got.dtype == dtype
    assert mx.allclose(got, exp, atol=1e-2, rtol=1e-2), \
        f"max diff: {mx.max(mx.abs(got.astype(mx.float32) - exp.astype(mx.float32))).item()}"


if __name__ == "__main__":
    for shp in SHAPES:
        for dt in DTYPES:
            test_add_rt(shp, dt)
    print("ok")
