"""Correctness test for the ThunderMittens softmax Metal kernel (oracle mx.softmax).

Run from kernels/:  python -m pytest softmax/correctness/test_softmax.py -v
"""

import mlx.core as mx
import pytest

from tk import softmax

SHAPES = [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)]


@pytest.mark.parametrize("shape", SHAPES)
def test_softmax_matches_mlx(shape):
    mx.random.seed(0)
    x = mx.random.normal(shape).astype(mx.bfloat16)
    got = softmax(x)
    exp = mx.softmax(x.astype(mx.float32), axis=-1).astype(mx.bfloat16)
    mx.eval(got, exp)
    assert got.shape == x.shape and got.dtype == mx.bfloat16
    assert mx.allclose(got, exp, atol=2e-2, rtol=2e-2), \
        f"max diff: {mx.max(mx.abs(got.astype(mx.float32)-exp.astype(mx.float32))).item()}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_softmax_matches_mlx(shp)
        print("ok", shp)
