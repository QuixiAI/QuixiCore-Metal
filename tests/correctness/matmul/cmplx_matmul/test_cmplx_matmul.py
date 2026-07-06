"""Correctness test for the complex GEMM (exercises the complex_mma_AB primitive).

Operands carry a leading size-2 (real, imag) axis: a (2,N,K), b (2,K,M) -> (2,N,M).
Reference: numpy complex matmul. Run from kernels/:
    python -m pytest cmplx_matmul/correctness/test_cmplx_matmul.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import cmplx_matmul

SHAPES = [(32, 16, 32), (64, 32, 64), (128, 64, 128)]


@pytest.mark.parametrize("dtype,tol", [(mx.float32, 2e-2), (mx.bfloat16, 6e-2)])
@pytest.mark.parametrize("shape", SHAPES)
def test_cmplx_matmul(shape, dtype, tol):
    N, K, M = shape
    rng = np.random.default_rng(0)
    Ar, Ai = rng.standard_normal((N, K)).astype(np.float32), rng.standard_normal((N, K)).astype(np.float32)
    Br, Bi = rng.standard_normal((K, M)).astype(np.float32), rng.standard_normal((K, M)).astype(np.float32)
    A = mx.array(np.stack([Ar, Ai])).astype(dtype)
    B = mx.array(np.stack([Br, Bi])).astype(dtype)
    got = cmplx_matmul(A, B)
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    with np.errstate(all="ignore"):    # macOS Accelerate emits spurious matmul RuntimeWarnings
        ref_r = Ar @ Br - Ai @ Bi      # real-arithmetic reference (no complex BLAS path)
        ref_i = Ar @ Bi + Ai @ Br
    assert got.shape == (2, N, M)
    scale = max(np.abs(ref_r).max(), np.abs(ref_i).max()) + 1e-9
    rel = max(np.abs(g[0] - ref_r).max(), np.abs(g[1] - ref_i).max()) / scale
    assert rel < tol, f"relative diff {rel}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_cmplx_matmul(shp, mx.float32, 2e-2)
        print("ok", shp)
