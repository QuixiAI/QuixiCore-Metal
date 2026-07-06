"""Correctness test for the quantized GEMM (Marlin's method, dequant-to-shared).

Oracle: out = dequantize(Wq) @ X (the exact kernel target — isolates kernel correctness from
quantization error, so the tolerance is format-independent). Parametrized over every packed
format in tk.quant.QUANT_FORMATS. Run from kernels/:
    python -m pytest qgemm/correctness/test_qgemm.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qgemm
from tk.quant import QUANT_FORMATS

# K is a multiple of 256 so q4_K's 256-weight super-block works for every format uniformly.
SHAPES = [(32, 256, 32), (64, 256, 64), (128, 512, 128), (256, 256, 64)]


@pytest.mark.parametrize("fmt", sorted(QUANT_FORMATS))
@pytest.mark.parametrize("shape", SHAPES)
def test_qgemm(shape, fmt):
    quantize, dequantize = QUANT_FORMATS[fmt]
    N, K, M = shape
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    Wq = quantize(W)
    got = qgemm(mx.array(Wq), mx.array(X).astype(mx.float16), format=fmt)
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    with np.errstate(all="ignore"):                          # macOS Accelerate matmul warnings
        ref = dequantize(Wq).astype(np.float32) @ X
    assert got.shape == (N, M)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 2e-2, f"{fmt} relative diff {rel}"


if __name__ == "__main__":
    for fmt in sorted(QUANT_FORMATS):
        for shp in SHAPES:
            test_qgemm(shp, fmt)
        print("ok", fmt)
