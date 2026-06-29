"""Correctness test for the quantized GEMM (Marlin's method, dequant-to-shared).

Oracle: out = dequantize(Wq) @ X (the exact kernel target — isolates kernel correctness from
quantization error). Run from kernels/:
    python -m pytest qgemm/correctness/test_qgemm.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qgemm
from tk.quant import quantize_q8_0, dequantize_q8_0

SHAPES = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 128, 64)]


@pytest.mark.parametrize("shape", SHAPES)
def test_qgemm_q8_0(shape):
    N, K, M = shape
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    Wq = quantize_q8_0(W)                                   # (N, K/32, 34) uint8
    got = qgemm(mx.array(Wq), mx.array(X).astype(mx.float16), format="q8_0")
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    with np.errstate(all="ignore"):                          # macOS Accelerate matmul warnings
        ref = dequantize_q8_0(Wq).astype(np.float32) @ X
    assert got.shape == (N, M)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 2e-2, f"relative diff {rel}"


if __name__ == "__main__":
    for shp in SHAPES:
        test_qgemm_q8_0(shp)
        print("ok", shp)
