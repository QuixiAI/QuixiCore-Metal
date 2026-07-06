"""Correctness test for the quantized fused GEMM+GELU (Phase-6 retrofit).

out = gelu(dequantize(Wq) @ X + bias), tanh-approx GELU. Oracle uses dequantize(Wq) so the
tolerance is format-independent. Parametrized over every packed format. Run from kernels/:
    python -m pytest qflux/correctness/test_qflux.py -v
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from tk import qflux_gelu
from tk.quant import QUANT_FORMATS

SHAPES = [(32, 256, 32), (128, 512, 64)]  # (N, K, M); K%256 for q4_K


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


@pytest.mark.parametrize("fmt", sorted(QUANT_FORMATS))
@pytest.mark.parametrize("shape", SHAPES)
def test_qflux_gelu(shape, fmt):
    quantize, dequantize = QUANT_FORMATS[fmt]
    N, K, M = shape
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    bias = rng.standard_normal((M,)).astype(np.float32)
    Wq = quantize(W)
    got = qflux_gelu(mx.array(Wq), mx.array(X).astype(mx.float16),
                     mx.array(bias).astype(mx.float16), format=fmt)
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    with np.errstate(all="ignore"):
        ref = _gelu_tanh(dequantize(Wq).astype(np.float32) @ X + bias)
    assert got.shape == (N, M)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 3e-2, f"{fmt} relative diff {rel}"


if __name__ == "__main__":
    for fmt in sorted(QUANT_FORMATS):
        test_qflux_gelu((128, 512, 64), fmt)
        print("ok", fmt)
