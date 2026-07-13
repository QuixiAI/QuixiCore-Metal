"""Correctness test for the quantized GEMV (batch-1 decode path).

Oracle: out = dequantize(Wq) @ x, x is (K, 1). Parametrized over every packed format in
tk.quant.QUANT_FORMATS. Also checks that tk.qgemm routes M==1 here.
Run from kernels/:  python -m pytest qgemv/correctness/test_qgemv.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qgemv, qgemm
from tk.quant import QUANT_FORMATS

SHAPES = [(32, 256), (128, 256), (256, 512), (1024, 256)]  # (N, K); K%256 for q4_K


@pytest.mark.parametrize("fmt", sorted(QUANT_FORMATS))
@pytest.mark.parametrize("shape", SHAPES)
def test_qgemv(shape, fmt):
    quantize, dequantize = QUANT_FORMATS[fmt]
    N, K = shape
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    x = rng.standard_normal((K, 1)).astype(np.float32)
    Wq = quantize(W)
    got = qgemv(mx.array(Wq), mx.array(x).astype(mx.float16), format=fmt)
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    with np.errstate(all="ignore"):
        ref = dequantize(Wq).astype(np.float32) @ x
    assert got.shape == (N, 1)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 2e-2, f"{fmt} relative diff {rel}"


def test_qgemm_routes_m1_to_qgemv():
    """tk.qgemm with M==1 must dispatch to the GEMV kernel and match the oracle."""
    from tk.quant import quantize_q8_0, dequantize_q8_0
    N, K = 128, 128
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    x = rng.standard_normal((K, 1)).astype(np.float32)
    Wq = quantize_q8_0(W)
    got = qgemm(mx.array(Wq), mx.array(x).astype(mx.float16))   # M==1 -> qgemv
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    with np.errstate(all="ignore"):
        ref = dequantize_q8_0(Wq).astype(np.float32) @ x
    assert got.shape == (N, 1)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


@pytest.mark.parametrize("fmt", ["q4_0", "q6_K"])
def test_qgemv_float32_paths(fmt):
    """The fp32 decode specializations preserve fp32 activation/output dtype."""
    quantize, dequantize = QUANT_FORMATS[fmt]
    N, K = 67, 512
    rng = np.random.default_rng(91)
    weight = (0.2 * rng.standard_normal((N, K))).astype(np.float32)
    x = (0.3 * rng.standard_normal((K, 1))).astype(np.float32)
    packed = quantize(weight)
    got = qgemv(mx.array(packed), mx.array(x), format=fmt)
    mx.eval(got)
    ref = dequantize(packed).astype(np.float32) @ x
    assert got.dtype == mx.float32 and got.shape == (N, 1)
    np.testing.assert_allclose(np.array(got), ref, rtol=3e-4, atol=3e-4)


if __name__ == "__main__":
    for fmt in sorted(QUANT_FORMATS):
        for shp in SHAPES:
            test_qgemv(shp, fmt)
        print("ok", fmt)
