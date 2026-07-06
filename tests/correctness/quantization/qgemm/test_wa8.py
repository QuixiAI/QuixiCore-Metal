"""Parity tests for the W·A8 schemes (quantized weights AND quantized activations):
fp8 W8A8, int8 W8A8, int8 W4A8.

On Apple there is no int8/fp8 matmul, so A8 = snap activations to the 8-bit grid then run the
dequant-to-half GEMM. The oracle is the W·A8 fake-quant: dequantize(Wq) @ round_act(X). This
matches what a W8A8/W4A8 inference produces (the standard fake-quant reference), which is the
"parity" target. Run from kernels/:  python -m pytest qgemm/correctness/test_wa8.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qmm
from tk.quant import QUANT_FORMATS, ACT_FORMATS

# (name, weight_format, activation_dtype)
SCHEMES = [
    ("fp8_W8A8", "fp8_e4m3", "fp8"),
    ("int8_W8A8", "q8_0", "int8"),
    ("int8_W4A8", "q4_0", "int8"),
    ("gptq_W4A8", "kU4B8", "int8"),
]


@pytest.mark.parametrize("name,w_fmt,act", SCHEMES)
@pytest.mark.parametrize("shape", [(64, 256, 64), (128, 256, 128)])
def test_wa8(shape, name, w_fmt, act):
    quantize_w, dequantize_w = QUANT_FORMATS[w_fmt]
    N, K, M = shape
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    Wq = quantize_w(W)
    Xr = ACT_FORMATS[act](X)[0]                       # activation snapped to the 8-bit grid

    got = qmm(mx.array(Wq), mx.array(X), w_format=w_fmt, act=act)
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    with np.errstate(all="ignore"):
        ref = dequantize_w(Wq).astype(np.float32) @ Xr      # W·A8 fake-quant reference
    assert got.shape == (N, M)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 2e-2, f"{name} relative diff {rel}"


def test_activations_are_actually_8bit():
    """Sanity: int8 activation column has <=256 distinct values; fp8 <= 256 codes."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((512, 4)).astype(np.float32)
    for act in ("int8", "fp8"):
        Xr, codes, scale = ACT_FORMATS[act](X)
        assert Xr.shape == X.shape
        # per-column distinct quantized levels are bounded by the code width
        for m in range(X.shape[1]):
            assert len(np.unique(codes[:, m])) <= 256


if __name__ == "__main__":
    for nm, wf, a in SCHEMES:
        test_wa8((128, 256, 128), nm, wf, a)
        print("ok", nm)
