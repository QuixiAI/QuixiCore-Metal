"""Correctness for the integer-path decode GEMVs (W8A8, BitNet W2A8).

These validate against the INTEGER oracle — (W_int8 @ x_int8) as int32, then * w_scale * a_scale —
NOT the dequant-to-half path. The int32 path and the half path produce genuinely different numbers
(int32 is exact; the half path has fp16 accumulation error), so they are NOT expected to agree to
half-tolerance — that gap is correct, not a bug. Run from kernels/:
    python -m pytest qgemv_int/correctness/test_qgemv_int.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qgemv_w8a8, qgemv_w2a8
from tk.quant import quantize_w8a8, quantize_act_int8, quantize_bitnet, dequantize_bitnet


@pytest.mark.parametrize("nk", [(64, 256), (128, 512), (256, 1024)])
def test_w8a8_vs_int_oracle(nk):
    N, K = nk
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, 1)).astype(np.float32)
    Wq, w_scale = quantize_w8a8(W)                       # int8 (N,K), float32 (N,)
    _, Xq, xs = quantize_act_int8(X)                     # int8 (K,1), scale (1,1)
    a_scale = float(xs[0, 0])
    got = qgemv_w8a8(mx.array(Wq), mx.array(Xq),
                     mx.array(w_scale).astype(mx.float16),
                     mx.array(np.array([a_scale], np.float16)))
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    # integer oracle: exact int32 accumulate, scales applied once at the end
    iacc = (Wq.astype(np.int32) @ Xq.astype(np.int32)).astype(np.float32)   # (N,1)
    ref = iacc * w_scale[:, None] * a_scale
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 2e-2, f"W8A8 vs int oracle rel {rel}"


@pytest.mark.parametrize("nk", [(64, 256), (128, 512)])
def test_w2a8_vs_int_oracle(nk):
    N, K = nk
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, 1)).astype(np.float32)
    Wq = quantize_bitnet(W)                              # packed 2-bit + per-group absmean
    _, Xq, xs = quantize_act_int8(X)
    a_scale = float(xs[0, 0])
    got = qgemv_w2a8(mx.array(Wq), mx.array(Xq), mx.array(np.array([a_scale], np.float16)))
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    # per-group int sums * absmean scale == dequantize(Wq) . Xq ; then * a_scale
    ref = (dequantize_bitnet(Wq).astype(np.float32) @ Xq.astype(np.float32)) * a_scale
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 2e-2, f"W2A8 vs int oracle rel {rel}"


if __name__ == "__main__":
    test_w8a8_vs_int_oracle((128, 512))
    test_w2a8_vs_int_oracle((128, 512))
    print("ok")
