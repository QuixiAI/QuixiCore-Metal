"""Integer-accumulate prefill GEMM (W8A8, BitNet W2A8), M>1. Validated against the INTEGER oracle
(W_int8 @ x_int8 as int32, then * w_scale * a_scale) — the exact-int32 numerics this path provides,
distinct from the dequant-to-half (fp16-accumulate) prefill. Run from kernels/:
    python -m pytest qgemm_int/correctness/test_qgemm_int.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qgemm_w8a8, qgemm_w2a8
from tk.quant import quantize_w8a8, quantize_act_int8, quantize_bitnet, dequantize_bitnet


@pytest.mark.parametrize("nkm", [(64, 256, 32), (128, 512, 64)])
def test_qgemm_w8a8_vs_int_oracle(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    Wq, ws = quantize_w8a8(W)                                  # int8 (N,K), (N,)
    _, Xq, xs = quantize_act_int8(X)                           # int8 (K,M), (1,M)
    Xqt = np.ascontiguousarray(Xq.T)                           # (M,K) token-major
    asc = xs[0, :].astype(np.float16)                          # (M,)
    got = qgemm_w8a8(mx.array(Wq), mx.array(Xqt), mx.array(ws).astype(mx.float16), mx.array(asc))
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    iacc = (Wq.astype(np.int32) @ Xq.astype(np.int32)).astype(np.float32)   # (N,M)
    ref = iacc * ws[:, None] * xs[0, None, :]
    assert got.shape == (N, M)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


@pytest.mark.parametrize("nkm", [(64, 256, 32), (128, 512, 64)])
def test_qgemm_w2a8_vs_int_oracle(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    Wq = quantize_bitnet(W)
    _, Xq, xs = quantize_act_int8(X)
    Xqt = np.ascontiguousarray(Xq.T)
    asc = xs[0, :].astype(np.float16)
    got = qgemm_w2a8(mx.array(Wq), mx.array(Xqt), mx.array(asc))
    mx.eval(got)
    g = np.array(got.astype(mx.float32))
    ref = (dequantize_bitnet(Wq).astype(np.float32) @ Xq.astype(np.float32)) * xs[0, None, :]
    assert got.shape == (N, M)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


if __name__ == "__main__":
    test_qgemm_w8a8_vs_int_oracle((128, 512, 64))
    test_qgemm_w2a8_vs_int_oracle((128, 512, 64))
    print("ok")
