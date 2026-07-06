"""fp8 rank-1 scaled GEMM: both operands fp8 e4m3, output scaled by w_scale[n] (per-channel) *
a_scale[m] (per-token) — the fp8 analog of W8A8/SmoothQuant (TK gemm/fp8_h100_scaled). Validated vs
the rank-1 oracle. Run from kernels/:
    python -m pytest qgemm/correctness/test_fp8_scaled.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qgemm_fp8_scaled
from tk.quant import quantize_fp8_scaled, quantize_act_fp8, _e4m3_decode_arr


@pytest.mark.parametrize("nkm", [(64, 256, 32), (128, 512, 128), (256, 256, 64)])
def test_qgemm_fp8_scaled(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    wq, w_scale = quantize_fp8_scaled(W)                       # codes (N,K), scale (N,)
    _, xq, s = quantize_act_fp8(X)                             # codes (K,M), s (1,M)
    a_scale = s[0, :].astype(np.float16)                       # (M,)
    got = qgemm_fp8_scaled(mx.array(wq), mx.array(xq),
                           mx.array(w_scale), mx.array(a_scale))
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    dW = _e4m3_decode_arr(wq).astype(np.float32)               # (N,K)
    dX = _e4m3_decode_arr(xq).astype(np.float32)               # (K,M)
    ref = w_scale.astype(np.float32)[:, None] * a_scale.astype(np.float32)[None, :] * (dW @ dX)
    assert got.shape == (N, M)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


if __name__ == "__main__":
    test_qgemm_fp8_scaled((128, 512, 128))
    print("ok")
