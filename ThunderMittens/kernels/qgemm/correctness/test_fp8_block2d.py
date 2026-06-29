"""fp8_block2d: storage-optimal fp8_block (codes-only weights + a separate (N/128,K/128) tile scale,
no per-row scale replication). Validated vs dequantize_fp8_block2d(codes,scale2d) @ X. Run from kernels/:
    python -m pytest qgemm/correctness/test_fp8_block2d.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qgemm_fp8_block2d
from tk.quant import quantize_fp8_block2d, dequantize_fp8_block2d


@pytest.mark.parametrize("nkm", [(128, 256, 64), (256, 512, 128)])   # N,K % 128 ; M % 32
def test_fp8_block2d(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    codes, scale2d = quantize_fp8_block2d(W)
    got = qgemm_fp8_block2d(mx.array(codes), mx.array(X).astype(mx.float16), mx.array(scale2d))
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    ref = dequantize_fp8_block2d(codes, scale2d).astype(np.float32) @ X
    assert got.shape == (N, M)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


if __name__ == "__main__":
    test_fp8_block2d((256, 512, 128))
    print("ok")
