"""GPTQ act-order (desc_act): the weight is quantized in g_idx-permuted K order so its groups are
contiguous; tk.qgemm_actorder gathers the activations by the same permutation and runs the standard
qgemm. The permutation cancels, recovering W@X (up to quant error). Run from kernels/:
    python -m pytest qgemm/correctness/test_actorder.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import qgemm_actorder
from tk.quant import quantize_kU4B8, dequantize_kU4B8


@pytest.mark.parametrize("nkm", [(64, 256, 64), (128, 512, 128)])
def test_actorder_kU4B8(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    perm = rng.permutation(K)                              # g_idx-derived column permutation
    Wq = quantize_kU4B8(W[:, perm])                        # weight stored in permuted (grouped) order
    got = qgemm_actorder(mx.array(Wq), mx.array(X).astype(mx.float16), perm, w_format="kU4B8")
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    # oracle: dequantize(permuted weight) @ permuted activations (isolates quant error)
    ref = dequantize_kU4B8(Wq).astype(np.float32) @ X[perm]
    assert got.shape == (N, M)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 2e-2, f"act-order vs oracle rel {rel}"
    # and the permutation cancels: result approximates the original (unpermuted) W @ X
    approx = W @ X
    assert np.abs(g - approx).max() / (np.abs(approx).max() + 1e-9) < 0.25  # + int4 quant error


@pytest.mark.parametrize("nkm", [(64, 256, 64), (128, 512, 128)])
def test_actorder_fused_in_kernel(nkm):
    """fused=True gathers X K-rows inside the kernel (qgemm_actorder_k) — no permuted-X copy."""
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    perm = rng.permutation(K)
    Wq = quantize_kU4B8(W[:, perm])
    got = qgemm_actorder(mx.array(Wq), mx.array(X).astype(mx.float16), perm, w_format="kU4B8", fused=True)
    mx.eval(got)
    g = np.array(got).astype(np.float32)
    ref = dequantize_kU4B8(Wq).astype(np.float32) @ X[perm]      # in-kernel gather == host gather
    assert got.shape == (N, M)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


if __name__ == "__main__":
    test_actorder_kU4B8((128, 512, 128))
    print("ok")
