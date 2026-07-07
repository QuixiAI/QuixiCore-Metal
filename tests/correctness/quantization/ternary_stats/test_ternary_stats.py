import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import code_flip_count, ternary_stats
from tk.quant import quantize_bitnet


def _unpack_bitnet_codes(wq):
    leading = wq.shape[:-2]
    rows = int(np.prod(leading, dtype=np.int64))
    nblocks = wq.shape[-2]
    flat = wq.reshape(rows, nblocks, 10)
    out = np.empty((rows, nblocks * 32), dtype=np.int8)
    for r in range(rows):
        for b in range(nblocks):
            for byte in range(8):
                v = int(flat[r, b, 2 + byte])
                for j in range(4):
                    out[r, b * 32 + byte * 4 + j] = ((v >> (2 * j)) & 3) - 1
    return out


def test_ternary_stats_counts_codes():
    rng = np.random.default_rng(83)
    w = (0.3 * rng.standard_normal((5, 96))).astype(np.float32)
    wq = quantize_bitnet(w)
    got = ternary_stats(mx.array(wq))
    mx.eval(got)
    codes = _unpack_bitnet_codes(wq)
    ref = np.stack([(codes == -1).sum(axis=1), (codes == 0).sum(axis=1),
                    (codes == 1).sum(axis=1)], axis=1).astype(np.int32)
    np.testing.assert_array_equal(np.array(got), ref)


def test_code_flip_count_counts_changed_codes():
    rng = np.random.default_rng(84)
    wq_a = quantize_bitnet((0.3 * rng.standard_normal((4, 64))).astype(np.float32))
    wq_b = wq_a.copy()
    wq_b[0, 0, 2] ^= np.uint8(0b00000011)
    wq_b[1, 1, 5] ^= np.uint8(0b11110000)
    got = code_flip_count(mx.array(wq_a), mx.array(wq_b))
    mx.eval(got)
    ref = (_unpack_bitnet_codes(wq_a) != _unpack_bitnet_codes(wq_b)).sum(axis=1).astype(np.int32)
    np.testing.assert_array_equal(np.array(got), ref)
