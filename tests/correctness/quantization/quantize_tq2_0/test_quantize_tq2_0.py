import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tk import quantize_tq2_0


def _round_half_away(x):
    return np.sign(x).astype(np.int32) * np.floor(np.abs(x) + 0.5).astype(np.int32)


def _quantize_tq2_0_ref(w):
    leading = w.shape[:-1]
    K = w.shape[-1]
    rows = int(np.prod(leading, dtype=np.int64))
    flat = w.reshape(rows, K).astype(np.float32)
    nblocks = K // 256
    wq = np.zeros((rows, nblocks, 66), dtype=np.uint8)
    w_deq = np.zeros_like(flat, dtype=np.float32)
    for r in range(rows):
        for b in range(nblocks):
            block = flat[r, b * 256:(b + 1) * 256]
            d = float(np.max(np.abs(block)))
            dh = np.float16(d)
            if d == 0.0:
                xi = np.zeros(256, dtype=np.int32)
            else:
                xi = _round_half_away(block / d)
            for j in range(2):
                for m in range(32):
                    q = 0
                    for n in range(4):
                        col = 128 * j + 32 * n + m
                        q |= int(xi[col] + 1) << (2 * n)
                    wq[r, b, 32 * j + m] = q
            wq[r, b, 64:66] = np.array([dh], dtype=np.float16).view(np.uint8)
            w_deq[r, b * 256:(b + 1) * 256] = np.float32(dh) * xi.astype(np.float32)
    return wq.reshape(*leading, nblocks, 66), w_deq.reshape(w.shape)


def _bf16_round(x):
    return np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))


@pytest.mark.parametrize("shape", [(4, 256), (2, 3, 512)])
def test_quantize_tq2_0_matches_reference(shape):
    rng = np.random.default_rng(77 + len(shape))
    w_np = (0.2 * rng.standard_normal(shape)).astype(np.float32)
    wq, w_deq = quantize_tq2_0(mx.array(w_np).astype(mx.float32))
    mx.eval(wq, w_deq)
    ref_q, ref_deq = _quantize_tq2_0_ref(w_np)
    np.testing.assert_array_equal(np.array(wq), ref_q)
    np.testing.assert_allclose(np.array(w_deq.astype(mx.float32)), _bf16_round(ref_deq), rtol=0, atol=0)


def test_quantize_tq2_0_zero_block():
    w = mx.zeros((2, 256), dtype=mx.float32)
    wq, w_deq = quantize_tq2_0(w)
    mx.eval(wq, w_deq)
    q = np.array(wq)
    assert np.all(q[:, :, :64] == 0x55)
    assert np.all(q[:, :, 64:66] == 0)
    assert np.all(np.array(w_deq.astype(mx.float32)) == 0.0)
