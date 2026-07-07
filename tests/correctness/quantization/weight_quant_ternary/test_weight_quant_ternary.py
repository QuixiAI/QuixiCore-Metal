import mlx.core as mx
import numpy as np
import pytest

from tk import weight_quant_ternary, weight_quant_ternary_pt
from tk.quant import dequantize_bitnet, quantize_bitnet

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _as_device_np(x, dtype):
    xd = mx.array(x).astype(_MX[dtype])
    return xd, np.array(xd.astype(mx.float32))


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(32, 64), (65, 128)])
def test_weight_quant_ternary_group32_matches_bitnet_packer(dtype, shape):
    rng = np.random.default_rng(100 + shape[0])
    w_host = (0.05 * rng.standard_normal(shape)).astype(np.float32)
    w, w_np = _as_device_np(w_host, dtype)
    wq, w_deq = weight_quant_ternary(w, group_k=32)
    mx.eval(wq, w_deq)

    got = np.array(wq)
    ref = quantize_bitnet(w_np)
    np.testing.assert_array_equal(got[:, :, 2:], ref[:, :, 2:])
    got_scale = np.ascontiguousarray(got[:, :, :2]).reshape(shape[0], -1).view(np.float16)
    ref_scale = np.ascontiguousarray(ref[:, :, :2]).reshape(shape[0], -1).view(np.float16)
    np.testing.assert_allclose(got_scale.astype(np.float32), ref_scale.astype(np.float32),
                               rtol=2 ** -10, atol=0)
    np.testing.assert_allclose(np.array(w_deq.astype(mx.float32)), dequantize_bitnet(got),
                               rtol=1 / 128, atol=1e-6)


def test_weight_quant_ternary_coarser_group_and_zero():
    rng = np.random.default_rng(101)
    w_host = (0.04 * rng.standard_normal((16, 128))).astype(np.float32)
    w = mx.array(w_host)
    wq, w_deq = weight_quant_ternary(w, group_k=64)
    mx.eval(wq, w_deq)
    got = np.array(wq)
    scales = got[:, :, :2].reshape(16, -1).view(np.float16).astype(np.float32)
    np.testing.assert_array_equal(scales[:, 0::2], scales[:, 1::2])

    wg = w_host.reshape(16, 2, 64)
    s = np.maximum(np.abs(wg).mean(axis=-1), 1e-5).astype(np.float32)
    q = np.clip(np.rint(wg / s[..., None]), -1, 1)
    ref = (q * s.astype(np.float16).astype(np.float32)[..., None]).reshape(16, 128)
    np.testing.assert_allclose(np.array(w_deq.astype(mx.float32)), ref, rtol=1 / 128, atol=1e-6)

    zq, zd = weight_quant_ternary(mx.zeros((8, 64), dtype=mx.float32))
    mx.eval(zq, zd)
    assert np.all(np.array(zd.astype(mx.float32)) == 0.0)
    codes = np.array(zq)[:, :, 2:]
    assert np.all(np.stack([(codes >> (2 * j)) & 3 for j in range(4)]) == 1)


def test_weight_quant_ternary_pt_2d_and_batched():
    rng = np.random.default_rng(102)
    w_host = (0.05 * rng.standard_normal((3, 16, 64))).astype(np.float32)
    w = mx.array(w_host)
    wq, w_deq = weight_quant_ternary_pt(w)
    mx.eval(wq, w_deq)

    got = np.array(wq)
    deq = np.array(w_deq.astype(mx.float32))
    for e in range(3):
        s = max(float(np.abs(w_host[e]).mean()), 1e-5)
        scales = got[e, :, :, :2].reshape(16, -1).view(np.float16).astype(np.float32)
        assert np.unique(scales).size == 1
        np.testing.assert_allclose(scales[0, 0], np.float16(s).astype(np.float32),
                                   rtol=2 ** -10, atol=0)
        q = np.clip(np.rint(w_host[e] / s), -1, 1)
        ref = q * np.float16(s).astype(np.float32)
        np.testing.assert_allclose(deq[e], ref, rtol=1 / 64, atol=1e-6)
