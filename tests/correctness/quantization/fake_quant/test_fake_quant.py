import mlx.core as mx
import numpy as np
import pytest

from tk import fake_quant_int8, silu_mul_fake_quant_int8

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def _fake_ref(x):
    amax = np.abs(x).max(axis=-1)
    scale = (amax / 127.0).astype(np.float32)
    inv = np.where(scale > 0, 1.0 / scale, 0.0).astype(np.float32)
    codes = np.clip(np.rint(x * inv[..., None]), -127, 127).astype(np.int8)
    deq = codes.astype(np.float32) * scale.astype(np.float16).astype(np.float32)[..., None]
    return deq, codes, scale


def _swiglu(x, gate):
    return x / (1.0 + np.exp(-x)) * gate


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_fake_quant_int8_matches_one_pass_reference(dtype):
    rng = np.random.default_rng(110)
    x = (2.0 * rng.standard_normal((9, 128))).astype(np.float32)
    xm = mx.array(x).astype(_MX[dtype])
    x_post = np.array(xm.astype(mx.float32))
    x_q, codes, scale = fake_quant_int8(xm)
    mx.eval(x_q, codes, scale)

    ref_deq, ref_codes, ref_scale = _fake_ref(x_post)
    np.testing.assert_array_equal(np.array(codes), ref_codes)
    np.testing.assert_allclose(np.array(scale), ref_scale, rtol=1e-3, atol=1e-8)
    np.testing.assert_allclose(np.array(x_q.astype(mx.float32)), ref_deq, rtol=1 / 128, atol=1e-6)


def test_silu_mul_fake_quant_int8_matches_composition():
    rng = np.random.default_rng(111)
    x = mx.array(rng.standard_normal((7, 64)).astype(np.float32)).astype(mx.bfloat16)
    gate = mx.array(rng.standard_normal((7, 64)).astype(np.float32)).astype(mx.bfloat16)
    x_q, codes, scale = silu_mul_fake_quant_int8(x, gate)
    mx.eval(x_q, codes, scale)

    act = _swiglu(np.array(x.astype(mx.float32)), np.array(gate.astype(mx.float32)))
    ref_deq, ref_codes, ref_scale = _fake_ref(act)
    np.testing.assert_array_equal(np.array(codes), ref_codes)
    np.testing.assert_allclose(np.array(scale), ref_scale, rtol=1e-3, atol=1e-8)
    np.testing.assert_allclose(np.array(x_q.astype(mx.float32)), ref_deq, rtol=1 / 128, atol=1e-6)
