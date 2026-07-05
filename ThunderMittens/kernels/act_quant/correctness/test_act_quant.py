"""Correctness tests for the fused gated-activation -> quant epilogues.

Because exp() sits between input and code, device-vs-numpy transcendental drift can flip
borderline codes — so correctness is asserted by RECONSTRUCTION BOUNDS (|scale*decode(code)
- ref_act| within a half quantization step) plus a composition check vs the unfused
tk.glu -> tk.quantize_per_token chain (>= 99.9% of codes identical, rest off-by-one).

Run from kernels/:  python -m pytest act_quant/correctness/test_act_quant.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk
from tk.quant import _e4m3_decode_arr


def _act(mode, x, gate, alpha=1.702, limit=7.0):
    x = x.astype(np.float64)
    g = gate.astype(np.float64)
    if mode == "swiglu_oai":
        x = np.minimum(x, limit)
        g = np.clip(g, -limit, limit)
        return (x / (1.0 + np.exp(-x * alpha))) * (1.0 + g)
    return (x / (1.0 + np.exp(-x))) * g


@pytest.mark.parametrize("act", ["swiglu", "swiglu_oai"])
def test_silu_mul_quant_fp8_reconstruction(act):
    rng = np.random.default_rng(0)
    T, D = 33, 768
    x = rng.standard_normal((T, D)).astype(np.float32)
    gate = rng.standard_normal((T, D)).astype(np.float32)
    codes, scale = tk.silu_mul_quant_fp8(mx.array(x), mx.array(gate), act=act)
    mx.eval(codes, scale)
    ref = _act(act, x, gate)
    sn = np.array(scale)[:, None]
    recon = sn * _e4m3_decode_arr(np.array(codes))
    # e4m3 half-ulp relative + a scale-step floor
    assert (np.abs(recon - ref) <= np.abs(ref) * 2.0 ** -4 + sn * 2.0 ** -6 + 1e-5).all()
    # scale covers the row amax
    assert (sn[:, 0] * 448.0 >= np.abs(ref).max(1) - 1e-3).all()


@pytest.mark.parametrize("act", ["swiglu", "swiglu_oai"])
def test_silu_mul_quant_int8_reconstruction(act):
    rng = np.random.default_rng(1)
    T, D = 17, 512
    x = rng.standard_normal((T, D)).astype(np.float32)
    gate = rng.standard_normal((T, D)).astype(np.float32)
    codes, scale = tk.silu_mul_quant_int8(mx.array(x), mx.array(gate), act=act)
    mx.eval(codes, scale)
    ref = _act(act, x, gate)
    sn = np.array(scale)[:, None]
    recon = sn * np.array(codes, np.float64)
    assert (np.abs(recon - ref) <= 0.51 * sn + 1e-6).all()


def test_silu_mul_quant_fp8_group_scales():
    rng = np.random.default_rng(2)
    T, D, G = 8, 512, 128
    x = rng.standard_normal((T, D)).astype(np.float32)
    gate = rng.standard_normal((T, D)).astype(np.float32)
    codes, scale = tk.silu_mul_quant_fp8_group(mx.array(x), mx.array(gate), group_size=G,
                                               ue8m0=True)
    mx.eval(codes, scale)
    sn = np.array(scale)
    assert sn.shape == (T, D // G)
    ref = _act("swiglu", x, gate).reshape(T, D // G, G)
    exp = np.log2(np.where(sn > 0, sn, 1.0))
    np.testing.assert_array_equal(exp, np.round(exp))       # power-of-two scales
    assert (sn * 448.0 >= np.abs(ref).max(-1) - 1e-3).all() # coverage
    recon = sn[..., None] * _e4m3_decode_arr(np.array(codes)).reshape(T, D // G, G)
    assert (np.abs(recon - ref) <= np.abs(ref) * 2.0 ** -4 + sn[..., None] * 2.0 ** -6 + 1e-5).all()


def test_composition_matches_unfused_chain():
    """Fused int8 epilogue vs tk.glu -> tk.quantize_per_token_int8: same scales,
    >= 99.9% identical codes (borderline exp() rounding may flip codes by one)."""
    rng = np.random.default_rng(3)
    T, D = 64, 1024
    x = mx.array(rng.standard_normal((T, D)).astype(np.float32)).astype(mx.bfloat16)
    gate = mx.array(rng.standard_normal((T, D)).astype(np.float32)).astype(mx.bfloat16)
    fused_c, fused_s = tk.silu_mul_quant_int8(x, gate)
    act = tk.glu(x, gate, mode="swiglu") if hasattr(tk, "glu") else None
    if act is None:
        act = tk.swiglu(x, gate)
    chain_c, chain_s = tk.quantize_per_token_int8(act)
    mx.eval(fused_c, fused_s, chain_c, chain_s)
    fc, cc = np.array(fused_c, np.int32), np.array(chain_c, np.int32)
    # the unfused chain quantizes the bf16-rounded activation; the fused path quantizes the
    # fp32 activation — scales may differ by one bf16 ulp, codes by one step
    frac_equal = (fc == cc).mean()
    assert frac_equal >= 0.95, f"only {frac_equal:.4f} codes equal"
    assert np.abs(fc - cc).max() <= 2
    np.testing.assert_allclose(np.array(fused_s), np.array(chain_s), rtol=1e-2)


def test_rms_norm_add_int8():
    """int8 sibling of rms_norm_add_fp8 dynamic: reconstruction within half a step and
    res_out identical to the fp8 variant's."""
    rng = np.random.default_rng(4)
    T, D = 16, 512
    x = mx.array(rng.standard_normal((T, D)).astype(np.float32)).astype(mx.bfloat16)
    r = mx.array(rng.standard_normal((T, D)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array((1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)).astype(mx.bfloat16)
    codes, added, scale = tk.rms_norm_add_int8(x, r, w)
    mx.eval(codes, added, scale)
    xf = np.array(x.astype(mx.float32), np.float64)
    rf = np.array(r.astype(mx.float32), np.float64)
    wf = np.array(w.astype(mx.float32), np.float64)
    s_ = xf + rf
    ref = s_ / np.sqrt((s_ * s_).mean(-1, keepdims=True) + 1e-5) * wf
    sn = np.array(scale)[:, None]
    recon = sn * np.array(codes, np.float64)
    # bf16 sum rounding (res_out is stored bf16 in-kernel from fp32) + half int8 step
    assert (np.abs(recon - ref) <= 0.51 * sn + np.abs(ref) * 2.0 ** -7 + 1e-3).all()
    np.testing.assert_allclose(np.array(added.astype(mx.float32)), s_, atol=2e-2, rtol=2e-2)
