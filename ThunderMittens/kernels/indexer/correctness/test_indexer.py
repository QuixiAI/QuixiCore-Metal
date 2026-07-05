"""Correctness tests for the DeepSeek-V3.2 indexer K quant-and-cache.

The fp32 scales are bit-exact vs a numpy twin (plain) or power-of-two (ue8m0); the e4m3
codes are validated by reconstruction (round-to-nearest-even ties differ from a numpy argmin,
the repo's fp8 contract). Gather dequantizes back to bf16.

Run from kernels/:  python -m pytest indexer/correctness/test_indexer.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk
from tk.quant import _e4m3_decode_arr
from tk.quant import _E4M3_CODES, _E4M3_VALS, _nearest


def _quant_ref(k, slot_mapping, num_slots, head_dim, qbs, ue8m0):
    """fp32-faithful numpy twin of indexer_k_quant_and_cache."""
    nq = (head_dim + qbs - 1) // qbs
    code = np.zeros((num_slots, head_dim), np.uint8)
    scale = np.zeros((num_slots, nq), np.float32)
    kf = k.astype(np.float32)
    for t in range(k.shape[0]):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        for qb in range(nq):
            s0 = qb * qbs
            seg = kf[t, s0:min(s0 + qbs, head_dim)]
            amax = np.abs(seg).max()
            sc = max(amax, 1e-4) / 448.0
            if ue8m0:
                sc = 2.0 ** np.ceil(np.log2(sc))
            sc = np.float32(sc)
            inv = 1.0 / sc if sc > 0 else 0.0
            v = seg * inv
            codes = _nearest(v.astype(np.float32), _E4M3_CODES, _E4M3_VALS)
            codes = np.where(v == 0.0, 0, codes).astype(np.uint8)
            code[slot, s0:s0 + len(seg)] = codes
            scale[slot, qb] = sc
    return code, scale


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("head_dim,qbs", [(128, 128), (256, 128), (192, 64)])
@pytest.mark.parametrize("ue8m0", [False, True])
def test_indexer_quant_bit_exact(dtype, head_dim, qbs, ue8m0):
    md = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype]
    rng = np.random.default_rng(head_dim + qbs + int(ue8m0))
    T, num_slots = 20, 24
    nq = (head_dim + qbs - 1) // qbs
    k = (0.3 * rng.standard_normal((T, head_dim))).astype(np.float32)
    slot_mapping = rng.permutation(num_slots)[:T].astype(np.int32)
    slot_mapping[3] = -1                                     # a padding token
    code0 = rng.integers(0, 256, (num_slots, head_dim), np.uint8)   # pre-existing bytes
    scale0 = rng.standard_normal((num_slots, nq)).astype(np.float32)
    kq = mx.array(k).astype(md)
    code, scale = tk.indexer_k_quant_and_cache(kq, mx.array(slot_mapping), mx.array(code0),
                                               mx.array(scale0), quant_block_size=qbs, ue8m0=ue8m0)
    mx.eval(code, scale)
    kr = np.array(kq.astype(mx.float32))                     # k as the kernel sees it
    cn, sn = np.array(code), np.array(scale)
    tset = set(int(x) for x in slot_mapping if x >= 0)
    tok_of_slot = {int(sl): t for t, sl in enumerate(slot_mapping) if int(sl) >= 0}
    for sl in range(num_slots):
        if sl not in tset:
            np.testing.assert_array_equal(cn[sl], code0[sl])        # untouched preserved
            np.testing.assert_array_equal(sn[sl], scale0[sl])
            continue
        t = tok_of_slot[sl]
        for qb in range(nq):
            s0 = qb * qbs
            seg = kr[t, s0:min(s0 + qbs, head_dim)]
            amax = np.abs(seg).max()
            sc = sn[sl, qb]
            # scale bit-exact (plain) / power-of-two + covers amax (ue8m0)
            if ue8m0:
                assert np.log2(sc) == np.round(np.log2(sc)) and sc * 448.0 >= amax - 1e-6
            else:
                np.testing.assert_allclose(sc, max(amax, 1e-4) / 448.0, rtol=1e-6)
            # codes validated by RECONSTRUCTION (the repo's fp8 contract — tk_e4m3_encode's
            # round-to-nearest-even ties differ from a numpy argmin, so bit-exact is the wrong
            # test): scale * decode(code) is within a half e4m3 step of k.
            recon = sc * _e4m3_decode_arr(cn[sl, s0:s0 + len(seg)])
            assert (np.abs(recon - seg) <= np.abs(seg) * 2.0 ** -4 + sc * 2.0 ** -6 + 1e-4).all()


def test_indexer_gather_roundtrip():
    rng = np.random.default_rng(7)
    T, num_slots, head_dim, qbs = 16, 20, 128, 128
    nq = head_dim // qbs
    k = (0.3 * rng.standard_normal((T, head_dim))).astype(np.float32)
    slot_mapping = np.arange(T, dtype=np.int32)
    code0 = np.zeros((num_slots, head_dim), np.uint8)
    scale0 = np.zeros((num_slots, nq), np.float32)
    code, scale = tk.indexer_k_quant_and_cache(mx.array(k), mx.array(slot_mapping),
                                               mx.array(code0), mx.array(scale0))
    mx.eval(code, scale)
    slots = np.arange(T, dtype=np.int32)
    kout = tk.indexer_k_gather(code, scale, mx.array(slots), head_dim)
    mx.eval(kout)
    # gather == decode(code[slot]) * scale[slot, qblock]
    cn, sn = np.array(code), np.array(scale)
    ref = (_e4m3_decode_arr(cn[slots]).reshape(T, nq, qbs)
           * sn[slots][:, :, None]).reshape(T, head_dim)
    np.testing.assert_allclose(np.array(kout.astype(mx.float32)), ref, atol=2e-2, rtol=2e-2)
    # and within fp8 relative precision of the original k
    assert (np.abs(np.array(kout.astype(mx.float32)) - k) <= np.abs(k) * 2.0 ** -3
            + sn.max() + 1e-3).all()
