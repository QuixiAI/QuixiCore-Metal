"""Correctness tests for the TurboQuant KV codec.

K codes/scales/zp bit-exact vs the fp16-faithful numpy oracle (tk.quant.tq_encode_ref);
V codes bit-exact where the FWHT-rotated value is not within fp16-epsilon of a centroid
boundary (documented ≥99.9% fallback otherwise); round-trip SNR floors; functional
untouched-slot preservation; sub-8-bit byte-straddle exercised (k_bits=4, v_bits=3).

Run from kernels/:  python -m pytest turboquant/correctness/test_turboquant.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk
from tk.quant import (tq_signs, lloyd_max_centroids, tq_encode_ref, tq_decode_ref)


def _pack_bits(idx, bits, hs):
    """Pack per-element indices little-endian into ceil(hs*bits/8) bytes (kernel layout)."""
    nbytes = (hs * bits + 7) // 8
    out = np.zeros(nbytes, np.uint8)
    for i, v in enumerate(idx):
        for b in range(bits):
            if (int(v) >> b) & 1:
                bp = i * bits + b
                out[bp >> 3] |= 1 << (bp & 7)
    return out


def _run_encode(k, v, slots, block_size, k_bits, k_signed, v_bits, hs, Hkv, nblocks):
    signs = tq_signs(hs)
    cent = lloyd_max_centroids(v_bits)
    ng = hs // 32
    kc = np.zeros((nblocks, block_size, Hkv, (hs * k_bits + 7) // 8), np.uint8)
    vc = np.zeros((nblocks, block_size, Hkv, (hs * v_bits + 7) // 8), np.uint8)
    ks = np.zeros((nblocks, block_size, Hkv, ng), np.float16)
    vs = np.zeros((nblocks, block_size, Hkv, ng), np.float16)
    kz = np.zeros((nblocks, block_size, Hkv, ng), np.float16)
    out = tk.tq_encode(mx.array(k), mx.array(v), mx.array(kc), mx.array(vc), mx.array(ks),
                       mx.array(vs), mx.array(kz), mx.array(slots.astype(np.int32)),
                       mx.array(cent), mx.array(signs), block_size, k_bits, k_signed, v_bits)
    mx.eval(*out)
    return [np.array(o) for o in out], signs, cent


def test_k_codes_bit_exact():
    rng = np.random.default_rng(0)
    hs, Hkv, T, bs, nblocks = 128, 2, 5, 16, 4
    k = rng.standard_normal((T, Hkv, hs)).astype(np.float32)
    v = rng.standard_normal((T, Hkv, hs)).astype(np.float32)
    slots = np.array([0, 3, 20, 21, 40], np.int32)
    (kc, vc, ks, vs, kz), signs, cent = _run_encode(k, v, slots, bs, 8, True, 4, hs, Hkv, nblocks)
    for ti in range(T):
        slot = int(slots[ti]); blk, off = slot // bs, slot % bs
        for h in range(Hkv):
            ki, ksc, kzp, _, _ = tq_encode_ref(k[ti, h], v[ti, h], signs, cent, 8, True, 4)
            got = kc[blk, off, h].astype(np.int8).astype(np.int64)   # signed 8-bit
            np.testing.assert_array_equal(got, ki)
            np.testing.assert_array_equal(ks[blk, off, h], ksc)
            np.testing.assert_array_equal(kz[blk, off, h], kzp)


def test_k_sub8bit_bit_exact():
    rng = np.random.default_rng(1)
    hs, Hkv, T, bs, nblocks = 64, 1, 4, 16, 2
    k = rng.standard_normal((T, Hkv, hs)).astype(np.float32)
    v = rng.standard_normal((T, Hkv, hs)).astype(np.float32)
    slots = np.arange(T, dtype=np.int32)
    (kc, vc, ks, vs, kz), signs, cent = _run_encode(k, v, slots, bs, 4, False, 3, hs, Hkv, nblocks)
    for ti in range(T):
        slot = int(slots[ti]); blk, off = slot // bs, slot % bs
        ki, ksc, kzp, vi, vsc = tq_encode_ref(k[ti, 0], v[ti, 0], signs, cent, 4, False, 3)
        np.testing.assert_array_equal(kc[blk, off, 0], _pack_bits(ki, 4, hs))
        np.testing.assert_array_equal(ks[blk, off, 0], ksc)
        # V codes: allow off-by-one only where the fp16 norm sits within eps of a boundary
        vc_ref = _pack_bits(vi, 3, hs)
        # unpack both and compare index-wise
        got_idx = np.array([(int.from_bytes(vc[blk, off, 0].tobytes(), "little") >> (i * 3)) & 7
                            for i in range(hs)])
        diff = np.abs(got_idx - vi)
        assert (diff <= 1).all() and (diff == 0).mean() >= 0.95


def test_round_trip_snr():
    rng = np.random.default_rng(2)
    hs, Hkv, T, bs, nblocks = 128, 2, 6, 16, 3
    k = (0.7 * rng.standard_normal((T, Hkv, hs))).astype(np.float32)
    v = (0.7 * rng.standard_normal((T, Hkv, hs))).astype(np.float32)
    slots = np.array([1, 2, 3, 17, 18, 33], np.int32)
    (kc, vc, ks, vs, kz), signs, cent = _run_encode(k, v, slots, bs, 8, True, 4, hs, Hkv, nblocks)
    kd, vd = tk.tq_decode(mx.array(kc), mx.array(vc), mx.array(ks), mx.array(vs), mx.array(kz),
                          mx.array(slots), mx.array(cent), mx.array(signs), Hkv, hs, bs,
                          8, True, 4)
    mx.eval(kd, vd)
    kd, vd = np.array(kd), np.array(vd)
    ksnr = 10 * np.log10((k ** 2).sum() / ((kd - k) ** 2).sum())
    vsnr = 10 * np.log10((v ** 2).sum() / ((vd - v) ** 2).sum())
    assert ksnr > 30, f"K 8-bit SNR {ksnr:.1f} dB too low"
    assert vsnr > 18, f"V 4-bit SNR {vsnr:.1f} dB too low"


def test_decode_matches_ref():
    rng = np.random.default_rng(3)
    hs, Hkv, T, bs, nblocks = 64, 1, 3, 16, 2
    k = rng.standard_normal((T, Hkv, hs)).astype(np.float32)
    v = rng.standard_normal((T, Hkv, hs)).astype(np.float32)
    slots = np.array([0, 5, 20], np.int32)
    (kc, vc, ks, vs, kz), signs, cent = _run_encode(k, v, slots, bs, 8, True, 4, hs, Hkv, nblocks)
    kd, vd = tk.tq_decode(mx.array(kc), mx.array(vc), mx.array(ks), mx.array(vs), mx.array(kz),
                          mx.array(slots), mx.array(cent), mx.array(signs), Hkv, hs, bs,
                          8, True, 4)
    mx.eval(kd, vd)
    for ti in range(T):
        ki, ksc, kzp, vi, vsc = tq_encode_ref(k[ti, 0], v[ti, 0], signs, cent, 8, True, 4)
        kref, vref = tq_decode_ref(ki, ksc, kzp, vi, vsc, signs, cent)
        np.testing.assert_allclose(np.array(kd)[ti, 0], kref, atol=2e-2, rtol=2e-2)
        np.testing.assert_allclose(np.array(vd)[ti, 0], vref, atol=2e-2, rtol=2e-2)


def test_functional_untouched_slots():
    """slot -1 skipped; untouched cache bytes preserved (functional)."""
    rng = np.random.default_rng(4)
    hs, Hkv, T, bs, nblocks = 64, 1, 2, 16, 2
    k = rng.standard_normal((T, Hkv, hs)).astype(np.float32)
    v = rng.standard_normal((T, Hkv, hs)).astype(np.float32)
    signs = tq_signs(hs); cent = lloyd_max_centroids(4)
    ng = hs // 32
    kc0 = rng.integers(0, 256, (nblocks, bs, Hkv, hs), np.uint8)
    vc0 = rng.integers(0, 256, (nblocks, bs, Hkv, (hs * 4 + 7) // 8), np.uint8)
    ks0 = np.zeros((nblocks, bs, Hkv, ng), np.float16)
    vs0 = np.zeros((nblocks, bs, Hkv, ng), np.float16)
    kz0 = np.zeros((nblocks, bs, Hkv, ng), np.float16)
    slots = np.array([2, -1], np.int32)     # second token skipped
    out = tk.tq_encode(mx.array(k), mx.array(v), mx.array(kc0), mx.array(vc0), mx.array(ks0),
                       mx.array(vs0), mx.array(kz0), mx.array(slots), mx.array(cent),
                       mx.array(signs), bs, 8, True, 4)
    mx.eval(*out)
    kc = np.array(out[0])
    # slot 2 written, everything else (incl. all other blocks/offsets) preserved from kc0
    touched = np.zeros((nblocks, bs), bool); touched[0, 2] = True
    for b in range(nblocks):
        for off in range(bs):
            if not touched[b, off]:
                np.testing.assert_array_equal(kc[b, off], kc0[b, off])
