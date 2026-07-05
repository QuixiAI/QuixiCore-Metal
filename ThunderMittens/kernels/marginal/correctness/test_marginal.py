"""Correctness tests for the marginal layout/bit utilities.

packbits vs np.packbits (exact); permute_cols vs x[:, perm] (exact); tau_tail vs a numpy
transcription of the tanh(gate)+tau_pos scaling; segment_packbits vs per-row np.packbits.

Run from kernels/:  python -m pytest marginal/correctness/test_marginal.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk


def test_packbits_big_and_little():
    rng = np.random.default_rng(0)
    for n in [8, 17, 100, 257]:
        x = (rng.random(n) > 0.5).astype(np.uint8)
        for big in [True, False]:
            got = tk.packbits(mx.array(x), bit_order_big=big)
            mx.eval(got)
            ref = np.packbits(x, bitorder="big" if big else "little")
            np.testing.assert_array_equal(np.array(got), ref)


def test_packbits_2d_flattened():
    rng = np.random.default_rng(1)
    x = (rng.random((5, 16)) > 0.5).astype(np.uint8)
    got = tk.packbits(mx.array(x))
    mx.eval(got)
    np.testing.assert_array_equal(np.array(got), np.packbits(x.reshape(-1), bitorder="big"))


def test_permute_cols():
    rng = np.random.default_rng(2)
    rows, cols = 7, 64
    x = rng.integers(0, 60000, (rows, cols)).astype(np.uint16)
    perm = rng.permutation(cols).astype(np.int32)
    got = tk.permute_cols(mx.array(x.view(np.int16)), mx.array(perm))
    mx.eval(got)
    np.testing.assert_array_equal(np.array(got).view(np.uint16), x[:, perm])


def test_permute_cols_bf16():
    rng = np.random.default_rng(3)
    rows, cols = 4, 128
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    xm = mx.array(x).astype(mx.bfloat16)
    perm = rng.permutation(cols).astype(np.int32)
    got = tk.permute_cols(xm, mx.array(perm))
    mx.eval(got)
    ref = np.array(xm.astype(mx.float32))[:, perm]
    np.testing.assert_array_equal(np.array(got.astype(mx.float32)), ref)


def test_tau_tail():
    rng = np.random.default_rng(4)
    T, n_heads, head_dim = 6, 4, 32
    q_dim = n_heads * head_dim
    qkv = rng.standard_normal((T, 3 * q_dim)).astype(np.float32)
    gate = rng.standard_normal((T, 2 * n_heads)).astype(np.float32)
    max_pos = 40
    tau = rng.standard_normal((max_pos, n_heads)).astype(np.float32)
    pos = rng.integers(0, max_pos, T).astype(np.int32)
    got = tk.tau_tail(mx.array(qkv), mx.array(gate), mx.array(tau), mx.array(pos),
                      n_heads, head_dim)
    mx.eval(got)
    ref = qkv.astype(np.float64).copy()
    for t in range(T):
        for h in range(n_heads):
            tq = np.tanh(gate[t, h])
            tv = np.tanh(gate[t, n_heads + h])
            tp = tau[pos[t], h]
            qs, vs = tq + tp, tv + tp
            qsl = slice(h * head_dim, (h + 1) * head_dim)
            ref[t, qsl] *= qs                                       # Q slice
            ref[t, 2 * q_dim + h * head_dim: 2 * q_dim + (h + 1) * head_dim] *= vs  # V slice
    np.testing.assert_allclose(np.array(got), ref, atol=1e-4, rtol=1e-5)
    # K slice (middle third) untouched
    np.testing.assert_allclose(np.array(got)[:, q_dim:2 * q_dim], qkv[:, q_dim:2 * q_dim],
                               atol=1e-6)


def test_segment_packbits():
    rng = np.random.default_rng(5)
    lens = [10, 3, 17, 8]
    x = (rng.random(sum(lens)) > 0.5).astype(np.uint8)
    in_ind = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    out_lens = [(l + 7) // 8 for l in lens]
    out_ind = np.concatenate([[0], np.cumsum(out_lens)]).astype(np.int32)
    total = int(out_ind[-1])
    got = tk.segment_packbits(mx.array(x), mx.array(in_ind), mx.array(out_ind), total)
    mx.eval(got)
    gn = np.array(got)
    for i, l in enumerate(lens):
        seg = x[in_ind[i]:in_ind[i] + l]
        ref = np.packbits(seg, bitorder="big")
        np.testing.assert_array_equal(gn[out_ind[i]:out_ind[i] + out_lens[i]], ref)
