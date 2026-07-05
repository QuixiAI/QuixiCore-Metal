"""Correctness tests for the EAGLE spec-decode input-prep metadata builders.

Each kernel vs a direct python transcription of the reference loop (exact int32 outputs).
cu_* are (B+1,) with a leading 0.

Run from kernels/:  python -m pytest sampling/correctness/test_eagle.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk


def test_eagle_prepare_inputs_padded():
    rng = np.random.default_rng(0)
    B, S = 8, 5
    lens = rng.integers(0, S + 1, B)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    valid = np.array([rng.integers(0, l + 2) for l in lens], np.int32)
    qlens = rng.integers(1, S + 2, B)
    qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    ti, nr = tk.eagle_prepare_inputs_padded(mx.array(cu), mx.array(valid), mx.array(qsl))
    mx.eval(ti, nr)
    ti_ref = np.zeros(B, np.int32); nr_ref = np.zeros(B, np.int32)
    for r in range(B):
        nd = int(cu[r + 1] - cu[r])
        rej = nd + 1 - int(valid[r]) if nd > 0 else 0
        nr_ref[r] = rej
        ti_ref[r] = int(qsl[r + 1]) - 1 - rej
    np.testing.assert_array_equal(np.array(ti), ti_ref)
    np.testing.assert_array_equal(np.array(nr), nr_ref)


def test_eagle_prepare_next_token_padded():
    rng = np.random.default_rng(1)
    B, ns, V = 8, 4, 200
    sampled = rng.integers(-1, V + 5, (B, ns)).astype(np.int32)   # some -1 / out-of-range
    discard = (rng.random(B) > 0.7).astype(np.uint8)
    backup = rng.integers(0, V, B).astype(np.int32)
    nt, vc = tk.eagle_prepare_next_token_padded(mx.array(sampled), mx.array(discard),
                                                mx.array(backup), V)
    mx.eval(nt, vc)
    nt_ref = np.zeros(B, np.int32); vc_ref = np.zeros(B, np.int32)
    for r in range(B):
        valid = 0; last = -1
        for pos in range(ns):
            tok = int(sampled[r, pos])
            if tok != -1 and tok < V:
                valid += 1; last = tok
        if discard[r] != 0:
            nt_ref[r] = int(backup[r]); vc_ref[r] = 0
        else:
            nt_ref[r] = last if valid > 0 else int(backup[r]); vc_ref[r] = valid
    np.testing.assert_array_equal(np.array(nt), nt_ref)
    np.testing.assert_array_equal(np.array(vc), vc_ref)


@pytest.mark.parametrize("ib_extra", [0, 3])
def test_eagle_step_slot_mapping_metadata(ib_extra):
    rng = np.random.default_rng(2)
    B, nblk, block_size, max_len = 6, 8, 16, 200
    ib = B + ib_extra
    pos = rng.integers(0, max_len + 5, B).astype(np.int32)   # some exceed max_len
    bt = rng.integers(0, 100, (B, nblk)).astype(np.int32)
    sl = rng.integers(1, max_len, B).astype(np.int32)
    pad_id = -1
    cp, sm, nsl = tk.eagle_step_slot_mapping_metadata(mx.array(pos), mx.array(bt), mx.array(sl),
                                                      block_size, max_len, pad_id,
                                                      input_batch_size=ib)
    mx.eval(cp, sm, nsl)
    sm_ref = np.zeros(ib, np.int32); cp_ref = np.zeros(ib, np.int32); nsl_ref = np.zeros(B, np.int32)
    for r in range(ib):
        if r >= B:
            sm_ref[r] = pad_id
            continue
        newp = int(pos[r]) + 1
        exceeds = newp >= max_len
        clamped = 0 if exceeds else newp
        cp_ref[r] = clamped
        bn = min(clamped // block_size, nblk - 1)
        slot = int(bt[r, bn]) * block_size + (clamped % block_size)
        sm_ref[r] = pad_id if exceeds else slot
        nsl_ref[r] = 1 if exceeds else min(int(sl[r]) + 1, max_len)
    np.testing.assert_array_equal(np.array(sm), sm_ref)
    # clamped positions + new_seq_lens only meaningful for r < B
    np.testing.assert_array_equal(np.array(cp)[:B], cp_ref[:B])
    np.testing.assert_array_equal(np.array(nsl), nsl_ref)


def test_eagle_expand_int32():
    rng = np.random.default_rng(3)
    B = 7
    lens = rng.integers(1, 6, B)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    total = int(cu[-1])
    inp = rng.integers(-1, 50, B).astype(np.int32)
    out = tk.eagle_expand_int32(mx.array(inp), mx.array(cu), total, replace_from=-1, replace_to=99)
    mx.eval(out)
    ref = np.zeros(total, np.int32)
    for r in range(B):
        v = 99 if int(inp[r]) == -1 else int(inp[r])
        ref[cu[r]:cu[r + 1]] = v
    np.testing.assert_array_equal(np.array(out), ref)
