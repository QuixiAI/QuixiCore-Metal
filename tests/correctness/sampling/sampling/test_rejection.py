"""Correctness tests for the vLLM v1 ragged rejection samplers.

Each kernel is compared against a direct python transcription of the reference loop — exact
integer token ids. cu_num_draft_tokens is (B+1,) with a leading 0.

Run from kernels/:  python -m pytest sampling/correctness/test_rejection.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk


def _greedy_ref(cu, draft, targ, bonus, S1, is_greedy):
    B = len(cu) - 1
    out = -np.ones((B, S1), np.int64)
    for r in range(B):
        if is_greedy is not None and is_greedy[r] == 0:
            continue
        st, nd = int(cu[r]), int(cu[r + 1]) - int(cu[r])
        rej = False
        for pos in range(nd):
            t = int(targ[st + pos])
            out[r, pos] = t
            if int(draft[st + pos]) != t:
                rej = True
                break
        if not rej:
            out[r, nd] = int(bonus[r])
    return out


def _random_ref(cu, draft, dp, tp, bonus, rec, unif, S1, no_dp, is_greedy):
    B, V = len(cu) - 1, tp.shape[1]
    out = -np.ones((B, S1), np.int64)
    for r in range(B):
        if is_greedy is not None and is_greedy[r] != 0:
            continue
        st, nd = int(cu[r]), int(cu[r + 1]) - int(cu[r])
        rej = False
        for pos in range(nd):
            ti = st + pos
            did = int(draft[ti])
            p = float(tp[ti, did])
            q = 1.0 if no_dp else float(dp[ti, did])
            ratio = p / q if q > 0 else 0.0
            if ratio >= float(unif[ti]):
                out[r, pos] = did
            else:
                out[r, pos] = int(rec[ti])
                rej = True
                break
        if not rej:
            out[r, nd] = int(bonus[r])
    return out


def _recovered_ref(cu, draft, dp, tp, iq, no_dp):
    B, V = len(cu) - 1, tp.shape[1]
    total = len(draft)
    out = np.zeros(total, np.int64)
    for ti in range(total):
        r = int(np.searchsorted(cu[1:], ti, side="right"))
        did = int(draft[ti])
        if no_dp:
            prob = tp[ti].copy(); prob[did] = 0.0
            prob = np.maximum(prob, 0.0)
        else:
            prob = np.maximum(tp[ti] - dp[ti], 0.0)
        val = prob * iq[r]
        out[ti] = int(np.argmax(val))       # numpy argmax picks first max (smaller id)
    return out


def _setup(rng, B, V, max_draft):
    lens = rng.integers(1, max_draft + 1, B)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    total = int(cu[-1])
    draft = rng.integers(0, V, total).astype(np.int32)
    tp = rng.random((total, V)).astype(np.float32); tp /= tp.sum(1, keepdims=True)
    dp = rng.random((total, V)).astype(np.float32); dp /= dp.sum(1, keepdims=True)
    bonus = rng.integers(0, V, B).astype(np.int32)
    return cu, total, draft, tp, dp, bonus


def test_rejection_greedy():
    rng = np.random.default_rng(0)
    B, V, S = 8, 200, 5
    cu, total, draft, tp, dp, bonus = _setup(rng, B, V, S)
    targ = rng.integers(0, V, total).astype(np.int32)
    # force some drafts to match target_argmax so acceptance varies
    for i in range(0, total, 3):
        draft[i] = targ[i]
    is_greedy = (rng.random(B) > 0.3).astype(np.uint8)
    got = tk.rejection_greedy_sample(mx.array(cu), mx.array(draft), mx.array(targ),
                                     mx.array(bonus), S, is_greedy=mx.array(is_greedy))
    mx.eval(got)
    ref = _greedy_ref(cu, draft, targ, bonus, S + 1, is_greedy)
    np.testing.assert_array_equal(np.array(got), ref)


@pytest.mark.parametrize("no_dp", [False, True])
def test_rejection_random(no_dp):
    rng = np.random.default_rng(1)
    B, V, S = 8, 200, 5
    cu, total, draft, tp, dp, bonus = _setup(rng, B, V, S)
    rec = rng.integers(0, V, total).astype(np.int32)
    unif = rng.random(total).astype(np.float32)
    is_greedy = (rng.random(B) > 0.5).astype(np.uint8)
    got = tk.rejection_random_sample(mx.array(cu), mx.array(draft), mx.array(tp), mx.array(bonus),
                                     mx.array(rec), mx.array(unif), S,
                                     draft_probs=None if no_dp else mx.array(dp),
                                     is_greedy=mx.array(is_greedy))
    mx.eval(got)
    ref = _random_ref(cu, draft, dp, tp, bonus, rec, unif, S + 1, no_dp, is_greedy)
    np.testing.assert_array_equal(np.array(got), ref)


@pytest.mark.parametrize("no_dp", [False, True])
def test_sample_recovered(no_dp):
    rng = np.random.default_rng(2)
    B, V, S = 6, 300, 4
    cu, total, draft, tp, dp, bonus = _setup(rng, B, V, S)
    iq = rng.random((B, V)).astype(np.float32) + 0.5      # positive noise
    got = tk.sample_recovered_tokens(mx.array(cu), mx.array(draft), mx.array(tp), mx.array(iq),
                                     draft_probs=None if no_dp else mx.array(dp))
    mx.eval(got)
    ref = _recovered_ref(cu, draft, dp, tp, iq, no_dp)
    np.testing.assert_array_equal(np.array(got), ref)


def test_recovered_feeds_random():
    """The two-kernel pipeline: sample_recovered_tokens -> rejection_random_sample."""
    rng = np.random.default_rng(3)
    B, V, S = 8, 256, 5
    cu, total, draft, tp, dp, bonus = _setup(rng, B, V, S)
    unif = rng.random(total).astype(np.float32)
    iq = rng.random((B, V)).astype(np.float32) + 0.5
    rec = tk.sample_recovered_tokens(mx.array(cu), mx.array(draft), mx.array(tp), mx.array(iq),
                                     draft_probs=mx.array(dp))
    mx.eval(rec)
    out = tk.rejection_random_sample(mx.array(cu), mx.array(draft), mx.array(tp), mx.array(bonus),
                                     rec, mx.array(unif), S, draft_probs=mx.array(dp))
    mx.eval(out)
    rec_ref = _recovered_ref(cu, draft, dp, tp, iq, False)
    ref = _random_ref(cu, draft, dp, tp, bonus, rec_ref, unif, S + 1, False, None)
    np.testing.assert_array_equal(np.array(out), ref)
