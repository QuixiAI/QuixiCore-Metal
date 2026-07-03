"""Correctness tests for the sampling kernels.

argmax (greedy): token index of the max logit over the vocab axis; ties resolve
to the smallest index (== numpy argmax first-occurrence).

Run from kernels/:  python -m pytest sampling/correctness/test_sampling.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import (argmax_sample, sample_categorical, top_k_sample, top_p_sample, apply_penalty,
                beam_advance, beam_reorder_kv, beam_build_copy_pairs, beam_length_penalty,
                spec_verify_linear, spec_compact, spec_update_kv_meta, spec_verify_tree,
                spec_build_tree_pointers, min_p_sample, typical_p_sample, apply_token_bitmask,
                apply_bad_words)


def _pack_bitmask(allow):
    """Pack a boolean (T, V) allow-mask into (T, ceil(V/32)) int32 words."""
    T, V = allow.shape
    nw = (V + 31) // 32
    m = np.zeros((T, nw), np.uint32)
    for t in range(T):
        for v in range(V):
            if allow[t, v]:
                m[t, v >> 5] |= np.uint32(1) << np.uint32(v & 31)
    return m.view(np.int32)


@pytest.mark.parametrize("B,BM", [(2, 3), (1, 4), (3, 2)])
def test_beam_reorder_kv(B, BM):
    rng = np.random.default_rng(B + BM)
    bs, H_KV, D, max_blocks = 4, 2, 32, 2
    nbeams = B * BM
    nb = nbeams * max_blocks
    kc = rng.standard_normal((nb, bs, H_KV, D)).astype(np.float32)
    vc = rng.standard_normal((nb, bs, H_KV, D)).astype(np.float32)
    bt = np.arange(nb, dtype=np.int32).reshape(nbeams, max_blocks)  # beam g owns [2g, 2g+1]
    # each new beam's parent is a random beam in its batch (allows fan-out / self-parent)
    pb = rng.integers(0, BM, size=(B, BM)).astype(np.int32)
    seq_lens = np.full(nbeams, 7, np.int32)   # ceil(7/4) = 2 blocks
    kc2, vc2 = beam_reorder_kv(mx.array(kc), mx.array(vc), mx.array(bt), mx.array(pb),
                               mx.array(seq_lens))
    mx.eval(kc2, vc2)
    ref_k, ref_v = kc.copy(), vc.copy()
    for b in range(B):
        for k in range(BM):
            p = pb[b, k]
            if p == k:
                continue
            for c in range(2):
                ref_k[bt[b * BM + k, c]] = kc[bt[b * BM + p, c]]
                ref_v[bt[b * BM + k, c]] = vc[bt[b * BM + p, c]]
    np.testing.assert_array_equal(np.array(kc2), ref_k)
    np.testing.assert_array_equal(np.array(vc2), ref_v)


def test_beam_reorder_kv_chain():
    """Reorder CHAIN: beam0<-beam1, beam1<-beam2 (a parent block that is also a copy destination).
    Correct semantics reads the ORIGINAL cache, so beam0 must get beam2's blocks, not the reordered
    beam1. Pins the read-from-original (race-free) copy against the read-from-clone hazard."""
    B, BM, bs, H_KV, D, max_blocks = 1, 3, 4, 1, 8, 1
    nbeams = B * BM
    rng = np.random.default_rng(0)
    kc = rng.standard_normal((nbeams, bs, H_KV, D)).astype(np.float32)
    vc = rng.standard_normal((nbeams, bs, H_KV, D)).astype(np.float32)
    bt = np.arange(nbeams, dtype=np.int32).reshape(nbeams, max_blocks)   # beam g owns block g
    pb = np.array([[1, 2, 2]], np.int32)     # 0<-1, 1<-2, 2 keeps itself
    seq_lens = np.full(nbeams, 3, np.int32)  # 1 block
    kc2, vc2 = beam_reorder_kv(mx.array(kc), mx.array(vc), mx.array(bt), mx.array(pb),
                               mx.array(seq_lens))
    mx.eval(kc2, vc2)
    ref_k, ref_v = kc.copy(), vc.copy()
    ref_k[0] = kc[1]; ref_k[1] = kc[2]       # both read the ORIGINAL cache
    ref_v[0] = vc[1]; ref_v[1] = vc[2]
    np.testing.assert_array_equal(np.array(kc2), ref_k)
    np.testing.assert_array_equal(np.array(vc2), ref_v)


@pytest.mark.parametrize("B,BM", [(2, 3), (3, 2)])
def test_beam_build_copy_pairs(B, BM):
    """The device pair builder emits the same (src,dst) set as the reference host loop."""
    max_blocks, block_size = 3, 4
    nbeams = B * BM
    rng = np.random.default_rng(B * 7 + BM)
    bt = np.arange(nbeams * max_blocks, dtype=np.int32).reshape(nbeams, max_blocks)
    pb = rng.integers(0, BM, size=(B, BM)).astype(np.int32)
    sl = rng.integers(1, max_blocks * block_size, size=nbeams).astype(np.int32)
    pairs = np.array(beam_build_copy_pairs(mx.array(pb), mx.array(bt), mx.array(sl), block_size))
    got = {(int(s), int(d)) for s, d in pairs if s >= 0 and d >= 0}
    want = set()
    for b in range(B):
        for k in range(BM):
            p = int(pb[b, k])
            if p == k:
                continue
            for c in range((int(sl[b * BM + k]) + block_size - 1) // block_size):
                want.add((int(bt[b * BM + p, c]), int(bt[b * BM + k, c])))
    assert got == want


@pytest.mark.parametrize("min_p", [0.1, 0.3, 0.7])
def test_min_p_sample(min_p):
    """Over many seeds, min-p only ever samples tokens with prob >= min_p * max_prob."""
    rng = np.random.default_rng(int(min_p * 100))
    V = 60
    logits = (rng.standard_normal(V) * 2).astype(np.float32)
    p = np.exp(logits - logits.max()); p /= p.sum()
    kept = set(np.where(p >= min_p * p.max())[0].tolist())
    x = mx.array(logits[None])
    seen = set()
    for s in range(2500):
        o = min_p_sample(x, min_p, seed=s)
        mx.eval(o)
        seen.add(int(np.array(o)[0]))
    assert seen <= kept
    assert int(np.argmax(logits)) in seen        # the top token is always kept


@pytest.mark.parametrize("V", [40, 70, 200])
def test_apply_token_bitmask(V):
    rng = np.random.default_rng(V)
    T = 4
    logits = rng.standard_normal((T, V)).astype(np.float32)
    allow = rng.integers(0, 2, size=(T, V)).astype(bool)
    allow[:, 0] = True                            # keep at least one token per row
    out = np.array(apply_token_bitmask(mx.array(logits), mx.array(_pack_bitmask(allow))))
    np.testing.assert_array_equal(out[allow], logits[allow])   # allowed logits untouched
    assert (out[~allow] < -1e30).all()                         # masked -> -inf sentinel


def _rng_uniform_np(seed, a, b):
    M = np.uint64(0xFFFFFFFF)
    x = (np.uint64(seed) * np.uint64(0x9E3779B9) + np.uint64(a) * np.uint64(0x85EBCA77)
         + np.uint64(b) * np.uint64(0xC2B2AE3D)) & M
    x = (x ^ (x >> np.uint64(16))) & M
    x = (x * np.uint64(0x7FEB352D)) & M
    x = (x ^ (x >> np.uint64(15))) & M
    x = (x * np.uint64(0x846CA68B)) & M
    x = (x ^ (x >> np.uint64(16))) & M
    return float(x >> np.uint64(8)) * (1.0 / 16777216.0)


def _tree_walk_ref(draft_b, tp_b, nxt_tok, nxt_sib, seed, b, N):
    last, num_acc, acc_idx, acc_tok, term, tried = 0, 0, [0], [], 0, []
    for j in range(1, N):
        fc = int(nxt_tok[last])
        if fc == -1:
            term = 1
            break
        coin = _rng_uniform_np(seed, b, j)
        pacc, accepted, child, tried = 0.0, False, fc, []
        while child != -1:
            tok = int(draft_b[child - 1])
            pacc += float(tp_b[last, tok])
            if coin <= pacc:
                acc_tok.append(tok); num_acc += 1; acc_idx.append(child); last = child; accepted = True
                break
            tried.append(tok); child = int(nxt_sib[child])
        if not accepted:
            term = 2
            break
    return acc_idx, acc_tok, num_acc, term, last, tried


@pytest.mark.parametrize("seed", [1, 7, 42])
def test_spec_verify_tree(seed):
    # A fixed 7-node binary-ish tree; peaked target so the coin walk is unambiguous vs the oracle.
    rng = np.random.default_rng(seed)
    B, V = 3, 400
    parents = [-1, 0, 0, 1, 1, 2, 2]           # node -> parent (0 = root)
    N = len(parents)
    nxt_tok, nxt_sib = spec_build_tree_pointers(parents, N)
    draft = rng.integers(0, V, size=(B, N - 1)).astype(np.int32)
    # peaked target dist per node (low-temp softmax of random logits)
    tp = np.zeros((B, N, V), np.float32)
    for b in range(B):
        for n in range(N):
            lg = 4.0 * rng.standard_normal(V)
            e = np.exp(lg - lg.max()); tp[b, n] = (e / e.sum()).astype(np.float32)
    nt_b = np.broadcast_to(nxt_tok, (B, N)).copy()   # shared topology -> per-request (B, N)
    ns_b = np.broadcast_to(nxt_sib, (B, N)).copy()
    ai, at, an = spec_verify_tree(mx.array(draft), mx.array(tp), mx.array(nt_b), mx.array(ns_b), seed)
    ai, at, an = np.array(ai), np.array(at), np.array(an)
    for b in range(B):
        acc_idx, acc_tok, num_acc, term, last, tried = _tree_walk_ref(draft[b], tp[b], nxt_tok,
                                                                      nxt_sib, seed, b, N)
        assert an[b] == num_acc, (b, an[b], num_acc)
        assert list(ai[b, :num_acc + 1]) == acc_idx                 # accepted tree path
        assert list(at[b, :num_acc]) == acc_tok                     # accepted draft tokens
        # terminal token validity
        if term != 0:
            tok = int(at[b, num_acc])
            assert tp[b, last, tok] > 0
            if term == 2:
                assert tok not in tried                             # residual excludes tried siblings


@pytest.mark.parametrize("B,S", [(3, 4), (8, 5), (1, 1)])
def test_spec_compact_and_kv_meta(B, S):
    rng = np.random.default_rng(B * 10 + S)
    Sp1 = S + 1
    accepted_cnt = rng.integers(0, S + 1, size=B).astype(np.int32)   # 0..S accepted
    seq_lens = rng.integers(1, 100, size=B).astype(np.int32)
    out_tokens = np.full((B, Sp1), -1, np.int32)
    for b in range(B):
        for j in range(int(accepted_cnt[b]) + 1):
            out_tokens[b, j] = rng.integers(0, 32000)
    pk, pos, cu = spec_compact(mx.array(out_tokens), mx.array(accepted_cnt), mx.array(seq_lens))
    pk, pos, cu = np.array(pk), np.array(pos), np.array(cu)
    # numpy reference
    vlen = accepted_cnt + 1
    cu_ref = np.concatenate([[0], np.cumsum(vlen)]).astype(np.int32)
    cap = B * Sp1
    pk_ref = -np.ones(cap, np.int32); pos_ref = -np.ones(cap, np.int32)
    for b in range(B):
        for j in range(int(vlen[b])):
            pk_ref[cu_ref[b] + j] = out_tokens[b, j]
            pos_ref[cu_ref[b] + j] = seq_lens[b] + j
    np.testing.assert_array_equal(cu, cu_ref)
    np.testing.assert_array_equal(pk, pk_ref)
    np.testing.assert_array_equal(pos, pos_ref)
    nsl = np.array(spec_update_kv_meta(mx.array(seq_lens), mx.array(accepted_cnt)))
    np.testing.assert_array_equal(nsl, seq_lens + accepted_cnt + 1)


def _typical_p_kept(logits_row, typical_p, invtemp):
    ls = logits_row.astype(np.float64) * invtemp
    ls = ls - ls.max()
    p = np.exp(ls); p /= p.sum()
    logp = np.log(p)
    H = -(p * logp).sum()
    surprise = np.abs(-logp - H)
    order = np.argsort(surprise, kind="stable")
    cum = np.cumsum(p[order])
    cutoff = min(int(np.searchsorted(cum, typical_p)), len(order) - 1)
    tau = surprise[order[cutoff]]
    return surprise, tau


@pytest.mark.parametrize("typical_p", [0.2, 0.9])
def test_typical_p_sample(typical_p):
    # over many seeds, the sampled token must lie in the numpy typical-p kept set (surprise <= tau).
    rng = np.random.default_rng(int(typical_p * 100))
    V, invtemp = 500, 1.0 / 0.8
    logits = (1.5 * rng.standard_normal(V)).astype(np.float32)
    surprise, tau = _typical_p_kept(logits, typical_p, invtemp)
    lg = mx.array(logits[None, :])
    for seed in range(60):
        tok = int(np.array(typical_p_sample(lg, typical_p, temperature=0.8, seed=seed))[0])
        assert surprise[tok] <= tau + 1e-3, (seed, tok, surprise[tok], tau)


@pytest.mark.parametrize("V", [50, 200])
def test_apply_bad_words(V):
    rng = np.random.default_rng(V + 5)
    T, maxbad = 4, 6
    logits = rng.standard_normal((T, V)).astype(np.float32)
    bad_lens = rng.integers(0, maxbad + 1, size=T).astype(np.int32)
    bad_ids = rng.integers(0, V, size=(T, maxbad)).astype(np.int32)
    out = np.array(apply_bad_words(mx.array(logits), mx.array(bad_ids), mx.array(bad_lens)))
    is_bad = np.zeros((T, V), bool)
    for t in range(T):
        for j in range(int(bad_lens[t])):
            is_bad[t, bad_ids[t, j]] = True
    np.testing.assert_array_equal(out[~is_bad], logits[~is_bad])   # non-bad untouched
    assert (out[is_bad] < -1e30).all()                            # bad -> -inf sentinel


def test_apply_token_bitmask_then_sample():
    """A grammar mask composed before argmax always yields an allowed token."""
    rng = np.random.default_rng(0)
    T, V = 5, 100
    logits = rng.standard_normal((T, V)).astype(np.float32)
    allow = rng.integers(0, 2, size=(T, V)).astype(bool)
    allow[:, 3] = True
    masked = apply_token_bitmask(mx.array(logits), mx.array(_pack_bitmask(allow)))
    tok = np.array(argmax_sample(masked)).reshape(-1)
    for t in range(T):
        assert allow[t, tok[t]]


def _spec_inputs(B, S, V, seed=0):
    rng = np.random.default_rng(seed)
    dp = rng.dirichlet(np.ones(V), size=(B, S)).astype(np.float32)
    tp = rng.dirichlet(np.ones(V), size=(B, S + 1)).astype(np.float32)
    dt = rng.integers(0, V, size=(B, S)).astype(np.int32)
    bonus = rng.integers(0, V, size=B).astype(np.int32)
    return dp, tp, dt, bonus


def _accept_oracle(dt, dp, tp, au):
    B, S = dt.shape
    cnt = np.zeros(B, np.int32)
    for b in range(B):
        c = S
        for i in range(S):
            pt = tp[b, i, dt[b, i]]; pd = dp[b, i, dt[b, i]]
            if not (au[b, i] * pd <= pt):
                c = i; break
        cnt[b] = c
    return cnt


@pytest.mark.parametrize("B,S,V", [(3, 4, 50), (2, 1, 128), (4, 6, 33)])
def test_spec_verify_accept_reject(B, S, V):
    """Deterministic accept path: accepted_cnt + accepted tokens + bonus + placeholder fill."""
    dp, tp, dt, bonus = _spec_inputs(B, S, V, seed=B + S + V)
    # all-accept (u ~ 0): every draft accepted, bonus appended at position S.
    au0 = np.full((B, S), 1e-9, np.float32)
    o, cnt = spec_verify_linear(mx.array(dt), mx.array(dp), mx.array(tp), mx.array(bonus),
                                mx.array(au0), seed=1)
    mx.eval(o, cnt); o = np.array(o); cnt = np.array(cnt)
    assert (cnt == S).all()
    np.testing.assert_array_equal(o[:, :S], dt)
    np.testing.assert_array_equal(o[:, S], bonus)
    # mixed accept (u = 0.99): accepted_cnt matches the u <= p_t/p_d oracle; tail is placeholder.
    au1 = np.full((B, S), 0.99, np.float32)
    o, cnt = spec_verify_linear(mx.array(dt), mx.array(dp), mx.array(tp), mx.array(bonus),
                                mx.array(au1), seed=1)
    mx.eval(o, cnt); o = np.array(o); cnt = np.array(cnt)
    np.testing.assert_array_equal(cnt, _accept_oracle(dt, dp, tp, au1))
    for b in range(B):
        np.testing.assert_array_equal(o[b, :cnt[b]], dt[b, :cnt[b]])   # accepted prefix
        if cnt[b] < S:
            assert (o[b, cnt[b] + 1:] == -1).all()                     # placeholder tail


def test_spec_verify_recovered_single_support():
    """When the residual (p_t - p_d)+ has a single positive token, the recovered token is that
    token for any seed (the Gumbel-max degenerates)."""
    V = 20
    dp = np.full((1, 1, V), 1.0 / V, np.float32)
    tp = np.full((1, 2, V), 1.0 / V, np.float32)
    tp[0, 0, 7] += 0.5           # token 7 is the only one with p_t > p_d at position 0
    dp[0, 0, 3] += 0.5           # draft token 3 has p_d > p_t -> forced reject at u=1
    dt = np.array([[3]], np.int32)
    bonus = np.array([0], np.int32)
    au = np.array([[1.0]], np.float32)
    for seed in (1, 2, 99):
        o, cnt = spec_verify_linear(mx.array(dt), mx.array(dp), mx.array(tp), mx.array(bonus),
                                    mx.array(au), seed=seed)
        mx.eval(o, cnt)
        assert int(np.array(cnt)[0]) == 0 and int(np.array(o)[0, 0]) == 7


def test_spec_verify_recovered_distribution():
    """Over many seeds, the recovered token is distributed as the normalized residual."""
    rng = np.random.default_rng(5)
    V = 40
    dp = rng.dirichlet(np.ones(V)).astype(np.float32)
    tp = rng.dirichlet(np.ones(V)).astype(np.float32)
    dt = int(np.argmax(dp - tp))            # p_t < p_d here -> reject at u=1
    resid = np.maximum(0, tp - dp); resid /= resid.sum()
    DT = mx.array(np.array([[dt]], np.int32)); DP = mx.array(dp[None, None])
    TP = mx.array(np.concatenate([tp[None, None], tp[None, None]], 1))
    BO = mx.array(np.array([0], np.int32)); AU = mx.array(np.array([[1.0]], np.float32))
    N = 6000
    cnt = np.zeros(V)
    for seed in range(N):
        o, _ = spec_verify_linear(DT, DP, TP, BO, AU, seed=seed)
        mx.eval(o); cnt[int(np.array(o)[0, 0])] += 1
    assert (cnt[resid == 0] == 0).all()      # never samples outside the support
    freq = cnt / N
    mask = resid > 0.01
    assert np.abs(freq[mask] - resid[mask]).max() < 0.03


def test_beam_length_penalty():
    cum = mx.array(np.array([[-2.0, -3.0, -4.0], [-1.0, -5.0, -2.0]], np.float32))
    lengths = mx.array(np.array([[5, 10, 3], [7, 7, 20]], np.float32))
    got = np.array(beam_length_penalty(cum, lengths, alpha=1.0))
    ref = np.array(cum) / (((5.0 + np.array(lengths)) / 6.0) ** 1.0)
    np.testing.assert_allclose(got, ref, atol=1e-5)
    # alpha=0 is a no-op (penalty == 1)
    got0 = np.array(beam_length_penalty(cum, lengths, alpha=0.0))
    np.testing.assert_allclose(got0, np.array(cum), atol=1e-5)


def _beam_oracle(logits, cum, B, BM, V):
    """log_softmax + cum, then flat top-BM over (BM*V) per batch (ties: lowest flat index)."""
    lg = logits.reshape(B, BM, V).astype(np.float64)
    mx_ = lg.max(2, keepdims=True)
    lse = np.log(np.exp(lg - mx_).sum(2, keepdims=True)) + mx_
    scores = (lg - lse) + cum.reshape(B, BM, 1)
    nt = np.zeros((B, BM), np.int32)
    pb = np.zeros((B, BM), np.int32)
    nc = np.zeros((B, BM))
    for b in range(B):
        flat = scores[b].reshape(-1)
        order = np.argsort(-flat, kind="stable")[:BM]
        for k, idx in enumerate(order):
            pb[b, k] = idx // V
            nt[b, k] = idx % V
            nc[b, k] = flat[idx]
    return nt, pb, nc


@pytest.mark.parametrize("B,BM,V", [(2, 4, 32000), (1, 1, 1000), (3, 8, 4000), (2, 16, 4000)])
def test_beam_advance(B, BM, V):
    rng = np.random.default_rng(B + BM + V)
    logits = (rng.standard_normal((B * BM, V)) * 2.0).astype(np.float32)
    cum = rng.standard_normal((B, BM)).astype(np.float32)
    nt, pb, nc = beam_advance(mx.array(logits), mx.array(cum), BM)
    mx.eval(nt, pb, nc)
    ont, opb, onc = _beam_oracle(logits.astype(np.float64), cum.astype(np.float64), B, BM, V)
    np.testing.assert_array_equal(np.array(nt), ont)      # exact tokens (f32, no ties)
    np.testing.assert_array_equal(np.array(pb), opb)      # exact parents
    np.testing.assert_allclose(np.array(nc), onc, atol=1e-4)


def test_beam_advance_step0():
    # step-0 duplicate-beam suppression: cum[:, 1:] = -inf -> all beams pick from beam 0.
    rng = np.random.default_rng(9)
    B, BM, V = 2, 4, 5000
    logits = (rng.standard_normal((B * BM, V)) * 2.0).astype(np.float32)
    cum = np.full((B, BM), -1e30, np.float32)
    cum[:, 0] = 0.0
    nt, pb, nc = beam_advance(mx.array(logits), mx.array(cum), BM)
    mx.eval(pb)
    assert np.all(np.array(pb) == 0)   # every selected beam descends from beam 0


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _nucleus(logits, p):
    sm = _softmax(logits)
    order = np.argsort(-sm)
    csum = np.cumsum(sm[order])
    n = int(np.searchsorted(csum, p)) + 1
    return order[:n], sm

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(4, 1000), (8, 32000), (2, 3, 257)])
def test_argmax_sample(dtype, shape):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    xq = mx.array(x).astype(_MX[dtype])
    got = argmax_sample(xq)
    mx.eval(got)
    xd = np.array(xq.astype(mx.float32))
    ref = np.argmax(xd, axis=-1).astype(np.int32)
    assert np.array_equal(np.array(got).reshape(ref.shape), ref)


def test_sample_categorical_distribution():
    # Each row shares the same logits but a distinct RNG stream (row index), so the
    # empirical token frequencies must converge to softmax(logits).
    V = 8
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = sample_categorical(mx.array(x), temperature=1.0, seed=1234)
    mx.eval(got)
    idx = np.array(got).reshape(-1)
    freq = np.bincount(idx, minlength=V).astype(np.float64) / N
    p = np.exp(logits - logits.max())
    p /= p.sum()
    assert np.max(np.abs(freq - p)) < 0.02, f"freq {freq} vs p {p}"


def test_sample_categorical_determinism():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((16, 100)).astype(np.float32))
    a = sample_categorical(x, temperature=0.8, seed=7)
    b = sample_categorical(x, temperature=0.8, seed=7)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


def test_sample_categorical_temperature_flattens():
    # High temperature -> closer to uniform than low temperature.
    V = 16
    rng = np.random.default_rng(1)
    logits = (rng.standard_normal(V) * 2).astype(np.float32)
    N = 20000
    x = np.broadcast_to(logits, (N, V)).copy()
    hot = np.bincount(np.array(sample_categorical(mx.array(x), temperature=5.0, seed=3)).reshape(-1),
                      minlength=V) / N
    cold = np.bincount(np.array(sample_categorical(mx.array(x), temperature=0.5, seed=3)).reshape(-1),
                       minlength=V) / N
    # entropy(hot) > entropy(cold)
    ent = lambda q: -np.sum(np.where(q > 0, q * np.log(q + 1e-12), 0.0))
    assert ent(hot) > ent(cold)


@pytest.mark.parametrize("K", [1, 5, 40])
def test_top_k_sample_in_topk(K):
    rng = np.random.default_rng(0)
    T, V = 200, 1000
    x = rng.standard_normal((T, V)).astype(np.float32)
    got = np.array(top_k_sample(mx.array(x), K, temperature=1.0, seed=42)).reshape(-1)
    topk_ids = np.argsort(-x, axis=1)[:, :K]
    for t in range(T):
        assert got[t] in topk_ids[t]


def test_top_k_sample_k1_is_argmax():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 500)).astype(np.float32)
    got = np.array(top_k_sample(mx.array(x), 1, seed=99)).reshape(-1)
    assert np.array_equal(got, np.argmax(x, axis=1))


def test_top_k_sample_distribution():
    V, K = 50, 5
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = np.array(top_k_sample(mx.array(x), K, temperature=1.0, seed=7)).reshape(-1)
    freq = np.bincount(got, minlength=V).astype(np.float64) / N
    order = np.argsort(-logits)[:K]
    p = np.zeros(V)
    ex = np.exp(logits[order] - logits[order].max())
    p[order] = ex / ex.sum()
    assert np.max(np.abs(freq - p)) < 0.02


def test_top_k_sample_determinism():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((16, 200)).astype(np.float32))
    a = top_k_sample(x, 8, seed=3)
    b = top_k_sample(x, 8, seed=3)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("p", [0.5, 0.9, 0.99])
def test_top_p_sample_in_nucleus(p):
    rng = np.random.default_rng(0)
    T, V = 200, 500
    x = rng.standard_normal((T, V)).astype(np.float32)
    got = np.array(top_p_sample(mx.array(x), p, temperature=1.0, seed=42)).reshape(-1)
    for t in range(T):
        nuc, _ = _nucleus(x[t], p)
        assert got[t] in set(nuc.tolist())


def test_top_p_sample_small_p_is_argmax():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 500)).astype(np.float32)
    got = np.array(top_p_sample(mx.array(x), 0.001, seed=99)).reshape(-1)
    assert np.array_equal(got, np.argmax(x, axis=1))


def test_top_p_sample_distribution():
    V, p = 40, 0.8
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = np.array(top_p_sample(mx.array(x), p, temperature=1.0, seed=7)).reshape(-1)
    freq = np.bincount(got, minlength=V).astype(np.float64) / N
    nuc, sm = _nucleus(logits, p)
    pn = np.zeros(V)
    pn[nuc] = sm[nuc] / sm[nuc].sum()
    assert np.max(np.abs(freq - pn)) < 0.02


def test_top_p_sample_determinism():
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((16, 200)).astype(np.float32))
    a = top_p_sample(x, 0.9, seed=3)
    b = top_p_sample(x, 0.9, seed=3)
    mx.eval(a, b)
    assert np.array_equal(np.array(a), np.array(b))


def _ref_penalty(ld, prev, temp, rep, presence, freq,
                 bias=None, eos_id=-1, min_length=0, gen_len=0):
    T, V = ld.shape
    ref = ld / temp
    for t in range(T):
        c = np.zeros(V)
        for tok in prev[t]:
            if 0 <= tok < V:
                c[int(tok)] += 1
        for v in range(V):
            if c[v] > 0:
                l = ref[t, v]
                l = l * rep if l < 0 else l / rep
                l -= presence
                l -= freq * c[v]
                ref[t, v] = l
    if bias is not None:
        ref = ref + bias[None, :]
    if eos_id >= 0 and gen_len < min_length:
        ref[:, eos_id] = -np.inf
    return ref


def test_apply_penalty_bias_minlen():
    rng = np.random.default_rng(3)
    T, V, L = 8, 500, 40
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(-1, V, size=(T, L)).astype(np.int32)
    bias = rng.standard_normal(V).astype(np.float32)
    kw = dict(temperature=0.8, repetition_penalty=1.3, presence_penalty=0.1, frequency_penalty=0.05)
    eos_id = 7
    # gen_len < min_length -> EOS forbidden
    got = np.array(apply_penalty(mx.array(logits), mx.array(prev), bias=mx.array(bias),
                                 eos_id=eos_id, min_length=10, gen_len=5, **kw))
    ref = _ref_penalty(logits, prev, kw["temperature"], kw["repetition_penalty"],
                       kw["presence_penalty"], kw["frequency_penalty"],
                       bias=bias, eos_id=eos_id, min_length=10, gen_len=5)
    m = np.arange(V) != eos_id
    np.testing.assert_allclose(got[:, m], ref[:, m], atol=1e-4, rtol=2e-3)
    assert np.all(got[:, eos_id] < -1e30)                 # EOS masked
    # gen_len >= min_length -> EOS not masked
    got2 = np.array(apply_penalty(mx.array(logits), mx.array(prev), bias=mx.array(bias),
                                  eos_id=eos_id, min_length=10, gen_len=15, **kw))
    assert got2[0, eos_id] > -1e30


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_apply_penalty(dtype):
    rng = np.random.default_rng(0)
    T, V, L = 8, 500, 40
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(-1, V, size=(T, L)).astype(np.int32)  # -1 = padding (ignored)
    temp, rep, presence, freq = 0.8, 1.3, 0.1, 0.05
    got = np.array(apply_penalty(
        mx.array(logits).astype(_MX[dtype]), mx.array(prev),
        temperature=temp, repetition_penalty=rep,
        presence_penalty=presence, frequency_penalty=freq).astype(mx.float32))
    ld = np.array(mx.array(logits).astype(_MX[dtype]).astype(mx.float32))
    ref = _ref_penalty(ld, prev, temp, rep, presence, freq)
    atol = 1e-4 if dtype == "float32" else 3e-2
    np.testing.assert_allclose(got, ref, atol=atol, rtol=2e-3)


def test_apply_penalty_identity():
    # temperature=1, rep=1, presence=freq=0 -> logits unchanged.
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((4, 300)).astype(np.float32)
    prev = rng.integers(0, 300, size=(4, 20)).astype(np.int32)
    got = np.array(apply_penalty(mx.array(logits), mx.array(prev)))
    np.testing.assert_allclose(got, logits, atol=1e-5)


def test_apply_penalty_beam_parent():
    # Beam search: each row's occurrence history is gathered from its parent beam (parent_ids).
    rng = np.random.default_rng(4)
    T, V, L = 6, 200, 12
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(0, V, size=(T, L)).astype(np.int32)
    parent = np.array([0, 1, 0, 1, 2, 3], dtype=np.int32)   # rows inherit from earlier rows
    temp, rep, presence, freq = 0.8, 1.3, 0.1, 0.05
    got = np.array(apply_penalty(mx.array(logits), mx.array(prev), temperature=temp,
                                 repetition_penalty=rep, presence_penalty=presence,
                                 frequency_penalty=freq, parent_ids=mx.array(parent)))
    # Reference: each row t uses its OWN logits but the parent beam's history prev[parent[t]].
    ref = _ref_penalty(logits, prev[parent], temp, rep, presence, freq)
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=2e-3)
    # Sanity: differs from the identity (non-beam) result on the redirected rows.
    ident = np.array(apply_penalty(mx.array(logits), mx.array(prev), temperature=temp,
                                   repetition_penalty=rep, presence_penalty=presence,
                                   frequency_penalty=freq))
    assert np.max(np.abs(got - ident)) > 1e-2


if __name__ == "__main__":
    for shp in [(4, 1000), (8, 32000), (2, 3, 257)]:
        test_argmax_sample("float32", shp)
        print("ok", shp)
    test_sample_categorical_distribution()
    test_top_k_sample_distribution()
    test_top_p_sample_distribution()
    test_apply_penalty("float32")
    print("ok sampling")
