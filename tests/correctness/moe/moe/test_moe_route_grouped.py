"""Correctness tests for DeepSeek-style grouped MoE routing (moe_route_grouped).

Oracle: a direct numpy transcription of HF DeepSeek-V3's MoEGate "noaux_tc" method —
sigmoid (or softmax / sqrt-softplus) scoring, e_score_correction_bias added for SELECTION
only, per-group score = sum of the group's top-2 biased scores, top `topk_group` groups
survive, top-k experts among survivors, weights from the UNBIASED scores (+ optional
renormalize, x routed_scaling_factor).

Run from kernels/:  python -m pytest moe/correctness/test_moe_route_grouped.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import moe_route_grouped

_SCORE = {
    "softmax": lambda x: np.exp(x - x.max(-1, keepdims=True))
                         / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "softplus_sqrt": lambda x: np.sqrt(np.log1p(np.exp(np.minimum(x, 20.0)))
                                       + np.maximum(x - 20.0, 0.0)),
}


def _ref(logits, bias, k, n_group, topk_group, renormalize, scaling, scoring):
    T, E = logits.shape
    scored = _SCORE[scoring](logits.astype(np.float64))
    biased = scored + (bias.astype(np.float64) if bias is not None else 0.0)
    ids = np.zeros((T, k), np.int32)
    w = np.zeros((T, k), np.float64)
    epg = E // n_group
    for t in range(T):
        sel = np.ones(E, bool)
        if n_group > 1 and topk_group < n_group:
            gs = np.array([np.sort(biased[t, g * epg:(g + 1) * epg])[-2:].sum()
                           if epg > 1 else biased[t, g * epg] for g in range(n_group)])
            # top groups; ties -> smaller group id (stable argsort on (-score, id))
            order = np.lexsort((np.arange(n_group), -gs))
            keep = order[:topk_group]
            sel[:] = False
            for g in keep:
                sel[g * epg:(g + 1) * epg] = True
        cand = np.where(sel, biased[t], -np.inf)
        order = np.lexsort((np.arange(E), -cand))       # ties -> smaller expert id
        ids[t] = order[:k].astype(np.int32)
        w[t] = scored[t, ids[t]]
        if renormalize:
            w[t] /= w[t].sum()
        w[t] *= scaling
    return ids, w


def _run(T, E, k, n_group, topk_group, scoring="sigmoid", renormalize=True, scaling=1.0,
         use_bias=True, seed=0, logits=None, bias=None):
    rng = np.random.default_rng(seed)
    if logits is None:
        logits = rng.standard_normal((T, E)).astype(np.float32)
    if bias is None and use_bias:
        bias = (0.1 * rng.standard_normal(E)).astype(np.float32)
    ids, w = moe_route_grouped(
        mx.array(logits), k, n_group, topk_group,
        bias=None if bias is None else mx.array(bias),
        renormalize=renormalize, routed_scaling_factor=scaling, scoring=scoring)
    mx.eval(ids, w)
    rids, rw = _ref(logits, bias, k, n_group, topk_group, renormalize, scaling, scoring)
    # the SET of selected experts must match exactly (order can differ only on exact ties,
    # which the tie-break rule also pins -> compare sorted for robustness, exact for f32)
    np.testing.assert_array_equal(np.sort(np.array(ids), 1), np.sort(rids, 1))
    got_w = np.take_along_axis(np.array(w, np.float64), np.argsort(np.array(ids), 1), 1)
    ref_w = np.take_along_axis(rw, np.argsort(rids, 1), 1)
    np.testing.assert_allclose(got_w, ref_w, atol=1e-5, rtol=1e-5)


def test_deepseek_v3_shape():
    _run(T=64, E=256, k=8, n_group=8, topk_group=4, scoring="sigmoid", scaling=2.5)


def test_kimi_k2_shape():
    # n_group=1 -> the group stage is skipped entirely
    _run(T=64, E=384, k=8, n_group=1, topk_group=1, scoring="sigmoid", scaling=2.827)


def test_softmax_scoring():
    _run(T=32, E=64, k=4, n_group=4, topk_group=2, scoring="softmax")


def test_softplus_sqrt_scoring():
    _run(T=32, E=64, k=4, n_group=4, topk_group=2, scoring="softplus_sqrt")


def test_no_renormalize():
    _run(T=32, E=128, k=6, n_group=8, topk_group=4, renormalize=False, scaling=1.5)


def test_no_bias():
    _run(T=32, E=128, k=6, n_group=8, topk_group=4, use_bias=False)


def test_bias_flips_selection_not_weights():
    """The classic porting bug: bias must steer WHICH experts win but never leak into the
    emitted weights. Construct a case where expert 1 outscores expert 0 raw, but bias
    flips the selection to expert 0 — whose weight must be its UNBIASED score."""
    E, k = 8, 1
    logits = np.full((1, E), -8.0, np.float32)
    logits[0, 0] = 1.0     # sigmoid ~ 0.731
    logits[0, 1] = 1.2     # sigmoid ~ 0.769 (raw winner)
    bias = np.zeros(E, np.float32)
    bias[0] = 0.5          # biased: e0 = 1.231 > e1 = 0.769 -> e0 selected
    ids, w = moe_route_grouped(mx.array(logits), k, 1, 1, bias=mx.array(bias),
                               renormalize=False, scoring="sigmoid")
    mx.eval(ids, w)
    assert int(np.array(ids)[0, 0]) == 0
    np.testing.assert_allclose(float(np.array(w)[0, 0]), 1.0 / (1.0 + np.exp(-1.0)),
                               atol=1e-6)


def test_exact_ties_break_to_smaller_id():
    """Equal biased scores -> the smaller expert id must win (simd_argmax semantics)."""
    E, k = 16, 4
    logits = np.zeros((1, E), np.float32)      # all sigmoid scores identical
    ids, _ = moe_route_grouped(mx.array(logits), k, 1, 1, bias=None, scoring="sigmoid")
    mx.eval(ids)
    np.testing.assert_array_equal(np.array(ids)[0], np.arange(k, dtype=np.int32))


def test_group_masking_excludes_losers():
    """Experts in non-selected groups must never appear, even with huge raw scores."""
    E, n_group, topk_group, k = 64, 8, 2, 4
    epg = E // n_group
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((4, E)).astype(np.float32)
    # make groups 0 and 3 dominate via bias so every selection lands inside them
    bias = np.zeros(E, np.float32)
    bias[0 * epg:1 * epg] = 10.0
    bias[3 * epg:4 * epg] = 10.0
    ids, _ = moe_route_grouped(mx.array(logits), k, n_group, topk_group,
                               bias=mx.array(bias), scoring="sigmoid")
    mx.eval(ids)
    groups = np.array(ids) // epg
    assert np.isin(groups, [0, 3]).all()
