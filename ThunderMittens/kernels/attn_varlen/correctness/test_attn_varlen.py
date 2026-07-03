"""Correctness tests for varlen / paged-prefill causal attention.

tk.attn_varlen_prefill runs causal attention over ragged packed queries that read K/V straight
from the paged KV cache (no dense (B,H,N,D) materialization), with a cached prefix
(context_len >= q_len), GQA, and D in {64,128}.

Oracle: per-sequence dense attention, gathering K/V rows [0, ctx) from the paged cache in numpy.
On a fresh cache with a single full sequence it must match tk.attn_causal bit-for-bit.

Run from kernels/:  python -m pytest attn_varlen/correctness/test_attn_varlen.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

import tk


def _build_cache(rng, ctxs, H_KV, D, bs):
    """A paged cache with each sequence's [0, ctx) positions laid into its own blocks."""
    B = len(ctxs)
    max_blocks = max((c + bs - 1) // bs for c in ctxs)
    nb = sum((c + bs - 1) // bs for c in ctxs) + 2  # +2 leaves block 0 and a spare unused
    kc = (0.3 * rng.standard_normal((nb, bs, H_KV, D))).astype(np.float32)
    vc = (0.3 * rng.standard_normal((nb, bs, H_KV, D))).astype(np.float32)
    bt = np.full((B, max_blocks), -1, np.int32)
    blk = 1
    for b in range(B):
        for c in range((ctxs[b] + bs - 1) // bs):
            bt[b, c] = blk
            blk += 1
    return kc, vc, bt


def _oracle(qn, kc, vc, bt, cu, ctxs, H, H_KV, bs, scale):
    out = np.zeros_like(qn)
    grp = H // H_KV
    for b in range(len(ctxs)):
        s, e = int(cu[b]), int(cu[b + 1])
        qlen, ctx = e - s, ctxs[b]
        past = ctx - qlen
        K = np.stack([kc[bt[b, t // bs], t % bs] for t in range(ctx)], 0)  # (ctx, H_KV, D)
        V = np.stack([vc[bt[b, t // bs], t % bs] for t in range(ctx)], 0)
        for h in range(H):
            kvh = h // grp
            for j in range(qlen):
                lim = past + j + 1
                sc = (qn[s + j, h].astype(np.float64) @ K[:lim, kvh].T.astype(np.float64)) * scale
                sc -= sc.max()
                w = np.exp(sc); w /= w.sum()
                out[s + j, h] = w @ V[:lim, kvh]
    return out


def _run(cu, ctxs, H, H_KV, D, bs=16, seed=0):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(D)
    total_q = cu[-1]
    q = (0.3 * rng.standard_normal((total_q, H, D))).astype(np.float32)
    kc, vc, bt = _build_cache(rng, ctxs, H_KV, D, bs)
    cl = np.array(ctxs, np.int32)
    o = tk.attn_varlen_prefill(
        mx.array(q).astype(mx.bfloat16), mx.array(kc).astype(mx.bfloat16),
        mx.array(vc).astype(mx.bfloat16), mx.array(bt), mx.array(cl), cu, scale=float(scale))
    mx.eval(o)
    on = np.array(o.astype(mx.float32))
    ref = _oracle(q, kc, vc, bt, cu, ctxs, H, H_KV, bs, scale)
    assert on.shape == (total_q, H, D)
    rel = np.abs(on - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel < 0.03, f"relerr {rel}"


def _run_device(cu, ctxs, H, H_KV, D, bs=16, seed=1):
    """Fully device-resident path (device cu_seqlens + device Q pad/gather + output re-gather)."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(D)
    total_q = cu[-1]
    q = (0.3 * rng.standard_normal((total_q, H, D))).astype(np.float32)
    kc, vc, bt = _build_cache(rng, ctxs, H_KV, D, bs)
    cl = np.array(ctxs, np.int32)
    qlens = np.diff(np.array(cu))
    max_tiles = int(sum((int(x) + 7) // 8 for x in qlens)) + 4
    o = tk.attn_varlen_prefill_device(
        mx.array(q).astype(mx.bfloat16), mx.array(kc).astype(mx.bfloat16),
        mx.array(vc).astype(mx.bfloat16), mx.array(bt), mx.array(cl),
        mx.array(np.array(cu, np.int32)), max_tiles, scale=float(scale))
    mx.eval(o)
    on = np.array(o.astype(mx.float32))
    ref = _oracle(q, kc, vc, bt, cu, ctxs, H, H_KV, bs, scale)
    assert on.shape == (total_q, H, D)
    rel = np.abs(on - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel < 0.03, f"device relerr {rel}"


@pytest.mark.parametrize("D", [64, 128])
def test_prefill_device(D):
    _run_device([0, 1, 8, 17, 117], [1, 7, 9, 100], H=4, H_KV=2, D=D)   # ragged
    _run_device([0, 16, 32], [16, 16], H=4, H_KV=4, D=D)               # equal lengths
    _run_device([0, 4, 20], [40, 60], H=4, H_KV=2, D=D)                # cached prefix


@pytest.mark.parametrize("D", [64, 128])
def test_equal_lengths(D):
    _run([0, 16, 32], [16, 16], H=4, H_KV=4, D=D)


@pytest.mark.parametrize("D", [64, 128])
def test_ragged(D):
    _run([0, 1, 8, 17, 117], [1, 7, 9, 100], H=2, H_KV=2, D=D)


@pytest.mark.parametrize("D", [64, 128])
def test_prefix(D):
    # context_len > q_len: the cached prefix is attended but not itself a query.
    _run([0, 4, 20], [40, 60], H=4, H_KV=2, D=D)


@pytest.mark.parametrize("H,H_KV", [(8, 1), (8, 2), (4, 4)])
def test_gqa(H, H_KV):
    _run([0, 8, 24], [20, 30], H=H, H_KV=H_KV, D=64)


def test_single_sequence():
    _run([0, 24], [24], H=4, H_KV=4, D=64)


def test_matches_attn_causal():
    # One full sequence on a fresh cache == dense tk.attn_causal, bit-for-bit.
    rng = np.random.default_rng(1)
    N, H, D, bs = 24, 4, 64, 16
    scale = 1.0 / np.sqrt(D)
    q = (0.3 * rng.standard_normal((N, H, D))).astype(np.float32)
    k = (0.3 * rng.standard_normal((N, H, D))).astype(np.float32)
    v = (0.3 * rng.standard_normal((N, H, D))).astype(np.float32)

    qd = mx.array(q.transpose(1, 0, 2)[None]).astype(mx.bfloat16)
    kd = mx.array(k.transpose(1, 0, 2)[None]).astype(mx.bfloat16)
    vd = mx.array(v.transpose(1, 0, 2)[None]).astype(mx.bfloat16)
    od = np.array(tk.attn_causal(qd, kd, vd).astype(mx.float32))[0].transpose(1, 0, 2)

    nb = (N + bs - 1) // bs + 1
    kc = np.zeros((nb, bs, H, D), np.float32)
    vc = np.zeros((nb, bs, H, D), np.float32)
    for t in range(N):
        kc[1 + t // bs, t % bs] = k[t]
        vc[1 + t // bs, t % bs] = v[t]
    bt = np.array([[1, 2]], np.int32)
    o = tk.attn_varlen_prefill(
        mx.array(q).astype(mx.bfloat16), mx.array(kc).astype(mx.bfloat16),
        mx.array(vc).astype(mx.bfloat16), mx.array(bt), mx.array([N], dtype=mx.int32),
        [0, N], scale=float(scale))
    mx.eval(o)
    np.testing.assert_array_equal(np.array(o.astype(mx.float32)), od)


@pytest.mark.parametrize("cu", [[0, 5, 5, 20, 37], [0, 8, 24], [0, 1, 2, 3, 4, 5, 6, 7],
                                [0, 64], [0, 3, 3, 3]])
def test_varlen_build_worklist(cu):
    """The on-device scheduler matches the numpy host worklist builder exactly (qlens, pad_off,
    tile_seq, tile_local0, n_tiles), and sentinel-fills unused tiles with -1."""
    cu = np.asarray(cu, np.int32)
    B = len(cu) - 1
    ql, _padded, poff, ts, tl = tk._varlen_worklist(cu)
    max_tiles = int((cu[-1] + 7 * B) // 8 + B)     # host upper bound
    q2, po2, ts2, tl2, nt2 = tk.varlen_build_worklist(mx.array(cu), max_tiles)
    mx.eval(q2, po2, ts2, tl2, nt2)
    q2, po2, ts2, tl2, nt2 = (np.array(x) for x in (q2, po2, ts2, tl2, nt2))
    n = len(ts)
    assert int(nt2[0]) == n
    np.testing.assert_array_equal(q2, ql)
    np.testing.assert_array_equal(po2, poff.astype(np.int32))
    np.testing.assert_array_equal(ts2[:n], ts)
    np.testing.assert_array_equal(tl2[:n], tl)
    assert (ts2[n:] == -1).all()


@pytest.mark.parametrize("B", [257, 512, 1000])
def test_varlen_build_worklist_large_B(B):
    """B > 256 exercises the chunked scan (each thread owns a contiguous batch chunk)."""
    rng = np.random.default_rng(B)
    qlens = rng.integers(1, 40, size=B).astype(np.int32)
    cu = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    ql, _padded, poff, ts, tl = tk._varlen_worklist(cu)
    max_tiles = int(sum((int(x) + 7) // 8 for x in qlens)) + B
    q2, po2, ts2, tl2, nt2 = tk.varlen_build_worklist(mx.array(cu), max_tiles)
    mx.eval(q2, po2, ts2, tl2, nt2)
    q2, po2, ts2, tl2, nt2 = (np.array(x) for x in (q2, po2, ts2, tl2, nt2))
    n = len(ts)
    assert int(nt2[0]) == n
    np.testing.assert_array_equal(q2, ql)
    np.testing.assert_array_equal(po2, poff.astype(np.int32))
    np.testing.assert_array_equal(ts2[:n], ts)
    np.testing.assert_array_equal(tl2[:n], tl)


if __name__ == "__main__":
    test_equal_lengths(64)
    test_ragged(64)
    test_prefix(64)
    test_matches_attn_causal()
    print("ok")
