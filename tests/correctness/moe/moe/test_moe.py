"""Correctness tests for MoE routing.

moe_route_topk: per token, select top-k experts by router logit (descending) and
return softmax weights over the k selected logits (Mixtral renormalized top-k).

Run from kernels/:  python -m pytest moe/correctness/test_moe.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import (moe_route_topk, moe_permute, moe_pad_schedule, moe_gather, moe_finalize,
                moe_grouped_gemm, moe_grouped_gemm_rect, moe_grouped_gemm_swiglu, moe_mlp)

_MX = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("E,K", [(8, 2), (64, 4), (16, 1), (128, 8)])
def test_moe_route_topk(dtype, E, K):
    rng = np.random.default_rng(0)
    T = 100
    x = rng.standard_normal((T, E)).astype(np.float32)
    xq = mx.array(x).astype(_MX[dtype])
    ids, w = moe_route_topk(xq, K)
    mx.eval(ids, w)
    ids = np.array(ids)
    w = np.array(w)
    xd = np.array(xq.astype(mx.float32))

    gathered = np.take_along_axis(xd, ids, axis=1)         # logits at chosen experts
    true_top = -np.sort(-xd, axis=1)[:, :K]                # true top-k values (descending)

    # the chosen set of logits matches the true top-k set
    np.testing.assert_allclose(np.sort(gathered, axis=1), np.sort(true_top, axis=1), atol=2e-2)
    # returned in descending order
    assert np.all(np.diff(gathered, axis=1) <= 1e-3)
    # weights == softmax over the chosen logits
    m = gathered.max(axis=1, keepdims=True)
    ex = np.exp(gathered - m)
    ref_w = ex / ex.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(w, ref_w, atol=2e-3, rtol=2e-3)
    # ids are a valid permutation subset (no duplicates per row)
    assert all(len(set(row)) == K for row in ids)
    # exact id order for float32 (no down-cast ties)
    if dtype == "float32":
        order = np.argsort(-xd, axis=1, kind="stable")[:, :K].astype(np.int32)
        np.testing.assert_array_equal(ids, order)


@pytest.mark.parametrize("E,K", [(8, 2), (4, 1), (16, 4), (300, 2)])  # E=300 spans >1 scan tile
def test_moe_permute(E, K):
    rng = np.random.default_rng(0)
    T = 50
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    sorted_idx, offsets, inv = moe_permute(mx.array(ids), E)
    mx.eval(sorted_idx, offsets, inv)
    sorted_idx, offsets, inv = np.array(sorted_idx), np.array(offsets), np.array(inv)
    flat = ids.reshape(-1)
    TK = T * K

    counts = np.bincount(flat, minlength=E)
    ref_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    np.testing.assert_array_equal(offsets, ref_off)
    # rows are grouped by expert in the permuted order
    assert np.all(np.diff(flat[sorted_idx]) >= 0)
    for e in range(E):
        s, en = offsets[e], offsets[e + 1]
        assert set(sorted_idx[s:en].tolist()) == set(np.where(flat == e)[0].tolist())
    # inv is the inverse permutation
    assert np.array_equal(sorted_idx[inv], np.arange(TK))


def _pad_schedule_ref(sorted_idx, offsets, K):
    """Numpy oracle for moe_pad_schedule (the former host-side glue)."""
    off = np.asarray(offsets, np.int64)
    E = len(off) - 1
    TK = len(sorted_idx)
    counts = np.diff(off)
    off_pad = np.concatenate([[0], np.cumsum(((counts + 31) // 32) * 32)]).astype(np.int32)
    total_pad_max = ((TK + 31 * E + 31) // 32) * 32
    eot = np.full(total_pad_max // 32, -1, np.int32)
    tb = off_pad // 32
    for e in range(E):
        eot[tb[e]:tb[e + 1]] = e
    gather_idx = np.full(total_pad_max, -1, np.int32)
    inv_pad = np.zeros(TK, np.int32)
    for e in range(E):
        s, en = int(off[e]), int(off[e + 1])
        for p in range(s, en):
            padpos = off_pad[e] + (p - s)
            r = int(sorted_idx[p])
            gather_idx[padpos] = r // K
            inv_pad[r] = padpos
    return eot, gather_idx, inv_pad, off_pad


def _route(rng, T, E, K, scenario):
    if scenario == "one_expert":
        return np.full((T, K), min(3, E - 1), np.int32)  # everything on one expert (max pad)
    if scenario == "aligned":
        # exactly 32 rows per expert (zero padding needed): T*K = 32*E, round-robin
        return (np.arange(T * K, dtype=np.int32) % E).reshape(T, K)
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    if scenario == "empty_expert" and E > 1:
        ids[ids == 0] = 1  # expert 0 gets no tokens
    return ids


@pytest.mark.parametrize("scenario", ["random", "empty_expert", "one_expert", "aligned"])
@pytest.mark.parametrize("E,K", [(8, 2), (16, 4), (1, 1), (300, 2)])
def test_moe_pad_schedule(scenario, E, K):
    rng = np.random.default_rng(3)
    T = 16 * E if scenario == "aligned" else 50
    ids = _route(rng, T, E, K, scenario)
    sidx, offsets, _ = moe_permute(mx.array(ids), E)
    eot, gidx, inv_pad, off_pad = moe_pad_schedule(sidx, offsets, K)
    mx.eval(eot, gidx, inv_pad, off_pad)
    eot_r, gidx_r, inv_pad_r, off_pad_r = _pad_schedule_ref(np.array(sidx), np.array(offsets), K)
    np.testing.assert_array_equal(np.array(off_pad), off_pad_r)
    np.testing.assert_array_equal(np.array(eot), eot_r)
    np.testing.assert_array_equal(np.array(gidx), gidx_r)
    np.testing.assert_array_equal(np.array(inv_pad), inv_pad_r)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("H", [64, 96, 4096])
def test_moe_gather(dtype, H):
    rng = np.random.default_rng(4)
    T, E, K = 37, 8, 2
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    sidx, offsets, _ = moe_permute(mx.array(ids), E)
    _, gidx, _, _ = moe_pad_schedule(sidx, offsets, K)
    x = rng.standard_normal((T, H)).astype(np.float32)
    xm = mx.array(x).astype(_MX[dtype])
    out = moe_gather(xm, gidx)
    mx.eval(out)
    assert out.dtype == xm.dtype
    gidx_np = np.array(gidx)
    xr = np.array(xm.astype(mx.float32))
    ref = np.zeros((len(gidx_np), H), np.float32)
    valid = gidx_np >= 0
    ref[valid] = xr[gidx_np[valid]]
    np.testing.assert_array_equal(np.array(out.astype(mx.float32)), ref)  # pure copy: exact


@pytest.mark.parametrize("K,H", [(2, 64), (4, 128), (1, 256)])
def test_moe_finalize(K, H):
    rng = np.random.default_rng(1)
    T, E = 20, 4
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    w = rng.random((T, K)).astype(np.float32)
    _, _, inv = moe_permute(mx.array(ids), E)
    mx.eval(inv)
    inv = np.array(inv)
    expert_out = rng.standard_normal((T * K, H)).astype(np.float32)
    got = moe_finalize(mx.array(expert_out), mx.array(inv), mx.array(w), K)
    mx.eval(got)
    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for k in range(K):
            ref[t] += w[t, k] * expert_out[inv[t * K + k]]
    np.testing.assert_allclose(np.array(got), ref, atol=1e-4)


@pytest.mark.parametrize("E,K", [(8, 2), (16, 4)])
def test_moe_forward_end_to_end(E, K):
    # route -> permute -> per-expert GEMM (host loop) -> finalize, vs a dense reference.
    rng = np.random.default_rng(2)
    T, H = 32, 64
    x = rng.standard_normal((T, H)).astype(np.float32)
    rl = rng.standard_normal((T, E)).astype(np.float32)
    W = (rng.standard_normal((E, H, H)) * 0.1).astype(np.float32)

    ids, w = moe_route_topk(mx.array(rl), K)
    mx.eval(ids, w)
    ids, w = np.array(ids), np.array(w)
    sidx, off, inv = moe_permute(mx.array(ids.astype(np.int32)), E)
    mx.eval(sidx, off, inv)
    sidx, off, inv = np.array(sidx), np.array(off), np.array(inv)

    permuted_x = x[sidx // K]
    out_perm = np.zeros((T * K, H), np.float32)
    for e in range(E):
        s, en = off[e], off[e + 1]
        if en > s:
            out_perm[s:en] = permuted_x[s:en] @ W[e]
    y = np.array(moe_finalize(mx.array(out_perm), mx.array(inv), mx.array(w), K))

    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for j in range(K):
            ref[t] += w[t, j] * (x[t] @ W[ids[t, j]])
    np.testing.assert_allclose(y, ref, atol=1e-3, rtol=1e-3)


def _padded_schedule(counts):
    padded = [((int(c) + 31) // 32) * 32 for c in counts]
    off_pad = np.concatenate([[0], np.cumsum(padded)]).astype(np.int64)
    total = int(off_pad[-1])
    tile_base = (off_pad // 32).astype(np.int64)
    eot = np.zeros(total // 32, np.int32)
    for e in range(len(counts)):
        eot[tile_base[e]:tile_base[e + 1]] = e
    return off_pad, total, eot


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-4), ("bfloat16", 8e-2)])
@pytest.mark.parametrize("H", [64, 128])
def test_moe_grouped_gemm(dtype, atol, H):
    rng = np.random.default_rng(5)
    E = 4
    counts = [40, 5, 70, 20]  # per-expert token counts -> padded [64,32,96,32], total 224
    off_pad, total, eot = _padded_schedule(counts)
    pi = (0.1 * rng.standard_normal((total, H))).astype(np.float32)
    W = (0.1 * rng.standard_normal((E, H, H))).astype(np.float32)
    md = {"float32": mx.float32, "bfloat16": mx.bfloat16}[dtype]
    pim, Wm = mx.array(pi).astype(md), mx.array(W).astype(md)
    out = moe_grouped_gemm(pim, Wm, mx.array(eot))
    mx.eval(out)
    pir = np.array(pim.astype(mx.float32))
    Wr = np.array(Wm.astype(mx.float32))
    ref = np.zeros((total, H), np.float32)
    for e in range(E):
        s, en = int(off_pad[e]), int(off_pad[e + 1])
        ref[s:en] = pir[s:en] @ Wr[e]
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref, atol=atol, rtol=2e-2)


@pytest.mark.parametrize("E,K", [(8, 2), (16, 4)])
def test_moe_forward_grouped_gemm(E, K):
    # Full fused MoE forward, all-GPU schedule: route -> permute -> pad_schedule -> gather
    # -> moe_grouped_gemm -> finalize(inv_pad), vs a dense per-expert reference.
    rng = np.random.default_rng(7)
    T, H = 40, 64
    x = (0.1 * rng.standard_normal((T, H))).astype(np.float32)
    rl = rng.standard_normal((T, E)).astype(np.float32)
    W = (0.1 * rng.standard_normal((E, H, H))).astype(np.float32)
    xm, Wm = mx.array(x), mx.array(W)

    ids, weights = moe_route_topk(mx.array(rl), K)
    sidx, offsets, _ = moe_permute(ids, E)
    eot, gather_idx, inv_pad, _ = moe_pad_schedule(sidx, offsets, K)
    permuted_x = moe_gather(xm, gather_idx)                    # (total_pad_max, H)
    out_pad = moe_grouped_gemm(permuted_x, Wm, eot)            # (total_pad_max, H)
    y = np.array(moe_finalize(out_pad, inv_pad, weights, K))
    ids_np, w_np = np.array(ids), np.array(weights)

    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for j in range(K):
            ref[t] += w_np[t, j] * (x[t] @ W[ids_np[t, j]])
    np.testing.assert_allclose(y, ref, atol=1e-2, rtol=1e-2)


def _silu(x):
    return x / (1.0 + np.exp(-x))


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-4), ("bfloat16", 8e-2)])
@pytest.mark.parametrize("K_dim,N_out", [(64, 96), (128, 64)])
def test_moe_grouped_gemm_rect(dtype, atol, K_dim, N_out):
    rng = np.random.default_rng(9)
    E = 4
    counts = [40, 5, 70, 20]
    off_pad, total, eot = _padded_schedule(counts)
    A = (0.1 * rng.standard_normal((total, K_dim))).astype(np.float32)
    W = (0.1 * rng.standard_normal((E, K_dim, N_out))).astype(np.float32)
    md = {"float32": mx.float32, "bfloat16": mx.bfloat16}[dtype]
    out = moe_grouped_gemm_rect(mx.array(A).astype(md), mx.array(W).astype(md), mx.array(eot))
    mx.eval(out)
    ref = np.zeros((total, N_out), np.float32)
    Ar = np.array(mx.array(A).astype(md).astype(mx.float32))
    Wr = np.array(mx.array(W).astype(md).astype(mx.float32))
    for e in range(E):
        s, en = int(off_pad[e]), int(off_pad[e + 1])
        ref[s:en] = Ar[s:en] @ Wr[e]
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref, atol=atol, rtol=2e-2)


@pytest.mark.parametrize("dtype,atol", [("float32", 3e-4), ("bfloat16", 8e-2)])
@pytest.mark.parametrize("H,inter", [(64, 32), (128, 64)])
def test_moe_grouped_gemm_swiglu(dtype, atol, H, inter):
    rng = np.random.default_rng(10)
    E = 4
    counts = [40, 5, 70, 20]
    off_pad, total, eot = _padded_schedule(counts)
    A = (0.1 * rng.standard_normal((total, H))).astype(np.float32)
    W1 = (0.1 * rng.standard_normal((E, H, 2 * inter))).astype(np.float32)
    md = {"float32": mx.float32, "bfloat16": mx.bfloat16}[dtype]
    out = moe_grouped_gemm_swiglu(mx.array(A).astype(md), mx.array(W1).astype(md), mx.array(eot))
    mx.eval(out)
    Ar = np.array(mx.array(A).astype(md).astype(mx.float32))
    Wr = np.array(mx.array(W1).astype(md).astype(mx.float32))
    ref = np.zeros((total, inter), np.float32)
    for e in range(E):
        s, en = int(off_pad[e]), int(off_pad[e + 1])
        g = Ar[s:en] @ Wr[e, :, :inter]
        u = Ar[s:en] @ Wr[e, :, inter:]
        ref[s:en] = _silu(g) * u
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), ref, atol=atol, rtol=2e-2)


def _moe_mlp_ref(x, rl_ids, rl_w, W1, W2):
    T, H = x.shape
    inter = W2.shape[1]
    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for j in range(rl_ids.shape[1]):
            e = rl_ids[t, j]
            g = x[t] @ W1[e, :, :inter]
            u = x[t] @ W1[e, :, inter:]
            ref[t] += rl_w[t, j] * ((_silu(g) * u) @ W2[e])
    return ref


@pytest.mark.parametrize("E,K", [(8, 2), (16, 4)])
def test_moe_mlp_swiglu_forward(E, K):
    # Full SwiGLU MoE MLP, all-GPU schedule: route -> permute -> pad_schedule -> gather
    # -> GEMM1(+silu-glu) -> GEMM2 -> finalize(inv_pad), vs a dense ref.
    rng = np.random.default_rng(11)
    T, H, inter = 40, 64, 128
    x = (0.1 * rng.standard_normal((T, H))).astype(np.float32)
    rl = rng.standard_normal((T, E)).astype(np.float32)
    W1 = (0.1 * rng.standard_normal((E, H, 2 * inter))).astype(np.float32)
    W2 = (0.1 * rng.standard_normal((E, inter, H))).astype(np.float32)

    ids, weights = moe_route_topk(mx.array(rl), K)
    sidx, offsets, _ = moe_permute(ids, E)
    eot, gather_idx, inv_pad, _ = moe_pad_schedule(sidx, offsets, K)
    px = moe_gather(mx.array(x), gather_idx)
    h = moe_grouped_gemm_swiglu(px, mx.array(W1), eot)     # (total_pad_max, inter)
    op = moe_grouped_gemm_rect(h, mx.array(W2), eot)       # (total_pad_max, H)
    y = np.array(moe_finalize(op, inv_pad, weights, K))

    ref = _moe_mlp_ref(x, np.array(ids), np.array(weights), W1, W2)
    np.testing.assert_allclose(y, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype,atol", [("float32", 1e-2), ("bfloat16", 8e-2)])
@pytest.mark.parametrize("E,K,T", [(8, 2, 40), (16, 4, 100), (1, 1, 33), (4, 2, 1)])
def test_moe_mlp(dtype, atol, E, K, T):
    # One-call tk.moe_mlp (the whole pipeline, no host sync) vs a dense per-expert reference.
    rng = np.random.default_rng(12)
    H, inter = 64, 96
    x = (0.1 * rng.standard_normal((T, H))).astype(np.float32)
    rl = rng.standard_normal((T, E)).astype(np.float32)
    W1 = (0.1 * rng.standard_normal((E, H, 2 * inter))).astype(np.float32)
    W2 = (0.1 * rng.standard_normal((E, inter, H))).astype(np.float32)
    md = _MX[dtype]
    xm, W1m, W2m = mx.array(x).astype(md), mx.array(W1).astype(md), mx.array(W2).astype(md)

    y = moe_mlp(xm, mx.array(rl), W1m, W2m, K)
    mx.eval(y)
    assert y.shape == (T, H) and y.dtype == md

    ids, weights = moe_route_topk(mx.array(rl), K)
    mx.eval(ids, weights)
    xr = np.array(xm.astype(mx.float32))
    W1r = np.array(W1m.astype(mx.float32))
    W2r = np.array(W2m.astype(mx.float32))
    ref = _moe_mlp_ref(xr, np.array(ids), np.array(weights), W1r, W2r)
    np.testing.assert_allclose(np.array(y.astype(mx.float32)), ref, atol=atol, rtol=5e-2)


if __name__ == "__main__":
    for E, K in [(8, 2), (64, 4), (16, 1), (128, 8)]:
        test_moe_route_topk("float32", E, K)
        print("ok", E, K)
    test_moe_permute(8, 2)
    test_moe_forward_end_to_end(8, 2)
    print("ok moe pipeline")
