"""Correctness tests for the ThunderMittens PyTorch MPS backend.

Run from the kernels/ directory:

    python -m pytest tk_torch/tests/test_mps.py -v
"""

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

import tk_torch  # noqa: E402


def _maxdiff(a, b):
    torch.mps.synchronize()
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("shape", [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)])
def test_layernorm(shape):
    D = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    w = torch.randn(D, dtype=torch.bfloat16, device="mps")
    b = torch.randn(D, dtype=torch.bfloat16, device="mps")
    got = tk_torch.layernorm(x, w, b, 1e-5)
    exp = F.layer_norm(x, (D,), w, b, 1e-5)
    assert _maxdiff(got, exp) < 0.06


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8, 8), (64, 128), (128, 64)])
def test_add_rt(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="mps")
    y = torch.randn(shape, dtype=dtype, device="mps")
    assert _maxdiff(tk_torch.add_rt(x, y), x + y) < 0.02


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-2), (torch.bfloat16, 0.4)])
@pytest.mark.parametrize("nkm", [(32, 16, 32), (128, 64, 128), (256, 128, 256)])
def test_matmul_custom(nkm, dtype, atol):
    N, K, M = nkm
    torch.manual_seed(0)
    x = torch.rand(N, K, dtype=dtype, device="mps")
    y = torch.rand(K, M, dtype=dtype, device="mps")
    got = tk_torch.matmul_custom(x, y)
    exp = (x.float() @ y.float()).to(dtype)
    assert got.shape == (N, M)
    assert _maxdiff(got, exp) < atol


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 4, 512, 64), (2, 2, 128, 128)])
def test_attn_fwd(shape):
    torch.manual_seed(0)
    q = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = tk_torch.attn_fwd(q, k, v)
    # both use the default scale 1/sqrt(D); non-causal, no mask.
    exp = F.scaled_dot_product_attention(q, k, v)
    assert _maxdiff(got, exp) < 0.06


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "fp8_e4m3"])
@pytest.mark.parametrize("D", [64, 128])
def test_attn_q(D, fmt):
    """Quantized-KV attention (MPS) vs reference attention on the dequantized K/V."""
    import numpy as np
    from tk.quant import quantize_kv, dequantize_kv
    B, H, N = 1, 2, 64
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    Kq, Vq = quantize_kv(k, fmt), quantize_kv(v, fmt)
    dk, dv = dequantize_kv(Kq, fmt), dequantize_kv(Vq, fmt)
    got = tk_torch.attn_q(torch.from_numpy(q).to(torch.bfloat16).to("mps"),
                          torch.from_numpy(Kq).to("mps"), torch.from_numpy(Vq).to("mps"), fmt)
    torch.mps.synchronize()
    g = got.float().cpu().numpy()
    s = (q @ np.swapaxes(dk, -1, -2)) / np.sqrt(D); s -= s.max(-1, keepdims=True)
    p = np.exp(s); p /= p.sum(-1, keepdims=True); ref = p @ dv
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 0.1


@pytest.mark.parametrize("shape", [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)])
def test_rms_norm(shape):
    D = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    w = torch.randn(D, dtype=torch.bfloat16, device="mps")
    eps = 1e-5
    got = tk_torch.rms_norm(x, w, eps)
    xf = x.float()
    exp = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)
    assert _maxdiff(got, exp) < 0.03


@pytest.mark.parametrize("shape", [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)])
def test_rms_norm_add(shape):
    D = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    r = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    w = torch.randn(D, dtype=torch.bfloat16, device="mps")
    eps = 1e-5
    out, added = tk_torch.rms_norm_add(x, r, w, eps)
    s = x.float() + r.float()
    exp_added = s.to(torch.bfloat16)
    exp_out = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)
    assert _maxdiff(added, exp_added) < 0.03
    assert _maxdiff(out, exp_out) < 0.03


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 2), (4, 1)])  # MHA, GQA group 2, MQA
def test_paged_attention_gqa(D, H, H_KV):
    import numpy as np
    rng = np.random.default_rng(7 + D + H + H_KV)
    B, num_blocks, block_size = 2, 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)
    group = H // H_KV

    qt = torch.from_numpy(q).to(torch.bfloat16).to("mps")
    kt = torch.from_numpy(kc).to(torch.bfloat16).to("mps")
    vt = torch.from_numpy(vc).to(torch.bfloat16).to("mps")
    got = tk_torch.paged_attention(qt, kt, vt,
                                   torch.from_numpy(block_table).to("mps"),
                                   torch.from_numpy(context_lens).to("mps"), 0.0)

    ref = np.zeros_like(q)
    for b in range(B):
        for h in range(H):
            kvh = h // group
            sc, vs = [], []
            for t in range(context_lens[b]):
                blk = block_table[b, t // block_size]
                slot = t % block_size
                sc.append(float(np.dot(q[b, h], kc[blk, slot, kvh]) * scale))
                vs.append(vc[blk, slot, kvh])
            s = np.array(sc, np.float32)
            p = np.exp(s - s.max()); p /= p.sum()
            ref[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    assert _maxdiff(got.float(), torch.from_numpy(ref).to("mps")) < 0.03


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 1)])
@pytest.mark.parametrize("partition_size", [4, 16])
def test_paged_attention_v2(D, H, H_KV, partition_size):
    import numpy as np
    rng = np.random.default_rng(20 + D + H + H_KV + partition_size)
    B, num_blocks, block_size = 2, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)
    group = H // H_KV
    got = tk_torch.paged_attention_v2(
        torch.from_numpy(q).to(torch.bfloat16).to("mps"),
        torch.from_numpy(kc).to(torch.bfloat16).to("mps"),
        torch.from_numpy(vc).to(torch.bfloat16).to("mps"),
        torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"), 0.0, partition_size)
    ref = np.zeros_like(q)
    for b in range(B):
        for h in range(H):
            kvh = h // group
            sc, vs = [], []
            for t in range(int(cl[b])):
                blk = bt[b, t // block_size]
                slot = t % block_size
                sc.append(float(np.dot(q[b, h], kc[blk, slot, kvh]) * scale))
                vs.append(vc[blk, slot, kvh])
            s = np.array(sc, np.float32)
            p = np.exp(s - s.max()); p /= p.sum()
            ref[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    assert _maxdiff(got.float(), torch.from_numpy(ref).to("mps")) < 0.03


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H_KV", [1, 2])
def test_rope_kv_insert(D, H_KV):
    import numpy as np
    rng = np.random.default_rng(3 + D + H_KV)
    num_blocks, block_size, num_tokens = 4, 4, 5
    P = num_blocks * block_size
    half = D // 2
    inv = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = np.arange(P)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    k = (0.3 * rng.normal(size=(num_tokens, H_KV, D))).astype(np.float32)
    v = (0.3 * rng.normal(size=(num_tokens, H_KV, D))).astype(np.float32)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot_mapping = np.array([0, 5, -1, 6, 11], dtype=np.int64)
    kc0 = (0.1 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc0 = (0.1 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)

    def bf(x):
        return torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).to("mps")

    kc, vc = tk_torch.rope_kv_insert(
        bf(k), bf(v), bf(cos), bf(sin),
        torch.from_numpy(positions).to("mps"), torch.from_numpy(slot_mapping).to("mps"),
        bf(kc0), bf(vc0))

    def to_bf(x):
        return torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    kb, vb, cb, sb = to_bf(k), to_bf(v), to_bf(cos), to_bf(sin)
    ref_k, ref_v = to_bf(kc0), to_bf(vc0)
    for t in range(num_tokens):
        slot = int(slot_mapping[t])
        if slot < 0:
            continue
        blk, boff = slot // block_size, slot % block_size
        for h in range(H_KV):
            x1, x2 = kb[t, h, :half], kb[t, h, half:]
            c, sn = cb[positions[t]], sb[positions[t]]
            ref_k[blk, boff, h] = np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn])
            ref_v[blk, boff, h] = vb[t, h]
    assert _maxdiff(kc.float(), torch.from_numpy(ref_k).to("mps")) < 0.03
    assert _maxdiff(vc.float(), torch.from_numpy(ref_v).to("mps")) < 0.03


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 1000), (8, 32000), (2, 3, 257)])
def test_argmax_sample(dtype, shape):
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    xt = torch.from_numpy(x).to(dtype).to("mps")
    got = tk_torch.argmax_sample(xt)
    xd = xt.float().cpu().numpy()
    ref = np.argmax(xd, axis=-1).astype(np.int32)
    assert np.array_equal(got.cpu().numpy().reshape(ref.shape), ref)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 1)])
def test_fp8_kv_roundtrip(D, H, H_KV):
    import numpy as np
    from tk.quant import _e4m3_decode_arr
    rng = np.random.default_rng(30 + D + H + H_KV)
    B, num_blocks, block_size = 2, 8, 4
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    block_table = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    context_lens = np.array([10, 16], dtype=np.int32)
    k_scale = float(np.abs(K).max() / 448.0)
    v_scale = float(np.abs(V).max() / 448.0)
    scale = 1.0 / math.sqrt(D)
    slot = np.arange(total, dtype=np.int64)
    group = H // H_KV

    kc, vc = tk_torch.kv_cache_scatter_fp8(
        torch.from_numpy(K).to(torch.bfloat16).to("mps"),
        torch.from_numpy(V).to(torch.bfloat16).to("mps"),
        torch.from_numpy(slot).to("mps"), num_blocks, block_size, k_scale, v_scale)
    got = tk_torch.paged_attention_fp8(
        torch.from_numpy(q).to(torch.bfloat16).to("mps"), kc, vc,
        torch.from_numpy(block_table).to("mps"), torch.from_numpy(context_lens).to("mps"),
        k_scale, v_scale, 0.0)

    kc_deq = _e4m3_decode_arr(kc.cpu().numpy()) * k_scale
    vc_deq = _e4m3_decode_arr(vc.cpu().numpy()) * v_scale
    q_bf = torch.from_numpy(q).to(torch.bfloat16).float().numpy()
    ref = np.zeros((B, H, D), np.float32)
    for b in range(B):
        for h in range(H):
            kvh = h // group
            sc, vs = [], []
            for t in range(int(context_lens[b])):
                blk = block_table[b, t // block_size]
                sl = t % block_size
                sc.append(float(np.dot(q_bf[b, h], kc_deq[blk, sl, kvh]) * scale))
                vs.append(vc_deq[blk, sl, kvh])
            s = np.array(sc); p = np.exp(s - s.max()); p /= p.sum()
            ref[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    assert _maxdiff(got.float(), torch.from_numpy(ref).to("mps")) < 0.04


def test_moe_route_topk():
    import numpy as np
    rng = np.random.default_rng(0)
    T, E, K = 100, 64, 4
    x = rng.standard_normal((T, E)).astype(np.float32)
    ids, w = tk_torch.moe_route_topk(torch.from_numpy(x).to("mps"), K)
    ids, w = ids.cpu().numpy(), w.cpu().numpy()
    gathered = np.take_along_axis(x, ids, axis=1)
    true_top = -np.sort(-x, axis=1)[:, :K]
    np.testing.assert_allclose(np.sort(gathered, 1), np.sort(true_top, 1), atol=1e-4)
    np.testing.assert_array_equal(ids, np.argsort(-x, axis=1, kind="stable")[:, :K])


def test_moe_permute_and_finalize():
    import numpy as np
    rng = np.random.default_rng(0)
    T, E, K, H = 50, 8, 2, 64
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    s_t, off_t, inv_t = tk_torch.moe_permute(torch.from_numpy(ids).to("mps"), E)
    s, off, inv = s_t.cpu().numpy(), off_t.cpu().numpy(), inv_t.cpu().numpy()
    flat = ids.reshape(-1)
    counts = np.bincount(flat, minlength=E)
    np.testing.assert_array_equal(off, np.concatenate([[0], np.cumsum(counts)]).astype(np.int32))
    assert np.array_equal(s[inv], np.arange(T * K))
    # finalize
    w = rng.random((T, K)).astype(np.float32)
    eo = rng.standard_normal((T * K, H)).astype(np.float32)
    y = tk_torch.moe_finalize(torch.from_numpy(eo).to("mps"), inv_t,
                              torch.from_numpy(w).to("mps"), K).cpu().numpy()
    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for k in range(K):
            ref[t] += w[t, k] * eo[inv[t * K + k]]
    np.testing.assert_allclose(y, ref, atol=1e-4)


@pytest.mark.parametrize("H", [64, 128])
def test_moe_grouped_gemm(H):
    import numpy as np
    rng = np.random.default_rng(5)
    E = 4
    counts = [40, 5, 70, 20]
    padded = [((c + 31) // 32) * 32 for c in counts]
    off_pad = np.concatenate([[0], np.cumsum(padded)]).astype(np.int64)
    total = int(off_pad[-1])
    tb = off_pad // 32
    eot = np.zeros(total // 32, np.int32)
    for e in range(E):
        eot[tb[e]:tb[e + 1]] = e
    pi = (0.1 * rng.standard_normal((total, H))).astype(np.float32)
    W = (0.1 * rng.standard_normal((E, H, H))).astype(np.float32)
    out = tk_torch.moe_grouped_gemm(
        torch.from_numpy(pi).to(torch.bfloat16).to("mps"),
        torch.from_numpy(W).to(torch.bfloat16).to("mps"),
        torch.from_numpy(eot).to("mps")).float().cpu().numpy()
    pir = torch.from_numpy(pi).to(torch.bfloat16).float().numpy()
    Wr = torch.from_numpy(W).to(torch.bfloat16).float().numpy()
    ref = np.zeros((total, H), np.float32)
    for e in range(E):
        s, en = int(off_pad[e]), int(off_pad[e + 1])
        ref[s:en] = pir[s:en] @ Wr[e]
    assert _maxdiff(torch.from_numpy(out).to("mps"), torch.from_numpy(ref).to("mps")) < 0.08


def test_moe_forward_end_to_end():
    import numpy as np
    rng = np.random.default_rng(2)
    T, H, E, K = 32, 64, 8, 2
    x = rng.standard_normal((T, H)).astype(np.float32)
    rl = rng.standard_normal((T, E)).astype(np.float32)
    W = (rng.standard_normal((E, H, H)) * 0.1).astype(np.float32)
    ids_t, w_t = tk_torch.moe_route_topk(torch.from_numpy(rl).to("mps"), K)
    ids, w = ids_t.cpu().numpy(), w_t.cpu().numpy()
    s_t, off_t, inv_t = tk_torch.moe_permute(ids_t, E)
    sidx, off = s_t.cpu().numpy(), off_t.cpu().numpy()
    permuted_x = x[sidx // K]
    out_perm = np.zeros((T * K, H), np.float32)
    for e in range(E):
        s, en = off[e], off[e + 1]
        if en > s:
            out_perm[s:en] = permuted_x[s:en] @ W[e]
    y = tk_torch.moe_finalize(torch.from_numpy(out_perm).to("mps"), inv_t,
                              torch.from_numpy(w).to("mps"), K).cpu().numpy()
    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for j in range(K):
            ref[t] += w[t, j] * (x[t] @ W[ids[t, j]])
    np.testing.assert_allclose(y, ref, atol=1e-3, rtol=1e-3)


def test_apply_penalty():
    import numpy as np
    rng = np.random.default_rng(0)
    T, V, L = 8, 500, 40
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(-1, V, size=(T, L)).astype(np.int32)
    temp, rep, presence, freq = 0.8, 1.3, 0.1, 0.05
    got = tk_torch.apply_penalty(torch.from_numpy(logits).to("mps"), torch.from_numpy(prev).to("mps"),
                                 temp, rep, presence, freq).cpu().numpy()
    ref = logits / temp
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
    np.testing.assert_allclose(got, ref, atol=1e-4, rtol=2e-3)


def test_sample_categorical_distribution():
    import numpy as np
    V = 8
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = tk_torch.sample_categorical(torch.from_numpy(x).to("mps"), 1.0, 1234)
    freq = np.bincount(got.cpu().numpy().reshape(-1), minlength=V).astype(np.float64) / N
    p = np.exp(logits - logits.max()); p /= p.sum()
    assert np.max(np.abs(freq - p)) < 0.02


def test_sample_categorical_determinism():
    import numpy as np
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((16, 100)).astype(np.float32)).to("mps")
    a = tk_torch.sample_categorical(x, 0.8, 7)
    b = tk_torch.sample_categorical(x, 0.8, 7)
    assert torch.equal(a, b)


def test_top_k_sample_distribution():
    import numpy as np
    V, K = 50, 5
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = tk_torch.top_k_sample(torch.from_numpy(x).to("mps"), K, 1.0, 7).cpu().numpy().reshape(-1)
    freq = np.bincount(got, minlength=V).astype(np.float64) / N
    order = np.argsort(-logits)[:K]
    p = np.zeros(V)
    ex = np.exp(logits[order] - logits[order].max())
    p[order] = ex / ex.sum()
    assert np.max(np.abs(freq - p)) < 0.02


def test_top_k_sample_in_topk():
    import numpy as np
    rng = np.random.default_rng(0)
    T, V, K = 100, 1000, 8
    x = rng.standard_normal((T, V)).astype(np.float32)
    got = tk_torch.top_k_sample(torch.from_numpy(x).to("mps"), K, 1.0, 42).cpu().numpy().reshape(-1)
    topk_ids = np.argsort(-x, axis=1)[:, :K]
    assert all(got[t] in topk_ids[t] for t in range(T))


def test_top_p_sample_distribution():
    import numpy as np
    V, p = 40, 0.8
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(V).astype(np.float32)
    N = 40000
    x = np.broadcast_to(logits, (N, V)).copy()
    got = tk_torch.top_p_sample(torch.from_numpy(x).to("mps"), p, 1.0, 7).cpu().numpy().reshape(-1)
    freq = np.bincount(got, minlength=V).astype(np.float64) / N
    sm = np.exp(logits - logits.max()); sm /= sm.sum()
    order = np.argsort(-sm); csum = np.cumsum(sm[order])
    n = int(np.searchsorted(csum, p)) + 1
    nuc = order[:n]
    pn = np.zeros(V); pn[nuc] = sm[nuc] / sm[nuc].sum()
    assert np.max(np.abs(freq - pn)) < 0.02


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8, 256), (4, 64, 128), (3, 513)])
def test_quantize_per_token_fp8(dtype, shape):
    import numpy as np
    from tk.quant import _e4m3_decode_arr
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    D = shape[-1]
    codes, scale = tk_torch.quantize_per_token_fp8(torch.from_numpy(x).to(dtype).to("mps"))
    xd = torch.from_numpy(x).to(dtype).float().numpy().reshape(-1, D)
    amax = np.abs(xd).max(axis=1)
    ref = amax / 448.0
    ssafe = np.maximum(ref, 1e-30)[:, None]
    np.testing.assert_allclose(scale.cpu().numpy().reshape(-1), ref, rtol=1e-3, atol=1e-8)
    deq = _e4m3_decode_arr(codes.cpu().numpy().astype(np.uint8).reshape(-1, D)) * ssafe
    assert np.all(np.abs(deq - xd) <= 0.0625 * np.abs(xd) + 2.0 * ssafe)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8, 256), (4, 64, 128), (3, 513)])
def test_quantize_per_token_int8(dtype, shape):
    import numpy as np
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    D = shape[-1]
    codes, scale = tk_torch.quantize_per_token_int8(torch.from_numpy(x).to(dtype).to("mps"))
    xd = torch.from_numpy(x).to(dtype).float().numpy().reshape(-1, D)
    amax = np.abs(xd).max(axis=1)
    ref = amax / 127.0
    ssafe = np.maximum(ref, 1e-30)[:, None]
    np.testing.assert_allclose(scale.cpu().numpy().reshape(-1), ref, rtol=1e-3, atol=1e-8)
    c = codes.cpu().numpy().astype(np.int32)
    assert c.min() >= -127 and c.max() <= 127
    deq = c.reshape(-1, D).astype(np.float32) * ssafe
    assert np.all(np.abs(deq - xd) <= 0.5 * ssafe + 1e-6)


@pytest.mark.parametrize("shape", [(8, 256), (3, 1024)])
def test_rms_norm_add_fp8(shape):
    import numpy as np
    from tk.quant import _e4m3_decode_arr
    D, eps = shape[-1], 1e-5
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(torch.bfloat16).to("mps")
    r = torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(torch.bfloat16).to("mps")
    w = torch.from_numpy(rng.standard_normal((D,)).astype(np.float32)).to(torch.bfloat16).to("mps")
    codes, added, scale = tk_torch.rms_norm_add_fp8(x, r, w)   # dynamic
    s = x.float().cpu().numpy() + r.float().cpu().numpy()
    ms = (s * s).mean(-1, keepdims=True)
    normed = s / np.sqrt(ms + eps) * w.float().cpu().numpy()
    ref_scale = np.abs(normed).max(-1) / 448.0
    ssafe = np.maximum(ref_scale, 1e-30)[:, None]
    np.testing.assert_allclose(scale.cpu().numpy().reshape(-1), ref_scale, rtol=1e-3, atol=1e-8)
    deq = _e4m3_decode_arr(codes.cpu().numpy()) * ssafe
    assert np.all(np.abs(deq - normed) <= 0.0625 * np.abs(normed) + 2.0 * ssafe)
    # static mode
    sc = float(np.abs(normed).max() / 448.0)
    codes2, _ = tk_torch.rms_norm_add_fp8(x, r, w, scale=sc)
    deq2 = _e4m3_decode_arr(codes2.cpu().numpy()) * np.float32(sc)
    assert np.all(np.abs(deq2 - normed) <= 0.0625 * np.abs(normed) + 2.0 * sc)


@pytest.mark.parametrize("shape", [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)])
def test_layernorm_add(shape):
    D = shape[-1]
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    r = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    w = torch.randn(D, dtype=torch.bfloat16, device="mps")
    b = torch.randn(D, dtype=torch.bfloat16, device="mps")
    eps = 1e-5
    out, added = tk_torch.layernorm_add(x, r, w, b, eps)
    s = x.float() + r.float()
    exp_added = s.to(torch.bfloat16)
    mean = s.mean(-1, keepdim=True)
    var = (s - mean).pow(2).mean(-1, keepdim=True)
    exp_out = ((s - mean) * torch.rsqrt(var + eps) * w.float() + b.float()).to(torch.bfloat16)
    assert _maxdiff(added, exp_added) < 0.03
    assert _maxdiff(out, exp_out) < 0.03


@pytest.mark.parametrize("shape", [(2, 128, 1024), (4, 64, 512), (1, 256, 768), (8, 256)])
def test_softmax(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    got = tk_torch.softmax(x)
    exp = F.softmax(x.float(), dim=-1).to(torch.bfloat16)
    assert _maxdiff(got, exp) < 0.02


def _cos_sin(N, D, device, base=10000.0):
    i = torch.arange(0, D, 2, dtype=torch.float32)
    inv_freq = base ** (-(i / D))                       # (D/2,)
    pos = torch.arange(N, dtype=torch.float32)[:, None]  # (N,1)
    ang = pos * inv_freq[None, :]                        # (N,D/2)
    return (torch.cos(ang).to(torch.bfloat16).to(device),
            torch.sin(ang).to(torch.bfloat16).to(device))


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 4, 128, 64), (1, 2, 256, 128)])
def test_rotary(shape):
    B, H, N, D = shape
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    cos, sin = _cos_sin(N, D, "mps")
    got = tk_torch.rotary(x, cos, sin)
    xf = x.float()
    x1, x2 = xf[..., :D // 2], xf[..., D // 2:]
    c, s = cos.float()[None, None], sin.float()[None, None]
    exp = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(torch.bfloat16)
    assert _maxdiff(got, exp) < 0.03


@pytest.mark.parametrize("shape", [(2, 128, 1024), (4, 64, 512), (8, 256)])
def test_gelu(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    got = tk_torch.gelu(x)
    exp = F.gelu(x.float(), approximate="tanh").to(torch.bfloat16)
    assert _maxdiff(got, exp) < 0.02


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 4, 512, 64), (2, 2, 128, 128)])
def test_attn_causal(shape):
    B, H, N, D = shape
    torch.manual_seed(0)
    q = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = tk_torch.attn_causal(q, k, v)
    exp = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # scale defaults to 1/sqrt(D)
    assert _maxdiff(got, exp) < 0.05


@pytest.mark.parametrize("nkm", [(40, 20, 48), (100, 50, 70), (33, 17, 65)])
def test_matmul_arbitrary(nkm):
    N, K, M = nkm
    torch.manual_seed(0)
    x = torch.rand(N, K, dtype=torch.float32, device="mps")
    y = torch.rand(K, M, dtype=torch.float32, device="mps")
    got = tk_torch.matmul_custom(x, y)
    assert got.shape == (N, M)
    assert _maxdiff(got, x @ y) < 1e-2


@pytest.mark.parametrize("nkm", [(32, 16, 32), (64, 32, 64), (128, 64, 128)])
def test_flux_gelu(nkm):
    N, K, M = nkm
    torch.manual_seed(0)
    x = torch.rand(N, K, dtype=torch.bfloat16, device="mps")
    w = torch.rand(K, M, dtype=torch.bfloat16, device="mps")
    bias = torch.randn(M, dtype=torch.bfloat16, device="mps")
    got = tk_torch.flux_gelu(x, w, bias)
    ref = F.gelu(x.float() @ w.float() + bias.float(), approximate="tanh").to(torch.bfloat16)
    assert got.shape == (N, M)
    assert _maxdiff(got, ref) < 0.5


@pytest.mark.parametrize("nkm", [(32, 16, 32), (64, 32, 64), (128, 64, 128)])
def test_flux_gate(nkm):
    N, K, M = nkm
    torch.manual_seed(0)
    x = torch.rand(N, K, dtype=torch.bfloat16, device="mps")
    w = torch.rand(K, M, dtype=torch.bfloat16, device="mps")
    bias = torch.randn(M, dtype=torch.bfloat16, device="mps")
    gate = torch.randn(M, dtype=torch.bfloat16, device="mps")
    res = torch.randn(N, M, dtype=torch.bfloat16, device="mps")
    got = tk_torch.flux_gate(x, w, bias, gate, res)
    ref = ((x.float() @ w.float() + bias.float()) * gate.float() + res.float()).to(torch.bfloat16)
    assert got.shape == (N, M)
    assert _maxdiff(got, ref) < 0.5


@pytest.mark.parametrize("nkm", [(32, 16, 32), (128, 64, 128), (256, 128, 256)])
def test_gemm_staged(nkm):
    N, K, M = nkm
    torch.manual_seed(0)
    x = torch.rand(N, K, dtype=torch.float32, device="mps")
    y = torch.rand(K, M, dtype=torch.float32, device="mps")
    got = tk_torch.gemm_staged(x, y)
    assert got.shape == (N, M)
    assert _maxdiff(got, x @ y) < 1e-2


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 4, 512, 64), (2, 2, 128, 128)])
def test_attn_multiwarp(shape):
    B, H, N, D = shape
    torch.manual_seed(0)
    q = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = tk_torch.attn_multiwarp(q, k, v)
    exp = F.scaled_dot_product_attention(q, k, v)  # scale defaults to 1/sqrt(D), non-causal
    assert _maxdiff(got, exp) < 0.05


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 4, 256, 64)])
def test_linear_attn(shape):
    B, H, N, D = shape
    torch.manual_seed(0)
    q = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = tk_torch.linear_attn(q, k, v)
    kv = k.float().transpose(-1, -2) @ v.float()
    exp = q.float() @ kv
    torch.mps.synchronize()
    diff = (got.float() - exp).abs().max().item()
    scale = exp.abs().max().item() + 1e-9
    assert diff / scale < 0.03


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 4, 256, 64)])
def test_hedgehog(shape):
    B, H, N, D = shape
    torch.manual_seed(0)
    q = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = tk_torch.hedgehog(q, k, v)

    def phi(x):
        xf = x.float()
        return torch.exp(xf - xf.max(dim=-1, keepdim=True).values)

    kv = phi(k).transpose(-1, -2) @ v.float()
    exp = phi(q) @ kv
    torch.mps.synchronize()
    diff = (got.float() - exp).abs().max().item()
    scale = exp.abs().max().item() + 1e-9
    assert diff / scale < 0.03


@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (2, 4, 128, 64)])
def test_lin_attn_causal(shape):
    B, H, N, D = shape
    torch.manual_seed(0)
    q = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = tk_torch.lin_attn_causal(q, k, v)
    scores = q.float() @ k.float().transpose(-1, -2)
    mask = torch.tril(torch.ones(N, N, device="mps"))
    exp = (scores * mask) @ v.float()
    torch.mps.synchronize()
    diff = (got.float() - exp).abs().max().item()
    scale = exp.abs().max().item() + 1e-9
    assert diff / scale < 0.03


@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (2, 2, 128, 64)])
def test_mamba2(shape):
    B, H, N, D = shape
    torch.manual_seed(0)
    C = torch.randn(shape, dtype=torch.bfloat16, device="mps") * 0.5
    Bm = torch.randn(shape, dtype=torch.bfloat16, device="mps") * 0.5
    X = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    a = torch.sigmoid(torch.randn(B, H, N, device="mps")) * 0.5 + 0.5
    cumlog = torch.cumsum(torch.log(a), dim=-1).float()
    got = tk_torch.mamba2(C, Bm, X, cumlog)
    scores = C.float() @ Bm.float().transpose(-1, -2)
    decay = torch.exp(cumlog[..., :, None] - cumlog[..., None, :])
    mask = torch.tril(torch.ones(N, N, device="mps"))
    exp = (scores * decay * mask) @ X.float()
    torch.mps.synchronize()
    diff = (got.float() - exp).abs().max().item()
    scale = exp.abs().max().item() + 1e-9
    assert diff / scale < 0.03


@pytest.mark.parametrize("nkm", [(32, 16, 32), (64, 32, 64), (128, 64, 128)])
def test_cmplx_matmul(nkm):
    N, K, M = nkm
    torch.manual_seed(0)
    A = torch.randn(2, N, K, dtype=torch.float32, device="mps")
    B = torch.randn(2, K, M, dtype=torch.float32, device="mps")
    got = tk_torch.cmplx_matmul(A, B)
    torch.mps.synchronize()
    a = torch.complex(A[0], A[1]).cpu()
    b = torch.complex(B[0], B[1]).cpu()
    ref = a @ b
    g = got.cpu()
    assert got.shape == (2, N, M)
    rel = max((g[0] - ref.real).abs().max().item(),
              (g[1] - ref.imag).abs().max().item()) / (ref.abs().max().item() + 1e-9)
    assert rel < 2e-2


@pytest.mark.parametrize("shape", [(1, 1, 16), (2, 2, 32)])
def test_fftconv(shape):
    import numpy as np
    B, H, S = shape
    N = S * S
    rng = np.random.default_rng(0)
    u = rng.standard_normal((B, H, N)).astype(np.float32)
    k = rng.standard_normal((H, N)).astype(np.float32)

    def fftm(sign):
        n = np.arange(S); kk = n.reshape(-1, 1)
        return np.exp(sign * 2j * np.pi * n * kk / S)

    def tw(sign):
        na = np.arange(S).reshape(-1, 1); ma = np.arange(S)
        return np.exp(sign * 2j * np.pi * na * ma / N)

    F, Finv, TW, TWI = fftm(-1), fftm(1), tw(-1), tw(1) / N
    kf = np.fft.fft(k, n=N).reshape(H, S, S).transpose(0, 2, 1)

    def t(m):
        return torch.from_numpy(np.stack([m.real, m.imag]).astype(np.float32)).to("mps")

    xr = u.reshape(B, H, S, S).astype(np.float32)
    X = torch.from_numpy(np.stack([xr, np.zeros_like(xr)])).to("mps")
    KF = torch.from_numpy(np.stack([kf.real, kf.imag]).astype(np.float32)).to("mps")
    got = tk_torch.fftconv(X, t(F), t(TW), t(Finv), t(TWI), KF)
    torch.mps.synchronize()
    g = got.cpu().numpy()
    ref = np.fft.ifft(np.fft.fft(u, n=N) * np.fft.fft(k, n=N)[None], n=N).real.reshape(B, H, S, S)
    assert got.shape == (B, H, S, S)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_K", "kU4B8", "kU4", "fp8_e4m3", "fp4_e2m1", "mxfp8", "nvfp4", "mxfp4", "bitnet", "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K", "q5_K", "q6_K", "e5m2", "fp8_block", "mxfp6_e3m2", "mxfp6_e2m3", "hqq"])
@pytest.mark.parametrize("nkm", [(32, 256, 32), (128, 256, 128), (256, 512, 64)])
def test_qgemm(nkm, fmt):
    import numpy as np
    from tk.quant import QUANT_FORMATS
    quantize, dequantize = QUANT_FORMATS[fmt]
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    Wq = quantize(W)
    wq = torch.from_numpy(Wq).to("mps")
    x = torch.from_numpy(X).to(torch.float16).to("mps")
    got = tk_torch.qgemm(wq, x, fmt)
    torch.mps.synchronize()
    g = got.float().cpu().numpy()
    with np.errstate(all="ignore"):
        ref = dequantize(Wq).astype(np.float32) @ X
    assert got.shape == (N, M)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_K", "kU4B8", "kU4", "fp8_e4m3", "fp4_e2m1", "mxfp8", "nvfp4", "mxfp4", "bitnet", "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K", "q5_K", "q6_K", "e5m2", "fp8_block", "mxfp6_e3m2", "mxfp6_e2m3", "hqq"])
@pytest.mark.parametrize("nk", [(32, 256), (128, 256), (256, 512)])
def test_qgemv(nk, fmt):
    import numpy as np
    from tk.quant import QUANT_FORMATS
    quantize, dequantize = QUANT_FORMATS[fmt]
    N, K = nk
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    x = rng.standard_normal((K, 1)).astype(np.float32)
    Wq = quantize(W)
    wq = torch.from_numpy(Wq).to("mps")
    xt = torch.from_numpy(x).to(torch.float16).to("mps")
    got = tk_torch.qgemv(wq, xt, fmt)
    torch.mps.synchronize()
    g = got.float().cpu().numpy()
    with np.errstate(all="ignore"):
        ref = dequantize(Wq).astype(np.float32) @ x
    assert got.shape == (N, 1)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


@pytest.mark.parametrize("nk", [(64, 256), (128, 512)])
def test_qgemv_w8a8(nk):
    """W8A8 int8xint8 decode (MPS), vs the INTEGER oracle (not the half path)."""
    import numpy as np
    from tk.quant import quantize_w8a8, quantize_act_int8
    N, K = nk
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, 1)).astype(np.float32)
    Wq, ws = quantize_w8a8(W)
    _, Xq, xs = quantize_act_int8(X)
    a_scale = float(xs[0, 0])
    got = tk_torch.qgemv_w8a8(torch.from_numpy(Wq).to("mps"), torch.from_numpy(Xq).to("mps"),
                              torch.from_numpy(ws).to(torch.float16).to("mps"),
                              torch.from_numpy(np.array([a_scale], np.float16)).to("mps"))
    torch.mps.synchronize()
    g = got.float().cpu().numpy()
    ref = (Wq.astype(np.int32) @ Xq.astype(np.int32)).astype(np.float32) * ws[:, None] * a_scale
    assert got.shape == (N, 1)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


@pytest.mark.parametrize("nk", [(64, 256), (128, 512)])
def test_qgemv_w2a8(nk):
    """BitNet W2A8 int decode (MPS), per-group int sums * absmean scale * a_scale."""
    import numpy as np
    from tk.quant import quantize_bitnet, dequantize_bitnet, quantize_act_int8
    N, K = nk
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, 1)).astype(np.float32)
    Wq = quantize_bitnet(W)
    _, Xq, xs = quantize_act_int8(X)
    a_scale = float(xs[0, 0])
    got = tk_torch.qgemv_w2a8(torch.from_numpy(Wq).to("mps"), torch.from_numpy(Xq).to("mps"),
                              torch.from_numpy(np.array([a_scale], np.float16)).to("mps"))
    torch.mps.synchronize()
    g = got.float().cpu().numpy()
    ref = (dequantize_bitnet(Wq).astype(np.float32) @ Xq.astype(np.float32)) * a_scale
    assert got.shape == (N, 1)
    assert np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9) < 2e-2


def test_dispatch_routes_torch_to_mps():
    """tk.<kernel>(torch.Tensor) routes to the MPS backend (no MLX needed)."""
    import tk

    D = 512
    torch.manual_seed(0)
    x = torch.randn(4, D, dtype=torch.bfloat16, device="mps")
    w = torch.randn(D, dtype=torch.bfloat16, device="mps")
    b = torch.randn(D, dtype=torch.bfloat16, device="mps")
    got = tk.layernorm(x, w, b)
    exp = F.layer_norm(x, (D,), w, b, 1e-5)
    assert _maxdiff(got, exp) < 0.06
