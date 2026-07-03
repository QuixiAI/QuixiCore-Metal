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


@pytest.mark.parametrize("window", [1, 16, 63, 640])
def test_paged_attention_window(window):
    import numpy as np
    rng = np.random.default_rng(70 + window)
    B, H, H_KV, D = 2, 4, 2, 64
    block_size, ctx = 16, 64
    nblocks = (ctx + block_size - 1) // block_size
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(B * nblocks + 1, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(B * nblocks + 1, block_size, H_KV, D))).astype(np.float32)
    bt = np.full((B, nblocks), -1, np.int32)
    blk = 1
    for b in range(B):
        for c in range(nblocks):
            bt[b, c] = blk; blk += 1
    cl = np.full((B,), ctx, np.int32)
    scale = 1.0 / math.sqrt(D)
    group = H // H_KV
    qt = torch.from_numpy(q).to(torch.bfloat16).to("mps")
    kt = torch.from_numpy(kc).to(torch.bfloat16).to("mps")
    vt = torch.from_numpy(vc).to(torch.bfloat16).to("mps")
    btt, clt = torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps")
    got = tk_torch.paged_attention(qt, kt, vt, btt, clt, 0.0, window)
    ref = np.zeros_like(q)
    for b in range(B):
        t0 = max(0, ctx - window)
        for h in range(H):
            kvh = h // group
            K = np.stack([kc[bt[b, t // block_size], t % block_size, kvh] for t in range(t0, ctx)], 0)
            V = np.stack([vc[bt[b, t // block_size], t % block_size, kvh] for t in range(t0, ctx)], 0)
            s = (q[b, h] @ K.T) * scale
            p = np.exp(s - s.max()); p /= p.sum()
            ref[b, h] = p @ V
    assert _maxdiff(got.float(), torch.from_numpy(ref).to("mps")) < 0.03
    if window >= ctx:  # full context -> equals window=0 exactly
        full = tk_torch.paged_attention(qt, kt, vt, btt, clt, 0.0, 0)
        assert torch.equal(got, full)


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


@pytest.mark.parametrize("D,H,H_KV", [(64, 4, 2), (128, 2, 2)])
@pytest.mark.parametrize("plen", [7, 16])
def test_cascade_attention(D, H, H_KV, plen):
    import numpy as np
    rng = np.random.default_rng(30 + D + H + plen)
    B, num_blocks, bs = 2, 8, 4
    scale = 1.0 / math.sqrt(D)
    group = H // H_KV
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    pk = (0.2 * rng.normal(size=(plen, H_KV, D))).astype(np.float32)
    pv = (0.2 * rng.normal(size=(plen, H_KV, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, bs, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, bs, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], np.int32)
    cl = np.array([10, 16], np.int32)
    mps = lambda a: torch.from_numpy(a).to(torch.bfloat16).to("mps")
    got = tk_torch.cascade_attention(mps(q), mps(pk), mps(pv), mps(kc), mps(vc),
                                     torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                                     0.0, 8)
    ref = np.zeros_like(q)
    for b in range(B):
        for h in range(H):
            kvh = h // group
            sc, vs = [], []
            for t in range(plen):
                sc.append(float(np.dot(q[b, h], pk[t, kvh]) * scale)); vs.append(pv[t, kvh])
            for t in range(int(cl[b])):
                blk = bt[b, t // bs]; slot = t % bs
                sc.append(float(np.dot(q[b, h], kc[blk, slot, kvh]) * scale)); vs.append(vc[blk, slot, kvh])
            s = np.array(sc, np.float32); p = np.exp(s - s.max()); p /= p.sum()
            ref[b, h] = np.sum(p[:, None] * np.stack(vs), axis=0)
    assert _maxdiff(got.float(), torch.from_numpy(ref).to("mps")) < 0.03


@pytest.mark.parametrize("window", [1, 16, 640])
@pytest.mark.parametrize("fp8", [False, True])
def test_paged_attention_v2_window(window, fp8):
    import numpy as np
    rng = np.random.default_rng(200 + window + int(fp8))
    B, H, H_KV, D, num_blocks, block_size, ctx = 2, 4, 2, 64, 8, 16, 64
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.arange(num_blocks, dtype=np.int32).reshape(B, num_blocks // B)
    cl = np.array([ctx, ctx], dtype=np.int32)
    scale = 1.0 / math.sqrt(D)
    group = H // H_KV
    qt = torch.from_numpy(q).to(torch.bfloat16).to("mps")
    btt, clt = torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps")
    if fp8:
        total = num_blocks * block_size
        K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
        V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
        ks, vs = float(np.abs(K).max() / 448.0), float(np.abs(V).max() / 448.0)
        kc, vc = tk_torch.kv_cache_scatter_fp8(
            torch.from_numpy(K).to(torch.bfloat16).to("mps"),
            torch.from_numpy(V).to(torch.bfloat16).to("mps"),
            torch.from_numpy(np.arange(total, dtype=np.int64)).to("mps"), num_blocks, block_size,
            ks, vs)
        got = tk_torch.paged_attention_v2_fp8(qt, kc, vc, btt, clt, ks, vs, 0.0, 32, "e4m3", window)
        from tk.quant import _e4m3_decode_arr
        kcn = _e4m3_decode_arr(kc.cpu().numpy()) * ks
        vcn = _e4m3_decode_arr(vc.cpu().numpy()) * vs
    else:
        kcn = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
        vcn = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
        got = tk_torch.paged_attention_v2(qt, torch.from_numpy(kcn).to(torch.bfloat16).to("mps"),
                                          torch.from_numpy(vcn).to(torch.bfloat16).to("mps"),
                                          btt, clt, 0.0, 32, window)
    t0 = max(0, ctx - window)
    ref = np.zeros_like(q)
    for b in range(B):
        for h in range(H):
            kvh = h // group
            sc = [float(np.dot(q[b, h], kcn[bt[b, t // block_size], t % block_size, kvh]) * scale)
                  for t in range(t0, ctx)]
            vs_ = [vcn[bt[b, t // block_size], t % block_size, kvh] for t in range(t0, ctx)]
            s = np.array(sc, np.float32); p = np.exp(s - s.max()); p /= p.sum()
            ref[b, h] = np.sum(p[:, None] * np.stack(vs_), axis=0)
    assert _maxdiff(got.float(), torch.from_numpy(ref).to("mps")) < 0.03


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("gemma", [False, True])
def test_rope_kv_insert_norm(D, gemma):
    import numpy as np
    rng = np.random.default_rng(4 + D + int(gemma))
    nb, bs, nt, H_KV = 4, 4, 5, 2
    P, eps, half = nb * bs, 1e-5, D // 2
    inv = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = np.arange(P)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    k = (0.3 * rng.normal(size=(nt, H_KV, D))).astype(np.float32)
    v = (0.3 * rng.normal(size=(nt, H_KV, D))).astype(np.float32)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot = np.array([0, 5, -1, 6, 11], dtype=np.int64)
    w = rng.normal(size=(D,)).astype(np.float32)
    kc0 = (0.1 * rng.normal(size=(nb, bs, H_KV, D))).astype(np.float32)
    vc0 = (0.1 * rng.normal(size=(nb, bs, H_KV, D))).astype(np.float32)

    def bf(x):
        return torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).to("mps")

    kc, vc = tk_torch.rope_kv_insert_norm(
        bf(k), bf(v), bf(cos), bf(sin), torch.from_numpy(positions).to("mps"),
        torch.from_numpy(slot).to("mps"), bf(kc0), bf(vc0), bf(w), eps, gemma)

    def tb(x):
        return torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    kb, vb, cb, sb, wb = tb(k), tb(v), tb(cos), tb(sin), tb(w)
    ref_k, ref_v = tb(kc0), tb(vc0)
    for t in range(nt):
        s = int(slot[t])
        if s < 0:
            continue
        blk, boff = s // bs, s % bs
        for h in range(H_KV):
            ms = (kb[t, h] ** 2).mean()
            weff = (1.0 + wb) if gemma else wb
            kn = kb[t, h] / np.sqrt(ms + eps) * weff
            x1, x2 = kn[:half], kn[half:]
            c, sn = cb[positions[t]], sb[positions[t]]
            ref_k[blk, boff, h] = np.concatenate([x1 * c - x2 * sn, x2 * c + x1 * sn])
            ref_v[blk, boff, h] = vb[t, h]
    assert _maxdiff(kc.float(), torch.from_numpy(ref_k).to("mps")) < 0.03
    assert _maxdiff(vc.float(), torch.from_numpy(ref_v).to("mps")) < 0.03


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


def test_moe_pad_schedule_and_gather():
    import numpy as np
    rng = np.random.default_rng(3)
    T, E, K, H = 50, 8, 2, 64
    ids = rng.integers(0, E, size=(T, K)).astype(np.int32)
    ids[ids == 0] = 1  # expert 0 empty (edge case)
    s_t, off_t, _ = tk_torch.moe_permute(torch.from_numpy(ids).to("mps"), E)
    eot_t, gidx_t, inv_pad_t, off_pad_t = tk_torch.moe_pad_schedule(s_t, off_t, K)
    sidx, off = s_t.cpu().numpy(), off_t.cpu().numpy()
    TK = T * K
    counts = np.diff(off)
    off_pad = np.concatenate([[0], np.cumsum(((counts + 31) // 32) * 32)]).astype(np.int32)
    total_pad_max = ((TK + 31 * E + 31) // 32) * 32
    eot = np.full(total_pad_max // 32, -1, np.int32)
    tb = off_pad // 32
    for e in range(E):
        eot[tb[e]:tb[e + 1]] = e
    gidx = np.full(total_pad_max, -1, np.int32)
    inv_pad = np.zeros(TK, np.int32)
    for e in range(E):
        for p in range(off[e], off[e + 1]):
            padpos = off_pad[e] + (p - off[e])
            gidx[padpos] = sidx[p] // K
            inv_pad[sidx[p]] = padpos
    np.testing.assert_array_equal(off_pad_t.cpu().numpy(), off_pad)
    np.testing.assert_array_equal(eot_t.cpu().numpy(), eot)
    np.testing.assert_array_equal(gidx_t.cpu().numpy(), gidx)
    np.testing.assert_array_equal(inv_pad_t.cpu().numpy(), inv_pad)
    # gather
    x = rng.standard_normal((T, H)).astype(np.float32)
    out = tk_torch.moe_gather(torch.from_numpy(x).to("mps"), gidx_t).cpu().numpy()
    ref = np.zeros((total_pad_max, H), np.float32)
    ref[gidx >= 0] = x[gidx[gidx >= 0]]
    np.testing.assert_array_equal(out, ref)


def test_moe_mlp_gpu_schedule():
    import numpy as np
    rng = np.random.default_rng(12)
    T, H, inter, E, K = 40, 64, 96, 8, 2
    x = (0.1 * rng.standard_normal((T, H))).astype(np.float32)
    rl = rng.standard_normal((T, E)).astype(np.float32)
    W1 = (0.1 * rng.standard_normal((E, H, 2 * inter))).astype(np.float32)
    W2 = (0.1 * rng.standard_normal((E, inter, H))).astype(np.float32)
    xt = torch.from_numpy(x).to("mps")
    ids_t, w_t = tk_torch.moe_route_topk(torch.from_numpy(rl).to("mps"), K)
    s_t, off_t, _ = tk_torch.moe_permute(ids_t, E)
    eot_t, gidx_t, inv_pad_t, _ = tk_torch.moe_pad_schedule(s_t, off_t, K)
    px = tk_torch.moe_gather(xt, gidx_t)
    h = tk_torch.moe_grouped_gemm_swiglu(px, torch.from_numpy(W1).to("mps"), eot_t)
    op = tk_torch.moe_grouped_gemm_rect(h, torch.from_numpy(W2).to("mps"), eot_t)
    y = tk_torch.moe_finalize(op, inv_pad_t, w_t, K).cpu().numpy()
    ids, w = ids_t.cpu().numpy(), w_t.cpu().numpy()
    ref = np.zeros((T, H), np.float32)
    for t in range(T):
        for j in range(K):
            e = ids[t, j]
            g = x[t] @ W1[e, :, :inter]
            u = x[t] @ W1[e, :, inter:]
            ref[t] += w[t, j] * ((g / (1 + np.exp(-g)) * u) @ W2[e])
    np.testing.assert_allclose(y, ref, atol=1e-2, rtol=1e-2)


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
@pytest.mark.parametrize("shape", [(8, 256), (3, 513)])
def test_quantize_per_tensor_fp8(dtype, shape):
    import numpy as np
    from tk.quant import _e4m3_decode_arr
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    codes, scale = tk_torch.quantize_per_tensor_fp8(torch.from_numpy(x).to(dtype).to("mps"))
    xd = torch.from_numpy(x).to(dtype).float().numpy()
    ref_scale = np.abs(xd).max() / 448.0
    np.testing.assert_allclose(float(scale.cpu().numpy().reshape(-1)[0]), ref_scale, rtol=1e-3, atol=1e-8)
    ssafe = max(ref_scale, 1e-30)
    deq = _e4m3_decode_arr(codes.cpu().numpy()) * ssafe
    assert np.all(np.abs(deq - xd) <= 0.0625 * np.abs(xd) + 2.0 * ssafe)


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


@pytest.mark.parametrize("window", [5, 100])
@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 2, 128, 128)])
def test_attn_window(shape, window):
    B, H, N, D = shape
    torch.manual_seed(0)
    q = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = tk_torch.attn_window(q, k, v, window)
    i = torch.arange(N, device="mps")[:, None]
    j = torch.arange(N, device="mps")[None, :]
    mask = (j <= i) & (j >= i - window + 1)
    exp = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert _maxdiff(got, exp) < 0.05
    full = tk_torch.attn_window(q, k, v, N + 1)
    causal = tk_torch.attn_causal(q, k, v)
    assert torch.equal(full, causal), "window >= N must match attn_causal exactly"


@pytest.mark.parametrize("cu", [[0, 5, 5, 20, 37], [0, 8, 24], [0, 64]])
def test_varlen_build_worklist(cu):
    import numpy as np
    import tk
    cu = np.asarray(cu, np.int32)
    B = len(cu) - 1
    ql, _padded, poff, ts, tl = tk._varlen_worklist(cu)
    max_tiles = int((cu[-1] + 7 * B) // 8 + B)
    q2, po2, ts2, tl2, nt2 = tk_torch.varlen_build_worklist(
        torch.from_numpy(cu).to("mps"), max_tiles)
    q2, po2, ts2, tl2, nt2 = (x.cpu().numpy() for x in (q2, po2, ts2, tl2, nt2))
    n = len(ts)
    assert int(nt2[0]) == n
    np.testing.assert_array_equal(q2, ql)
    np.testing.assert_array_equal(po2, poff.astype(np.int32))
    np.testing.assert_array_equal(ts2[:n], ts)
    np.testing.assert_array_equal(tl2[:n], tl)
    assert (ts2[n:] == -1).all()


@pytest.mark.parametrize("D,H,H_KV", [(64, 4, 4), (64, 8, 2), (128, 4, 4)])
def test_attn_varlen_prefill(D, H, H_KV):
    import numpy as np
    rng = np.random.default_rng(2)
    cu = [0, 4, 20, 37]
    ctxs = [10, 30, 25]
    bs = 16
    scale = 1.0 / np.sqrt(D)
    total_q = cu[-1]
    q = (0.3 * rng.standard_normal((total_q, H, D))).astype(np.float32)
    nb = sum((c + bs - 1) // bs for c in ctxs) + 2
    mbk = max((c + bs - 1) // bs for c in ctxs)
    kc = (0.3 * rng.standard_normal((nb, bs, H_KV, D))).astype(np.float32)
    vc = (0.3 * rng.standard_normal((nb, bs, H_KV, D))).astype(np.float32)
    bt = np.full((len(ctxs), mbk), -1, np.int32)
    blk = 1
    for b in range(len(ctxs)):
        for c in range((ctxs[b] + bs - 1) // bs):
            bt[b, c] = blk
            blk += 1
    # tk.attn_varlen_prefill builds the head-major worklist + pad/transpose, then calls the MPS op.
    import tk
    o = tk.attn_varlen_prefill(
        torch.from_numpy(q).to(torch.bfloat16).to("mps"),
        torch.from_numpy(kc).to(torch.bfloat16).to("mps"),
        torch.from_numpy(vc).to(torch.bfloat16).to("mps"),
        torch.from_numpy(bt).to("mps"), torch.from_numpy(np.array(ctxs, np.int32)).to("mps"),
        cu, scale=float(scale))
    on = o.float().cpu().numpy()
    grp = H // H_KV
    ref = np.zeros((total_q, H, D), np.float32)
    for b in range(len(ctxs)):
        s, e = cu[b], cu[b + 1]
        qlen, ctx = e - s, ctxs[b]
        past = ctx - qlen
        K = np.stack([kc[bt[b, t // bs], t % bs] for t in range(ctx)], 0)
        V = np.stack([vc[bt[b, t // bs], t % bs] for t in range(ctx)], 0)
        for h in range(H):
            kvh = h // grp
            for j in range(qlen):
                lim = past + j + 1
                sc = (q[s + j, h].astype(np.float64) @ K[:lim, kvh].T.astype(np.float64)) * scale
                sc -= sc.max()
                w = np.exp(sc); w /= w.sum()
                ref[s + j, h] = w @ V[:lim, kvh]
    assert np.abs(on - ref).max() / (np.abs(ref).max() + 1e-6) < 0.03


@pytest.mark.parametrize("mode,k", [("argmax", 0), ("categorical", 0), ("topk", 8)])
def test_lm_head_sample(mode, k):
    import numpy as np
    import tk
    rng = np.random.default_rng(5)
    T, V, K = 4, 8000, 512
    h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
    W = (0.5 * rng.standard_normal((V, K))).astype(np.float32)
    hm = torch.from_numpy(h).to(torch.bfloat16).to("mps")
    Wm = torch.from_numpy(W).to(torch.bfloat16).to("mps")
    tok = tk.lm_head_sample(hm, Wm, mode=mode, k=k, temperature=0.8, seed=11).cpu().numpy()
    L = (h.astype(np.float64)) @ (W.astype(np.float64)).T
    assert tok.shape == (T,)
    if mode == "topk":
        for t in range(T):
            top = set(int(v) for v in np.argsort(-L[t], kind="stable")[:k])
            assert tok[t] in top or (L[t].max() - L[t, tok[t]]) < 1e-2
    else:
        for t in range(T):
            # both argmax and categorical select a high-logit token; check it's a plausible pick
            assert (L[t].max() - L[t, tok[t]]) < (1e-2 if mode == "argmax" else 40.0)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 3e-3), (torch.bfloat16, 6e-2)])
def test_cross_entropy(dtype, tol):
    import numpy as np
    import torch.nn.functional as F
    import tk
    rng = np.random.default_rng(6)
    T, V = 8, 4000
    logits = (rng.standard_normal((T, V)) * 2.0).astype(np.float32)
    tgt = rng.integers(0, V, size=(T,)).astype(np.int32)
    tgt[:2] = -100
    lm = torch.from_numpy(logits).to(dtype).to("mps")
    tm = torch.from_numpy(tgt.astype(np.int64)).to("mps")
    loss, lse = tk.cross_entropy(lm, tm, reduction="none", return_lse=True)
    ref = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(tgt.astype(np.int64)),
                          ignore_index=-100, reduction="none").numpy()
    assert np.abs(loss.float().cpu().numpy() - ref).max() < tol
    n = max(int((tgt != -100).sum()), 1)
    g = tk.cross_entropy_grad(lm, tm, lse, torch.full((T,), 1.0 / n, device="mps"))
    lt = torch.from_numpy(logits).requires_grad_(True)
    F.cross_entropy(lt, torch.from_numpy(tgt.astype(np.int64)), ignore_index=-100,
                    reduction="mean").backward()
    assert np.abs(g.float().cpu().numpy() - lt.grad.numpy()).max() < tol


def test_beam_reorder_kv():
    import numpy as np
    import tk
    rng = np.random.default_rng(0)
    B, BM, bs, H_KV, D, max_blocks = 2, 3, 4, 2, 32, 2
    nbeams = B * BM
    nb = nbeams * max_blocks
    kc = rng.standard_normal((nb, bs, H_KV, D)).astype(np.float32)
    vc = rng.standard_normal((nb, bs, H_KV, D)).astype(np.float32)
    bt = np.arange(nb, dtype=np.int32).reshape(nbeams, max_blocks)
    pb = np.array([[1, 1, 0], [0, 2, 1]], np.int32)
    seq_lens = np.full(nbeams, 7, np.int32)
    kc2, vc2 = tk.beam_reorder_kv(
        torch.from_numpy(kc).to(torch.bfloat16).to("mps"),
        torch.from_numpy(vc).to(torch.bfloat16).to("mps"),
        torch.from_numpy(bt).to("mps"), torch.from_numpy(pb).to("mps"),
        torch.from_numpy(seq_lens).to("mps"))
    kb = torch.from_numpy(kc).to(torch.bfloat16).float().numpy()
    vb = torch.from_numpy(vc).to(torch.bfloat16).float().numpy()
    ref_k, ref_v = kb.copy(), vb.copy()
    for b in range(B):
        for k in range(BM):
            p = pb[b, k]
            if p == k:
                continue
            for c in range(2):
                ref_k[bt[b * BM + k, c]] = kb[bt[b * BM + p, c]]
                ref_v[bt[b * BM + k, c]] = vb[bt[b * BM + p, c]]
    np.testing.assert_array_equal(kc2.float().cpu().numpy(), ref_k)
    np.testing.assert_array_equal(vc2.float().cpu().numpy(), ref_v)


def test_beam_reorder_kv_chain():
    """Reorder chain beam0<-beam1<-beam2: read-from-original must give beam0 the ORIGINAL beam1
    (not the reordered one). Race-free copy pin, matching the MLX chain test."""
    import numpy as np
    import tk
    rng = np.random.default_rng(0)
    nbeams, bs, H_KV, D = 3, 4, 1, 8
    kc = rng.standard_normal((nbeams, bs, H_KV, D)).astype(np.float32)
    vc = rng.standard_normal((nbeams, bs, H_KV, D)).astype(np.float32)
    bt = np.arange(nbeams, dtype=np.int32).reshape(nbeams, 1)
    pb = np.array([[1, 2, 2]], np.int32)
    sl = np.full(nbeams, 3, np.int32)
    kc2, vc2 = tk.beam_reorder_kv(torch.from_numpy(kc).to("mps"), torch.from_numpy(vc).to("mps"),
                                  torch.from_numpy(bt).to("mps"), torch.from_numpy(pb).to("mps"),
                                  torch.from_numpy(sl).to("mps"))
    ref_k, ref_v = kc.copy(), vc.copy()
    ref_k[0] = kc[1]; ref_k[1] = kc[2]
    ref_v[0] = vc[1]; ref_v[1] = vc[2]
    np.testing.assert_array_equal(kc2.float().cpu().numpy(), ref_k)
    np.testing.assert_array_equal(vc2.float().cpu().numpy(), ref_v)


@pytest.mark.parametrize("B,BM", [(2, 3), (3, 2)])
def test_beam_build_copy_pairs(B, BM):
    import numpy as np
    import tk_torch
    max_blocks, block_size = 3, 4
    nbeams = B * BM
    rng = np.random.default_rng(B * 7 + BM)
    bt = np.arange(nbeams * max_blocks, dtype=np.int32).reshape(nbeams, max_blocks)
    pb = rng.integers(0, BM, size=(B, BM)).astype(np.int32)
    sl = rng.integers(1, max_blocks * block_size, size=nbeams).astype(np.int32)
    pairs = tk_torch.beam_build_copy_pairs(torch.from_numpy(pb).to("mps"),
                                           torch.from_numpy(bt).to("mps"),
                                           torch.from_numpy(sl).to("mps"), block_size).cpu().numpy()
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


@pytest.mark.parametrize("min_p", [0.2, 0.6])
def test_min_p_sample(min_p):
    import numpy as np
    rng = np.random.default_rng(int(min_p * 100))
    V = 60
    logits = (rng.standard_normal(V) * 2).astype(np.float32)
    p = np.exp(logits - logits.max()); p /= p.sum()
    kept = set(np.where(p >= min_p * p.max())[0].tolist())
    x = torch.from_numpy(logits[None]).to("mps")
    seen = set()
    for s in range(1500):
        seen.add(int(tk_torch.min_p_sample(x, min_p, 1.0, s).cpu().numpy()[0]))
    assert seen <= kept


@pytest.mark.parametrize("V", [40, 70])
def test_apply_token_bitmask(V):
    import numpy as np
    rng = np.random.default_rng(V)
    T = 4
    logits = rng.standard_normal((T, V)).astype(np.float32)
    allow = rng.integers(0, 2, size=(T, V)).astype(bool)
    allow[:, 0] = True
    nw = (V + 31) // 32
    m = np.zeros((T, nw), np.uint32)
    for t in range(T):
        for v in range(V):
            if allow[t, v]:
                m[t, v >> 5] |= np.uint32(1) << np.uint32(v & 31)
    out = tk_torch.apply_token_bitmask(torch.from_numpy(logits).to("mps"),
                                       torch.from_numpy(m.view(np.int32)).to("mps")).cpu().numpy()
    np.testing.assert_array_equal(out[allow], logits[allow])
    assert (out[~allow] < -1e30).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_embedding_lookup(dtype):
    import numpy as np
    rng = np.random.default_rng(0)
    vocab, D, T = 200, 128, 12
    table = (0.3 * rng.standard_normal((vocab, D))).astype(np.float32)
    tok = rng.integers(0, vocab, size=T).astype(np.int32); tok[4] = -1
    o = tk_torch.embedding_lookup(torch.from_numpy(tok).to("mps"),
                                  torch.from_numpy(table).to(dtype).to("mps"),
                                  scale=1.5).float().cpu().numpy()
    ref = np.where(tok[:, None] >= 0, table[np.clip(tok, 0, vocab - 1)] * 1.5, 0.0)
    assert np.allclose(o, ref, atol=1e-4 if dtype == torch.float32 else 3e-2)


def test_merge_multimodal_spans():
    import numpy as np
    rng = np.random.default_rng(2)
    T, M, D = 16, 6, 128
    text = (0.3 * rng.standard_normal((T, D))).astype(np.float32)
    modal = (0.3 * rng.standard_normal((M, D))).astype(np.float32)
    src = np.full(T, -1, np.int32); src[2:5] = np.arange(3); src[9:11] = np.arange(3, 5)
    o = tk_torch.merge_multimodal_spans(torch.from_numpy(text).to("mps"),
                                        torch.from_numpy(modal).to("mps"),
                                        torch.from_numpy(src).to("mps")).cpu().numpy()
    ref = np.where(src[:, None] >= 0, modal[np.clip(src, 0, M - 1)], text)
    np.testing.assert_allclose(o, ref, atol=1e-5)


@pytest.mark.parametrize("B,S,V", [(3, 4, 50), (2, 1, 128)])
def test_spec_verify_linear(B, S, V):
    import numpy as np
    rng = np.random.default_rng(B + S + V)
    dp = rng.dirichlet(np.ones(V), size=(B, S)).astype(np.float32)
    tp = rng.dirichlet(np.ones(V), size=(B, S + 1)).astype(np.float32)
    dt = rng.integers(0, V, size=(B, S)).astype(np.int32)
    bonus = rng.integers(0, V, size=B).astype(np.int32)
    mps = lambda a: torch.from_numpy(a).to("mps")
    # all-accept -> drafts + bonus
    au0 = np.full((B, S), 1e-9, np.float32)
    o, cnt = tk_torch.spec_verify_linear(mps(dt), mps(dp), mps(tp), mps(bonus), mps(au0), 1)
    o = o.cpu().numpy(); cnt = cnt.cpu().numpy()
    assert (cnt == S).all()
    np.testing.assert_array_equal(o[:, :S], dt)
    np.testing.assert_array_equal(o[:, S], bonus)
    # mixed accept -> accepted_cnt matches the u <= p_t/p_d oracle
    au1 = np.full((B, S), 0.99, np.float32)
    o, cnt = tk_torch.spec_verify_linear(mps(dt), mps(dp), mps(tp), mps(bonus), mps(au1), 1)
    o = o.cpu().numpy(); cnt = cnt.cpu().numpy()
    ref = np.zeros(B, np.int32)
    for b in range(B):
        c = S
        for i in range(S):
            if not (au1[b, i] * dp[b, i, dt[b, i]] <= tp[b, i, dt[b, i]]):
                c = i; break
        ref[b] = c
    np.testing.assert_array_equal(cnt, ref)


@pytest.mark.parametrize("B,BM,V", [(2, 4, 4000), (3, 8, 4000)])
def test_beam_advance(B, BM, V):
    import numpy as np
    rng = np.random.default_rng(B + BM)
    logits = (rng.standard_normal((B * BM, V)) * 2.0).astype(np.float32)
    cum = rng.standard_normal((B, BM)).astype(np.float32)
    nt, pb, nc = tk_torch.beam_advance(torch.from_numpy(logits).to("mps"),
                                       torch.from_numpy(cum).to("mps"), BM)
    lg = logits.reshape(B, BM, V).astype(np.float64)
    mx_ = lg.max(2, keepdims=True)
    lse = np.log(np.exp(lg - mx_).sum(2, keepdims=True)) + mx_
    scores = (lg - lse) + cum.reshape(B, BM, 1)
    ont = np.zeros((B, BM), np.int32)
    opb = np.zeros((B, BM), np.int32)
    for b in range(B):
        order = np.argsort(-scores[b].reshape(-1), kind="stable")[:BM]
        opb[b] = order // V
        ont[b] = order % V
    np.testing.assert_array_equal(nt.cpu().numpy(), ont)
    np.testing.assert_array_equal(pb.cpu().numpy(), opb)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0"])
def test_lm_head_sample_quant(fmt):
    import numpy as np
    import tk
    from tk.quant import QUANT_FORMATS
    quant, dequant = QUANT_FORMATS[fmt]
    T, V, K = 4, 4000, 512
    rng = np.random.default_rng(11)
    W = (0.3 * rng.standard_normal((V, K))).astype(np.float32)
    Wq = quant(W)
    h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
    hm = torch.from_numpy(h).to(torch.bfloat16).to("mps")
    tok = tk.lm_head_sample(hm, torch.from_numpy(Wq).to("mps"), mode="argmax",
                            format=fmt).cpu().numpy()
    L = (h.astype(np.float64)) @ dequant(Wq).astype(np.float64).T
    for t in range(T):
        assert tok[t] == L[t].argmax() or (L[t].max() - L[t, tok[t]]) < 1e-2
    # fused quant top-k vs the dequant-logits top-k oracle
    for k in (1, 8):
        tk_tok = tk.lm_head_sample(hm, torch.from_numpy(Wq).to("mps"), mode="topk", k=k,
                                   temperature=0.8, seed=3, format=fmt).cpu().numpy()
        for t in range(T):
            top = set(int(v) for v in np.argsort(-L[t], kind="stable")[:k])
            assert tk_tok[t] in top or (L[t].max() - L[t, tk_tok[t]]) < 1e-2


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


# NOTE the mamba2 oracles mask the decay EXPONENT (with -inf) before exp: exp(cl_i - cl_j)
# overflows to inf in the upper triangle once N*|log a| exceeds ~88, and inf * 0 = NaN (forward)
# / 0 * inf = NaN (through tril's backward) would poison the reference. The kernels never form
# the upper triangle, so they are unaffected.
def _mamba2_fwd_ref(C, Bm, X, cumlog, N):
    scores = C.float() @ Bm.float().transpose(-1, -2)
    expo = cumlog[..., :, None] - cumlog[..., None, :]
    causal = torch.tril(torch.ones(N, N, dtype=torch.bool, device=C.device))
    expo = expo.masked_fill(~causal, float("-inf"))
    return (scores * torch.exp(expo)) @ X.float()


@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (2, 2, 128, 64),
                                   # auto-routed chunked (N >= threshold, N%64==0), both head dims:
                                   (1, 1, 2048, 64), (1, 1, 4096, 128)])
def test_mamba2(shape):
    B, H, N, D = shape
    torch.manual_seed(0)
    C = torch.randn(shape, dtype=torch.bfloat16, device="mps") * 0.5
    Bm = torch.randn(shape, dtype=torch.bfloat16, device="mps") * 0.5
    X = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    a = torch.sigmoid(torch.randn(B, H, N, device="mps")) * 0.5 + 0.5
    cumlog = torch.cumsum(torch.log(a), dim=-1).float()
    got = tk_torch.mamba2(C, Bm, X, cumlog)
    exp = _mamba2_fwd_ref(C, Bm, X, cumlog, N)
    torch.mps.synchronize()
    diff = (got.float() - exp).abs().max().item()
    scale = exp.abs().max().item() + 1e-9
    assert diff / scale < 0.03


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 2, 192, 64),
                                   (1, 2, 128, 128), (1, 1, 256, 128)])
def test_mamba2_chunked_forced(shape):
    """The forced chunked route at small N (below the auto thresholds), both head dims."""
    B, H, N, D = shape
    torch.manual_seed(N + D)
    C = torch.randn(shape, dtype=torch.bfloat16, device="mps") * 0.5
    Bm = torch.randn(shape, dtype=torch.bfloat16, device="mps") * 0.5
    X = torch.randn(shape, dtype=torch.bfloat16, device="mps")
    a = torch.sigmoid(torch.randn(B, H, N, device="mps")) * 0.5 + 0.5
    cumlog = torch.cumsum(torch.log(a), dim=-1).float()
    got = tk_torch.mamba2_chunked(C, Bm, X, cumlog)
    exp = _mamba2_fwd_ref(C, Bm, X, cumlog, N)
    torch.mps.synchronize()
    diff = (got.float() - exp).abs().max().item()
    scale = exp.abs().max().item() + 1e-9
    assert diff / scale < 0.03


def _mamba2_bwd_oracle(C, Bn, X, cl, dY):
    import numpy as np  # noqa: F401
    Ct = torch.tensor(C, requires_grad=True)
    Bt = torch.tensor(Bn, requires_grad=True)
    Xt = torch.tensor(X, requires_grad=True)
    clt = torch.tensor(cl, requires_grad=True)
    N = C.shape[2]
    G = torch.einsum("bhid,bhjd->bhij", Ct, Bt)
    expo = clt[:, :, :, None] - clt[:, :, None, :]
    causal = torch.arange(N)[None, :] <= torch.arange(N)[:, None]
    expo = expo.masked_fill(~causal, float("-inf"))
    Y = torch.einsum("bhij,bhjd->bhid", G * torch.exp(expo), Xt)
    Y.backward(torch.tensor(dY))
    return Ct.grad.numpy(), Bt.grad.numpy(), Xt.grad.numpy(), clt.grad.numpy()


def _mamba2_bwd_inputs(shape, seed):
    import numpy as np
    Bh, H, N, D = shape
    rng = np.random.default_rng(seed)
    C = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    Bn = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    X = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    dY = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    a = rng.uniform(0.9, 1.0, (Bh, H, N)).astype(np.float32)
    cl = np.cumsum(np.log(a), axis=2).astype(np.float32)
    return C, Bn, X, cl, dY


@pytest.mark.parametrize("shape", [(2, 2, 64, 64), (1, 1, 128, 128), (1, 2, 128, 64),
                                   # auto-routed chunked (N >= threshold, N%64==0), both head dims:
                                   (1, 1, 2048, 64), (1, 1, 4096, 128)])
def test_mamba2_bwd(shape):
    import numpy as np
    Bh, H, N, D = shape
    C, Bn, X, cl, dY = _mamba2_bwd_inputs(shape, Bh + N + D)
    rC, rB, rX, rcl = _mamba2_bwd_oracle(C, Bn, X, cl, dY)
    dC, dB, dX, dcl = tk_torch.mamba2_bwd(
        torch.tensor(C).to(torch.bfloat16).to("mps"), torch.tensor(Bn).to(torch.bfloat16).to("mps"),
        torch.tensor(X).to(torch.bfloat16).to("mps"), torch.tensor(cl).to("mps"),
        torch.tensor(dY).to(torch.bfloat16).to("mps"))

    def rel(g, ref):
        return np.abs(g.float().cpu().numpy() - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel(dC, rC) < 0.06
    assert rel(dB, rB) < 0.06
    assert rel(dX, rX) < 0.06
    assert rel(dcl, rcl) < 0.08


@pytest.mark.parametrize("shape", [(1, 1, 128, 64), (2, 2, 192, 64),
                                   (1, 2, 128, 128), (1, 1, 256, 128)])
def test_mamba2_bwd_chunked_forced(shape):
    """The forced chunked linear-time backward at small N, both head dims, vs autograd."""
    import numpy as np
    Bh, H, N, D = shape
    C, Bn, X, cl, dY = _mamba2_bwd_inputs(shape, 10 + N + D)
    rC, rB, rX, rcl = _mamba2_bwd_oracle(C, Bn, X, cl, dY)
    dC, dB, dX, dcl = tk_torch.mamba2_bwd_chunked(
        torch.tensor(C).to(torch.bfloat16).to("mps"), torch.tensor(Bn).to(torch.bfloat16).to("mps"),
        torch.tensor(X).to(torch.bfloat16).to("mps"), torch.tensor(cl).to("mps"),
        torch.tensor(dY).to(torch.bfloat16).to("mps"))

    def rel(g, ref):
        return np.abs(g.float().cpu().numpy() - ref).max() / (np.abs(ref).max() + 1e-6)
    assert rel(dC, rC) < 0.06
    assert rel(dB, rB) < 0.06
    assert rel(dX, rX) < 0.06
    assert rel(dcl, rcl) < 0.08


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("decay", [True, False])
def test_ssd_decode(D, decay):
    """T iterated decode steps == the fp32 recurrence oracle; the state updates IN PLACE. decay=False
    pins alpha == 1 (pure accumulation / undecayed linear attention)."""
    import numpy as np
    B, H, T = 2, 2, 5
    rng = np.random.default_rng(3 + D + (0 if decay else 100))
    S_ref = np.zeros((B, H, D, D), np.float32)
    S = torch.zeros(B, H, D, D, device="mps")
    for t in range(T):
        alpha = (rng.uniform(0.9, 1.0, (B, H)).astype(np.float32) if decay
                 else np.ones((B, H), np.float32))
        x = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
        k = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
        q = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
        y, S_out = tk_torch.ssd_decode(S, torch.tensor(alpha, device="mps"),
                                       torch.tensor(x, device="mps"),
                                       torch.tensor(k, device="mps"),
                                       torch.tensor(q, device="mps"))
        torch.mps.synchronize()
        assert S_out.data_ptr() == S.data_ptr()      # in-place state update
        S_ref = alpha[..., None, None] * S_ref + np.einsum("bhp,bhn->bhpn", x, k)
        ref = np.einsum("bhpn,bhn->bhp", S_ref, q)
        scale = np.abs(ref).max() + 1e-6
        assert np.abs(y.cpu().numpy() - ref).max() / scale < 1e-4, f"step {t}"
        assert np.abs(S.cpu().numpy() - S_ref).max() / (np.abs(S_ref).max() + 1e-6) < 1e-4


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


@pytest.mark.parametrize("R,D", [(8, 256), (16, 64)])
def test_rms_norm_backward(R, D):
    import numpy as np, tk
    rng = np.random.default_rng(R + D)
    x = (0.5 * rng.standard_normal((R, D))).astype(np.float32)
    w = (0.5 * rng.standard_normal((D,))).astype(np.float32)
    dy = (0.3 * rng.standard_normal((R, D))).astype(np.float32)
    xt = torch.tensor(x, requires_grad=True); wt = torch.tensor(w, requires_grad=True)
    (xt * torch.rsqrt((xt ** 2).mean(-1, keepdim=True) + 1e-5) * wt).backward(torch.tensor(dy))
    dx, dw = tk.rms_norm_backward(torch.from_numpy(x).to("mps"), torch.from_numpy(w).to("mps"),
                                  torch.from_numpy(dy).to("mps"), 1e-5)
    rel = lambda g, r: np.abs(g.float().cpu().numpy() - r).max() / (np.abs(r).max() + 1e-9)
    assert rel(dx, xt.grad.numpy()) < 1e-4 and rel(dw, wt.grad.numpy()) < 1e-4


@pytest.mark.parametrize("R,D", [(8, 256), (16, 64)])
def test_layernorm_backward(R, D):
    import numpy as np, tk
    rng = np.random.default_rng(R + D + 1)
    x = (0.5 * rng.standard_normal((R, D))).astype(np.float32)
    w = (0.5 * rng.standard_normal((D,))).astype(np.float32)
    b = (0.3 * rng.standard_normal((D,))).astype(np.float32)
    dy = (0.3 * rng.standard_normal((R, D))).astype(np.float32)
    xt = torch.tensor(x, requires_grad=True); wt = torch.tensor(w, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    torch.nn.functional.layer_norm(xt, (D,), wt, bt, 1e-5).backward(torch.tensor(dy))
    dx, dw, db = tk.layernorm_backward(torch.from_numpy(x).to("mps"), torch.from_numpy(w).to("mps"),
                                       torch.from_numpy(dy).to("mps"), 1e-5)
    rel = lambda g, r: np.abs(g.float().cpu().numpy() - r).max() / (np.abs(r).max() + 1e-9)
    assert rel(dx, xt.grad.numpy()) < 1e-4 and rel(dw, wt.grad.numpy()) < 1e-4 and rel(db, bt.grad.numpy()) < 1e-4


def test_gelu_backward():
    import numpy as np, tk
    rng = np.random.default_rng(2)
    x = (1.5 * rng.standard_normal((6, 128))).astype(np.float32)
    dy = (0.4 * rng.standard_normal((6, 128))).astype(np.float32)
    xt = torch.tensor(x, requires_grad=True)
    torch.nn.functional.gelu(xt, approximate="tanh").backward(torch.tensor(dy))
    dx = tk.gelu_backward(torch.from_numpy(x).to("mps"), torch.from_numpy(dy).to("mps"))
    assert np.abs(dx.float().cpu().numpy() - xt.grad.numpy()).max() / (np.abs(xt.grad.numpy()).max() + 1e-9) < 1e-4
