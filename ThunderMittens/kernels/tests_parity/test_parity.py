"""Cross-backend parity tests: the MLX and PyTorch-MPS backends run the SAME
compiled metallib kernel, so for identical inputs they must produce (near) identical
output. This is the strongest guarantee for the dual-backend design and catches any
host-ABI drift between <kernel>.cpp (MLX) and torch_kernels.mm (Torch).

Requires both mlx and torch; skips cleanly if either is missing. Run from kernels/:

    python -m pytest tests_parity/test_parity.py -v
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

if not torch.backends.mps.is_available():
    pytest.skip("MPS not available", allow_module_level=True)

import tk  # the type-dispatching API  # noqa: E402


def _mk(arr, fw, dtype="bf16"):
    """Build a matched input on each framework from one numpy fp32 array."""
    if fw == "torch":
        t = torch.from_numpy(arr)
        t = t.to(torch.bfloat16) if dtype == "bf16" else t.to(torch.float32)
        return t.to("mps")
    a = mx.array(arr)
    return a.astype(mx.bfloat16) if dtype == "bf16" else a.astype(mx.float32)


def _np(x):
    """Bring an mlx array or torch tensor back to fp32 numpy."""
    if type(x).__module__.split(".")[0] == "torch":
        return x.detach().float().cpu().numpy()
    mx.eval(x)
    return np.array(x.astype(mx.float32))


def _assert_parity(o_mlx, o_torch, atol):
    mx.eval(o_mlx)
    torch.mps.synchronize()
    a, b = _np(o_mlx), _np(o_torch)
    assert a.shape == b.shape, (a.shape, b.shape)
    d = float(np.max(np.abs(a - b)))
    assert d <= atol, f"MLX vs MPS max|diff|={d} (atol={atol})"


@pytest.mark.parametrize("shape", [(2, 128, 1024), (1, 256, 768), (8, 256)])
def test_layernorm_parity(shape):
    D = shape[-1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal((D,)).astype(np.float32)
    b = rng.standard_normal((D,)).astype(np.float32)
    om = tk.layernorm(_mk(x, "mlx"), _mk(w, "mlx"), _mk(b, "mlx"))
    ot = tk.layernorm(_mk(x, "torch"), _mk(w, "torch"), _mk(b, "torch"))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("shape", [(64, 128), (128, 64)])
def test_add_rt_parity(shape, dtype):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    y = rng.standard_normal(shape).astype(np.float32)
    om = tk.add_rt(_mk(x, "mlx", dtype), _mk(y, "mlx", dtype))
    ot = tk.add_rt(_mk(x, "torch", dtype), _mk(y, "torch", dtype))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("dtype,atol", [("f32", 1e-3), ("bf16", 1e-2)])
@pytest.mark.parametrize("nkm", [(32, 16, 32), (128, 64, 128)])
def test_matmul_parity(nkm, dtype, atol):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    x = rng.random((N, K), dtype=np.float32)
    y = rng.random((K, M), dtype=np.float32)
    om = tk.matmul_custom(_mk(x, "mlx", dtype), _mk(y, "mlx", dtype))
    ot = tk.matmul_custom(_mk(x, "torch", dtype), _mk(y, "torch", dtype))
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (2, 2, 128, 128)])
def test_attn_fwd_parity(shape):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.attn_fwd(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"))
    ot = tk.attn_fwd(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "fp8_e4m3"])
def test_attn_q_parity(fmt):
    from tk.quant import quantize_kv
    B, H, N, D = 1, 2, 64, 64
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    Kq = quantize_kv((rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32), fmt)
    Vq = quantize_kv((rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32), fmt)
    om = tk.attn_q(mx.array(q).astype(mx.bfloat16), mx.array(Kq), mx.array(Vq), format=fmt)
    ot = tk.attn_q(torch.from_numpy(q).to(torch.bfloat16).to("mps"),
                   torch.from_numpy(Kq).to("mps"), torch.from_numpy(Vq).to("mps"), format=fmt)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("D,H,H_KV", [(64, 4, 4), (64, 8, 2), (128, 4, 4)])
def test_attn_varlen_prefill_parity(D, H, H_KV):
    rng = np.random.default_rng(3)
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
    cl = np.array(ctxs, np.int32)
    om = tk.attn_varlen_prefill(_mk(q, "mlx"), _mk(kc, "mlx"), _mk(vc, "mlx"),
                                mx.array(bt), mx.array(cl), cu, scale=float(scale))
    ot = tk.attn_varlen_prefill(_mk(q, "torch"), _mk(kc, "torch"), _mk(vc, "torch"),
                                torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                                cu, scale=float(scale))
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("mode,k", [("argmax", 0), ("categorical", 0), ("topk", 8)])
def test_lm_head_sample_parity(mode, k):
    # fused=True runs the SAME metallib kernel + same serial-dot logit path on both backends, so
    # token ids match exactly (the default matmul path can differ by ULP-ties across frameworks).
    rng = np.random.default_rng(4)
    T, V, K = 4, 4096, 512
    h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
    W = (0.5 * rng.standard_normal((V, K))).astype(np.float32)
    om = tk.lm_head_sample(_mk(h, "mlx"), _mk(W, "mlx"), mode=mode, k=k,
                           temperature=0.8, seed=99, fused=True)
    ot = tk.lm_head_sample(_mk(h, "torch"), _mk(W, "torch"), mode=mode, k=k,
                           temperature=0.8, seed=99, fused=True)
    _assert_parity(om, ot, atol=0)   # integer token ids: exact


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0"])
@pytest.mark.parametrize("mode,k", [("argmax", 0), ("topk", 8)])
def test_lm_head_sample_quant_parity(fmt, mode, k):
    from tk.quant import QUANT_FORMATS
    quant, _ = QUANT_FORMATS[fmt]
    rng = np.random.default_rng(fmt.__hash__() % 1000 + k)
    T, V, K = 4, 4000, 512
    h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
    Wq = quant((0.3 * rng.standard_normal((V, K))).astype(np.float32))
    om = tk.lm_head_sample(_mk(h, "mlx"), mx.array(Wq), mode=mode, k=k, temperature=0.8, seed=9,
                           format=fmt)
    ot = tk.lm_head_sample(_mk(h, "torch"), torch.from_numpy(Wq).to("mps"), mode=mode, k=k,
                           temperature=0.8, seed=9, format=fmt)
    _assert_parity(om, ot, atol=0)   # same dequant + reduce metallib -> exact token ids


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0"])
@pytest.mark.parametrize("p", [0.5, 0.9])
def test_lm_head_sample_quant_topp_parity(fmt, p):
    from tk.quant import QUANT_FORMATS
    quant, _ = QUANT_FORMATS[fmt]
    rng = np.random.default_rng(fmt.__hash__() % 1000 + int(p * 10))
    T, V, K = 4, 4000, 512
    h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
    Wq = quant((0.3 * rng.standard_normal((V, K))).astype(np.float32))
    om = tk.lm_head_sample(_mk(h, "mlx"), mx.array(Wq), mode="topp", k=32, temperature=0.8,
                           seed=9, format=fmt, top_p=p)
    ot = tk.lm_head_sample(_mk(h, "torch"), torch.from_numpy(Wq).to("mps"), mode="topp", k=32,
                           temperature=0.8, seed=9, format=fmt, top_p=p)
    _assert_parity(om, ot, atol=0)   # same pool + bisection + Gumbel -> exact token ids


@pytest.mark.parametrize("eps,z,softcap", [(0.0, 0.0, 0.0), (0.1, 1e-4, 0.0), (0.0, 0.0, 30.0)])
def test_cross_entropy_parity(eps, z, softcap):
    rng = np.random.default_rng(6)
    T, V = 8, 4000
    logits = (rng.standard_normal((T, V)) * 2.0).astype(np.float32)
    tgt = rng.integers(0, V, size=(T,)).astype(np.int32)
    tgt[:2] = -100
    lm_loss, lm_lse = tk.cross_entropy(_mk(logits, "mlx"), mx.array(tgt), reduction="none",
                                       label_smoothing=eps, z_loss=z, softcap=softcap,
                                       return_lse=True)
    lt_loss, lt_lse = tk.cross_entropy(_mk(logits, "torch"), torch.from_numpy(tgt).to("mps"),
                                       reduction="none", label_smoothing=eps, z_loss=z,
                                       softcap=softcap, return_lse=True)
    _assert_parity(lm_loss, lt_loss, atol=6e-2)
    _assert_parity(lm_lse, lt_lse, atol=6e-2)
    # backward parity
    go_m = mx.full((T,), 0.25, dtype=mx.float32)
    gm = tk.cross_entropy_grad(_mk(logits, "mlx"), mx.array(tgt), lm_lse, go_m,
                               label_smoothing=eps, z_loss=z, softcap=softcap)
    gt = tk.cross_entropy_grad(_mk(logits, "torch"), torch.from_numpy(tgt).to("mps"), lt_lse,
                               torch.full((T,), 0.25, device="mps"), label_smoothing=eps, z_loss=z,
                               softcap=softcap)
    _assert_parity(gm, gt, atol=6e-2)


@pytest.mark.parametrize("B,BM,V", [(2, 4, 4000), (3, 8, 4000)])
def test_beam_advance_parity(B, BM, V):
    rng = np.random.default_rng(7)
    logits = (rng.standard_normal((B * BM, V)) * 2.0).astype(np.float32)
    cum = rng.standard_normal((B, BM)).astype(np.float32)
    om = tk.beam_advance(_mk(logits, "mlx", "f32"), mx.array(cum), BM)
    ot = tk.beam_advance(_mk(logits, "torch", "f32"), torch.from_numpy(cum).to("mps"), BM)
    _assert_parity(om[0], ot[0], atol=0)     # token ids: exact
    _assert_parity(om[1], ot[1], atol=0)     # parents: exact
    _assert_parity(om[2], ot[2], atol=1e-4)  # scores


def test_embedding_lookup_parity():
    rng = np.random.default_rng(0)
    vocab, D, T = 200, 128, 12
    table = (0.3 * rng.standard_normal((vocab, D))).astype(np.float32)
    tok = rng.integers(0, vocab, size=T).astype(np.int32); tok[4] = -1
    pos = (0.2 * rng.standard_normal((T, D))).astype(np.float32)
    om = tk.embedding_lookup(mx.array(tok), _mk(table, "mlx", "f32"), pos_table=mx.array(pos), scale=1.5)
    ot = tk.embedding_lookup(torch.from_numpy(tok).to("mps"), _mk(table, "torch", "f32"),
                             pos_table=torch.from_numpy(pos).to("mps"), scale=1.5)
    _assert_parity(om, ot, atol=0)


def test_adamw_parity():
    rng = np.random.default_rng(23)
    D = 1024
    p = (0.1 * rng.standard_normal(D)).astype(np.float32)
    g = rng.standard_normal(D).astype(np.float32)
    m = np.abs(rng.standard_normal(D)).astype(np.float32)
    v = np.abs(rng.standard_normal(D)).astype(np.float32)
    kw = dict(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, step=3)
    pm, mm, vm = tk.adamw(_mk(p, "mlx", "f32"), _mk(g, "mlx", "f32"),
                          _mk(m, "mlx", "f32"), _mk(v, "mlx", "f32"), **kw)
    ptt, mtt, vtt = tk.adamw(_mk(p, "torch", "f32"), _mk(g, "torch", "f32"),
                             _mk(m, "torch", "f32"), _mk(v, "torch", "f32"), **kw)
    _assert_parity(pm, ptt, atol=1e-6)
    _assert_parity(mm, mtt, atol=1e-6)
    _assert_parity(vm, vtt, atol=1e-6)


@pytest.mark.parametrize("p", [0.0, 0.3, 0.7])
def test_dropout_parity(p):
    # same (seed, index) hash on both backends -> IDENTICAL mask, so fwd/bwd are bit-parity.
    rng = np.random.default_rng(int(p * 10) + 4)
    x = rng.standard_normal((32, 256)).astype(np.float32)
    dy = rng.standard_normal((32, 256)).astype(np.float32)
    seed = 77
    om = tk.dropout(_mk(x, "mlx", "f32"), p, seed)
    ot = tk.dropout(_mk(x, "torch", "f32"), p, seed)
    _assert_parity(om, ot, atol=1e-6)
    bm = tk.dropout_backward(_mk(dy, "mlx", "f32"), p, seed)
    bt = tk.dropout_backward(_mk(dy, "torch", "f32"), p, seed)
    _assert_parity(bm, bt, atol=1e-6)


@pytest.mark.parametrize("mode", ["reglu", "geglu", "swiglu", "swiglu_oai", "geglu_erf",
                                  "geglu_quick"])
def test_glu_backward_parity(mode):
    rng = np.random.default_rng(31)
    x = rng.standard_normal((4, 512)).astype(np.float32)
    g = rng.standard_normal((4, 512)).astype(np.float32)
    dc = rng.standard_normal((4, 512)).astype(np.float32)
    dam, dbm = tk.glu_backward(_mk(x, "mlx", "f32"), _mk(g, "mlx", "f32"),
                               _mk(dc, "mlx", "f32"), mode=mode, alpha=1.3, limit=2.5)
    dat, dbt = tk.glu_backward(_mk(x, "torch", "f32"), _mk(g, "torch", "f32"),
                               _mk(dc, "torch", "f32"), mode=mode, alpha=1.3, limit=2.5)
    _assert_parity(dam, dat, atol=1e-5)
    _assert_parity(dbm, dbt, atol=1e-5)


def test_embedding_backward_parity():
    rng = np.random.default_rng(19)
    vocab, D, T = 60, 128, 40
    tok = rng.integers(0, vocab, size=T).astype(np.int32); tok[3] = -1
    tok[7] = tok[11] = tok[19] = 5                  # duplicate id -> atomic accumulation
    dY = (0.5 * rng.standard_normal((T, D))).astype(np.float32)
    om = tk.embedding_backward(mx.array(tok), _mk(dY, "mlx", "f32"), vocab, scale=1.5)
    ot = tk.embedding_backward(torch.from_numpy(tok).to("mps"), _mk(dY, "torch", "f32"),
                               vocab, scale=1.5)
    _assert_parity(om, ot, atol=1e-5)   # atomic add order differs -> not bit-exact


@pytest.mark.parametrize("BM", [4, 8])
def test_beam_remap_block_table_parity(BM):
    rng = np.random.default_rng(BM + 3)
    B, ctx, block_size = 2, 512, 16
    max_blocks = ctx // block_size
    nbeams = B * BM
    bt = np.arange(nbeams * max_blocks, dtype=np.int32).reshape(nbeams, max_blocks)
    parent = rng.integers(0, BM, size=(B, BM)).astype(np.int32)
    om = tk.beam_remap_block_table(mx.array(bt), mx.array(parent))
    ot = tk.beam_remap_block_table(torch.from_numpy(bt).to("mps"), torch.from_numpy(parent).to("mps"))
    _assert_parity(om, ot, atol=0)


def test_build_multimodal_src_parity():
    so = np.array([5, 15, 30], np.int32)
    sl = np.array([4, 6, 5], np.int32)
    ms = np.array([0, 4, 10], np.int32)
    T = 40
    om = tk.build_multimodal_src(mx.array(so), mx.array(sl), mx.array(ms), T)
    ot = tk.build_multimodal_src(torch.from_numpy(so).to("mps"), torch.from_numpy(sl).to("mps"),
                                 torch.from_numpy(ms).to("mps"), T)
    _assert_parity(om, ot, atol=0)


def test_merge_multimodal_spans_parity():
    rng = np.random.default_rng(3)
    T, M, D = 16, 6, 128
    text = (0.3 * rng.standard_normal((T, D))).astype(np.float32)
    modal = (0.3 * rng.standard_normal((M, D))).astype(np.float32)
    src = np.full(T, -1, np.int32); src[2:5] = np.arange(3); src[9:11] = np.arange(3, 5)
    om = tk.merge_multimodal_spans(_mk(text, "mlx", "f32"), _mk(modal, "mlx", "f32"), mx.array(src))
    ot = tk.merge_multimodal_spans(_mk(text, "torch", "f32"), _mk(modal, "torch", "f32"),
                                   torch.from_numpy(src).to("mps"))
    _assert_parity(om, ot, atol=0)


@pytest.mark.parametrize("min_p", [0.2, 0.5])
def test_min_p_sample_parity(min_p):
    rng = np.random.default_rng(int(min_p * 100))
    logits = (rng.standard_normal((4, 500)) * 2).astype(np.float32)
    om = tk.min_p_sample(_mk(logits, "mlx", "f32"), min_p, seed=9)
    ot = tk.min_p_sample(_mk(logits, "torch", "f32"), min_p, seed=9)
    _assert_parity(om, ot, atol=0)           # token ids: exact (same Gumbel)


@pytest.mark.parametrize("typical_p", [0.3, 0.9])
def test_typical_p_sample_parity(typical_p):
    rng = np.random.default_rng(int(typical_p * 100))
    logits = (rng.standard_normal((4, 500)) * 1.5).astype(np.float32)
    om = tk.typical_p_sample(_mk(logits, "mlx", "f32"), typical_p, temperature=0.8, seed=9)
    ot = tk.typical_p_sample(_mk(logits, "torch", "f32"), typical_p, temperature=0.8, seed=9)
    _assert_parity(om, ot, atol=0)           # token ids: exact (same threshold + Gumbel)


@pytest.mark.parametrize("V", [50, 200])
def test_apply_bad_words_parity(V):
    rng = np.random.default_rng(V + 9)
    T, maxbad = 3, 6
    logits = rng.standard_normal((T, V)).astype(np.float32)
    bad_lens = rng.integers(0, maxbad + 1, size=T).astype(np.int32)
    bad_ids = rng.integers(0, V, size=(T, maxbad)).astype(np.int32)
    om = tk.apply_bad_words(_mk(logits, "mlx", "f32"), mx.array(bad_ids), mx.array(bad_lens))
    ot = tk.apply_bad_words(_mk(logits, "torch", "f32"), torch.from_numpy(bad_ids).to("mps"),
                            torch.from_numpy(bad_lens).to("mps"))
    _assert_parity(om, ot, atol=0)           # masked logits: bit-identical


@pytest.mark.parametrize("V", [40, 200])
def test_apply_token_bitmask_parity(V):
    rng = np.random.default_rng(V)
    T = 3
    logits = rng.standard_normal((T, V)).astype(np.float32)
    allow = rng.integers(0, 2, size=(T, V)).astype(bool)
    allow[:, 0] = True
    nw = (V + 31) // 32
    m = np.zeros((T, nw), np.uint32)
    for t in range(T):
        for v in range(V):
            if allow[t, v]:
                m[t, v >> 5] |= np.uint32(1) << np.uint32(v & 31)
    mi = m.view(np.int32)
    om = tk.apply_token_bitmask(_mk(logits, "mlx", "f32"), mx.array(mi))
    ot = tk.apply_token_bitmask(_mk(logits, "torch", "f32"), torch.from_numpy(mi).to("mps"))
    _assert_parity(om, ot, atol=0)           # masked logits: bit-identical


@pytest.mark.parametrize("seed,parents", [
    (3, [-1, 0, 0, 1, 1, 2, 2]),
    (11, [-1, 0, 0, 1, 1, 2, 2]),
    (5, [-1] + [0] * 130),                       # wide star: 130 siblings (>64), exercises no-cap residual
])
def test_spec_verify_tree_parity(seed, parents):
    from tk import spec_build_tree_pointers
    rng = np.random.default_rng(seed)
    B, V = 3, 400
    N = len(parents)
    nt, ns = spec_build_tree_pointers(parents, N)
    nt = np.broadcast_to(nt, (B, N)).copy(); ns = np.broadcast_to(ns, (B, N)).copy()
    draft = rng.integers(0, V, size=(B, N - 1)).astype(np.int32)
    tp = np.abs(rng.standard_normal((B, N, V))).astype(np.float32)
    tp /= tp.sum(-1, keepdims=True)
    tv = rng.integers(0, 2, size=B).astype(np.int32)      # mixed valid/invalid rows
    om = tk.spec_verify_tree(mx.array(draft), mx.array(tp), mx.array(nt), mx.array(ns), seed,
                             tree_valid=mx.array(tv))
    ot = tk.spec_verify_tree(torch.from_numpy(draft).to("mps"), torch.from_numpy(tp).to("mps"),
                             torch.from_numpy(nt).to("mps"), torch.from_numpy(ns).to("mps"), seed,
                             tree_valid=torch.from_numpy(tv).to("mps"))
    for a, b in zip(om, ot):                     # accept path + terminal Gumbel-max both deterministic
        _assert_parity(a, b, atol=0)


@pytest.mark.parametrize("B,S", [(3, 4), (8, 5), (300, 4)])
def test_spec_compact_parity(B, S):
    rng = np.random.default_rng(B * 7 + S)
    Sp1 = S + 1
    accepted_cnt = rng.integers(0, S + 1, size=B).astype(np.int32)
    seq_lens = rng.integers(1, 100, size=B).astype(np.int32)
    out_tokens = np.full((B, Sp1), -1, np.int32)
    for b in range(B):
        for j in range(int(accepted_cnt[b]) + 1):
            out_tokens[b, j] = rng.integers(0, 32000)
    om = tk.spec_compact(mx.array(out_tokens), mx.array(accepted_cnt), mx.array(seq_lens))
    ot = tk.spec_compact(torch.from_numpy(out_tokens).to("mps"),
                         torch.from_numpy(accepted_cnt).to("mps"),
                         torch.from_numpy(seq_lens).to("mps"))
    for a, b in zip(om, ot):
        _assert_parity(a, b, atol=0)
    nm = tk.spec_update_kv_meta(mx.array(seq_lens), mx.array(accepted_cnt))
    nt = tk.spec_update_kv_meta(torch.from_numpy(seq_lens).to("mps"),
                                torch.from_numpy(accepted_cnt).to("mps"))
    _assert_parity(nm, nt, atol=0)


@pytest.mark.parametrize("B,S,V", [(3, 4, 50), (2, 5, 200)])
def test_spec_verify_linear_parity(B, S, V):
    rng = np.random.default_rng(B + S + V)
    dp = rng.dirichlet(np.ones(V), size=(B, S)).astype(np.float32)
    tp = rng.dirichlet(np.ones(V), size=(B, S + 1)).astype(np.float32)
    dt = rng.integers(0, V, size=(B, S)).astype(np.int32)
    bonus = rng.integers(0, V, size=B).astype(np.int32)
    au = rng.uniform(0.0, 1.0, size=(B, S)).astype(np.float32)   # mixed accept + rejections
    om = tk.spec_verify_linear(mx.array(dt), mx.array(dp), mx.array(tp), mx.array(bonus),
                               mx.array(au), 7)
    ot = tk.spec_verify_linear(torch.from_numpy(dt).to("mps"), torch.from_numpy(dp).to("mps"),
                               torch.from_numpy(tp).to("mps"), torch.from_numpy(bonus).to("mps"),
                               torch.from_numpy(au).to("mps"), 7)
    _assert_parity(om[0], ot[0], atol=0)     # out_tokens (incl. recovered Gumbel token): exact
    _assert_parity(om[1], ot[1], atol=0)     # accepted_cnt: exact


@pytest.mark.parametrize("B,BM", [(2, 3), (1, 4)])
def test_beam_reorder_kv_parity(B, BM):
    rng = np.random.default_rng(B + BM)
    bs, H_KV, D, max_blocks = 4, 2, 16, 2
    nbeams = B * BM
    nb = nbeams * max_blocks
    kc = rng.standard_normal((nb, bs, H_KV, D)).astype(np.float32)
    vc = rng.standard_normal((nb, bs, H_KV, D)).astype(np.float32)
    bt = np.arange(nb, dtype=np.int32).reshape(nbeams, max_blocks)
    pb = rng.integers(0, BM, size=(B, BM)).astype(np.int32)   # random parents -> fan-out + chains
    sl = np.full(nbeams, 7, np.int32)
    km, vm = tk.beam_reorder_kv(mx.array(kc).astype(mx.bfloat16), mx.array(vc).astype(mx.bfloat16),
                                mx.array(bt), mx.array(pb), mx.array(sl))
    kt, vt = tk.beam_reorder_kv(torch.from_numpy(kc).to(torch.bfloat16).to("mps"),
                                torch.from_numpy(vc).to(torch.bfloat16).to("mps"),
                                torch.from_numpy(bt).to("mps"), torch.from_numpy(pb).to("mps"),
                                torch.from_numpy(sl).to("mps"))
    _assert_parity(km, kt, atol=0)           # same metallib -> bit-identical reorder
    _assert_parity(vm, vt, atol=0)


@pytest.mark.parametrize("cu", [[0, 5, 5, 20, 37], [0, 8, 24], [0, 64]])
def test_varlen_build_worklist_parity(cu):
    cu = np.asarray(cu, np.int32)
    B = len(cu) - 1
    max_tiles = int((cu[-1] + 7 * B) // 8 + B)
    om = tk.varlen_build_worklist(mx.array(cu), max_tiles)
    ot = tk.varlen_build_worklist(torch.from_numpy(cu).to("mps"), max_tiles)
    for a, b in zip(om, ot):                 # qlens, pad_off, tile_seq, tile_local0, n_tiles: exact
        _assert_parity(a, b, atol=0)


@pytest.mark.parametrize("D,H,H_KV", [(64, 4, 2), (128, 2, 2)])
@pytest.mark.parametrize("plen", [7, 16])
def test_cascade_attention_parity(D, H, H_KV, plen):
    rng = np.random.default_rng(D + H + plen)
    B, num_blocks, bs = 2, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    pk = (0.2 * rng.normal(size=(plen, H_KV, D))).astype(np.float32)
    pv = (0.2 * rng.normal(size=(plen, H_KV, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, bs, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, bs, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], np.int32)
    cl = np.array([10, 16], np.int32)
    om = tk.cascade_attention(_mk(q, "mlx", "f32"), _mk(pk, "mlx", "f32"), _mk(pv, "mlx", "f32"),
                              _mk(kc, "mlx", "f32"), _mk(vc, "mlx", "f32"),
                              mx.array(bt), mx.array(cl), partition_size=8)
    ot = tk.cascade_attention(_mk(q, "torch", "f32"), _mk(pk, "torch", "f32"), _mk(pv, "torch", "f32"),
                              _mk(kc, "torch", "f32"), _mk(vc, "torch", "f32"),
                              torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                              partition_size=8)
    _assert_parity(om, ot, atol=1e-5)


@pytest.mark.parametrize("plens", [[4, 8, 3], [1, 16, 7]])
def test_cascade_attention_multi_parity(plens):
    rng = np.random.default_rng(sum(plens))
    B, H, H_KV, D, num_blocks, bs = 2, 4, 2, 64, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    pks = [(0.2 * rng.normal(size=(pl, H_KV, D))).astype(np.float32) for pl in plens]
    pvs = [(0.2 * rng.normal(size=(pl, H_KV, D))).astype(np.float32) for pl in plens]
    kc = (0.2 * rng.normal(size=(num_blocks, bs, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, bs, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], np.int32)
    cl = np.array([10, 16], np.int32)
    om = tk.cascade_attention(_mk(q, "mlx", "f32"), [_mk(x, "mlx", "f32") for x in pks],
                              [_mk(x, "mlx", "f32") for x in pvs], _mk(kc, "mlx", "f32"),
                              _mk(vc, "mlx", "f32"), mx.array(bt), mx.array(cl), partition_size=8)
    ot = tk.cascade_attention(_mk(q, "torch", "f32"), [_mk(x, "torch", "f32") for x in pks],
                              [_mk(x, "torch", "f32") for x in pvs], _mk(kc, "torch", "f32"),
                              _mk(vc, "torch", "f32"), torch.from_numpy(bt).to("mps"),
                              torch.from_numpy(cl).to("mps"), partition_size=8)
    _assert_parity(om, ot, atol=1e-5)


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_cascade_attention_fp8_parity(fmt):
    rng = np.random.default_rng(len(fmt) + 3)
    B, H, H_KV, D, plen, num_blocks, bs = 2, 4, 2, 64, 12, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    # shared fp8 prefix codes (encode once on MLX -> the SAME uint8 fed to both backends)
    pk = (0.2 * rng.normal(size=(plen, H_KV, D))).astype(np.float32)
    pv = (0.2 * rng.normal(size=(plen, H_KV, D))).astype(np.float32)
    qmax = 448.0 if fmt == "e4m3" else 57344.0
    ks = float(np.abs(pk).max() / qmax); vs = float(np.abs(pv).max() / qmax)
    pkc, pvc = tk.kv_cache_scatter_fp8(mx.array(pk).astype(mx.bfloat16), mx.array(pv).astype(mx.bfloat16),
                                       mx.array(np.arange(plen, dtype=np.int64)), 1, plen, ks, vs, fmt=fmt)
    pk8 = np.array(pkc).reshape(plen, H_KV, D); pv8 = np.array(pvc).reshape(plen, H_KV, D)
    kc = (0.2 * rng.normal(size=(num_blocks, bs, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, bs, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], np.int32); cl = np.array([10, 16], np.int32)
    args = lambda be: (_mk(q, be, "bf16"), mx.array(pk8) if be == "mlx" else torch.from_numpy(pk8).to("mps"),
                       mx.array(pv8) if be == "mlx" else torch.from_numpy(pv8).to("mps"),
                       _mk(kc, be, "bf16"), _mk(vc, be, "bf16"),
                       mx.array(bt) if be == "mlx" else torch.from_numpy(bt).to("mps"),
                       mx.array(cl) if be == "mlx" else torch.from_numpy(cl).to("mps"))
    om = tk.cascade_attention_fp8(*args("mlx"), ks, vs, partition_size=8, fmt=fmt)
    ot = tk.cascade_attention_fp8(*args("torch"), ks, vs, partition_size=8, fmt=fmt)
    _assert_parity(om, ot, atol=2e-2)          # bf16 out + fp8 prefix; same kernels, ~fp16 accum order


@pytest.mark.parametrize("shape", [(2, 128, 1024), (8, 256)])
def test_rms_norm_parity(shape):
    D = shape[-1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal((D,)).astype(np.float32)
    om = tk.rms_norm(_mk(x, "mlx"), _mk(w, "mlx"))
    ot = tk.rms_norm(_mk(x, "torch"), _mk(w, "torch"))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("shape", [(2, 128, 1024), (8, 256)])
def test_rms_norm_add_parity(shape):
    D = shape[-1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    r = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal((D,)).astype(np.float32)
    om, am = tk.rms_norm_add(_mk(x, "mlx"), _mk(r, "mlx"), _mk(w, "mlx"))
    ot, at = tk.rms_norm_add(_mk(x, "torch"), _mk(r, "torch"), _mk(w, "torch"))
    _assert_parity(om, ot, atol=1e-2)
    _assert_parity(am, at, atol=1e-2)


@pytest.mark.parametrize("H", [64, 128])
def test_moe_grouped_gemm_parity(H):
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
    om = tk.moe_grouped_gemm(_mk(pi, "mlx"), _mk(W, "mlx"), mx.array(eot))
    ot = tk.moe_grouped_gemm(_mk(pi, "torch"), _mk(W, "torch"), torch.from_numpy(eot).to("mps"))
    _assert_parity(om, ot, atol=6e-2)


def _moe_padded_eot(counts):
    padded = [((c + 31) // 32) * 32 for c in counts]
    off_pad = np.concatenate([[0], np.cumsum(padded)]).astype(np.int64)
    total = int(off_pad[-1])
    tb = off_pad // 32
    eot = np.zeros(total // 32, np.int32)
    for e in range(len(counts)):
        eot[tb[e]:tb[e + 1]] = e
    return total, eot


@pytest.mark.parametrize("K_dim,N_out", [(64, 96), (128, 64)])
def test_moe_grouped_gemm_rect_parity(K_dim, N_out):
    rng = np.random.default_rng(51)
    E = 4
    total, eot = _moe_padded_eot([40, 5, 70, 20])
    A = (0.1 * rng.standard_normal((total, K_dim))).astype(np.float32)
    W = (0.1 * rng.standard_normal((E, K_dim, N_out))).astype(np.float32)
    om = tk.moe_grouped_gemm_rect(_mk(A, "mlx"), _mk(W, "mlx"), mx.array(eot))
    ot = tk.moe_grouped_gemm_rect(_mk(A, "torch"), _mk(W, "torch"), torch.from_numpy(eot).to("mps"))
    _assert_parity(om, ot, atol=6e-2)


@pytest.mark.parametrize("H,inter", [(64, 32), (128, 64)])
def test_moe_grouped_gemm_swiglu_parity(H, inter):
    rng = np.random.default_rng(52)
    E = 4
    total, eot = _moe_padded_eot([40, 5, 70, 20])
    A = (0.1 * rng.standard_normal((total, H))).astype(np.float32)
    W1 = (0.1 * rng.standard_normal((E, H, 2 * inter))).astype(np.float32)
    om = tk.moe_grouped_gemm_swiglu(_mk(A, "mlx"), _mk(W1, "mlx"), mx.array(eot))
    ot = tk.moe_grouped_gemm_swiglu(_mk(A, "torch"), _mk(W1, "torch"), torch.from_numpy(eot).to("mps"))
    _assert_parity(om, ot, atol=6e-2)


@pytest.mark.parametrize("E,K", [(8, 2), (64, 4)])
def test_moe_route_topk_parity(E, K):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, E)).astype(np.float32)
    im, wm = tk.moe_route_topk(_mk(x, "mlx", "f32"), K)
    it, wt = tk.moe_route_topk(_mk(x, "torch", "f32"), K)
    _assert_parity(im, it, atol=0)        # exact ids (f32, no ties)
    _assert_parity(wm, wt, atol=1e-4)


@pytest.mark.parametrize("E,K", [(8, 2), (16, 4)])
def test_moe_permute_offsets_parity(E, K):
    # sorted_row_idx/inv_idx order is atomic-nondeterministic; offsets are deterministic.
    rng = np.random.default_rng(0)
    ids = rng.integers(0, E, size=(50, K)).astype(np.int32)
    om = tk.moe_permute(mx.array(ids), E)[1]
    ot = tk.moe_permute(torch.from_numpy(ids).to("mps"), E)[1]
    _assert_parity(om, ot, atol=0)


@pytest.mark.parametrize("E,K", [(8, 2), (16, 4)])
def test_moe_pad_schedule_parity(E, K):
    # Feed both backends the SAME (deterministic) sorted/offsets so all four outputs
    # must match exactly (moe_permute's own within-segment order is nondeterministic).
    rng = np.random.default_rng(3)
    T = 50
    flat = rng.integers(0, E, size=(T * K,)).astype(np.int32)
    sidx = np.argsort(flat, kind="stable").astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=E))]).astype(np.int32)
    om = tk.moe_pad_schedule(mx.array(sidx), mx.array(offsets), K)
    ot = tk.moe_pad_schedule(torch.from_numpy(sidx).to("mps"),
                             torch.from_numpy(offsets).to("mps"), K)
    for a, b in zip(om, ot):
        _assert_parity(a, b, atol=0)


@pytest.mark.parametrize("H", [64, 96])
def test_moe_gather_parity(H):
    rng = np.random.default_rng(4)
    T = 40
    gidx = np.full(96, -1, np.int32)
    gidx[:T] = rng.permutation(T).astype(np.int32)
    x = rng.standard_normal((T, H)).astype(np.float32)
    om = tk.moe_gather(_mk(x, "mlx", "f32"), mx.array(gidx))
    ot = tk.moe_gather(_mk(x, "torch", "f32"), torch.from_numpy(gidx).to("mps"))
    _assert_parity(om, ot, atol=0)  # pure copy: exact


@pytest.mark.parametrize("E,K", [(8, 2), (16, 4)])
def test_moe_mlp_parity(E, K):
    # Whole pipeline. Output is invariant to permute's within-segment order (each padded
    # row's GEMM depends only on its own token row), so cross-backend values must agree.
    rng = np.random.default_rng(5)
    T, H, inter = 40, 64, 96
    x = (0.1 * rng.standard_normal((T, H))).astype(np.float32)
    rl = rng.standard_normal((T, E)).astype(np.float32)
    W1 = (0.1 * rng.standard_normal((E, H, 2 * inter))).astype(np.float32)
    W2 = (0.1 * rng.standard_normal((E, inter, H))).astype(np.float32)
    om = tk.moe_mlp(_mk(x, "mlx", "f32"), _mk(rl, "mlx", "f32"),
                    _mk(W1, "mlx", "f32"), _mk(W2, "mlx", "f32"), K)
    ot = tk.moe_mlp(_mk(x, "torch", "f32"), _mk(rl, "torch", "f32"),
                    _mk(W1, "torch", "f32"), _mk(W2, "torch", "f32"), K)
    _assert_parity(om, ot, atol=2e-3)


@pytest.mark.parametrize("K,H", [(2, 64), (4, 128)])
def test_moe_finalize_parity(K, H):
    rng = np.random.default_rng(1)
    T = 20
    inv = rng.permutation(T * K).astype(np.int32)
    eo = rng.standard_normal((T * K, H)).astype(np.float32)
    w = rng.random((T, K)).astype(np.float32)
    ym = tk.moe_finalize(_mk(eo, "mlx", "f32"), mx.array(inv), _mk(w, "mlx", "f32"), K)
    yt = tk.moe_finalize(_mk(eo, "torch", "f32"), torch.from_numpy(inv).to("mps"),
                         _mk(w, "torch", "f32"), K)
    _assert_parity(ym, yt, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 1000), (2, 3, 257)])
def test_argmax_sample_parity(shape):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    om = tk.argmax_sample(_mk(x, "mlx", "f32"))
    ot = tk.argmax_sample(_mk(x, "torch", "f32"))
    _assert_parity(om, ot, atol=0)


@pytest.mark.parametrize("shape,K", [((16, 256), 5), ((4, 1000), 20)])
def test_top_k_sample_parity(shape, K):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    om = tk.top_k_sample(_mk(x, "mlx", "f32"), K, temperature=0.9, seed=99)
    ot = tk.top_k_sample(_mk(x, "torch", "f32"), K, temperature=0.9, seed=99)
    _assert_parity(om, ot, atol=0)


@pytest.mark.parametrize("shape,p", [((16, 256), 0.9), ((4, 1000), 0.7)])
def test_top_p_sample_parity(shape, p):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    om = tk.top_p_sample(_mk(x, "mlx", "f32"), p, temperature=0.9, seed=99)
    ot = tk.top_p_sample(_mk(x, "torch", "f32"), p, temperature=0.9, seed=99)
    _assert_parity(om, ot, atol=0)


def test_apply_penalty_parity():
    rng = np.random.default_rng(0)
    T, V, L = 8, 300, 30
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(-1, V, size=(T, L)).astype(np.int32)
    bias = rng.standard_normal(V).astype(np.float32)
    kw = dict(temperature=0.8, repetition_penalty=1.3, presence_penalty=0.1, frequency_penalty=0.05,
              eos_id=5, min_length=10, gen_len=3)
    om = tk.apply_penalty(_mk(logits, "mlx", "f32"), mx.array(prev), bias=mx.array(bias), **kw)
    ot = tk.apply_penalty(_mk(logits, "torch", "f32"), torch.from_numpy(prev).to("mps"),
                          bias=torch.from_numpy(bias).to("mps"), **kw)
    # eos_id=5 is -inf in both backends; compare the rest.
    _assert_parity(om[:, :5], ot[:, :5], atol=1e-5)
    _assert_parity(om[:, 6:], ot[:, 6:], atol=1e-5)


def test_apply_penalty_beam_parity():
    rng = np.random.default_rng(1)
    T, V, L = 6, 300, 20
    logits = rng.standard_normal((T, V)).astype(np.float32)
    prev = rng.integers(0, V, size=(T, L)).astype(np.int32)
    parent = np.array([0, 1, 0, 1, 2, 3], dtype=np.int32)
    kw = dict(temperature=0.8, repetition_penalty=1.3, presence_penalty=0.1, frequency_penalty=0.05)
    om = tk.apply_penalty(_mk(logits, "mlx", "f32"), mx.array(prev), parent_ids=mx.array(parent), **kw)
    ot = tk.apply_penalty(_mk(logits, "torch", "f32"), torch.from_numpy(prev).to("mps"),
                          parent_ids=torch.from_numpy(parent).to("mps"), **kw)
    _assert_parity(om, ot, atol=1e-5)


@pytest.mark.parametrize("shape,temp", [((16, 256), 1.0), ((4, 1000), 0.7)])
def test_sample_categorical_parity(shape, temp):
    # Same metallib kernel + same seed -> identical RNG stream -> identical tokens.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    om = tk.sample_categorical(_mk(x, "mlx", "f32"), temperature=temp, seed=99)
    ot = tk.sample_categorical(_mk(x, "torch", "f32"), temperature=temp, seed=99)
    _assert_parity(om, ot, atol=0)


@pytest.mark.parametrize("shape", [(8, 256), (3, 513)])
def test_quantize_per_tensor_fp8_parity(shape):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    cm, sm = tk.quantize_per_tensor_fp8(_mk(x, "mlx", "f32"))
    ct, st = tk.quantize_per_tensor_fp8(_mk(x, "torch", "f32"))
    _assert_parity(cm, ct, atol=0)        # same metallib -> identical codes
    _assert_parity(sm, st, atol=1e-6)


@pytest.mark.parametrize("shape", [(8, 256), (3, 513)])
def test_quantize_per_token_fp8_parity(shape):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    cm, sm = tk.quantize_per_token_fp8(_mk(x, "mlx", "f32"))
    ct, st = tk.quantize_per_token_fp8(_mk(x, "torch", "f32"))
    _assert_parity(cm, ct, atol=0)       # same metallib -> bit-identical codes
    _assert_parity(sm, st, atol=1e-6)


@pytest.mark.parametrize("shape", [(8, 256), (3, 513)])
def test_quantize_per_token_int8_parity(shape):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    cm, sm = tk.quantize_per_token_int8(_mk(x, "mlx", "f32"))
    ct, st = tk.quantize_per_token_int8(_mk(x, "torch", "f32"))
    _assert_parity(cm, ct, atol=0)
    _assert_parity(sm, st, atol=1e-6)


@pytest.mark.parametrize("H,H_KV,ps", [(2, 2, 4), (4, 1, 8)])
def test_paged_attention_v2_parity(H, H_KV, ps):
    rng = np.random.default_rng(3)
    B, D, num_blocks, block_size = 2, 64, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    om = tk.paged_attention_v2(
        _mk(q, "mlx"), _mk(kc, "mlx"), _mk(vc, "mlx"),
        mx.array(bt), mx.array(cl), partition_size=ps)
    ot = tk.paged_attention_v2(
        _mk(q, "torch"), _mk(kc, "torch"), _mk(vc, "torch"),
        torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"), partition_size=ps)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
@pytest.mark.parametrize("H,H_KV,ps", [(2, 2, 4), (4, 1, 8)])
def test_paged_attention_v2_fp8_parity(fmt, H, H_KV, ps):
    # Long-context fp8 decode must match across MLX and MPS for both fp8 formats.
    rng = np.random.default_rng(33)
    B, D, num_blocks, block_size = 2, 64, 8, 4
    total = num_blocks * block_size
    qmax = 448.0 if fmt == "e4m3" else 57344.0
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    ks = float(np.abs(K).max() / qmax)
    vs = float(np.abs(V).max() / qmax)
    slot = np.arange(total, dtype=np.int64)

    kcm, vcm = tk.kv_cache_scatter_fp8(_mk(K, "mlx"), _mk(V, "mlx"), mx.array(slot),
                                       num_blocks, block_size, ks, vs, fmt=fmt)
    om = tk.paged_attention_v2_fp8(_mk(q, "mlx"), kcm, vcm, mx.array(bt), mx.array(cl),
                                   ks, vs, partition_size=ps, fmt=fmt)
    kct, vct = tk.kv_cache_scatter_fp8(_mk(K, "torch"), _mk(V, "torch"),
                                       torch.from_numpy(slot).to("mps"), num_blocks, block_size,
                                       ks, vs, fmt=fmt)
    ot = tk.paged_attention_v2_fp8(_mk(q, "torch"), kct, vct,
                                   torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                                   ks, vs, partition_size=ps, fmt=fmt)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("window", [1, 5])
def test_paged_attention_fp8_window_parity(window):
    rng = np.random.default_rng(35)
    B, H, H_KV, D, num_blocks, block_size = 2, 4, 2, 64, 8, 16
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.arange(num_blocks, dtype=np.int32).reshape(B, num_blocks // B)
    cl = np.array([64, 64], dtype=np.int32)
    ks, vs = float(np.abs(K).max() / 448.0), float(np.abs(V).max() / 448.0)
    slot = np.arange(total, dtype=np.int64)
    kcm, vcm = tk.kv_cache_scatter_fp8(_mk(K, "mlx"), _mk(V, "mlx"), mx.array(slot), num_blocks,
                                       block_size, ks, vs)
    kct, vct = tk.kv_cache_scatter_fp8(_mk(K, "torch"), _mk(V, "torch"),
                                       torch.from_numpy(slot).to("mps"), num_blocks, block_size,
                                       ks, vs)
    om = tk.paged_attention_fp8(_mk(q, "mlx"), kcm, vcm, mx.array(bt), mx.array(cl), ks, vs,
                                window=window)
    ot = tk.paged_attention_fp8(_mk(q, "torch"), kct, vct, torch.from_numpy(bt).to("mps"),
                                torch.from_numpy(cl).to("mps"), ks, vs, window=window)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("window,fp8", [(1, False), (5, False), (16, True)])
def test_paged_attention_v2_window_parity(window, fp8):
    rng = np.random.default_rng(36 + window)
    B, H, H_KV, D, num_blocks, block_size = 2, 4, 2, 64, 8, 16
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.arange(num_blocks, dtype=np.int32).reshape(B, num_blocks // B)
    cl = np.array([64, 64], dtype=np.int32)
    if fp8:
        total = num_blocks * block_size
        K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
        V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
        ks, vs = float(np.abs(K).max() / 448.0), float(np.abs(V).max() / 448.0)
        slot = np.arange(total, dtype=np.int64)
        kcm, vcm = tk.kv_cache_scatter_fp8(_mk(K, "mlx"), _mk(V, "mlx"), mx.array(slot), num_blocks,
                                           block_size, ks, vs)
        kct, vct = tk.kv_cache_scatter_fp8(_mk(K, "torch"), _mk(V, "torch"),
                                           torch.from_numpy(slot).to("mps"), num_blocks, block_size,
                                           ks, vs)
        om = tk.paged_attention_v2_fp8(_mk(q, "mlx"), kcm, vcm, mx.array(bt), mx.array(cl), ks, vs,
                                       partition_size=32, window=window)
        ot = tk.paged_attention_v2_fp8(_mk(q, "torch"), kct, vct, torch.from_numpy(bt).to("mps"),
                                       torch.from_numpy(cl).to("mps"), ks, vs, partition_size=32,
                                       window=window)
    else:
        kc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
        vc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
        om = tk.paged_attention_v2(_mk(q, "mlx"), _mk(kc, "mlx"), _mk(vc, "mlx"), mx.array(bt),
                                   mx.array(cl), partition_size=32, window=window)
        ot = tk.paged_attention_v2(_mk(q, "torch"), _mk(kc, "torch"), _mk(vc, "torch"),
                                   torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                                   partition_size=32, window=window)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("D,gemma", [(64, False), (128, True)])
def test_rope_kv_insert_norm_parity(D, gemma):
    rng = np.random.default_rng(6)
    nb, bs, nt, H_KV = 4, 4, 5, 2
    P, half = nb * bs, D // 2
    inv = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = np.arange(P)[:, None] * inv[None, :]
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    k = (0.3 * rng.normal(size=(nt, H_KV, D))).astype(np.float32)
    v = (0.3 * rng.normal(size=(nt, H_KV, D))).astype(np.float32)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot = np.array([0, 5, -1, 6, 11], dtype=np.int64)
    w = rng.normal(size=(D,)).astype(np.float32)
    kc0 = (0.1 * rng.normal(size=(nb, bs, H_KV, D))).astype(np.float32)
    vc0 = (0.1 * rng.normal(size=(nb, bs, H_KV, D))).astype(np.float32)
    km, vm = tk.rope_kv_insert_norm(
        _mk(k, "mlx"), _mk(v, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"),
        mx.array(positions), mx.array(slot), _mk(kc0, "mlx"), _mk(vc0, "mlx"),
        _mk(w, "mlx"), 1e-5, gemma)
    kt, vt = tk.rope_kv_insert_norm(
        _mk(k, "torch"), _mk(v, "torch"), _mk(cos, "torch"), _mk(sin, "torch"),
        torch.from_numpy(positions).to("mps"), torch.from_numpy(slot).to("mps"),
        _mk(kc0, "torch"), _mk(vc0, "torch"), _mk(w, "torch"), 1e-5, gemma)
    _assert_parity(km, kt, atol=2e-2)
    _assert_parity(vm, vt, atol=2e-2)


@pytest.mark.parametrize("norm_mode", [0, 1, 2])
@pytest.mark.parametrize("T,H,nope,rope", [(2, 16, 128, 64), (2, 4, 448, 64)])
def test_mla_q_norm_rope_parity(norm_mode, T, H, nope, rope):
    Dh = nope + rope
    rng = np.random.default_rng(21)
    inv = 10000.0 ** (-(np.arange(0, rope, 2) / rope))
    ang = np.arange(64)[:, None] * inv[None, :]
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    q = (0.5 * rng.standard_normal((T, H, Dh))).astype(np.float32)
    w = (0.5 + 0.1 * rng.standard_normal(Dh)).astype(np.float32)
    positions = np.arange(T, dtype=np.int32)
    wm = _mk(w, "mlx") if norm_mode == 2 else None
    wt = _mk(w, "torch") if norm_mode == 2 else None
    om = tk.mla_q_norm_rope(_mk(q, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"), mx.array(positions),
                            H, nope, rope, norm_mode=norm_mode, norm_weight=wm)
    ot = tk.mla_q_norm_rope(_mk(q, "torch"), _mk(cos, "torch"), _mk(sin, "torch"),
                            torch.from_numpy(positions).to("mps"), H, nope, rope,
                            norm_mode=norm_mode, norm_weight=wt)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("norm_mode", [0, 2])
def test_mla_kv_insert_parity(norm_mode):
    latent, rope = 512, 64
    T, nb, bs = 5, 4, 4
    W = latent + rope
    rng = np.random.default_rng(22)
    inv = 10000.0 ** (-(np.arange(0, rope, 2) / rope))
    ang = np.arange(64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    kv_c = (0.3 * rng.standard_normal((T, latent))).astype(np.float32)
    k_pe = (0.3 * rng.standard_normal((T, rope))).astype(np.float32)
    w = (0.5 + 0.1 * rng.standard_normal(latent)).astype(np.float32)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot = np.array([0, 5, -1, 6, 11], dtype=np.int64)
    cache0 = (0.1 * rng.standard_normal((nb, bs, W))).astype(np.float32)
    wm = _mk(w, "mlx") if norm_mode == 2 else None
    wt = _mk(w, "torch") if norm_mode == 2 else None
    om = tk.mla_kv_insert(_mk(kv_c, "mlx"), _mk(k_pe, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"),
                          mx.array(positions), mx.array(slot), _mk(cache0, "mlx"),
                          rope_dim=rope, norm_mode=norm_mode, norm_weight=wm)
    ot = tk.mla_kv_insert(_mk(kv_c, "torch"), _mk(k_pe, "torch"), _mk(cos, "torch"), _mk(sin, "torch"),
                          torch.from_numpy(positions).to("mps"), torch.from_numpy(slot).to("mps"),
                          _mk(cache0, "torch"), rope_dim=rope, norm_mode=norm_mode, norm_weight=wt)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("N", [8, 16])
def test_mla_decode_parity(N):
    B, nb, bs = 2, 8, 4
    rng = np.random.default_rng(24)
    q = (0.3 * rng.standard_normal((B, N, 576))).astype(np.float32)
    cache = (0.3 * rng.standard_normal((nb, bs, 576))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    om = tk.mla_decode(_mk(q, "mlx"), _mk(cache, "mlx"), mx.array(bt), mx.array(cl))
    ot = tk.mla_decode(_mk(q, "torch"), _mk(cache, "torch"),
                       torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"))
    _assert_parity(om, ot, atol=2e-2)


def test_mla_decode_fp8_parity():
    # Build a packed cache (identical bytes across backends), then decode -> bit-close outputs.
    B, N, nb, bs = 2, 8, 8, 4
    total = nb * bs
    rng = np.random.default_rng(25)
    inv = 10000.0 ** (-(np.arange(0, 64, 2) / 64))
    ang = np.arange(64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    kv = (0.3 * rng.standard_normal((total, 512))).astype(np.float32)
    positions = np.arange(total, dtype=np.int32)
    slot = np.arange(total, dtype=np.int64)
    d0 = np.zeros((nb, bs, 576), np.uint8)
    s0 = np.zeros((nb, bs, 8), np.uint8)
    q = (0.3 * rng.standard_normal((B, N, 512))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)

    dm, sm_ = tk.mla_kv_insert_fp8(_mk(kv, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"),
                                   mx.array(positions), mx.array(slot), mx.array(d0), mx.array(s0))
    om = tk.mla_decode_fp8(_mk(q, "mlx"), dm, sm_, mx.array(bt), mx.array(cl))
    dt, st = tk.mla_kv_insert_fp8(_mk(kv, "torch"), _mk(cos, "torch"), _mk(sin, "torch"),
                                  torch.from_numpy(positions).to("mps"), torch.from_numpy(slot).to("mps"),
                                  torch.from_numpy(d0).to("mps"), torch.from_numpy(s0).to("mps"))
    ot = tk.mla_decode_fp8(_mk(q, "torch"), dt, st,
                           torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"))
    _assert_parity(om, ot, atol=2e-2)


def test_mla_decode_fp8_sparse_parity():
    B, N, nb, bs = 2, 8, 8, 4
    total = nb * bs
    rng = np.random.default_rng(26)
    inv = 10000.0 ** (-(np.arange(0, 64, 2) / 64))
    ang = np.arange(64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    kv = (0.3 * rng.standard_normal((total, 512))).astype(np.float32)
    positions = np.arange(total, dtype=np.int32)
    slot = np.arange(total, dtype=np.int64)
    d0 = np.zeros((nb, bs, 576), np.uint8)
    s0 = np.zeros((nb, bs, 8), np.uint8)
    q = (0.3 * rng.standard_normal((B, N, 512))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    idx = np.full((B, 5), -1, np.int32)
    idx[0, :4] = [0, 2, 5, 9]
    idx[1, :5] = [1, 3, 6, 10, 15]
    lens = np.array([4, 5], dtype=np.int32)

    dm, sm_ = tk.mla_kv_insert_fp8(_mk(kv, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"),
                                   mx.array(positions), mx.array(slot), mx.array(d0), mx.array(s0))
    om = tk.mla_decode_fp8_sparse(_mk(q, "mlx"), dm, sm_, mx.array(bt), mx.array(idx), mx.array(lens))
    dt, st = tk.mla_kv_insert_fp8(_mk(kv, "torch"), _mk(cos, "torch"), _mk(sin, "torch"),
                                  torch.from_numpy(positions).to("mps"), torch.from_numpy(slot).to("mps"),
                                  torch.from_numpy(d0).to("mps"), torch.from_numpy(s0).to("mps"))
    ot = tk.mla_decode_fp8_sparse(_mk(q, "torch"), dt, st, torch.from_numpy(bt).to("mps"),
                                  torch.from_numpy(idx).to("mps"), torch.from_numpy(lens).to("mps"))
    _assert_parity(om, ot, atol=2e-2)


def test_mla_kv_insert_fp8_parity():
    T, nb, bs = 5, 4, 4
    rng = np.random.default_rng(23)
    inv = 10000.0 ** (-(np.arange(0, 64, 2) / 64))
    ang = np.arange(64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    kv = (0.5 * rng.standard_normal((T, 512))).astype(np.float32)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot = np.array([0, 5, -1, 6, 11], dtype=np.int64)
    d0 = np.zeros((nb, bs, 576), dtype=np.uint8)
    s0 = np.zeros((nb, bs, 8), dtype=np.uint8)

    dm, sm_ = tk.mla_kv_insert_fp8(_mk(kv, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"),
                                   mx.array(positions), mx.array(slot), mx.array(d0), mx.array(s0))
    dt, st = tk.mla_kv_insert_fp8(_mk(kv, "torch"), _mk(cos, "torch"), _mk(sin, "torch"),
                                  torch.from_numpy(positions).to("mps"), torch.from_numpy(slot).to("mps"),
                                  torch.from_numpy(d0).to("mps"), torch.from_numpy(s0).to("mps"))
    _assert_parity(dm, dt, atol=0)     # same metallib -> identical packed bytes
    _assert_parity(sm_, st, atol=0)


@pytest.mark.parametrize("dt", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("do_norm", [False, True])
def test_rope_q_parity(dt, do_norm):
    D, H, nt, P = 128, 4, 5, 16
    rng = np.random.default_rng(8)
    half = D // 2
    inv = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = np.arange(P)[:, None] * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    q = (0.3 * rng.normal(size=(nt, H, D))).astype(np.float32)
    w = (0.5 + 0.1 * rng.normal(size=(D,))).astype(np.float32)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    wm = _mk(w, "mlx", dt) if do_norm else None
    wt = _mk(w, "torch", dt) if do_norm else None
    om = tk.rope_q(_mk(q, "mlx", dt), _mk(cos, "mlx", dt), _mk(sin, "mlx", dt),
                   mx.array(positions), norm_weight=wm, gemma=True)
    ot = tk.rope_q(_mk(q, "torch", dt), _mk(cos, "torch", dt), _mk(sin, "torch", dt),
                   torch.from_numpy(positions).to("mps"), norm_weight=wt, gemma=True)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("dt", ["f32", "f16", "bf16"])
@pytest.mark.parametrize("D,H_KV", [(64, 2), (128, 1)])
def test_rope_kv_insert_parity(dt, D, H_KV):
    rng = np.random.default_rng(5)
    num_blocks, block_size, num_tokens = 4, 4, 5
    P = num_blocks * block_size
    half = D // 2
    inv = 1.0 / (10000.0 ** (np.arange(half) / half))
    ang = np.arange(P)[:, None] * inv[None, :]
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    k = (0.3 * rng.normal(size=(num_tokens, H_KV, D))).astype(np.float32)
    v = (0.3 * rng.normal(size=(num_tokens, H_KV, D))).astype(np.float32)
    positions = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    slot_mapping = np.array([0, 5, -1, 6, 11], dtype=np.int64)
    kc0 = (0.1 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc0 = (0.1 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)

    km, vm = tk.rope_kv_insert(
        _mk(k, "mlx", dt), _mk(v, "mlx", dt), _mk(cos, "mlx", dt), _mk(sin, "mlx", dt),
        mx.array(positions), mx.array(slot_mapping), _mk(kc0, "mlx", dt), _mk(vc0, "mlx", dt))
    kt, vt = tk.rope_kv_insert(
        _mk(k, "torch", dt), _mk(v, "torch", dt), _mk(cos, "torch", dt), _mk(sin, "torch", dt),
        torch.from_numpy(positions).to("mps"), torch.from_numpy(slot_mapping).to("mps"),
        _mk(kc0, "torch", dt), _mk(vc0, "torch", dt))
    _assert_parity(km, kt, atol=2e-2)
    _assert_parity(vm, vt, atol=2e-2)


@pytest.mark.parametrize("shape", [(8, 256), (3, 1024)])
def test_rms_norm_add_fp8_parity(shape):
    D = shape[-1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    r = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal((D,)).astype(np.float32)
    cm, am, sm = tk.rms_norm_add_fp8(_mk(x, "mlx"), _mk(r, "mlx"), _mk(w, "mlx"))
    ct, at, st = tk.rms_norm_add_fp8(_mk(x, "torch"), _mk(r, "torch"), _mk(w, "torch"))
    _assert_parity(cm, ct, atol=0)        # same metallib -> identical codes
    _assert_parity(am, at, atol=2e-2)
    _assert_parity(sm, st, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 128, 1024), (8, 256)])
def test_layernorm_add_parity(shape):
    D = shape[-1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    r = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal((D,)).astype(np.float32)
    b = rng.standard_normal((D,)).astype(np.float32)
    om, am = tk.layernorm_add(_mk(x, "mlx"), _mk(r, "mlx"), _mk(w, "mlx"), _mk(b, "mlx"))
    ot, at = tk.layernorm_add(_mk(x, "torch"), _mk(r, "torch"), _mk(w, "torch"), _mk(b, "torch"))
    _assert_parity(om, ot, atol=1e-2)
    _assert_parity(am, at, atol=1e-2)


@pytest.mark.parametrize("shape", [(2, 128, 1024), (1, 256, 768)])
def test_softmax_parity(shape):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    om = tk.softmax(_mk(x, "mlx"))
    ot = tk.softmax(_mk(x, "torch"))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (1, 2, 128, 128)])
def test_rotary_parity(shape):
    B, H, N, D = shape
    rng = np.random.default_rng(0)
    base = 10000.0
    inv_freq = base ** (-(np.arange(0, D, 2).astype(np.float32) / D))
    ang = np.arange(N, dtype=np.float32)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x = rng.standard_normal((B, H, N, D)).astype(np.float32)
    om = tk.rotary(_mk(x, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"))
    ot = tk.rotary(_mk(x, "torch"), _mk(cos, "torch"), _mk(sin, "torch"))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (1, 2, 128, 128)])
def test_rotary_interleaved_parity(shape):
    B, H, N, D = shape
    rng = np.random.default_rng(1)
    base = 10000.0
    inv_freq = base ** (-(np.arange(0, D, 2).astype(np.float32) / D))
    ang = np.arange(N, dtype=np.float32)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    x = rng.standard_normal((B, H, N, D)).astype(np.float32)
    om = tk.rotary(_mk(x, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"), interleaved=True)
    ot = tk.rotary(_mk(x, "torch"), _mk(cos, "torch"), _mk(sin, "torch"), interleaved=True)
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("shape", [(2, 128, 1024), (1, 256, 768)])
def test_gelu_parity(shape):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    _assert_parity(tk.gelu(_mk(x, "mlx")), tk.gelu(_mk(x, "torch")), atol=1e-2)


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (1, 2, 128, 128)])
def test_attn_causal_parity(shape):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.attn_causal(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"))
    ot = tk.attn_causal(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"))
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("window", [5, 13, 100])
@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (1, 2, 128, 128)])
def test_attn_window_parity(shape, window):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.attn_window(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"), window)
    ot = tk.attn_window(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"), window)
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("nkm", [(40, 20, 48), (33, 17, 65)])
def test_matmul_arbitrary_parity(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    x = rng.random((N, K), dtype=np.float32)
    y = rng.random((K, M), dtype=np.float32)
    om = tk.matmul_custom(_mk(x, "mlx", "f32"), _mk(y, "mlx", "f32"))
    ot = tk.matmul_custom(_mk(x, "torch", "f32"), _mk(y, "torch", "f32"))
    _assert_parity(om, ot, atol=1e-3)


@pytest.mark.parametrize("nkm", [(32, 16, 32), (64, 32, 64)])
def test_flux_gelu_parity(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    x = rng.random((N, K), dtype=np.float32)
    w = rng.random((K, M), dtype=np.float32)
    b = rng.standard_normal((M,)).astype(np.float32)
    om = tk.flux_gelu(_mk(x, "mlx"), _mk(w, "mlx"), _mk(b, "mlx"))
    ot = tk.flux_gelu(_mk(x, "torch"), _mk(w, "torch"), _mk(b, "torch"))
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (2, 2, 128, 64)])
def test_lin_attn_causal_parity(shape):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.lin_attn_causal(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"))
    ot = tk.lin_attn_causal(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"))
    _assert_parity(om, ot, atol=1.0)


@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (2, 2, 128, 64)])
def test_lin_attn_decay_parity(shape):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    slopes = np.linspace(0.05, 0.5, shape[1]).astype(np.float32)
    om = tk.lin_attn_decay(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"), slopes)
    ot = tk.lin_attn_decay(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"), slopes)
    _assert_parity(om, ot, atol=1.0)


@pytest.mark.parametrize("shape", [(1, 2, 64), (2, 2, 128)])
def test_based_parity(shape):
    B, H, N = shape
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, N, 16)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, 16)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, 64)) * 0.5).astype(np.float32)
    om = tk.based(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"))
    ot = tk.based(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"))
    _assert_parity(om, ot, atol=1.0)


@pytest.mark.parametrize("causal", [False, True])
def test_attn_bwd_parity(causal):
    B, H, N, D = 1, 2, 64, 64
    rng = np.random.default_rng(0)
    q = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    do = (rng.standard_normal((B, H, N, D)) * 0.5).astype(np.float32)
    om, Lm = tk.attn_fwd_l(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"), causal=causal)
    ot, Lt = tk.attn_fwd_l(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"), causal=causal)
    dqm, dkm, dvm = tk.attn_bwd(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"), om, _mk(do, "mlx"), Lm, causal=causal)
    dqt, dkt, dvt = tk.attn_bwd(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"), ot, _mk(do, "torch"), Lt, causal=causal)
    for am, at_ in [(dqm, dqt), (dkm, dkt), (dvm, dvt)]:
        _assert_parity(am, at_, atol=1.0)


@pytest.mark.parametrize("shape", [(1, 1, 16), (2, 2, 32)])
def test_fftconv_parity(shape):
    B, H, S = shape
    N = S * S
    rng = np.random.default_rng(0)

    def fftm(sign):
        n = np.arange(S); k = n.reshape(-1, 1)
        return np.exp(sign * 2j * np.pi * n * k / S)

    def tw(sign):
        na = np.arange(S).reshape(-1, 1); ma = np.arange(S)
        return np.exp(sign * 2j * np.pi * na * ma / N)

    u = rng.standard_normal((B, H, S, S)).astype(np.float32)
    kf = rng.standard_normal((2, H, S, S)).astype(np.float32)
    X = np.stack([u, np.zeros_like(u)]).astype(np.float32)
    F, Finv, TW, TWI = fftm(-1), fftm(1), tw(-1), tw(1) / N

    def cs(m):
        return np.stack([m.real, m.imag]).astype(np.float32)

    args_np = [X, cs(F), cs(TW), cs(Finv), cs(TWI), kf]
    om = tk.fftconv(*[_mk(a, "mlx", "f32") for a in args_np])
    ot = tk.fftconv(*[_mk(a, "torch", "f32") for a in args_np])
    _assert_parity(om, ot, atol=1e-2)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_K", "kU4B8", "kU4", "fp8_e4m3", "fp4_e2m1", "mxfp8", "nvfp4", "mxfp4", "bitnet", "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K", "q5_K", "q6_K", "e5m2", "fp8_block", "mxfp6_e3m2", "mxfp6_e2m3", "hqq"])
@pytest.mark.parametrize("nkm", [(64, 256, 64), (128, 512, 128)])
def test_qgemm_parity(nkm, fmt):
    # same packed weights + same fp16 activations -> MLX and MPS run the same kernel ≈ identical
    from tk.quant import QUANT_FORMATS
    quantize, _ = QUANT_FORMATS[fmt]
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    Wq = quantize(W)
    om = tk.qgemm(mx.array(Wq), mx.array(X).astype(mx.float16), format=fmt)
    ot = tk.qgemm(torch.from_numpy(Wq).to("mps"),
                  torch.from_numpy(X).to(torch.float16).to("mps"), format=fmt)
    # the two backends use separately-compiled metallibs, so allow a tiny magnitude-relative diff
    mx.eval(om)
    atol = max(1e-2, 3e-3 * float(mx.max(mx.abs(om)).item()))
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_K", "kU4B8", "kU4", "fp8_e4m3", "fp4_e2m1", "mxfp8", "nvfp4", "mxfp4", "bitnet", "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K", "q5_K", "q6_K", "e5m2", "fp8_block", "mxfp6_e3m2", "mxfp6_e2m3", "hqq"])
@pytest.mark.parametrize("nk", [(64, 256), (128, 256)])
def test_qgemv_parity(nk, fmt):
    from tk.quant import QUANT_FORMATS
    quantize, _ = QUANT_FORMATS[fmt]
    N, K = nk
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    x = rng.standard_normal((K, 1)).astype(np.float32)
    Wq = quantize(W)
    om = tk.qgemv(mx.array(Wq), mx.array(x).astype(mx.float16), format=fmt)
    ot = tk.qgemv(torch.from_numpy(Wq).to("mps"),
                  torch.from_numpy(x).to(torch.float16).to("mps"), format=fmt)
    mx.eval(om)
    atol = max(1e-2, 3e-3 * float(mx.max(mx.abs(om)).item()))
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("nk", [(64, 256), (128, 512)])
def test_qgemv_w8a8_parity(nk):
    from tk.quant import quantize_w8a8, quantize_act_int8
    N, K = nk
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, 1)).astype(np.float32)
    Wq, ws = quantize_w8a8(W)
    _, Xq, xs = quantize_act_int8(X)
    asc = np.array([xs[0, 0]], np.float16)
    om = tk.qgemv_w8a8(mx.array(Wq), mx.array(Xq), mx.array(ws).astype(mx.float16), mx.array(asc))
    ot = tk.qgemv_w8a8(torch.from_numpy(Wq).to("mps"), torch.from_numpy(Xq).to("mps"),
                       torch.from_numpy(ws).to(torch.float16).to("mps"),
                       torch.from_numpy(asc).to("mps"))
    mx.eval(om)
    atol = max(1e-2, 3e-3 * float(mx.max(mx.abs(om)).item()))
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("nk", [(64, 256), (128, 512)])
def test_qgemv_w2a8_parity(nk):
    from tk.quant import quantize_bitnet, quantize_act_int8
    N, K = nk
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, 1)).astype(np.float32)
    Wq = quantize_bitnet(W)
    _, Xq, xs = quantize_act_int8(X)
    asc = np.array([xs[0, 0]], np.float16)
    om = tk.qgemv_w2a8(mx.array(Wq), mx.array(Xq), mx.array(asc))
    ot = tk.qgemv_w2a8(torch.from_numpy(Wq).to("mps"), torch.from_numpy(Xq).to("mps"),
                       torch.from_numpy(asc).to("mps"))
    mx.eval(om)
    atol = max(1e-2, 3e-3 * float(mx.max(mx.abs(om)).item()))
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("nkm", [(64, 256, 64), (128, 512, 128)])
def test_qgemm_fp8_scaled_parity(nkm):
    from tk.quant import quantize_fp8_scaled, quantize_act_fp8
    N, K, M = nkm
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    X = rng.standard_normal((K, M)).astype(np.float32)
    wq, ws = quantize_fp8_scaled(W)
    _, xq, s = quantize_act_fp8(X)
    asc = s[0, :].astype(np.float16)
    om = tk.qgemm_fp8_scaled(mx.array(wq), mx.array(xq), mx.array(ws), mx.array(asc))
    ot = tk.qgemm_fp8_scaled(torch.from_numpy(wq).to("mps"), torch.from_numpy(xq).to("mps"),
                             torch.from_numpy(ws).to("mps"), torch.from_numpy(asc).to("mps"))
    mx.eval(om)
    atol = max(1e-2, 3e-3 * float(mx.max(mx.abs(om)).item()))
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("nkm", [(32, 16, 32), (64, 32, 64)])
def test_cmplx_matmul_parity(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, N, K)).astype(np.float32)
    B = rng.standard_normal((2, K, M)).astype(np.float32)
    om = tk.cmplx_matmul(_mk(A, "mlx", "f32"), _mk(B, "mlx", "f32"))
    ot = tk.cmplx_matmul(_mk(A, "torch", "f32"), _mk(B, "torch", "f32"))
    _assert_parity(om, ot, atol=1e-3)


@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (2, 2, 128, 64)])
def test_mamba2_parity(shape):
    B, H, N, D = shape
    rng = np.random.default_rng(0)
    C = rng.standard_normal(shape).astype(np.float32) * 0.5
    Bm = rng.standard_normal(shape).astype(np.float32) * 0.5
    X = rng.standard_normal(shape).astype(np.float32)
    a = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, N)))) * 0.5 + 0.5
    cumlog = np.cumsum(np.log(a), axis=-1).astype(np.float32)
    om = tk.mamba2(_mk(C, "mlx"), _mk(Bm, "mlx"), _mk(X, "mlx"), _mk(cumlog, "mlx", "f32"))
    ot = tk.mamba2(_mk(C, "torch"), _mk(Bm, "torch"), _mk(X, "torch"), _mk(cumlog, "torch", "f32"))
    _assert_parity(om, ot, atol=1.0)


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (1, 1, 128, 128)])
def test_mamba2_chunked_parity(shape):
    """Forced chunked linear-time route (both head dims) — same metallib kernels on both hosts."""
    B, H, N, D = shape
    rng = np.random.default_rng(2)
    C = rng.standard_normal(shape).astype(np.float32) * 0.5
    Bm = rng.standard_normal(shape).astype(np.float32) * 0.5
    X = rng.standard_normal(shape).astype(np.float32)
    a = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, N)))) * 0.5 + 0.5
    cumlog = np.cumsum(np.log(a), axis=-1).astype(np.float32)
    om = tk.mamba2_chunked(_mk(C, "mlx"), _mk(Bm, "mlx"), _mk(X, "mlx"),
                           _mk(cumlog, "mlx", "f32"))
    ot = tk.mamba2_chunked(_mk(C, "torch"), _mk(Bm, "torch"), _mk(X, "torch"),
                           _mk(cumlog, "torch", "f32"))
    _assert_parity(om, ot, atol=1.0)


@pytest.mark.parametrize("shape", [(1, 2, 64, 64), (2, 1, 64, 128),
                                   (1, 2, 128, 64), (2, 2, 192, 64)])
def test_mamba2_bwd_parity(shape):
    B, H, N, D = shape
    rng = np.random.default_rng(1)
    C = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    Bm = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    X = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    dY = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    a = rng.uniform(0.9, 1.0, (B, H, N)).astype(np.float32)
    cumlog = np.cumsum(np.log(a), axis=-1).astype(np.float32)
    om = tk.mamba2_bwd(_mk(C, "mlx"), _mk(Bm, "mlx"), _mk(X, "mlx"), _mk(cumlog, "mlx", "f32"),
                       _mk(dY, "mlx"))
    ot = tk.mamba2_bwd(_mk(C, "torch"), _mk(Bm, "torch"), _mk(X, "torch"),
                       _mk(cumlog, "torch", "f32"), _mk(dY, "torch"))
    for a_, b_ in zip(om, ot):
        _assert_parity(a_, b_, atol=6e-2)


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 2, 192, 64), (1, 1, 128, 128)])
def test_mamba2_bwd_chunked_parity(shape):
    """Forced chunked linear-time backward (both head dims)."""
    B, H, N, D = shape
    rng = np.random.default_rng(3)
    C = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    Bm = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    X = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    dY = (0.3 * rng.standard_normal(shape)).astype(np.float32)
    a = rng.uniform(0.9, 1.0, (B, H, N)).astype(np.float32)
    cumlog = np.cumsum(np.log(a), axis=-1).astype(np.float32)
    om = tk.mamba2_bwd_chunked(_mk(C, "mlx"), _mk(Bm, "mlx"), _mk(X, "mlx"),
                               _mk(cumlog, "mlx", "f32"), _mk(dY, "mlx"))
    ot = tk.mamba2_bwd_chunked(_mk(C, "torch"), _mk(Bm, "torch"), _mk(X, "torch"),
                               _mk(cumlog, "torch", "f32"), _mk(dY, "torch"))
    for a_, b_ in zip(om, ot):
        _assert_parity(a_, b_, atol=6e-2)


@pytest.mark.parametrize("D", [64, 128])
def test_ssd_decode_parity(D):
    """One decode step: same kernel, functional (MLX) vs in-place (torch) state update."""
    B, H = 2, 2
    rng = np.random.default_rng(4 + D)
    S = (0.1 * rng.standard_normal((B, H, D, D))).astype(np.float32)
    alpha = rng.uniform(0.9, 1.0, (B, H)).astype(np.float32)
    x = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
    k = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
    q = (0.3 * rng.standard_normal((B, H, D))).astype(np.float32)
    ym, Sm = tk.ssd_decode(_mk(S, "mlx", "f32"), _mk(alpha, "mlx", "f32"), _mk(x, "mlx", "f32"),
                           _mk(k, "mlx", "f32"), _mk(q, "mlx", "f32"))
    yt, St = tk.ssd_decode(_mk(S, "torch", "f32"), _mk(alpha, "torch", "f32"),
                           _mk(x, "torch", "f32"), _mk(k, "torch", "f32"), _mk(q, "torch", "f32"))
    _assert_parity(ym, yt, atol=1e-4)
    _assert_parity(Sm, St, atol=1e-4)


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 2, 256, 64)])
def test_hedgehog_parity(shape):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.hedgehog(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"), use_kernel=True)
    ot = tk.hedgehog(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"), use_kernel=True)
    _assert_parity(om, ot, atol=1.0)


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 2, 256, 64)])
def test_linear_attn_parity(shape):
    # same kernel + deterministic bf16 input rounding => MLX and MPS outputs match closely
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.linear_attn(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"), use_kernel=True)
    ot = tk.linear_attn(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"), use_kernel=True)
    _assert_parity(om, ot, atol=1.0)  # values are O(N*D); same-kernel parity is ~exact


@pytest.mark.parametrize("shape", [(1, 2, 256, 64), (1, 2, 128, 128)])
def test_attn_multiwarp_parity(shape):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.attn_multiwarp(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"))
    ot = tk.attn_multiwarp(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"))
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("nkm", [(64, 32, 64), (128, 64, 128)])
def test_gemm_staged_parity(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    x = rng.random((N, K), dtype=np.float32)
    y = rng.random((K, M), dtype=np.float32)
    om = tk.gemm_staged(_mk(x, "mlx", "f32"), _mk(y, "mlx", "f32"))
    ot = tk.gemm_staged(_mk(x, "torch", "f32"), _mk(y, "torch", "f32"))
    _assert_parity(om, ot, atol=1e-3)


@pytest.mark.parametrize("nkm", [(32, 16, 32), (64, 32, 64)])
def test_flux_gate_parity(nkm):
    N, K, M = nkm
    rng = np.random.default_rng(0)
    x = rng.random((N, K), dtype=np.float32)
    w = rng.random((K, M), dtype=np.float32)
    b = rng.standard_normal((M,)).astype(np.float32)
    g = rng.standard_normal((M,)).astype(np.float32)
    r = rng.standard_normal((N, M)).astype(np.float32)
    om = tk.flux_gate(_mk(x, "mlx"), _mk(w, "mlx"), _mk(b, "mlx"), _mk(g, "mlx"), _mk(r, "mlx"))
    ot = tk.flux_gate(_mk(x, "torch"), _mk(w, "torch"), _mk(b, "torch"), _mk(g, "torch"), _mk(r, "torch"))
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("dtype,atol", [("f32", 1e-6), ("bf16", 0.0)])
@pytest.mark.parametrize("D", [64, 256])
def test_hadamard_parity(dtype, atol, D):
    rng = np.random.default_rng(D)
    x = rng.standard_normal((3, D)).astype(np.float32)
    om = tk.hadamard(_mk(x, "mlx", dtype))
    ot = tk.hadamard(_mk(x, "torch", dtype))
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("dtype,atol", [("f32", 1e-6), ("bf16", 0.0)])
def test_kv_cache_parity(dtype, atol):
    rng = np.random.default_rng(0)
    T, H, D = 7, 2, 64
    num_blocks, block_size = 3, 4
    key = rng.normal(size=(T, H, D)).astype(np.float32)
    value = rng.normal(size=(T, H, D)).astype(np.float32)
    slots = np.array([0, 2, -1, 5, 8, 1, 7], dtype=np.int64)
    block_table = np.array([[0, 1], [2, 0]], dtype=np.int32)
    cu_seq_lens = np.array([0, 5, 9], dtype=np.int32)
    block_mapping = np.array([[0, 2], [1, 0]], dtype=np.int64)

    km, vm = _mk(key, "mlx", dtype), _mk(value, "mlx", dtype)
    kt, vt = _mk(key, "torch", dtype), _mk(value, "torch", dtype)
    sm = mx.array(slots)
    st = torch.from_numpy(slots).to("mps")

    m_kc, m_vc = tk.kv_cache_scatter(km, vm, sm, num_blocks, block_size)
    t_kc, t_vc = tk.kv_cache_scatter(kt, vt, st, num_blocks, block_size)
    _assert_parity(m_kc, t_kc, atol=atol)
    _assert_parity(m_vc, t_vc, atol=atol)

    bm = mx.array(block_table)
    bt = torch.from_numpy(block_table).to("mps")
    lm = mx.array(cu_seq_lens)
    lt = torch.from_numpy(cu_seq_lens).to("mps")
    m_gk, m_gv = tk.kv_cache_gather(m_kc, m_vc, bm, lm, int(cu_seq_lens[-1]))
    t_gk, t_gv = tk.kv_cache_gather(t_kc, t_vc, bt, lt, int(cu_seq_lens[-1]))
    _assert_parity(m_gk, t_gk, atol=atol)
    _assert_parity(m_gv, t_gv, atol=atol)

    mm = mx.array(block_mapping)
    mt = torch.from_numpy(block_mapping).to("mps")
    m_ck, m_cv = tk.kv_cache_copy_blocks(m_kc, m_vc, mm)
    t_ck, t_cv = tk.kv_cache_copy_blocks(t_kc, t_vc, mt)
    _assert_parity(m_ck, t_ck, atol=atol)
    _assert_parity(m_cv, t_cv, atol=atol)

    m_ks, m_vs = tk.kv_cache_scales(km, vm)
    t_ks, t_vs = tk.kv_cache_scales(kt, vt)
    _assert_parity(m_ks, t_ks, atol=1e-7)
    _assert_parity(m_vs, t_vs, atol=1e-7)


@pytest.mark.parametrize("dtype,atol", [("f32", 2e-5), ("bf16", 2e-2)])
def test_paged_attention_parity(dtype, atol):
    rng = np.random.default_rng(1)
    B, H, D = 2, 2, 64
    num_blocks, block_size = 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)

    om = tk.paged_attention(
        _mk(q, "mlx", dtype),
        _mk(key_cache, "mlx", dtype),
        _mk(value_cache, "mlx", dtype),
        mx.array(block_table),
        mx.array(context_lens),
    )
    ot = tk.paged_attention(
        _mk(q, "torch", dtype),
        _mk(key_cache, "torch", dtype),
        _mk(value_cache, "torch", dtype),
        torch.from_numpy(block_table).to("mps"),
        torch.from_numpy(context_lens).to("mps"),
    )
    _assert_parity(om, ot, atol=atol)


@pytest.mark.parametrize("window", [1, 3, 5])
def test_paged_attention_window_parity(window):
    rng = np.random.default_rng(2)
    B, H, D = 2, 2, 64
    block_size, ctx = 16, 40
    nblocks = (ctx + block_size - 1) // block_size
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(B * nblocks + 1, block_size, H, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(B * nblocks + 1, block_size, H, D))).astype(np.float32)
    bt = np.full((B, nblocks), -1, np.int32)
    blk = 1
    for b in range(B):
        for c in range(nblocks):
            bt[b, c] = blk; blk += 1
    cl = np.full((B,), ctx, np.int32)
    om = tk.paged_attention(_mk(q, "mlx"), _mk(kc, "mlx"), _mk(vc, "mlx"),
                            mx.array(bt), mx.array(cl), window=window)
    ot = tk.paged_attention(_mk(q, "torch"), _mk(kc, "torch"), _mk(vc, "torch"),
                            torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                            window=window)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 1)])
def test_paged_attention_fp8_parity(H, H_KV):
    rng = np.random.default_rng(3)
    B, D, num_blocks, block_size = 2, 64, 8, 4
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    ks = float(np.abs(K).max() / 448.0)
    vs = float(np.abs(V).max() / 448.0)
    slot = np.arange(total, dtype=np.int64)

    kcm, vcm = tk.kv_cache_scatter_fp8(_mk(K, "mlx"), _mk(V, "mlx"), mx.array(slot),
                                       num_blocks, block_size, ks, vs)
    om = tk.paged_attention_fp8(_mk(q, "mlx"), kcm, vcm, mx.array(bt), mx.array(cl), ks, vs)
    kct, vct = tk.kv_cache_scatter_fp8(_mk(K, "torch"), _mk(V, "torch"),
                                       torch.from_numpy(slot).to("mps"), num_blocks, block_size, ks, vs)
    ot = tk.paged_attention_fp8(_mk(q, "torch"), kct, vct,
                                torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"), ks, vs)
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("H,H_KV", [(2, 2), (4, 1)])
def test_paged_attention_fp8_e5m2_parity(H, H_KV):
    # e5m2 format must match bit-for-bit across MLX and MPS.
    rng = np.random.default_rng(41)
    B, D, num_blocks, block_size = 2, 64, 8, 4
    total = num_blocks * block_size
    K = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D))).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    ks = float(np.abs(K).max() / 57344.0)
    vs = float(np.abs(V).max() / 57344.0)
    slot = np.arange(total, dtype=np.int64)

    kcm, vcm = tk.kv_cache_scatter_fp8(_mk(K, "mlx"), _mk(V, "mlx"), mx.array(slot),
                                       num_blocks, block_size, ks, vs, fmt="e5m2")
    om = tk.paged_attention_fp8(_mk(q, "mlx"), kcm, vcm, mx.array(bt), mx.array(cl),
                                ks, vs, fmt="e5m2")
    kct, vct = tk.kv_cache_scatter_fp8(_mk(K, "torch"), _mk(V, "torch"),
                                       torch.from_numpy(slot).to("mps"), num_blocks, block_size,
                                       ks, vs, fmt="e5m2")
    ot = tk.paged_attention_fp8(_mk(q, "torch"), kct, vct,
                                torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                                ks, vs, fmt="e5m2")
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("H,H_KV", [(4, 2), (6, 3)])
def test_paged_attention_fp8_perhead_parity(H, H_KV):
    # Per-head scale arrays must match bit-for-bit across MLX and MPS (same metallib).
    rng = np.random.default_rng(31)
    B, D, num_blocks, block_size = 2, 64, 8, 4
    total = num_blocks * block_size
    gain = (1.0 + np.arange(H_KV)).astype(np.float32)[None, :, None]
    K = (0.2 * rng.normal(size=(total, H_KV, D)) * gain).astype(np.float32)
    V = (0.2 * rng.normal(size=(total, H_KV, D)) * gain).astype(np.float32)
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    ks = (np.abs(K).max(axis=(0, 2)) / 448.0).astype(np.float32)   # (H_KV,)
    vs = (np.abs(V).max(axis=(0, 2)) / 448.0).astype(np.float32)
    slot = np.arange(total, dtype=np.int64)

    kcm, vcm = tk.kv_cache_scatter_fp8(_mk(K, "mlx"), _mk(V, "mlx"), mx.array(slot),
                                       num_blocks, block_size, mx.array(ks), mx.array(vs))
    om = tk.paged_attention_fp8(_mk(q, "mlx"), kcm, vcm, mx.array(bt), mx.array(cl),
                                mx.array(ks), mx.array(vs))
    kct, vct = tk.kv_cache_scatter_fp8(_mk(K, "torch"), _mk(V, "torch"),
                                       torch.from_numpy(slot).to("mps"), num_blocks, block_size,
                                       torch.from_numpy(ks).to("mps"), torch.from_numpy(vs).to("mps"))
    ot = tk.paged_attention_fp8(_mk(q, "torch"), kct, vct,
                                torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                                torch.from_numpy(ks).to("mps"), torch.from_numpy(vs).to("mps"))
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("H,H_KV", [(4, 2), (4, 1)])  # GQA group 2, MQA
def test_paged_attention_gqa_parity(H, H_KV):
    rng = np.random.default_rng(2)
    B, D = 2, 64
    num_blocks, block_size = 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)

    om = tk.paged_attention(
        _mk(q, "mlx", "bf16"),
        _mk(key_cache, "mlx", "bf16"),
        _mk(value_cache, "mlx", "bf16"),
        mx.array(block_table),
        mx.array(context_lens),
    )
    ot = tk.paged_attention(
        _mk(q, "torch", "bf16"),
        _mk(key_cache, "torch", "bf16"),
        _mk(value_cache, "torch", "bf16"),
        torch.from_numpy(block_table).to("mps"),
        torch.from_numpy(context_lens).to("mps"),
    )
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("H,H_KV,x", [(4, 2, 8), (4, 1, 4)])
def test_paged_attention_xcache_parity(H, H_KV, x):
    rng = np.random.default_rng(15)
    B, D, num_blocks, block_size = 2, 64, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    dk = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    dv = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    xk = dk.transpose(0, 2, 3, 1).reshape(num_blocks, H_KV, D // x, x, block_size).transpose(0, 1, 2, 4, 3).copy()
    xv = dv.transpose(0, 2, 3, 1).copy()

    om = tk.paged_attention_xcache(_mk(q, "mlx", "bf16"), _mk(xk, "mlx", "bf16"), _mk(xv, "mlx", "bf16"),
                                   mx.array(bt), mx.array(cl))
    ot = tk.paged_attention_xcache(_mk(q, "torch", "bf16"), _mk(xk, "torch", "bf16"), _mk(xv, "torch", "bf16"),
                                   torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"))
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("H,H_KV", [(4, 2), (4, 1)])
def test_paged_attention_block_sparse_parity(H, H_KV):
    rng = np.random.default_rng(12)
    B, D, num_blocks, block_size = 2, 64, 8, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    cl = np.array([10, 16], dtype=np.int32)
    mask = np.zeros((B, 4), dtype=np.int32); mask[:, ::2] = 1; mask[:, 0] = 1

    om = tk.paged_attention_block_sparse(_mk(q, "mlx", "bf16"), _mk(kc, "mlx", "bf16"), _mk(vc, "mlx", "bf16"),
                                         mx.array(bt), mx.array(cl), mx.array(mask))
    ot = tk.paged_attention_block_sparse(_mk(q, "torch", "bf16"), _mk(kc, "torch", "bf16"), _mk(vc, "torch", "bf16"),
                                         torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                                         torch.from_numpy(mask).to("mps"))
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("H,H_KV", [(4, 2), (4, 1)])
def test_paged_attention_alibi_parity(H, H_KV):
    rng = np.random.default_rng(9)
    B, D = 2, 64
    num_blocks, block_size = 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    kc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    vc = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    bt = np.array([[0, 1], [2, 3]], dtype=np.int32)
    cl = np.array([6, 7], dtype=np.int32)
    slopes = (0.1 * (1.0 + np.arange(H))).astype(np.float32)

    om = tk.paged_attention_alibi(_mk(q, "mlx", "bf16"), _mk(kc, "mlx", "bf16"), _mk(vc, "mlx", "bf16"),
                                  mx.array(bt), mx.array(cl), mx.array(slopes))
    ot = tk.paged_attention_alibi(_mk(q, "torch", "bf16"), _mk(kc, "torch", "bf16"), _mk(vc, "torch", "bf16"),
                                  torch.from_numpy(bt).to("mps"), torch.from_numpy(cl).to("mps"),
                                  torch.from_numpy(slopes).to("mps"))
    _assert_parity(om, ot, atol=2e-2)


@pytest.mark.parametrize("H,H_KV", [(8, 2), (4, 1)])  # GQA group 4, MQA
def test_paged_attention_staged_parity(H, H_KV):
    rng = np.random.default_rng(7)
    B, D = 2, 64
    num_blocks, block_size = 4, 4
    q = (0.2 * rng.normal(size=(B, H, D))).astype(np.float32)
    key_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    value_cache = (0.2 * rng.normal(size=(num_blocks, block_size, H_KV, D))).astype(np.float32)
    block_table = np.array([[0, 1], [2, 3]], dtype=np.int32)
    context_lens = np.array([6, 7], dtype=np.int32)

    om = tk.paged_attention_staged(
        _mk(q, "mlx", "bf16"), _mk(key_cache, "mlx", "bf16"), _mk(value_cache, "mlx", "bf16"),
        mx.array(block_table), mx.array(context_lens))
    ot = tk.paged_attention_staged(
        _mk(q, "torch", "bf16"), _mk(key_cache, "torch", "bf16"), _mk(value_cache, "torch", "bf16"),
        torch.from_numpy(block_table).to("mps"), torch.from_numpy(context_lens).to("mps"))
    _assert_parity(om, ot, atol=2e-2)


def test_norm_backward_parity():
    rng = np.random.default_rng(11)
    R, D = 8, 256
    x = (0.5 * rng.standard_normal((R, D))).astype(np.float32)
    w = (0.5 * rng.standard_normal((D,))).astype(np.float32)
    dy = (0.3 * rng.standard_normal((R, D))).astype(np.float32)
    # rms
    mdx, mdw = tk.rms_norm_backward(_mk(x, "mlx", "f32"), _mk(w, "mlx", "f32"), _mk(dy, "mlx", "f32"), 1e-5)
    tdx, tdw = tk.rms_norm_backward(_mk(x, "torch", "f32"), _mk(w, "torch", "f32"), _mk(dy, "torch", "f32"), 1e-5)
    _assert_parity(mdx, tdx, atol=1e-5); _assert_parity(mdw, tdw, atol=1e-5)
    # layernorm
    b = (0.3 * rng.standard_normal((D,))).astype(np.float32)  # noqa: F841
    ldx, ldw, ldb = tk.layernorm_backward(_mk(x, "mlx", "f32"), _mk(w, "mlx", "f32"), _mk(dy, "mlx", "f32"), 1e-5)
    tldx, tldw, tldb = tk.layernorm_backward(_mk(x, "torch", "f32"), _mk(w, "torch", "f32"), _mk(dy, "torch", "f32"), 1e-5)
    _assert_parity(ldx, tldx, atol=1e-5); _assert_parity(ldw, tldw, atol=1e-5); _assert_parity(ldb, tldb, atol=1e-5)
    # gelu
    gm = tk.gelu_backward(_mk(x, "mlx", "f32"), _mk(dy, "mlx", "f32"))
    gt = tk.gelu_backward(_mk(x, "torch", "f32"), _mk(dy, "torch", "f32"))
    _assert_parity(gm, gt, atol=1e-5)
