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
    kw = dict(temperature=0.8, repetition_penalty=1.3, presence_penalty=0.1, frequency_penalty=0.05)
    om = tk.apply_penalty(_mk(logits, "mlx", "f32"), mx.array(prev), **kw)
    ot = tk.apply_penalty(_mk(logits, "torch", "f32"), torch.from_numpy(prev).to("mps"), **kw)
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


@pytest.mark.parametrize("D,H_KV", [(64, 2), (128, 1)])
def test_rope_kv_insert_parity(D, H_KV):
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
        _mk(k, "mlx"), _mk(v, "mlx"), _mk(cos, "mlx"), _mk(sin, "mlx"),
        mx.array(positions), mx.array(slot_mapping), _mk(kc0, "mlx"), _mk(vc0, "mlx"))
    kt, vt = tk.rope_kv_insert(
        _mk(k, "torch"), _mk(v, "torch"), _mk(cos, "torch"), _mk(sin, "torch"),
        torch.from_numpy(positions).to("mps"), torch.from_numpy(slot_mapping).to("mps"),
        _mk(kc0, "torch"), _mk(vc0, "torch"))
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


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 2, 256, 64)])
def test_hedgehog_parity(shape):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.hedgehog(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"))
    ot = tk.hedgehog(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"))
    _assert_parity(om, ot, atol=1.0)


@pytest.mark.parametrize("shape", [(1, 2, 128, 64), (2, 2, 256, 64)])
def test_linear_attn_parity(shape):
    # same kernel + deterministic bf16 input rounding => MLX and MPS outputs match closely
    rng = np.random.default_rng(0)
    q = rng.standard_normal(shape).astype(np.float32)
    k = rng.standard_normal(shape).astype(np.float32)
    v = rng.standard_normal(shape).astype(np.float32)
    om = tk.linear_attn(_mk(q, "mlx"), _mk(k, "mlx"), _mk(v, "mlx"))
    ot = tk.linear_attn(_mk(q, "torch"), _mk(k, "torch"), _mk(v, "torch"))
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
