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


@pytest.mark.parametrize("shape", [(2, 128, 1024), (8, 256)])
def test_rms_norm_parity(shape):
    D = shape[-1]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal((D,)).astype(np.float32)
    om = tk.rms_norm(_mk(x, "mlx"), _mk(w, "mlx"))
    ot = tk.rms_norm(_mk(x, "torch"), _mk(w, "torch"))
    _assert_parity(om, ot, atol=1e-2)


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


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_K", "kU4B8", "kU4", "fp8_e4m3", "fp4_e2m1", "mxfp8", "nvfp4", "mxfp4"])
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


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_K", "kU4B8", "kU4", "fp8_e4m3", "fp4_e2m1", "mxfp8", "nvfp4", "mxfp4"])
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
