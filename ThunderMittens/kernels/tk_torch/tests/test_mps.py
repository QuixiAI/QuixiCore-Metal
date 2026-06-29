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


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_K", "kU4B8", "kU4", "fp8_e4m3", "fp4_e2m1", "mxfp8", "nvfp4", "mxfp4", "bitnet", "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K", "q5_K", "q6_K"])
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


@pytest.mark.parametrize("fmt", ["q8_0", "q4_0", "q4_K", "kU4B8", "kU4", "fp8_e4m3", "fp4_e2m1", "mxfp8", "nvfp4", "mxfp4", "bitnet", "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq3_xxs", "iq1_s", "q4_1", "q5_0", "q5_1", "q2_K", "q3_K", "q5_K", "q6_K"])
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
