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
