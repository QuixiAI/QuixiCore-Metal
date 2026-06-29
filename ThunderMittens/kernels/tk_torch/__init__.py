"""PyTorch MPS backend for the ThunderMittens kernels.

The compute lives in the shared, framework-agnostic .metal kernels. This package:
  1. compiles them into a standalone metallib with `xcrun metal` (no MLX, no CMake), and
  2. JIT-compiles a thin ObjC++ extension (torch.utils.cpp_extension.load) that dispatches
     those kernels onto PyTorch's MPS stream.

So a PyTorch user needs neither MLX nor the Xcode/CMake build — only Xcode's Metal toolchain.
"""

import os
import subprocess

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNELS = os.path.dirname(_HERE)              # ThunderMittens/kernels
_INCLUDE = os.path.abspath(os.path.join(_KERNELS, "..", "include"))
_METALLIB = os.path.join(_HERE, "tk.metallib")

# The shared .metal kernel sources (single source of truth, also used by the MLX build).
_METAL_SOURCES = [
    os.path.join(_KERNELS, "add_rt", "add_rt.metal"),
    os.path.join(_KERNELS, "attn_fwd", "attn_fwd.metal"),
    os.path.join(_KERNELS, "matmul_custom", "matmul_custom.metal"),
    os.path.join(_KERNELS, "layernorm", "layernorm.metal"),
    os.path.join(_KERNELS, "rms_norm", "rms_norm.metal"),
    os.path.join(_KERNELS, "softmax", "softmax.metal"),
    os.path.join(_KERNELS, "rotary", "rotary.metal"),
    os.path.join(_KERNELS, "gelu", "gelu.metal"),
    os.path.join(_KERNELS, "attn_causal", "attn_causal.metal"),
    os.path.join(_KERNELS, "flux", "flux.metal"),
    os.path.join(_KERNELS, "gemm_staged", "gemm_staged.metal"),
    os.path.join(_KERNELS, "attn_multiwarp", "attn_multiwarp.metal"),
    os.path.join(_KERNELS, "linear_attn", "linear_attn.metal"),
    os.path.join(_KERNELS, "hedgehog", "hedgehog.metal"),
    os.path.join(_KERNELS, "lin_attn_causal", "lin_attn_causal.metal"),
    os.path.join(_KERNELS, "mamba2", "mamba2.metal"),
    os.path.join(_KERNELS, "cmplx_matmul", "cmplx_matmul.metal"),
    os.path.join(_KERNELS, "fftconv", "fftconv.metal"),
    os.path.join(_KERNELS, "qgemm", "qgemm.metal"),
    os.path.join(_KERNELS, "qgemv", "qgemv.metal"),
]


def build_metallib(force: bool = False) -> str:
    """Compile the shared .metal kernels into tk.metallib via xcrun metal. MLX-independent."""
    if not force and os.path.exists(_METALLIB):
        newest_src = max(os.path.getmtime(s) for s in _METAL_SOURCES)
        if os.path.getmtime(_METALLIB) >= newest_src:
            return _METALLIB
    cmd = ["xcrun", "metal", "-std=metal3.1", "-O2", "-I", _INCLUDE,
           *_METAL_SOURCES, "-o", _METALLIB]
    subprocess.run(cmd, check=True)
    return _METALLIB


# Build the metallib (if missing/stale) and the ObjC++ extension on import.
build_metallib()

_ext = load(
    name="tk_torch_ext",
    sources=[os.path.join(_HERE, "torch_kernels.mm")],
    extra_cflags=["-std=c++17"],
    extra_include_paths=[_KERNELS],  # for "tk_launch.h" (shared host ABI)
    extra_ldflags=["-framework", "Metal", "-framework", "Foundation", "-framework", "QuartzCore"],
    verbose=False,
)
_ext._set_library(_METALLIB)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """LayerNorm over the last axis. bf16 MPS tensors; D in {256,512,768,1024}."""
    return _ext.layernorm(x, weight, bias, float(eps))


def add_rt(x: torch.Tensor, y: torch.Tensor):
    """Elementwise x + y over 2D tensors whose dims are multiples of 8 (f32/f16/bf16, MPS)."""
    return _ext.add_rt(x, y)


def _ceil(a, m):
    return ((a + m - 1) // m) * m


def matmul_custom(x: torch.Tensor, y: torch.Tensor):
    """(N,K) @ (K,M) GEMM, arbitrary shapes (f32/bf16, MPS). The tile-blocked kernel needs
    N%32, M%32, K%16; arbitrary shapes are zero-padded to the next tile multiple and sliced."""
    import torch.nn.functional as F

    N, K = x.shape[-2], x.shape[-1]
    M = y.shape[-1]
    Np, Kp, Mp = _ceil(N, 32), _ceil(K, 16), _ceil(M, 32)
    xp = F.pad(x, (0, Kp - K, 0, Np - N)) if (Np != N or Kp != K) else x
    yp = F.pad(y, (0, Mp - M, 0, Kp - K)) if (Kp != K or Mp != M) else y
    out = _ext.matmul_custom(xp.contiguous(), yp.contiguous())
    return out[:N, :M].contiguous()


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Non-causal attention forward. bf16 (B,H,N,D) MPS tensors; D in {64,128}, N%8==0."""
    return _ext.attn_fwd(q, k, v)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """RMSNorm over the last axis. bf16 MPS tensors; D in {256,512,768,1024}."""
    return _ext.rms_norm(x, weight, float(eps))


def softmax(x: torch.Tensor):
    """Softmax over the last axis. bf16 MPS tensors; D in {256,512,768,1024}."""
    return _ext.softmax(x)


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """RoPE (split-half). x bf16 (B,H,N,D); cos/sin bf16 (N,D/2); D in {64,128}."""
    return _ext.rotary(x, cos, sin)


def gelu(x: torch.Tensor):
    """GELU (tanh approx) over the last axis. bf16 MPS; D in {256,512,768,1024}."""
    return _ext.gelu(x)


def attn_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Causal attention forward. bf16 (B,H,N,D) MPS tensors; D in {64,128}, N%8==0."""
    return _ext.attn_causal(q, k, v)


def flux_gelu(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor):
    """Fused gelu(x @ w + bias). f32/bf16 MPS; N%32, M%32, K%16."""
    return _ext.flux_gelu(x, w, bias)


def flux_gate(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
              gate: torch.Tensor, residual: torch.Tensor):
    """Fused (x @ w + bias) * gate + residual. f32/bf16 MPS; N%32, M%32, K%16."""
    return _ext.flux_gate(x, w, bias, gate, residual)


def gemm_staged(x: torch.Tensor, y: torch.Tensor):
    """Multi-simdgroup threadgroup-staged GEMM (x @ y). f32/bf16 MPS; N%32, M%32, K%16."""
    return _ext.gemm_staged(x, y)


def attn_multiwarp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Multi-warp flash attention forward (shared K/V). bf16 (B,H,N,D) MPS; D in {64,128}, N%32."""
    return _ext.attn_multiwarp(q, k, v)


def linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Non-causal linear attention Q@(K^T@V). bf16 (B,H,N,D) MPS; D=64, N%8."""
    return _ext.linear_attn(q, k, v)


def hedgehog(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Hedgehog feature-map linear attention. bf16 (B,H,N,D) MPS; D=64, N%8."""
    return _ext.hedgehog(q, k, v)


def lin_attn_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Causal linear attention (chunked scan). bf16 (B,H,N,D) MPS; D=64, N%8."""
    return _ext.lin_attn_causal(q, k, v)


def mamba2(C: torch.Tensor, B: torch.Tensor, X: torch.Tensor, cumlog: torch.Tensor):
    """Mamba-2 / SSD forward. C,B,X bf16 (B,H,N,D); cumlog fp32 (B,H,N). MPS; D=64, N%8."""
    return _ext.mamba2(C, B, X, cumlog)


def cmplx_matmul(a: torch.Tensor, b: torch.Tensor):
    """Complex GEMM D=A@B; leading size-2 (real,imag) axis: a (2,N,K), b (2,K,M) -> (2,N,M).
    f32/bf16 MPS; N%32, M%32, K%16."""
    return _ext.cmplx_matmul(a, b)


def fftconv(x: torch.Tensor, fmat: torch.Tensor, twf: torch.Tensor, finv: torch.Tensor,
            twi: torch.Tensor, kf: torch.Tensor):
    """Monarch FFT convolution (N=S*S, S in {16,32}). float32 MPS; complex inputs carry a
    leading size-2 (real,imag) axis: x (2,B,H,S,S), fmat/twf/finv/twi (2,S,S), kf (2,H,S,S)
    -> real (B,H,S,S)."""
    return _ext.fftconv(x, fmat, twf, finv, twi, kf)


def qgemm(wq: torch.Tensor, x: torch.Tensor, format: str = "q8_0"):
    """Quantized GEMM (Marlin's method): out = dequantize(wq) @ x. wq packed weight blocks
    (N, K//block_k, block_bytes) uint8; x (K, M) float16 -> (N, M) float16. MPS."""
    return _ext.qgemm(wq, x, format)


def qgemv(wq: torch.Tensor, x: torch.Tensor, format: str = "q8_0"):
    """Quantized GEMV (batch-1 decode): out = dequantize(wq) @ x. x (K, 1) float16 -> (N, 1). MPS."""
    return _ext.qgemv(wq, x, format)
