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


def matmul_custom(x: torch.Tensor, y: torch.Tensor):
    """(N,K) @ (K,M) GEMM. f32/bf16 MPS tensors; N%32==0, M%32==0, K%16==0."""
    return _ext.matmul_custom(x, y)


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Non-causal attention forward. bf16 (B,H,N,D) MPS tensors; D in {64,128}, N%8==0."""
    return _ext.attn_fwd(q, k, v)
