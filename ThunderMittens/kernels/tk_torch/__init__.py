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
    os.path.join(_KERNELS, "add_norm", "add_norm.metal"),
    os.path.join(_KERNELS, "softmax", "softmax.metal"),
    os.path.join(_KERNELS, "rotary", "rotary.metal"),
    os.path.join(_KERNELS, "rope_kv", "rope_kv.metal"),
    os.path.join(_KERNELS, "mla", "mla.metal"),
    os.path.join(_KERNELS, "gelu", "gelu.metal"),
    os.path.join(_KERNELS, "glu", "glu.metal"),
    os.path.join(_KERNELS, "hadamard", "hadamard.metal"),
    os.path.join(_KERNELS, "kv_cache", "kv_cache.metal"),
    os.path.join(_KERNELS, "paged_attn_v2", "paged_attn_v2.metal"),
    os.path.join(_KERNELS, "quant_rt", "quant_rt.metal"),
    os.path.join(_KERNELS, "sampling", "sampling.metal"),
    os.path.join(_KERNELS, "moe", "moe.metal"),
    os.path.join(_KERNELS, "attn_causal", "attn_causal.metal"),
    os.path.join(_KERNELS, "flux", "flux.metal"),
    os.path.join(_KERNELS, "gemm_staged", "gemm_staged.metal"),
    os.path.join(_KERNELS, "attn_multiwarp", "attn_multiwarp.metal"),
    os.path.join(_KERNELS, "linear_attn", "linear_attn.metal"),
    os.path.join(_KERNELS, "hedgehog", "hedgehog.metal"),
    os.path.join(_KERNELS, "lin_attn_causal", "lin_attn_causal.metal"),
    os.path.join(_KERNELS, "mamba2", "mamba2.metal"),
    os.path.join(_KERNELS, "lin_attn_decay", "lin_attn_decay.metal"),
    os.path.join(_KERNELS, "based", "based.metal"),
    os.path.join(_KERNELS, "attn_bwd", "attn_bwd.metal"),
    os.path.join(_KERNELS, "cmplx_matmul", "cmplx_matmul.metal"),
    os.path.join(_KERNELS, "fftconv", "fftconv.metal"),
    os.path.join(_KERNELS, "qgemm", "qgemm.metal"),
    os.path.join(_KERNELS, "qgemv", "qgemv.metal"),
    os.path.join(_KERNELS, "qflux", "qflux.metal"),
    os.path.join(_KERNELS, "qgemv_int", "qgemv_int.metal"),
    os.path.join(_KERNELS, "attn_q", "attn_q.metal"),
    os.path.join(_KERNELS, "qgemm_int", "qgemm_int.metal"),
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


def rms_norm_add(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """Fused residual-add + RMSNorm. Returns (out, x+residual). bf16 MPS; D in {256,512,768,1024}."""
    return _ext.rms_norm_add(x, residual, weight, float(eps))


def layernorm_add(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                  bias: torch.Tensor, eps: float = 1e-5):
    """Fused residual-add + LayerNorm. Returns (out, x+residual). bf16 MPS; D in {256,512,768,1024}."""
    return _ext.layernorm_add(x, residual, weight, bias, float(eps))


def rms_norm_add_fp8(x, residual, weight, eps: float = 1e-5, scale=None):
    """Fused add + rms_norm + fp8. scale=None -> dynamic per-row (codes,added,scale);
    else static per-tensor (codes,added). codes are e4m3 uint8. bf16 MPS."""
    if scale is None:
        return _ext.rms_norm_add_fp8_dyn(x, residual, weight, float(eps))
    return _ext.rms_norm_add_fp8(x, residual, weight, float(eps), float(scale))


def layernorm_add_fp8(x, residual, weight, bias, eps: float = 1e-5, scale=None):
    """Fused add + layernorm + fp8. scale=None -> dynamic (codes,added,scale); else static (codes,added)."""
    if scale is None:
        return _ext.layernorm_add_fp8_dyn(x, residual, weight, bias, float(eps))
    return _ext.layernorm_add_fp8(x, residual, weight, bias, float(eps), float(scale))


def softmax(x: torch.Tensor):
    """Softmax over the last axis. bf16 MPS tensors; D in {256,512,768,1024}."""
    return _ext.softmax(x)


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False):
    """RoPE. x bf16 (B,H,N,D); cos/sin bf16 (N,D/2); D in {64,128}.
    interleaved=False: split-half (GPT-NeoX); True: GPT-J adjacent pairs."""
    return _ext.rotary(x, cos, sin, interleaved)


def rope_kv_insert(k: torch.Tensor, v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                   positions: torch.Tensor, slot_mapping: torch.Tensor,
                   key_cache: torch.Tensor, value_cache: torch.Tensor):
    """Fused RoPE on K + paged-KV insert. Returns updated (key_cache, value_cache).

    k/v bf16 (num_tokens, num_kv_heads, D); cos/sin (P, D/2); positions/slot_mapping
    (num_tokens,); caches (num_blocks, block_size, num_kv_heads, D) bf16; D in {64,128}.
    """
    return _ext.rope_kv_insert(k, v, cos, sin, positions, slot_mapping, key_cache, value_cache)


def rope_kv_insert_norm(k, v, cos, sin, positions, slot_mapping, key_cache, value_cache,
                        norm_weight, eps=1e-5, gemma=False):
    """Fused K RMSNorm + RoPE + paged-KV insert. gemma=True uses (1+weight). Returns (kc, vc). MPS."""
    return _ext.rope_kv_insert_norm(k, v, cos, sin, positions, slot_mapping, key_cache,
                                    value_cache, norm_weight, float(eps), bool(gemma))


def mla_q_norm_rope(q, cos, sin, positions, norm_weight, num_heads, nope_dim, rope_dim,
                    norm_mode, eps):
    """DeepSeek MLA Q-path: optional RMSNorm + GPT-J interleaved RoPE on the last rope_dim dims. MPS."""
    return _ext.mla_q_norm_rope(q, cos, sin, positions, norm_weight, int(num_heads),
                                int(nope_dim), int(rope_dim), int(norm_mode), float(eps))


def mla_kv_insert(kv_c, k_pe, cos, sin, positions, slot_mapping, kv_cache, norm_weight,
                  rope_dim, norm_mode, eps):
    """DeepSeek MLA classic KV-insert: latent + interleaved-RoPE k_pe into a paged bf16 cache. MPS."""
    return _ext.mla_kv_insert(kv_c, k_pe, cos, sin, positions, slot_mapping, kv_cache,
                              norm_weight, int(rope_dim), int(norm_mode), float(eps))


def gelu(x: torch.Tensor):
    """GELU (tanh approx) over the last axis. bf16 MPS; D in {256,512,768,1024}."""
    return _ext.gelu(x)


def glu(x: torch.Tensor, gate: torch.Tensor, mode: str = "swiglu",
        alpha: float = 1.0, limit: float = 1.0e20):
    """GLU-family activation. mode in reglu/geglu/swiglu/swiglu_oai/geglu_erf/geglu_quick."""
    return _ext.glu(x, gate, mode, float(alpha), float(limit))


def reglu(x: torch.Tensor, gate: torch.Tensor):
    return glu(x, gate, "reglu")


def geglu(x: torch.Tensor, gate: torch.Tensor):
    return glu(x, gate, "geglu")


def swiglu(x: torch.Tensor, gate: torch.Tensor):
    return glu(x, gate, "swiglu")


def swiglu_oai(x: torch.Tensor, gate: torch.Tensor, alpha: float = 1.0, limit: float = 1.0e20):
    return glu(x, gate, "swiglu_oai", alpha, limit)


def geglu_erf(x: torch.Tensor, gate: torch.Tensor):
    return glu(x, gate, "geglu_erf")


def geglu_quick(x: torch.Tensor, gate: torch.Tensor):
    return glu(x, gate, "geglu_quick")


def hadamard(x: torch.Tensor, scale: float = 0.0):
    """Walsh-Hadamard transform over the final axis. Default scale is 1/sqrt(D)."""
    return _ext.hadamard(x, float(scale))


def kv_cache_scatter(key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor,
                     num_blocks: int, block_size: int):
    """Scatter key/value rows (T,H,D) into paged KV caches. MPS tensors."""
    return _ext.kv_cache_scatter(key, value, slot_mapping, int(num_blocks), int(block_size))


def kv_cache_gather(key_cache: torch.Tensor, value_cache: torch.Tensor,
                    block_table: torch.Tensor, cu_seq_lens: torch.Tensor, num_tokens: int):
    """Gather paged KV caches back to contiguous key/value tensors. MPS tensors."""
    return _ext.kv_cache_gather(key_cache, value_cache, block_table, cu_seq_lens, int(num_tokens))


def kv_cache_copy_blocks(key_cache: torch.Tensor, value_cache: torch.Tensor,
                         block_mapping: torch.Tensor):
    """Copy paged KV cache blocks according to (src, dst) pairs. MPS tensors."""
    return _ext.kv_cache_copy_blocks(key_cache, value_cache, block_mapping)


def kv_cache_scales(key: torch.Tensor, value: torch.Tensor):
    """Return fp8 KV-cache scales `(key_scale, value_scale)` as absmax / 240. MPS tensors."""
    return _ext.kv_cache_scales(key, value)


def paged_attention(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                    block_table: torch.Tensor, context_lens: torch.Tensor, scale: float = 0.0):
    """Decode paged attention. q/out (B,H,D), caches (num_blocks, block_size, H, D)."""
    return _ext.paged_attention(q, key_cache, value_cache, block_table, context_lens, float(scale))


def paged_attention_alibi(q, key_cache, value_cache, block_table, context_lens, alibi_slopes,
                          scale=0.0):
    """Paged decode with a per-head ALiBi linear position bias (alibi_slopes is (num_heads,)). MPS."""
    return _ext.paged_attention_alibi(q, key_cache, value_cache, block_table, context_lens,
                                      alibi_slopes, float(scale))


def paged_attention_block_sparse(q, key_cache, value_cache, block_table, context_lens, block_mask,
                                 scale=0.0):
    """Block-sparse paged decode; block_mask (batch, max_blocks) int (1=attend, 0=skip). MPS."""
    return _ext.paged_attention_block_sparse(q, key_cache, value_cache, block_table, context_lens,
                                             block_mask, float(scale))


def paged_attention_xcache(q, key_cache, value_cache, block_table, context_lens, scale=0.0):
    """Paged decode over a vLLM x-packed KV cache: key (nb, nkv, hd/x, bs, x), value (nb, nkv, hd, bs). MPS."""
    return _ext.paged_attention_xcache(q, key_cache, value_cache, block_table, context_lens,
                                       float(scale))


def paged_attention_staged(q, key_cache, value_cache, block_table, context_lens, scale=0.0):
    """GQA KV-reuse staged decode; bit-equivalent to paged_attention. MPS."""
    return _ext.paged_attention_staged(q, key_cache, value_cache, block_table, context_lens,
                                       float(scale))


def _scale_vec_t(scale, num, ref):
    """Broadcast a python scalar into a (num,) float32 tensor on ref's device; tensors pass through."""
    if isinstance(scale, (int, float)):
        return torch.full((num,), float(scale), dtype=torch.float32, device=ref.device)
    return scale.to(dtype=torch.float32)


def _fmt_code(fmt):
    """Map an fp8 format ('e4m3'/'e5m2' or 0/1) to the kernel's integer format code."""
    return {"e4m3": 0, "e5m2": 1}.get(fmt, fmt) if isinstance(fmt, str) else int(fmt)


def kv_cache_scatter_fp8(key, value, slot_mapping, num_blocks, block_size, k_scale, v_scale,
                         fmt="e4m3"):
    """Scatter K/V into a uint8 paged cache. Returns (kc, vc). MPS.

    k_scale/v_scale: plain float (per-tensor) or a (num_heads,) tensor (per-head).
    fmt: 'e4m3' (default) or 'e5m2'.
    """
    H = key.shape[1]
    return _ext.kv_cache_scatter_fp8(key, value, slot_mapping, int(num_blocks), int(block_size),
                                     _scale_vec_t(k_scale, H, key), _scale_vec_t(v_scale, H, key),
                                     _fmt_code(fmt))


def paged_attention_fp8(q, key_cache, value_cache, block_table, context_lens,
                        k_scale, v_scale, scale=0.0, fmt="e4m3"):
    """Decode paged attention over fp8 (uint8) caches, dequantized on read. GQA aware. MPS.

    k_scale/v_scale: plain float (per-tensor) or a (num_kv_heads,) tensor (per-head).
    fmt: 'e4m3' (default) or 'e5m2' — must match the format the cache was written with.
    """
    H_KV = key_cache.shape[2]
    return _ext.paged_attention_fp8(q, key_cache, value_cache, block_table, context_lens,
                                    _scale_vec_t(k_scale, H_KV, q), _scale_vec_t(v_scale, H_KV, q),
                                    float(scale), _fmt_code(fmt))


def attn_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Causal attention forward. bf16 (B,H,N,D) MPS tensors; D in {64,128}, N%8==0."""
    return _ext.attn_causal(q, k, v)


def paged_attention_v2(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                       block_table: torch.Tensor, context_lens: torch.Tensor,
                       scale: float = 0.0, partition_size: int = 512):
    """Long-context paged decode attention (partition/reduce). GQA/MQA aware.
    q/out (B,H,D); caches (num_blocks, block_size, num_kv_heads, D); D in {64,128}."""
    return _ext.paged_attention_v2(q, key_cache, value_cache, block_table, context_lens,
                                   float(scale), int(partition_size))


def paged_attention_v2_fp8(q, key_cache, value_cache, block_table, context_lens,
                           k_scale, v_scale, scale=0.0, partition_size=512, fmt="e4m3"):
    """Long-context paged decode over an fp8 (uint8) cache, dequantized on read. GQA aware. MPS.

    k_scale/v_scale: plain float (per-tensor) or a (num_kv_heads,) tensor (per-head).
    fmt: 'e4m3' (default) or 'e5m2' — must match how the cache was written.
    """
    H_KV = key_cache.shape[2]
    return _ext.paged_attention_v2_fp8(q, key_cache, value_cache, block_table, context_lens,
                                       _scale_vec_t(k_scale, H_KV, q), _scale_vec_t(v_scale, H_KV, q),
                                       float(scale), int(partition_size), _fmt_code(fmt))


def moe_route_topk(logits: torch.Tensor, k: int):
    """MoE routing: top-k experts + renormalized softmax weights. Returns (ids int32, weights f32).
    logits (num_tokens, num_experts) float; k <= min(16, num_experts). MPS."""
    return _ext.moe_route_topk(logits, int(k))


def moe_permute(topk_ids: torch.Tensor, num_experts: int):
    """Group T*k routing rows by expert. Returns (sorted_row_idx, offsets, inv_idx) int32. MPS."""
    return _ext.moe_permute(topk_ids, int(num_experts))


def moe_grouped_gemm(permuted_input, W, expert_of_tile):
    """Fused grouped expert GEMM: out = permuted_input @ W[expert]. Returns (total_rows, H). MPS."""
    return _ext.moe_grouped_gemm(permuted_input, W, expert_of_tile)


def moe_finalize(expert_out: torch.Tensor, inv_idx: torch.Tensor, topk_weights: torch.Tensor, k: int):
    """out[t] = sum_k weight[t,k] * expert_out[inv_idx[t*k+k]]. Returns (T, Hdim). MPS."""
    return _ext.moe_finalize(expert_out, inv_idx, topk_weights, int(k))


def argmax_sample(logits: torch.Tensor):
    """Greedy sampling: argmax token index over the last (vocab) axis. Returns int32. MPS."""
    return _ext.argmax_sample(logits)


def sample_categorical(logits: torch.Tensor, temperature: float = 1.0, seed: int = 0):
    """Gumbel-max categorical sampling from softmax(logits/temperature). Returns int32. MPS."""
    return _ext.sample_categorical(logits, float(temperature), int(seed))


def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0, seed: int = 0):
    """Top-k sampling: Gumbel-max from softmax over the k highest logits. Returns int32. MPS."""
    return _ext.top_k_sample(logits, int(k), float(temperature), int(seed))


def top_p_sample(logits: torch.Tensor, p: float, temperature: float = 1.0, seed: int = 0):
    """Top-p (nucleus) sampling: Gumbel-max from the smallest top-prob set with mass >= p. int32. MPS."""
    return _ext.top_p_sample(logits, float(p), float(temperature), int(seed))


def apply_penalty(logits: torch.Tensor, prev_tokens: torch.Tensor, temperature: float = 1.0,
                  repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                  frequency_penalty: float = 0.0, bias=None, eos_id: int = -1,
                  min_length: int = 0, gen_len: int = 0):
    """Temperature + rep/presence/freq penalties + logit bias + min-length EOS mask (forbids eos_id
    while gen_len < min_length). bias is (V,) or None. Returns penalized logits (T,V). MPS."""
    if bias is None:
        bias = torch.zeros(logits.shape[-1], dtype=torch.float32, device=logits.device)
    return _ext.apply_penalty(logits, prev_tokens, bias, float(temperature),
                              float(repetition_penalty), float(presence_penalty),
                              float(frequency_penalty), int(eos_id), int(min_length), int(gen_len))


def quantize_per_tensor_fp8(x: torch.Tensor):
    """Per-tensor fp8 e4m3 quant (global absmax/448). Returns (codes uint8, scale scalar). MPS."""
    return _ext.quantize_per_tensor_fp8(x)


def quantize_per_tensor_int8(x: torch.Tensor):
    """Per-tensor symmetric int8 quant (global absmax/127). Returns (codes int8, scale scalar). MPS."""
    return _ext.quantize_per_tensor_int8(x)


def quantize_per_token_fp8(x: torch.Tensor):
    """Per-row fp8 e4m3 quant. Returns (codes uint8, scale f32), scale=absmax/448. MPS, x float."""
    return _ext.quantize_per_token_fp8(x)


def quantize_per_token_int8(x: torch.Tensor):
    """Per-row symmetric int8 quant. Returns (codes int8, scale f32), scale=absmax/127. MPS, x float."""
    return _ext.quantize_per_token_int8(x)


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


def lin_attn_decay(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cl: torch.Tensor):
    """Decay/retention linear attention. q,k,v bf16 (B,H,N,D); cl fp32 (B,H,N) = -slope*pos. MPS; D=64."""
    return _ext.lin_attn_decay(q, k, v, cl)


def based(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Based Taylor-map linear attention. q,k bf16 (B,H,N,16); v bf16 (B,H,N,64). MPS; N%8."""
    return _ext.based(q, k, v)


def attn_fwd_l(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False):
    """Flash-attn forward + logsumexp -> (o, L). q,k,v bf16 (B,H,N,D); L fp32 (B,H,N). MPS; D in {64,128}."""
    return _ext.attn_fwd_l(q, k, v, causal)


def attn_bwd_prep(o: torch.Tensor, do: torch.Tensor):
    """Backward prep: delta = rowsum(dO . O) (B,H,N) fp32. MPS."""
    return _ext.attn_bwd_prep(o, do)


def attn_bwd_dq(q, k, v, do, L, delta, causal=False):
    """Flash-attn backward dQ. bf16 (B,H,N,D); L,delta fp32 (B,H,N). MPS."""
    return _ext.attn_bwd_dq(q, k, v, do, L, delta, causal)


def attn_bwd_dkv(q, k, v, do, L, delta, causal=False):
    """Flash-attn backward -> (dK, dV). bf16 (B,H,N,D); L,delta fp32 (B,H,N). MPS."""
    return _ext.attn_bwd_dkv(q, k, v, do, L, delta, causal)


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


def qgemm_actorder_k(wq, x, perm, format="kU4B8"):
    """GPTQ act-order qgemm with in-kernel g_idx gather. wq uint8; x f16 (K,M); perm int32 (K,). MPS."""
    return _ext.qgemm_actorder_k(wq, x, perm, format)


def qgemm_blockscale(wq, x, scale2d):
    """fp8_block2d GEMM: codes-only fp8 + separate (N/128,K/128) tile scale. wq uint8; x,scale2d f16. MPS."""
    return _ext.qgemm_blockscale(wq, x, scale2d)


def qgemm_fp8_scaled(wq, xq, w_scale, a_scale):
    """fp8 rank-1 scaled GEMM: both operands fp8 e4m3 codes; w_scale (N,), a_scale (M,) f16. MPS."""
    return _ext.qgemm_fp8_scaled(wq, xq, w_scale, a_scale)


def qgemv(wq: torch.Tensor, x: torch.Tensor, format: str = "q8_0"):
    """Quantized GEMV (batch-1 decode): out = dequantize(wq) @ x. x (K, 1) float16 -> (N, 1). MPS."""
    return _ext.qgemv(wq, x, format)


def qflux_gelu(wq: torch.Tensor, x: torch.Tensor, bias: torch.Tensor, format: str = "q8_0"):
    """Quantized fused GEMM+GELU: gelu(dequantize(wq) @ x + bias). x (K,M) f16; bias (M,) f16. MPS."""
    return _ext.qflux_gelu(wq, x, bias, format)


def attn_q(q, kq, vq, format="q8_0", causal=False, multiwarp=False):
    """Quantized-KV flash attention: softmax(QK^T)V, K/V from blocks. q bf16 (B,H,N,D); kq/vq uint8. MPS."""
    return _ext.attn_q(q, kq, vq, format, causal, multiwarp)


def qgemm_w8a8(wq, xq, w_scale, a_scale):
    """W8A8 prefill GEMM (M>1, bit-exact int32). wq int8 (N,K); xq int8 (M,K) token-major. MPS."""
    return _ext.qgemm_w8a8(wq, xq, w_scale, a_scale)


def qgemm_w2a8(wq, xq, a_scale):
    """BitNet W2A8 prefill GEMM (M>1): ternary 2-bit weight x int8 act (M,K). MPS."""
    return _ext.qgemm_w2a8(wq, xq, a_scale)


def qgemv_w8a8(wq, xq, w_scale, a_scale):
    """W8A8 decode GEMV: int8 weight (N,K) x int8 act (K,1) -> int32 * w_scale[n] * a_scale. MPS."""
    return _ext.qgemv_w8a8(wq, xq, w_scale, a_scale)


def qgemv_w2a8(wq, xq, a_scale):
    """BitNet W2A8 decode GEMV: ternary 2-bit weight x int8 act -> int32, per-group scale. MPS."""
    return _ext.qgemv_w2a8(wq, xq, a_scale)
