"""PyTorch MPS backend for the QuixiCore Metal kernels.

The compute lives in the shared, framework-agnostic .metal kernels. This package:
  1. compiles them into a standalone metallib with `xcrun metal` (no MLX, no CMake), and
  2. JIT-compiles a thin ObjC++ extension (torch.utils.cpp_extension.load) that dispatches
     those kernels onto PyTorch's MPS stream.

So a PyTorch user needs neither MLX nor the Xcode/CMake build — only Xcode's Metal toolchain.
"""

import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_KERNELS = _REPO_ROOT / "kernels"
_INCLUDE = _REPO_ROOT / "include" / "metal"
_KERNEL_COMMON = _KERNELS / "common"
_METALLIB = _HERE / "tk.metallib"


def _kernel_source(path: str) -> Path:
    return _KERNELS / path

# The shared .metal kernel sources (single source of truth, also used by the MLX build).
_METAL_SOURCES = [
    _kernel_source("utils/add_rt/add_rt.metal"),
    _kernel_source("attention/attn_fwd/attn_fwd.metal"),
    _kernel_source("attention/attn_fwd_sg/attn_fwd_sg.metal"),
    _kernel_source("matmul/matmul_custom/matmul_custom.metal"),
    _kernel_source("norms/layernorm/layernorm.metal"),
    _kernel_source("norms/rms_norm/rms_norm.metal"),
    _kernel_source("norms/rms_norm_residual_next/rms_norm_residual_next.metal"),
    _kernel_source("serving/mean_pool_rms_l2/mean_pool_rms_l2.metal"),
    _kernel_source("norms/add_norm/add_norm.metal"),
    _kernel_source("activations/softmax/softmax.metal"),
    _kernel_source("attention/rotary/rotary.metal"),
    _kernel_source("attention/rope_kv/rope_kv.metal"),
    _kernel_source("norms/qk_norm_rope/qk_norm_rope.metal"),
    _kernel_source("norms/qk_norm_rope/qk_norm_rope_kv_f16.metal"),
    _kernel_source("ssm/selective_scan/selective_scan.metal"),
    _kernel_source("linear_attention/gdn/gdn.metal"),
    _kernel_source("quantization/act_quant/act_quant.metal"),
    _kernel_source("quantization/fake_quant/fake_quant.metal"),
    _kernel_source("quantization/fake_quant_fp8/fake_quant_fp8.metal"),
    _kernel_source("quantization/weight_quant_ternary/weight_quant_ternary.metal"),
    _kernel_source("quantization/quantize_tq2_0/quantize_tq2_0.metal"),
    _kernel_source("quantization/ternary_stats/ternary_stats.metal"),
    _kernel_source("serving/minference/minference.metal"),
    _kernel_source("quantization/turboquant/turboquant.metal"),
    _kernel_source("utils/marginal/marginal.metal"),
    _kernel_source("serving/indexer/indexer.metal"),
    _kernel_source("attention/mla/mla.metal"),
    _kernel_source("activations/gelu/gelu.metal"),
    _kernel_source("utils/dropout/dropout.metal"),
    _kernel_source("optimizers/optim/adamw.metal"),
    _kernel_source("serving/embedding/embedding.metal"),
    _kernel_source("activations/glu/glu.metal"),
    _kernel_source("utils/hadamard/hadamard.metal"),
    _kernel_source("serving/kv_cache/kv_cache.metal"),
    _kernel_source("attention/paged_attn_v2/paged_attn_v2.metal"),
    _kernel_source("quantization/quant_rt/quant_rt.metal"),
    _kernel_source("sampling/sampling/sampling.metal"),
    _kernel_source("sampling/sampling/sampling_transforms.metal"),
    _kernel_source("moe/moe/moe.metal"),
    _kernel_source("attention/attn_causal/attn_causal.metal"),
    _kernel_source("attention/attn_varlen/attn_varlen.metal"),
    _kernel_source("quantization/lm_head/lm_head.metal"),
    _kernel_source("utils/cross_entropy/cross_entropy.metal"),
    _kernel_source("utils/kd_kl_topk/kd_kl_topk.metal"),
    _kernel_source("utils/kd_kl_dense/kd_kl_dense.metal"),
    _kernel_source("matmul/flux/flux.metal"),
    _kernel_source("matmul/gemm_staged/gemm_staged.metal"),
    _kernel_source("matmul/gemm_v3/gemm_v3.metal"),
    _kernel_source("attention/attn_multiwarp/attn_multiwarp.metal"),
    _kernel_source("linear_attention/linear_attn/linear_attn.metal"),
    _kernel_source("linear_attention/hedgehog/hedgehog.metal"),
    _kernel_source("linear_attention/lin_attn_causal/lin_attn_causal.metal"),
    _kernel_source("ssm/mamba2/mamba2.metal"),
    _kernel_source("linear_attention/lin_attn_decay/lin_attn_decay.metal"),
    _kernel_source("linear_attention/based/based.metal"),
    _kernel_source("attention/attn_bwd/attn_bwd.metal"),
    _kernel_source("matmul/cmplx_matmul/cmplx_matmul.metal"),
    _kernel_source("ssm/fftconv/fftconv.metal"),
    _kernel_source("quantization/qgemm/qgemm.metal"),
    _kernel_source("quantization/qgemm_bwd/qgemm_bwd.metal"),
    _kernel_source("quantization/qgemm_fused/qgemm_fused.metal"),
    _kernel_source("quantization/qgemv/qgemv.metal"),
    _kernel_source("quantization/qgemv_fused/qgemv_fused.metal"),
    _kernel_source("quantization/qflux/qflux.metal"),
    _kernel_source("quantization/qgemv_int/qgemv_int.metal"),
    _kernel_source("attention/attn_q/attn_q.metal"),
    _kernel_source("attention/attn_decode/attn_decode.metal"),
    _kernel_source("quantization/qgemm_int/qgemm_int.metal"),
    _kernel_source("attention/swin_attn/swin_attn.metal"),
    _kernel_source("matmul/decode_linear/decode_linear.metal"),
    _kernel_source("quantization/dequant_gather/dequant_gather.metal"),
    _kernel_source("vision/edge_mlp/edge_mlp.metal"),
    _kernel_source("vision/patch_merge/patch_merge.metal"),
]


def build_metallib(force: bool = False) -> str:
    """Compile the shared .metal kernels into tk.metallib via xcrun metal. MLX-independent."""
    if not force and _METALLIB.exists():
        # staleness must also track the header-only substrate under include/ (tk.metal pulls
        # in everything there), not just the listed kernel sources
        deps = list(_METAL_SOURCES)
        deps.extend(_INCLUDE.rglob("*.metal"))
        newest_src = max(s.stat().st_mtime for s in deps)
        if _METALLIB.stat().st_mtime >= newest_src:
            return str(_METALLIB)
    cmd = ["xcrun", "metal", "-std=metal3.1", "-O2", "-I", str(_INCLUDE),
           *map(str, _METAL_SOURCES), "-o", str(_METALLIB)]
    subprocess.run(cmd, check=True)
    return str(_METALLIB)


# Build the metallib (if missing/stale) and the ObjC++ extension on import.
build_metallib()

_ext = load(
    name="tk_torch_ext",
    sources=[str(_HERE / "torch_kernels.mm")],
    extra_cflags=["-std=c++17"],
    extra_include_paths=[str(_KERNEL_COMMON)],
    extra_ldflags=["-framework", "Metal", "-framework", "Foundation", "-framework", "QuartzCore"],
    verbose=False,
)
_ext._set_library(str(_METALLIB))


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """LayerNorm over the last axis. bf16 MPS tensors; D divisible by four."""
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


def attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softcap: float = 0.0,
             sinks: torch.Tensor | None = None):
    """Non-causal attention forward. bf16 (B,H,N,D) MPS tensors; D in {64,128}, N%8==0.
    softcap > 0 = Gemma-style logit capping; sinks (H,) = gpt-oss attention sinks."""
    return _ext.attn_fwd(q, k, v, float(softcap), sinks)


def attn_fwd_sg_d256(q, k, v, scale: float = 0.0, window: int = 0):
    """simdgroup_matrix flash attention, head-dim 256, GQA, f16 KV. q/o (T, Hq, 256) f32,
    k/v (T, Hkv, 256) (cast to f16); Hq a multiple of Hkv. Bidirectional; window>0 keeps keys
    within window/2 of the query. scale<=0 defaults to 1/sqrt(256). Returns (T, Hq, 256). MPS."""
    return _ext.attn_fwd_sg_d256(q, k, v, float(scale), int(window))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """RMSNorm over the last axis. bf16 MPS tensors; static kernels for D in {256,512,768,1024},
    dynamic kernel for other D multiples of 4."""
    return _ext.rms_norm(x, weight, float(eps))


def mean_pool_rms_l2(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """Mean-pool an (M, D) block of token states into one D embedding, apply RMSNorm(weight),
    then L2-normalize. bf16 MPS tensors; D in {256,512,768,1024}. Returns (D,)."""
    return _ext.mean_pool_rms_l2(x, weight, float(eps))


def rms_norm_bwd_dx(x, weight, dy, rstd):
    """RMSNorm backward dX kernel (rstd (rows,) precomputed). Use tk.rms_norm_backward for the full
    (dx, dweight). MPS tensors."""
    return _ext.rms_norm_bwd_dx(x, weight, dy, rstd)


def rms_norm_bwd_fused(x, weight, dy, eps):
    """Fused RMSNorm backward -> (dx, dweight) in one pass (rstd in-kernel + atomic dweight). MPS."""
    return tuple(_ext.rms_norm_bwd_fused(x, weight, dy, float(eps)))


def layernorm_bwd_dx(x, weight, dy, mean, rstd):
    """LayerNorm backward dX kernel (mean/rstd (rows,) precomputed). Use tk.layernorm_backward for
    the full (dx, dweight, dbias). MPS tensors."""
    return _ext.layernorm_bwd_dx(x, weight, dy, mean, rstd)


def layernorm_bwd_fused(x, weight, dy, eps):
    """Fused LayerNorm backward -> (dx, dweight, dbias) in one pass (mean/rstd in-kernel + atomic
    dweight/dbias). MPS."""
    return tuple(_ext.layernorm_bwd_fused(x, weight, dy, float(eps)))


def rms_norm_add(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
    """Fused residual-add + RMSNorm. Returns (out, x+residual). bf16 MPS; D in {256,512,768,1024}."""
    return _ext.rms_norm_add(x, residual, weight, float(eps))


def rms_norm_residual_next(x, post_weight, residual, next_weight, eps: float = 1e-5):
    """Fused residual-stream seam: res_out = residual + rms_norm(x) * post_weight, then
    next_out = rms_norm(res_out) * next_weight. Returns (res_out, next_out). bf16 MPS;
    D in {256,512,768,1024}."""
    return tuple(_ext.rms_norm_residual_next(x, post_weight, residual, next_weight, float(eps)))


def layernorm_add(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                  bias: torch.Tensor, eps: float = 1e-5):
    """Fused residual-add + LayerNorm. Returns (out, x+residual). bf16 MPS; D in {256,512,768,1024}."""
    return _ext.layernorm_add(x, residual, weight, bias, float(eps))


def decode_layernorm_add(x, residual, weight, bias, eps=1e-5):
    """Decode-compatible add + LayerNorm with input-dtype rounding before statistics."""
    return tuple(_ext.decode_layernorm_add(x, residual, weight, bias, float(eps)))


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


def rope_q(q, cos, sin, positions, norm_weight, do_norm, gemma, eps):
    """Q-path RoPE (+optional weighted RMSNorm) into a contiguous q_out. MPS."""
    return _ext.rope_q(q, cos, sin, positions, norm_weight, bool(do_norm), bool(gemma), float(eps))


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


def mla_kv_insert_fp8(kv, cos, sin, positions, slot_mapping, data_cache, scale_cache):
    """DeepSeek-V4 packed fp8 MLA KV-insert. Returns (data_cache u8 (…,576), scale_cache u8 (…,8)). MPS."""
    return _ext.mla_kv_insert_fp8(kv, cos, sin, positions, slot_mapping, data_cache, scale_cache)


def mla_decode_fp8(q, data_cache, scale_cache, block_table, context_lens, scale=0.0):
    """DeepSeek-V4 dense latent decode over the UE8M0-packed cache. q (B,N,512) -> o (B,N,512). MPS."""
    return _ext.mla_decode_fp8(q, data_cache, scale_cache, block_table, context_lens, float(scale))


def mla_decode_fp8_sparse(q, data_cache, scale_cache, block_table, indices, topk_length, scale=0.0):
    """DeepSeek-V4 sparse latent decode: attend only indices[b, 0:topk_length[b]]. -> o (B,N,512). MPS."""
    return _ext.mla_decode_fp8_sparse(q, data_cache, scale_cache, block_table, indices, topk_length,
                                      float(scale))


def mla_decode(q, kv_cache, block_table, context_lens, scale=0.0):
    """DeepSeek MLA absorb-path latent flash-decode (MQA). q (B,N,576), cache (nb,bs,576) -> o (B,N,512). MPS."""
    return _ext.mla_decode(q, kv_cache, block_table, context_lens, float(scale))


def gelu_bwd(x, dy):
    """GELU (tanh approx) backward: dx = dy * gelu'(x). MPS tensors."""
    return _ext.gelu_bwd(x, dy)


def gelu(x: torch.Tensor):
    """GELU (tanh approx) over the last axis. bf16 MPS; D in {256,512,768,1024}."""
    return _ext.gelu(x)


def dropout(x: torch.Tensor, p: float, seed: int):
    """Inverted dropout: out = keep ? x/(1-p) : 0, keep from (seed, index). p in [0,1). MPS."""
    return _ext.dropout(x, float(p), int(seed), False)


def dropout_backward(dy: torch.Tensor, p: float, seed: int):
    """Dropout backward: dx = keep ? dy/(1-p) : 0 (same mask from seed). MPS."""
    return _ext.dropout(dy, float(p), int(seed), True)


def adamw(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step):
    """AdamW step. Returns (param', m', v'); m/v are fp32 moment state, step (t) >= 1. MPS."""
    return tuple(_ext.adamw(param, grad, m, v, float(lr), float(beta1), float(beta2),
                            float(eps), float(weight_decay), int(step)))


def adamw_masked(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step,
                 mask, seg_size, mask_mode=0):
    """Segment-masked AdamW. mask_mode=0 skips inactive segment updates; mask_mode=1 skips only decay."""
    return tuple(_ext.adamw_masked(param, grad, m, v, float(lr), float(beta1), float(beta2),
                                   float(eps), float(weight_decay), int(step),
                                   mask, int(seg_size), int(mask_mode)))


def glu(x: torch.Tensor, gate: torch.Tensor, mode: str = "swiglu",
        alpha: float = 1.0, limit: float = 1.0e20):
    """GLU-family activation. mode in reglu/geglu/swiglu/swiglu_oai/geglu_erf/geglu_quick."""
    return _ext.glu(x, gate, mode, float(alpha), float(limit))


def glu_backward(x: torch.Tensor, gate: torch.Tensor, dc: torch.Tensor, mode: str = "swiglu",
                 alpha: float = 1.0, limit: float = 1.0e20):
    """GLU-family backward. Returns (da, db) = grads wrt x, gate given upstream grad dc. MPS."""
    return tuple(_ext.glu_backward(x, gate, dc, mode, float(alpha), float(limit)))


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


def kv_cache_gather_fp8(key_cache, value_cache, block_table, cu_seq_lens, k_scale, v_scale,
                        num_tokens, fmt=0):
    """fp8 KV gather + upconvert to bf16 (per-kv_head scales). Returns (key_out, value_out). MPS."""
    return _ext.kv_cache_gather_fp8(key_cache, value_cache, block_table, cu_seq_lens, k_scale,
                                    v_scale, int(num_tokens), int(fmt))


def kv_cache_scale_update(key, value, old_key_scale, old_value_scale):
    """Incremental per-tensor KV scale running-max update. Returns (new_key_scale, new_value_scale). MPS."""
    return _ext.kv_cache_scale_update(key, value, old_key_scale, old_value_scale)


def kv_cache_copy_blocks(key_cache: torch.Tensor, value_cache: torch.Tensor,
                         block_mapping: torch.Tensor):
    """Copy paged KV cache blocks according to (src, dst) pairs. MPS tensors."""
    return _ext.kv_cache_copy_blocks(key_cache, value_cache, block_mapping)


def varlen_build_worklist(cu_seqlens: torch.Tensor, max_tiles: int):
    """On-device varlen prefill worklist builder from cu_seqlens (B+1,). Returns
    (qlens, pad_off, tile_seq, tile_local0, n_tiles); tile_seq is -1 past n_tiles. max_tiles is a
    host upper bound on the tile count. MPS tensors."""
    return tuple(_ext.varlen_build_worklist(cu_seqlens, int(max_tiles)))


def spec_verify_linear(draft_tokens, draft_probs, target_probs, bonus_tokens, accept_u, seed):
    """Speculative decoding linear rejection-sampling verification (vLLM contract). Returns
    (out_tokens (B,S+1) int32, accepted_cnt (B,) int32). MPS tensors."""
    return tuple(_ext.spec_verify_linear(draft_tokens, draft_probs, target_probs, bonus_tokens,
                                         accept_u, int(seed)))


def build_dynamic_tree(parents):
    """Device draft-tree builder -> (retrieve_next_token, retrieve_next_sibling, positions) int32. MPS."""
    return tuple(_ext.build_dynamic_tree(parents))


def spec_verify_tree(draft_tokens, target_probs, retrieve_next_token, retrieve_next_sibling, seed,
                     tree_valid=None):
    """Speculative tree verification -> (accept_index, accept_token, accept_num) int32, -1-pad. MPS.
    tree_valid (B,) int optional (default all ones); where 0 the request samples the target root
    token (accept_num=0). Exact for any sibling count."""
    import torch
    if tree_valid is None:
        tree_valid = torch.ones(target_probs.shape[0], dtype=torch.int32, device=target_probs.device)
    return tuple(_ext.spec_verify_tree(draft_tokens, target_probs, retrieve_next_token,
                                       retrieve_next_sibling, tree_valid, int(seed)))


def spec_compact(out_tokens, accepted_cnt, seq_lens):
    """Compact accepted spec tokens -> (packed_tokens, packed_pos, cu_accepted) int32. Any B. MPS."""
    return tuple(_ext.spec_compact(out_tokens, accepted_cnt, seq_lens))


def eagle_prepare_inputs_padded(cu_num_draft_tokens, valid_sampled_tokens_count, query_start_loc):
    """EAGLE prep -> (token_indices_to_sample, num_rejected) int32. MPS."""
    return tuple(_ext.eagle_prepare_inputs_padded(cu_num_draft_tokens, valid_sampled_tokens_count,
                                                  query_start_loc))


def eagle_prepare_next_token_padded(sampled_token_ids, discard_request_mask,
                                    backup_next_token_ids, vocab_size):
    """EAGLE next seed token -> (next_token_ids, valid_sampled_tokens_count) int32. MPS."""
    return tuple(_ext.eagle_prepare_next_token_padded(sampled_token_ids, discard_request_mask,
                                                      backup_next_token_ids, int(vocab_size)))


def eagle_step_slot_mapping_metadata(positions, block_table, seq_lens, block_size, max_model_len,
                                     pad_id, input_batch_size=-1):
    """EAGLE step slot -> (out_clamped_positions, out_slot_mapping, new_seq_lens) int32. MPS."""
    return tuple(_ext.eagle_step_slot_mapping_metadata(positions, block_table, seq_lens,
                                                       int(block_size), int(max_model_len),
                                                       int(pad_id), int(input_batch_size)))


def eagle_expand_int32(input, cu_num_tokens, total, replace_from=-1, replace_to=-1):
    """EAGLE ragged broadcast -> (total,) int32. MPS."""
    return _ext.eagle_expand_int32(input, cu_num_tokens, int(total), int(replace_from),
                                   int(replace_to))


def spec_update_kv_meta(seq_lens, accepted_cnt):
    """new_seq_lens[b] = seq_lens[b] + accepted_cnt[b] + 1. Returns (B,) int32. MPS."""
    return _ext.spec_update_kv_meta(seq_lens, accepted_cnt)


def rejection_greedy_sample(cu_num_draft_tokens, draft_token_ids, target_argmax, bonus_token_ids,
                            max_draft, is_greedy=None):
    """vLLM greedy rejection verify -> out (B, max_draft+1) int32. MPS."""
    return _ext.rejection_greedy_sample(cu_num_draft_tokens, draft_token_ids, target_argmax,
                                        bonus_token_ids, int(max_draft), is_greedy)


def rejection_random_sample(cu_num_draft_tokens, draft_token_ids, target_probs, bonus_token_ids,
                            recovered_token_ids, uniform_probs, max_draft, draft_probs=None,
                            is_greedy=None):
    """vLLM stochastic rejection verify -> out (B, max_draft+1) int32. MPS."""
    return _ext.rejection_random_sample(cu_num_draft_tokens, draft_token_ids, target_probs,
                                        bonus_token_ids, recovered_token_ids, uniform_probs,
                                        int(max_draft), draft_probs, is_greedy)


def sample_recovered_tokens(cu_num_draft_tokens, draft_token_ids, target_probs, inv_q,
                            draft_probs=None):
    """Recovered token per draft position -> (total_draft,) int32. MPS."""
    return _ext.sample_recovered_tokens(cu_num_draft_tokens, draft_token_ids, target_probs, inv_q,
                                        draft_probs)


def beam_build_copy_pairs(parent_beam: torch.Tensor, block_table: torch.Tensor,
                          seq_lens: torch.Tensor, block_size: int):
    """Build the (src,dst) block-copy pairs for a beam KV reorder on-device (no host readback).
    Returns a fixed (B*BM*max_blocks, 2) int64 tensor of pairs (sentinel (-1,-1) for empty slots)
    for kv_cache_copy_blocks. MPS tensors."""
    return _ext.beam_build_copy_pairs(parent_beam, block_table, seq_lens, int(block_size))


def beam_remap_block_table(block_table, parent_beam):
    """Zero-copy beam reorder: new_block_table[b*BM+k] = block_table[b*BM+parent_beam[b,k]]. MPS."""
    return _ext.beam_remap_block_table(block_table, parent_beam)


def kv_cache_scales(key: torch.Tensor, value: torch.Tensor):
    """Return fp8 KV-cache scales `(key_scale, value_scale)` as absmax / 240. MPS tensors."""
    return _ext.kv_cache_scales(key, value)


def paged_attention(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                    block_table: torch.Tensor, context_lens: torch.Tensor, scale: float = 0.0,
                    window: int = 0):
    """Decode paged attention. q/out (B,H,D), caches (num_blocks, block_size, H, D).
    window > 0 restricts to the `window` most recent keys (Mistral sliding window). MPS."""
    return _ext.paged_attention(q, key_cache, value_cache, block_table, context_lens,
                                float(scale), int(window))


def paged_attention_alibi(q, key_cache, value_cache, block_table, context_lens, alibi_slopes,
                          scale=0.0, window=0):
    """Paged decode with a per-head ALiBi linear position bias (alibi_slopes is (num_heads,)). MPS.
    window > 0 restricts to the `window` most recent keys."""
    return _ext.paged_attention_alibi(q, key_cache, value_cache, block_table, context_lens,
                                      alibi_slopes, float(scale), int(window))


def paged_attention_block_sparse(q, key_cache, value_cache, block_table, context_lens, block_mask,
                                 scale=0.0, window=0):
    """Block-sparse paged decode; block_mask (batch, max_blocks) int (1=attend, 0=skip). MPS.
    window > 0 restricts to the `window` most recent keys."""
    return _ext.paged_attention_block_sparse(q, key_cache, value_cache, block_table, context_lens,
                                             block_mask, float(scale), int(window))


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
                        k_scale, v_scale, scale=0.0, fmt="e4m3", window=0):
    """Decode paged attention over fp8 (uint8) caches, dequantized on read. GQA aware. MPS.

    k_scale/v_scale: plain float (per-tensor) or a (num_kv_heads,) tensor (per-head).
    fmt: 'e4m3' (default) or 'e5m2' — must match the format the cache was written with.
    window > 0 restricts to the `window` most recent keys (Mistral sliding window).
    """
    H_KV = key_cache.shape[2]
    return _ext.paged_attention_fp8(q, key_cache, value_cache, block_table, context_lens,
                                    _scale_vec_t(k_scale, H_KV, q), _scale_vec_t(v_scale, H_KV, q),
                                    float(scale), _fmt_code(fmt), int(window))


def attn_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softcap: float = 0.0,
                sinks: torch.Tensor | None = None):
    """Causal attention forward. bf16 (B,H,N,D) MPS tensors; D in {64,128}, N%8==0.
    softcap > 0 = Gemma-style logit capping; sinks (H,) = gpt-oss attention sinks."""
    return _ext.attn_causal(q, k, v, softcap=float(softcap), sinks=sinks)


def attn_varlen_prefill(q_hm, key_cache, value_cache, block_table, context_lens,
                        tile_seq, tile_local0, seq_qlen, scale, softcap=0.0, sinks=None):
    """Low-level varlen/paged-prefill attention (head-major q/o + host worklist). MPS.

    q_hm (H, total_padded, D) bf16; key_cache/value_cache (nb, bs, H_KV, D) bf16;
    block_table (B, max_blocks), context_lens (B,), tile_seq/tile_local0 (n_tiles,),
    seq_qlen (B,) int32. Returns o_hm (H, total_padded, D). Prefer tk.attn_varlen_prefill,
    which builds the worklist and pads/transposes for you."""
    return _ext.attn_varlen_prefill(q_hm, key_cache, value_cache, block_table, context_lens,
                                    tile_seq, tile_local0, seq_qlen, float(scale),
                                    softcap=float(softcap), sinks=sinks)


def varlen_pad_q(q_packed, cu_seqlens, pad_off, total_padded):
    """Device varlen Q pad/gather: packed (total_q,H,D) bf16 -> padded head-major (H,total_padded,D). MPS."""
    return _ext.varlen_pad_q(q_packed, cu_seqlens, pad_off, int(total_padded))


def varlen_regather_o(o_hm, cu_seqlens, pad_off, total_q):
    """Device varlen output re-gather: (H,total_padded,D) bf16 -> packed (total_q,H,D). MPS."""
    return _ext.varlen_regather_o(o_hm, cu_seqlens, pad_off, int(total_q))


def attn_window(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
                softcap: float = 0.0, sinks: torch.Tensor | None = None):
    """Sliding-window causal attention: query i attends keys [max(0, i-window+1), i].
    window <= 0 disables the window. bf16 (B,H,N,D) MPS tensors; D in {64,128}, N%8==0.
    softcap > 0 = Gemma-style logit capping; sinks (H,) = gpt-oss attention sinks."""
    return _ext.attn_window(q, k, v, window, softcap=float(softcap), sinks=sinks)


def attn_decode(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor):
    """Batch-1 GQA attention decode. q (Hq,D), caches (Tk,Hkv,D), D <= 128. MPS."""
    return _ext.attn_decode(q, key_cache, value_cache)


def attn_decode_bh(q, key_cache, value_cache, context_length):
    """Batched head-major GQA decode. q (B,Hq,D), caches (B,Hkv,cache_T,D)."""
    return _ext.attn_decode_bh(q, key_cache, value_cache, int(context_length))


def decode_cache_attention(q, new_k, new_v, cos, sin, positions, context_lengths,
                           q_weight, k_weight, key_cache, value_cache, eps=1e-6,
                           do_q_norm=False, do_k_norm=False, gemma=False,
                           softmax_scale=0.0):
    """Fused norm, RoPE, cache append, and multi-simdgroup decode attention."""
    return tuple(_ext.decode_cache_attention(
        q, new_k, new_v, cos, sin, positions, context_lengths,
        q_weight, k_weight, key_cache, value_cache, float(eps),
        bool(do_q_norm), bool(do_k_norm), bool(gemma), float(softmax_scale)))


def swin_attn_d32(qkv, relative_bias, mask, windows_per_image=0):
    """Swin packed-QKV window attention for head dimension 32."""
    return _ext.swin_attn_d32(qkv, relative_bias, mask, int(windows_per_image))


def patch_merge_layernorm(x, weight, bias, height, width, eps=1e-5):
    """Fused Swin 2x2 patch gather + LayerNorm."""
    return _ext.patch_merge_layernorm(x, weight, bias, int(height), int(width), float(eps))


def space_to_depth_norm_linear(x, norm_weight, norm_bias, projection_weight,
                               projection_bias, height, width, block_size=2,
                               eps=1e-5, use_norm_bias=True,
                               use_projection_bias=False):
    """Fused space-to-depth, LayerNorm, and dense projection."""
    return _ext.space_to_depth_norm_linear(
        x, norm_weight, norm_bias, projection_weight, projection_bias,
        int(height), int(width), int(block_size), float(eps),
        bool(use_norm_bias), bool(use_projection_bias))


def edge_mlp_256x7(hidden, first_weight, first_bias, second_weight, second_bias):
    """Fixed pairwise 512->256->7 edge MLP; returns (B,7,L,L)."""
    return _ext.edge_mlp_256x7(
        hidden, first_weight, first_bias, second_weight, second_bias)


def decode_linear(x, weight, bias, gelu=False):
    """Latency-oriented decode linear with optional erf GELU."""
    return _ext.decode_linear(x, weight, bias, bool(gelu))


def decode_linear_residual(x, weight, bias, residual):
    """Decode linear with materialized-order residual addition."""
    return _ext.decode_linear_residual(x, weight, bias, residual)


def decode_linear_q8(x, weight, bias, residual, gelu=False, use_residual=False):
    """q8_0 decode linear with optional erf GELU and residual."""
    return _ext.decode_linear_q8(
        x, weight, bias, residual, bool(gelu), bool(use_residual))


def decode_linear_epilogue(x, weight, bias, residual, format="", activation=0,
                           use_bias=False, use_residual=False):
    """Dense/packed decode linear with fused bias, activation, and residual."""
    return _ext.decode_linear_epilogue(
        x, weight, bias, residual, str(format), int(activation),
        bool(use_bias), bool(use_residual))


def decode_swiglu(x, gate_weight, up_weight, gate_bias, up_bias,
                  format="", use_bias=False):
    """Dense/packed decode projections with fused SwiGLU."""
    return _ext.decode_swiglu(
        x, gate_weight, up_weight, gate_bias, up_bias,
        str(format), bool(use_bias))


def lm_head_sample(h, W, bias, mode, k, temperature, seed):
    """Fused LM-head + sampling: token id per row of h without materializing (T,V) logits.
    mode 0=argmax, 1=categorical, 2=top-k. bias (V,) or a 1-elem dummy. Returns (T,) int32. MPS."""
    return _ext.lm_head_sample(h, W, bias, int(mode), int(k), float(temperature), int(seed))


def lm_head_sample_q(h, Wq, bias, V, K, fmt, mode, topk, temperature, seed, top_p=0.0):
    """Fused LM-head + sampling over quantized q8_0/q4_0/q6_K/mxfp8/nvfp4/mxfp4 weights. mode 0=argmax, 1=categorical,
    2=topk, 3=topp (nucleus over the top-k candidate pool, top_p in (0,1]). Returns (T,) int32. MPS."""
    return _ext.lm_head_sample_q(h, Wq, bias, int(V), int(K), str(fmt), int(mode), int(topk),
                                 float(temperature), int(seed), float(top_p))


def lm_head_beam_advance(h, Wq, bias, cum_log_probs, beam_width, fmt="q4_0"):
    """Quantized LM-head + exact beam-search advance without full logits. MPS."""
    return _ext.lm_head_beam_advance(
        h, Wq, bias, cum_log_probs, int(beam_width), str(fmt))


def lm_head_constrained(h, W, bias, forbidden, previous, eos_id=-1, forbid_eos=False):
    """Dense grammar-constrained LM head; returns (token ids, selected log-probabilities)."""
    return tuple(_ext.lm_head_constrained(
        h, W, bias, forbidden, previous, int(eos_id), bool(forbid_eos)))


def lm_head_masked(h, W, bias, allow_mask, format="", topk=1,
                   normalize_allowed=True):
    """Masked dense/packed LM head returning top-k ids and log-probabilities."""
    return tuple(_ext.lm_head_masked(
        h, W, bias, allow_mask, str(format), int(topk), bool(normalize_allowed)))


def lm_head_candidates(h, W, bias, candidate_ids, offsets, format="", topk=1):
    """CSR candidate dense/packed LM head returning top-k ids and log-probabilities."""
    return tuple(_ext.lm_head_candidates(
        h, W, bias, candidate_ids, offsets, str(format), int(topk)))


def cross_entropy_fwd(logits, targets, ignore_index, label_smoothing, z_loss, softcap=0.0):
    """Fused cross-entropy forward. Returns (loss (T,), lse (T,)) f32. MPS."""
    return _ext.cross_entropy_fwd(logits, targets, int(ignore_index),
                                  float(label_smoothing), float(z_loss), float(softcap))


def cross_entropy_bwd(logits, targets, lse, grad_out, ignore_index, label_smoothing, z_loss,
                      softcap=0.0):
    """Fused cross-entropy backward -> grad_logits (T,V), out-of-place. MPS."""
    return _ext.cross_entropy_bwd(logits, targets, lse, grad_out, int(ignore_index),
                                  float(label_smoothing), float(z_loss), float(softcap))


def kd_kl_topk_fwd(logits, t_idx, t_prob, invtemp=1.0, tail_mode=0):
    """Sparse-teacher KD-KL forward over a top-k teacher cache. Returns (loss, lse). MPS."""
    return tuple(_ext.kd_kl_topk_fwd(logits, t_idx, t_prob, float(invtemp), int(tail_mode)))


def kd_kl_topk_bwd(logits, t_idx, t_prob, lse, grad_out, invtemp=1.0, tail_mode=0):
    """Sparse-teacher KD-KL backward. Returns grad_logits. MPS."""
    return _ext.kd_kl_topk_bwd(logits, t_idx, t_prob, lse, grad_out, float(invtemp),
                               int(tail_mode))


def kd_kl_dense_fwd(t_logits, s_logits, invtemp=1.0):
    """Dense-teacher KD-KL forward. Returns (loss, lse_t, lse_s). MPS."""
    return tuple(_ext.kd_kl_dense_fwd(t_logits, s_logits, float(invtemp)))


def kd_kl_dense_bwd(t_logits, s_logits, lse_t, lse_s, grad_out, invtemp=1.0):
    """Dense-teacher KD-KL backward. Returns grad wrt student logits. MPS."""
    return _ext.kd_kl_dense_bwd(t_logits, s_logits, lse_t, lse_s, grad_out, float(invtemp))


def kd_ce_fused_fwd(t_logits, s_logits, targets, invtemp=1.0):
    """Fused CE + dense-KD forward. Returns (ce, kd, lse_sr, lse_st, lse_t). MPS."""
    return tuple(_ext.kd_ce_fused_fwd(t_logits, s_logits, targets, float(invtemp)))


def kd_ce_fused_bwd(t_logits, s_logits, targets, lse_sr, lse_st, lse_t,
                    go_ce, go_kd, invtemp=1.0):
    """Fused CE + dense-KD backward. Returns combined grad wrt student logits. MPS."""
    return _ext.kd_ce_fused_bwd(t_logits, s_logits, targets, lse_sr, lse_st, lse_t,
                                go_ce, go_kd, float(invtemp))


def paged_attention_v2(q: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                       block_table: torch.Tensor, context_lens: torch.Tensor,
                       scale: float = 0.0, partition_size: int = 512, window: int = 0,
                       softcap: float = 0.0, sinks: torch.Tensor | None = None):
    """Long-context paged decode attention (partition/reduce). GQA/MQA aware.
    q/out (B,H,D); caches (num_blocks, block_size, num_kv_heads, D); D in {64,128}.
    window > 0 restricts to the `window` most recent keys."""
    return _ext.paged_attention_v2(q, key_cache, value_cache, block_table, context_lens,
                                   float(scale), int(partition_size), int(window),
                                   softcap=float(softcap), sinks=sinks)


def cascade_attention(q: torch.Tensor, prefix_k: torch.Tensor, prefix_v: torch.Tensor,
                      key_cache: torch.Tensor, value_cache: torch.Tensor, block_table: torch.Tensor,
                      context_lens: torch.Tensor, scale: float = 0.0, partition_size: int = 512):
    """Cascade / shared-prefix attention: shared contiguous prefix KV + per-request paged suffix,
    merged via the shared log-sum-exp reduce == full attention over [prefix ++ suffix]. MPS tensors.
    q/out (B,H,D); D in {64,128}; partition_size a multiple of block_size."""
    return _ext.cascade_attention(q, prefix_k, prefix_v, key_cache, value_cache, block_table,
                                  context_lens, float(scale), int(partition_size))


def cascade_attention_multi(q, prefix_ks, prefix_vs, key_cache, value_cache, block_table,
                            context_lens, scale: float = 0.0, partition_size: int = 512):
    """N-level cascade: a list of shared-prefix levels + the paged suffix, merged in one reduce ==
    full attention over [level0 ++ ... ++ suffix]. MPS tensors."""
    return _ext.cascade_attention_multi(q, list(prefix_ks), list(prefix_vs), key_cache, value_cache,
                                        block_table, context_lens, float(scale), int(partition_size))


def cascade_attention_fp8(q, prefix_k, prefix_v, key_cache, value_cache, block_table, context_lens,
                          k_scale, v_scale, scale: float = 0.0, partition_size: int = 512,
                          fmt: int = 0):
    """Cascade over a uint8 fp8 shared prefix (per-kv-head dequant on read) + regular paged suffix.
    k_scale/v_scale (num_kv_heads,) float, fmt 0=e4m3 / 1=e5m2. MPS tensors."""
    return _ext.cascade_attention_fp8(q, prefix_k, prefix_v, key_cache, value_cache, block_table,
                                      context_lens, k_scale, v_scale, float(scale),
                                      int(partition_size), int(fmt))


def paged_attention_v2_fp8(q, key_cache, value_cache, block_table, context_lens,
                           k_scale, v_scale, scale=0.0, partition_size=512, fmt="e4m3", window=0,
                           softcap=0.0, sinks=None):
    """Long-context paged decode over an fp8 (uint8) cache, dequantized on read. GQA aware. MPS.

    k_scale/v_scale: plain float (per-tensor) or a (num_kv_heads,) tensor (per-head).
    fmt: 'e4m3' (default) or 'e5m2' — must match how the cache was written.
    window > 0 restricts to the `window` most recent keys.
    """
    H_KV = key_cache.shape[2]
    return _ext.paged_attention_v2_fp8(q, key_cache, value_cache, block_table, context_lens,
                                       _scale_vec_t(k_scale, H_KV, q), _scale_vec_t(v_scale, H_KV, q),
                                       float(scale), int(partition_size), _fmt_code(fmt), int(window),
                                       softcap=float(softcap), sinks=sinks)


def moe_route_topk(logits: torch.Tensor, k: int):
    """MoE routing: top-k experts + renormalized softmax weights. Returns (ids int32, weights f32).
    logits (num_tokens, num_experts) float; k <= min(16, num_experts). MPS."""
    return _ext.moe_route_topk(logits, int(k))



def tau_tail(qkv, tok_qv_lin, tau_pos_table, positions, n_heads, head_dim):
    """tau_tail: scale Q and V slices of a packed (T, 3*q_dim) QKV. Returns new qkv. MPS."""
    return _ext.tau_tail(qkv, tok_qv_lin, tau_pos_table, positions, int(n_heads),
                         int(head_dim))


def packbits(x, bit_order_big=True):
    """Pack bool/uint8 into bits (np.packbits). Returns uint8 (ceil(N/8),). MPS."""
    return _ext.packbits(x, bool(bit_order_big))


def segment_packbits(x, input_indptr, output_indptr, total_output_bytes, bit_order_big=True):
    """Ragged per-row packbits (output_indptr = host cumsum of ceil(len/8)). MPS."""
    return _ext.segment_packbits(x, input_indptr, output_indptr, int(total_output_bytes),
                                 bool(bit_order_big))


def permute_cols(x, perm):
    """16-bit column gather x[:, perm] (dtype-agnostic on 2-byte elements). MPS."""
    return _ext.permute_cols(x, perm)


def tq_encode(key, value, key_cache, value_cache, key_scale, value_scale, key_zero,
              slot_mapping, v_centroids, signs, block_size, k_bits, k_signed, v_bits):
    """TurboQuant KV encode. Returns [key_cache, value_cache, key_scale, value_scale,
    key_zero] (functional). MPS."""
    return list(_ext.tq_encode(key, value, key_cache, value_cache, key_scale, value_scale,
                               key_zero, slot_mapping, v_centroids, signs, int(block_size),
                               int(k_bits), bool(k_signed), int(v_bits)))


def tq_decode(key_cache, value_cache, key_scale, value_scale, key_zero, slots, v_centroids,
              signs, num_kv_heads, head_size, block_size, k_bits, k_signed, v_bits):
    """TurboQuant KV decode (gather + dequantize). Returns [k_out, v_out] float32. MPS."""
    return list(_ext.tq_decode(key_cache, value_cache, key_scale, value_scale, key_zero,
                               slots, v_centroids, signs, int(num_kv_heads), int(head_size),
                               int(block_size), int(k_bits), bool(k_signed), int(v_bits)))


def indexer_k_quant_and_cache(k, slot_mapping, code_cache, scale_cache, quant_block_size=128,
                             ue8m0=False):
    """DeepSeek-V3.2 indexer K quant. Returns [code_cache u8, scale_cache f32] (functional). MPS."""
    return list(_ext.indexer_k_quant_and_cache(k, slot_mapping, code_cache, scale_cache,
                                               int(quant_block_size), bool(ue8m0)))


def indexer_k_gather(code_cache, scale_cache, slots, head_dim, quant_block_size=128):
    """Gather + dequantize the indexer cache to bf16 K for a slot list. MPS."""
    return _ext.indexer_k_gather(code_cache, scale_cache, slots, int(head_dim),
                                 int(quant_block_size))


def minference_block_mask(vertical_indexes, slash_indexes, context_lens, max_blocks,
                          block_size, vertical_topk=1 << 30, slash_topk=1 << 30,
                          last_n_blocks=1):
    """MInference decode block-mask builder -> (B, H, max_blocks) int32. MPS."""
    return _ext.minference_block_mask(vertical_indexes, slash_indexes, context_lens,
                                      int(max_blocks), int(block_size), int(vertical_topk),
                                      int(slash_topk), int(last_n_blocks))


def quadratic_transform(logits, factor, curve=1.0, temperature=1.0):
    """Quadratic/smoothing logit transform (factor 0 = identity). MPS."""
    return _ext.quadratic_transform(logits, float(factor), float(curve), float(temperature))


def top_nsigma_mask(logits, nsigma, temperature=1.0):
    """Top-nsigma: mask logits below max - nsigma*std. MPS."""
    return _ext.top_nsigma_mask(logits, float(nsigma), float(temperature))


def top_a_mask(logits, top_a, temperature=1.0):
    """Top-A: mask probs below top_a * pmax^2. MPS."""
    return _ext.top_a_mask(logits, float(top_a), float(temperature))


def epsilon_cutoff_mask(logits, epsilon, temperature=1.0):
    """Epsilon cutoff: mask probs below epsilon (argmax survives). MPS."""
    return _ext.epsilon_cutoff_mask(logits, float(epsilon), float(temperature))


def eta_cutoff_mask(logits, eta, temperature=1.0):
    """Eta sampling: mask probs below min(eta, sqrt(eta)*exp(-entropy)). MPS."""
    return _ext.eta_cutoff_mask(logits, float(eta), float(temperature))


def xtc_mask(logits, threshold, probability, seed=0, temperature=1.0):
    """XTC: with on-device coin, remove probs >= threshold except the least likely. MPS."""
    return _ext.xtc_mask(logits, float(threshold), float(probability), int(seed),
                         float(temperature))


def skew_transform(probs, skew):
    """Skew: pow(index-order CDF, exp(skew)) over probability rows. MPS."""
    return _ext.skew_transform(probs, float(skew))


def top_k_renorm(probs, k):
    """Keep top-k probs, renormalize, zero elsewhere (k <= 64). MPS."""
    return _ext.top_k_renorm(probs, int(k))


def top_p_renorm(probs, p):
    """Keep the top-p mass (bisection, no sort), renormalize. MPS."""
    return _ext.top_p_renorm(probs, float(p))


def no_repeat_ngram_mask(logits, prev_tokens, lens, ngram_size, temperature=1.0):
    """Ban tokens completing an already-seen ngram (n >= 2). MPS."""
    return _ext.no_repeat_ngram_mask(logits, prev_tokens, lens, int(ngram_size),
                                     float(temperature))


def dry_penalty(logits, prev_tokens, lens, breakers, multiplier, base=1.75,
                allowed_length=2, range=0, max_ngram=64, max_occurrences=64,
                early_exit_match_len=64, temperature=1.0):
    """DRY repetition penalty (breakers (NB,) int32, pad -1). MPS."""
    return _ext.dry_penalty(logits, prev_tokens, lens, breakers, float(multiplier),
                            float(base), int(allowed_length), int(range), int(max_ngram),
                            int(max_occurrences), int(early_exit_match_len),
                            float(temperature))


def rms_norm_add_int8(x, residual, weight, eps=1e-5):
    """Fused add + rms_norm + dynamic per-row int8. Returns (codes i8, x+residual, scale). MPS."""
    return tuple(_ext.rms_norm_add_int8_dyn(x, residual, weight, float(eps)))


def layernorm_add_int8(x, residual, weight, bias, eps=1e-5):
    """Fused add + layernorm + dynamic per-row int8. Returns (codes i8, x+residual, scale). MPS."""
    return tuple(_ext.layernorm_add_int8_dyn(x, residual, weight, bias, float(eps)))


def rms_norm_add_per_block(x, residual, weight, eps=1e-5, int8=False, ue8m0=False):
    """Fused add + rms_norm + per-128-block quant. Returns (codes, x+residual, scale (rows,D/128)). MPS."""
    return tuple(_ext.rms_norm_add_per_block(x, residual, weight, float(eps), bool(int8), bool(ue8m0)))


def layernorm_add_per_block(x, residual, weight, bias, eps=1e-5, int8=False, ue8m0=False):
    """Fused add + layernorm + per-128-block quant. Returns (codes, x+residual, scale). MPS."""
    return tuple(_ext.layernorm_add_per_block(x, residual, weight, bias, float(eps), bool(int8),
                                              bool(ue8m0)))


_ACTQ_MODES = {"swiglu": 0, "swiglu_oai": 1}


def silu_mul_quant_fp8(x, gate, act="swiglu", alpha=1.702, limit=7.0):
    """Fused gated-activation -> dynamic per-token fp8. Returns (codes u8, scale). MPS."""
    return tuple(_ext.silu_mul_quant_fp8(x, gate, _ACTQ_MODES[act], float(alpha), float(limit)))


def silu_mul_quant_int8(x, gate, act="swiglu", alpha=1.702, limit=7.0):
    """Fused gated-activation -> dynamic per-token int8 (feeds qgemm_w8a8). MPS."""
    return tuple(_ext.silu_mul_quant_int8(x, gate, _ACTQ_MODES[act], float(alpha), float(limit)))


def silu_mul_quant_fp8_group(x, gate, group_size=128, ue8m0=False, act="swiglu",
                             alpha=1.702, limit=7.0):
    """Fused gated-activation -> per-group fp8 (scale (rows, D/G); ue8m0 = 2^k scales). MPS."""
    return tuple(_ext.silu_mul_quant_fp8_group(x, gate, group_size, ue8m0,
                                               _ACTQ_MODES[act], float(alpha), float(limit)))


def fake_quant_int8(x: torch.Tensor):
    """One-pass per-token int8 fake quant. Returns (x_q bf16, codes int8, scale f32). MPS."""
    return tuple(_ext.fake_quant_int8(x))


def silu_mul_fake_quant_int8(x, gate, act="swiglu", alpha=1.702, limit=7.0):
    """Fused gated activation plus per-token int8 fake quant. Returns (x_q, codes, scale). MPS."""
    return tuple(_ext.silu_mul_fake_quant_int8(x, gate, _ACTQ_MODES[act],
                                               float(alpha), float(limit)))


def weight_quant_ternary(w: torch.Tensor, group_k: int = 32):
    """Latent weight (N,K) or (E,N,K) -> (packed bitnet blocks, dequantized bf16). MPS."""
    return tuple(_ext.weight_quant_ternary(w, int(group_k)))


def weight_quant_ternary_pt(w: torch.Tensor):
    """Per-tensor BitNet ternary quantization -> (packed bitnet blocks, dequantized bf16). MPS."""
    return tuple(_ext.weight_quant_ternary_pt(w))


def gemm_v3(a: torch.Tensor, b: torch.Tensor):
    """Staged GEMM variant. A (N,K), B (K,M), with N/M multiples of 64 and K multiple of 32. MPS."""
    return _ext.gemm_v3(a, b)


def qgemm_bwd(grad_y: torch.Tensor, wq: torch.Tensor):
    """BitNet ternary backward GEMM: grad_y (M,N) @ dequant(wq) -> (M,K). MPS."""
    return _ext.qgemm_bwd(grad_y, wq)


def quantize_tq2_0(w: torch.Tensor):
    """Pack W (N,K) or (E,N,K) into llama.cpp TQ2_0 blocks. Returns (wq, w_deq). MPS."""
    return tuple(_ext.quantize_tq2_0(w))


def qdequant(wq: torch.Tensor, format: str = "q8_0"):
    """Packed quant blocks (N, K/block_k, block_bytes) -> dense fp16 (N,K). MPS."""
    return _ext.qdequant(wq, format)


def dequantize_tq2_0(wq: torch.Tensor):
    """TQ2_0 packed blocks (N, K/256, 66) -> dense fp16 (N,K). MPS."""
    return _ext.qdequant(wq, "tq2_0")


def ternary_stats(wq: torch.Tensor):
    """Packed BitNet blocks -> int32 (rows, 3) counts of {-1, 0, +1} codes per row. MPS."""
    return _ext.ternary_stats(wq)


def code_flip_count(wq_a: torch.Tensor, wq_b: torch.Tensor):
    """Per-row count of ternary code differences between two identically-shaped packs. MPS."""
    return _ext.code_flip_count(wq_a, wq_b)


def fake_quant_fp8(x: torch.Tensor):
    """Per-tensor e4m3 fake quant. Returns (x_fq, scale). MPS."""
    return tuple(_ext.fake_quant_fp8(x))


def qgemm_w2a8_fused(wq, x):
    """Fused per-token int8 activation quantization plus BitNet W2A8 GEMM. MPS."""
    return _ext.qgemm_w2a8_fused(wq, x)


def quantize_per_group_fp8(x, group_size=128, ue8m0=False):
    """Per-group dynamic fp8 e4m3. Returns (codes u8, scale (rows, D/G) f32). MPS."""
    return tuple(_ext.quantize_per_group_fp8(x, group_size, ue8m0))


def quantize_per_group_int8(x, group_size=128):
    """Per-group dynamic symmetric int8. Returns (codes i8, scale). MPS."""
    return tuple(_ext.quantize_per_group_int8(x, group_size))


def quantize_per_token_int8_azp(x):
    """Asymmetric per-token int8 (vLLM azp). Returns (codes, scale, azp i32). MPS."""
    return tuple(_ext.quantize_per_token_int8_azp(x))


def qgemm_w8a8_azp(wq, xq, w_scale, a_scale, w_rowsum, azp):
    """azp-corrected W8A8 GEMM: y = s_w*s_a*(W@Xq^T - azp*rowsum(W)). MPS."""
    return _ext.qgemm_w8a8_azp(wq, xq, w_scale, a_scale, w_rowsum, azp)


def gdn_recur(q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping, load_initial=True):
    """GatedDeltaNet delta-rule linear attention (varlen packed, fp32 state pool). MPS."""
    return tuple(_ext.gdn_recur(q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping,
                                load_initial))


def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, z=None, state=None,
                   delta_softplus=True):
    """Mamba-1 (S6) selective scan, dense batch (channel-major). Returns (out, new_state). MPS."""
    return tuple(_ext.selective_scan(u, delta, A, B, C, D=D, delta_bias=delta_bias, z=z,
                                     state=state, delta_softplus=delta_softplus))


def selective_scan_varlen_apc(u, delta, A, B, C, query_start_loc, cache_indices,
                              has_initial_state, state, block_idx_first_scheduled_token,
                              block_idx_last_scheduled_token, initial_state_idx,
                              cu_chunk_seqlen, last_chunk_indices, block_size,
                              cache_indices_stride, use_chunk_metadata, D=None,
                              delta_bias=None, z=None, delta_softplus=True, null_block_id=-1):
    """Varlen S6 scan with automatic prefix caching (paged state checkpointing). MPS."""
    return tuple(_ext.selective_scan_varlen_apc(
        u, delta, A, B, C, query_start_loc, cache_indices, has_initial_state, state,
        block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx,
        cu_chunk_seqlen, last_chunk_indices, int(block_size), int(cache_indices_stride),
        bool(use_chunk_metadata), D=D, delta_bias=delta_bias, z=z,
        delta_softplus=delta_softplus, null_block_id=null_block_id))


def selective_scan_varlen(u, delta, A, B, C, query_start_loc, state, D=None, delta_bias=None,
                          z=None, cache_indices=None, has_initial_state=None,
                          delta_softplus=True, null_block_id=-1):
    """Varlen S6 scan over flattened tokens with a per-request paged state pool. MPS."""
    return tuple(_ext.selective_scan_varlen(u, delta, A, B, C, query_start_loc, state,
                                            D=D, delta_bias=delta_bias, z=z,
                                            cache_indices=cache_indices,
                                            has_initial_state=has_initial_state,
                                            delta_softplus=delta_softplus,
                                            null_block_id=null_block_id))


def qk_norm_rope(qkv, q_weight, k_weight, cos, sin, positions, num_heads_q, num_heads_k,
                 num_heads_v, eps=1e-6, interleaved=False, gemma=False):
    """Fused per-head QK-RMSNorm + RoPE over packed QKV (T, (Hq+Hk+Hv)*D) bf16; V heads
    copied through. interleaved: False = NeoX split-half, True = GPT-J pairs. MPS."""
    return _ext.qk_norm_rope(qkv, q_weight, k_weight, cos, sin, positions,
                             int(num_heads_q), int(num_heads_k), int(num_heads_v),
                             float(eps), bool(interleaved), bool(gemma))


def qk_norm_rope_kv_f16(qkv, q_weight, k_weight, cos, sin, positions, num_heads_q, num_heads_k,
                        num_heads_v, eps=1e-6, interleaved=False, gemma=False):
    """qk_norm_rope with a fused f16 KV split-store: the normed+roped result is split into
    Q (T, Hq*D) bf16 and contiguous K/V (T, Hk*D)/(T, Hv*D) f16 KV-cache tensors in one pass.
    Returns (q_out, k_out, v_out). MPS."""
    return tuple(_ext.qk_norm_rope_kv_f16(
        qkv, q_weight, k_weight, cos, sin, positions,
        int(num_heads_q), int(num_heads_k), int(num_heads_v),
        float(eps), bool(interleaved), bool(gemma)))


def moe_route_grouped(logits, k, n_group, topk_group, bias=None, renormalize=True,
                      routed_scaling_factor=1.0, scoring="sigmoid"):
    """DeepSeek-style grouped routing (noaux_tc): scoring + bias-corrected selection, group
    top-2 ranking, weights from unbiased scores. Returns (ids int32, weights f32). MPS."""
    sf = {"softmax": 0, "sigmoid": 1, "softplus_sqrt": 2}[scoring]
    return _ext.moe_route_grouped(logits, bias, int(k), int(n_group), int(topk_group),
                                  bool(renormalize), float(routed_scaling_factor), sf)


def moe_permute(topk_ids: torch.Tensor, num_experts: int):
    """Group T*k routing rows by expert. Returns (sorted_row_idx, offsets, inv_idx) int32. MPS."""
    return _ext.moe_permute(topk_ids, int(num_experts))


def moe_pad_schedule(sorted_row_idx: torch.Tensor, offsets: torch.Tensor, k: int):
    """32-row-padded per-expert schedule for the grouped GEMMs.

    Returns (expert_of_tile, gather_idx, inv_pad, off_pad) int32; -1 sentinels mark
    pad tiles/rows beyond the real (data-dependent) total. MPS."""
    return _ext.moe_pad_schedule(sorted_row_idx, offsets, int(k))


def moe_gather(x: torch.Tensor, gather_idx: torch.Tensor):
    """out[p, :] = x[gather_idx[p], :] (zeros where gather_idx[p] < 0). MPS."""
    return _ext.moe_gather(x, gather_idx)


def moe_grouped_gemm(permuted_input, W, expert_of_tile):
    """Fused grouped expert GEMM: out = permuted_input @ W[expert]. Returns (total_rows, H). MPS."""
    return _ext.moe_grouped_gemm(permuted_input, W, expert_of_tile)


def moe_grouped_gemm_rect(A, W, expert_of_tile):
    """Rectangular grouped GEMM: out(rows,N_out) = A(rows,K_dim) @ W[e](K_dim,N_out). MPS."""
    return _ext.moe_grouped_gemm_rect(A, W, expert_of_tile)


def moe_grouped_gemm_swiglu(A, W1, expert_of_tile):
    """Fused SiLU-GLU GEMM1: out(rows,inter) = silu(A@W1_gate)*(A@W1_up); W1[e] (H,2*inter). MPS."""
    return _ext.moe_grouped_gemm_swiglu(A, W1, expert_of_tile)


_MOE_Q_ACT_MODES = {"swiglu": 0, "swiglu_oai": 1}


def moe_grouped_gemm_rect_q(A, Wq, expert_of_tile, format="mxfp4", bias=None):
    """Quantized grouped expert GEMM: out(rows,N_out) = A @ dequant(Wq[e])^T [+ bias[e]].
    A bf16 (rows,K); Wq (E,N_out,row_bytes) uint8 (tk.quant.quantize_expert_stack). MPS."""
    has_bias = bias is not None
    if bias is None:
        bias = torch.zeros(1, dtype=torch.bfloat16, device=A.device)
    return _ext.moe_grouped_gemm_rect_q(A, Wq, expert_of_tile, bias, has_bias, format)


def moe_grouped_gemm_swiglu_q(A, W1q, expert_of_tile, format="mxfp4", bias=None,
                              act="swiglu", alpha=1.702, limit=7.0):
    """Quantized fused SwiGLU GEMM1 from a packed [gate|up] stack (E,2*inter,row_bytes) uint8.
    act "swiglu" or "swiglu_oai" (gpt-oss: clamped, alpha-sigmoid, (1+up)). MPS."""
    has_bias = bias is not None
    if bias is None:
        bias = torch.zeros(1, dtype=torch.bfloat16, device=A.device)
    return _ext.moe_grouped_gemm_swiglu_q(A, W1q, expert_of_tile, bias, has_bias,
                                          _MOE_Q_ACT_MODES[act], float(alpha), float(limit),
                                          format)


def moe_finalize(expert_out: torch.Tensor, inv_idx: torch.Tensor, topk_weights: torch.Tensor, k: int):
    """out[t] = sum_k weight[t,k] * expert_out[inv_idx[t*k+k]]. Returns (T, Hdim). MPS."""
    return _ext.moe_finalize(expert_out, inv_idx, topk_weights, int(k))


def moe_grouped_gemm_bwd_dx(dy, W, expert_of_tile):
    """MoE grouped GEMM backward dX over the padded expert schedule. MPS."""
    return _ext.moe_grouped_gemm_bwd_dx(dy, W, expert_of_tile)


def moe_grouped_gemm_bwd_dw(A, dy, off_pad, num_experts):
    """MoE grouped GEMM backward dW per padded expert segment. MPS."""
    return _ext.moe_grouped_gemm_bwd_dw(A, dy, off_pad, int(num_experts))


def moe_finalize_bwd(grad_out, expert_out, inv_pad, topk_weights):
    """Backward of moe_finalize. Returns (grad_expert_out, grad_weights). MPS."""
    return tuple(_ext.moe_finalize_bwd(grad_out, expert_out, inv_pad, topk_weights))


def moe_gather_bwd(dA, inv_pad, k):
    """Backward of moe_gather: sum routed copies back to token order. MPS."""
    return _ext.moe_gather_bwd(dA, inv_pad, int(k))


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


def min_p_sample(logits: torch.Tensor, min_p: float, temperature: float = 1.0, seed: int = 0):
    """min-p sampling: Gumbel-max over tokens with prob >= min_p * max_prob. int32. MPS."""
    return _ext.min_p_sample(logits, float(min_p), float(temperature), int(seed))


def typical_p_sample(logits, typical_p, temperature: float = 1.0, seed: int = 0):
    """typical-p sampling: Gumbel-max over the smallest-surprise mass typical_p. int32. MPS."""
    return _ext.typical_p_sample(logits, float(typical_p), float(temperature), int(seed))


def embedding_lookup(token_ids, table, pos_table=None, scale: float = 1.0):
    """Token embedding lookup: out[t] = scale*table[token_ids[t]] (+ pos_table[t] if given). MPS."""
    pt = pos_table if pos_table is not None else torch.zeros(1, dtype=table.dtype, device=table.device)
    return _ext.embedding_lookup(token_ids, table, pt, float(scale))


def embedding_backward(token_ids, dY, vocab, scale: float = 1.0, method: str = "atomic"):
    """Embedding backward: scatter-add dY (num_tok, D) rows into a (vocab, D) fp32 grad table by
    token id (out[token_ids[t]] += scale*dY[t]); padding/oob ids contribute nothing. MPS.
    method="atomic" (default) = per-element atomic-add; method="sorted" = presort by id so each id is
    summed by one threadgroup (atomic-free, wins under heavy duplication). Same result."""
    if method == "sorted":
        import torch
        tok = token_ids.to(torch.int32)
        perm = torch.argsort(tok)
        return _ext.embedding_backward_sorted(tok[perm], perm, dY, int(vocab), float(scale))
    if method != "atomic":
        raise ValueError(f"embedding_backward: method must be 'atomic' or 'sorted', got {method!r}")
    return _ext.embedding_backward(token_ids, dY, int(vocab), float(scale))


def build_multimodal_src(span_offsets, span_lengths, modal_starts, num_tok):
    """Build the multimodal src map on-device: src[t]=modal_starts[k]+off in span k, else -1. MPS."""
    return _ext.build_multimodal_src(span_offsets, span_lengths, modal_starts, int(num_tok))


def merge_multimodal_spans(text, modal, src):
    """Multimodal span merge: out[t] = modal[src[t]] if src[t] >= 0 else text[t]. MPS."""
    return _ext.merge_multimodal_spans(text, modal, src)


def apply_token_bitmask(logits: torch.Tensor, bitmask: torch.Tensor):
    """Grammar / structured-output masking: logits[v] = -inf where the packed allow-bitmask bit for
    token v is 0. bitmask (T, ceil(V/32)) int32. Returns masked logits, same dtype. MPS."""
    return _ext.apply_token_bitmask(logits, bitmask)


def apply_bad_words(logits, bad_ids, bad_lens):
    """Bad / stop-word masking: logits[t, bad_ids[t,j]] = -inf for j < bad_lens[t]. bad_ids
    (T, maxbad) int, bad_lens (T,) int. Returns masked logits, same dtype. MPS."""
    return _ext.apply_bad_words(logits, bad_ids, bad_lens)


def beam_advance(logits, cum_log_probs, beam_width):
    """Beam-search advance: fused log-softmax + cumulative score + top-beam_width with parent
    tracking. logits (B*BM, V), cum_log_probs (B, BM). Returns (next_token, parent_beam,
    cum_log_probs') each (B, BM). beam_width <= 16. MPS."""
    return _ext.beam_advance(logits, cum_log_probs, int(beam_width))


def apply_penalty(logits: torch.Tensor, prev_tokens: torch.Tensor, temperature: float = 1.0,
                  repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                  frequency_penalty: float = 0.0, bias=None, eos_id: int = -1,
                  min_length: int = 0, gen_len: int = 0, parent_ids=None):
    """Temperature + rep/presence/freq penalties + logit bias + min-length EOS mask (forbids eos_id
    while gen_len < min_length). bias (V,) or None; parent_ids (T,) redirects each row's occurrence
    history (beam search; None = identity). Returns penalized logits (T,V). MPS."""
    if bias is None:
        bias = torch.zeros(logits.shape[-1], dtype=torch.float32, device=logits.device)
    if parent_ids is None:
        parent_ids = torch.arange(logits.shape[0], dtype=torch.int32, device=logits.device)
    return _ext.apply_penalty(logits, prev_tokens, bias, parent_ids, float(temperature),
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


def flux_gelu_erf(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor):
    """Fused erf-GELU(x @ w + bias)."""
    return _ext.flux_gelu_erf(x, w, bias)


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
    """Mamba-2 / SSD forward. C,B,X bf16 (B,H,N,D); cumlog fp32 (B,H,N). MPS; D in {64,128}, N%8.
    Auto-routed between the quadratic kernel and the chunked linear-time pipeline at the
    MEASURED crossovers: N>=2048 for D=64, N>=4096 for D=128 (chunked needs N%64==0)."""
    return _ext.mamba2(C, B, X, cumlog)


def mamba2_chunked(C, B, X, cumlog):
    """The chunked linear-time forward, forced (tests/benchmarks). N%64==0, N>=128."""
    return _ext.mamba2_chunked(C, B, X, cumlog)


def mamba2_bwd(C, B, X, cumlog, dY, force_quadratic=False):
    """Mamba-2 / SSD backward. Returns (dC, dB, dX, dcumlog) matching the forward shapes. MPS.
    Auto-routed like the forward (chunked linear-time above the measured crossovers);
    force_quadratic pins the quadratic path (matches MLX)."""
    return _ext.mamba2_bwd(C, B, X, cumlog, dY, force_quadratic)


def mamba2_bwd_chunked(C, B, X, cumlog, dY):
    """The chunked linear-time backward, forced (tests/benchmarks). N%64==0, N>=128."""
    return _ext.mamba2_bwd_chunked(C, B, X, cumlog, dY)


def ssd_decode(S, alpha, x, k, q):
    """Single-token SSD decode step: S <- alpha*S + x⊗k ; y = S·q (readout after the write).

    S (B,H,D,D) fp32 is updated IN PLACE; alpha (B,H), x/k/q (B,H,D) fp32. MPS; D in {64,128}.
    Returns (y (B,H,D), S). The O(D^2) generation step for mamba2 / lin_attn_decay
    (q=C_t, k=B_t, x=X_t; alpha=1 for undecayed linear attention)."""
    return _ext.ssd_decode(S, alpha, x, k, q)


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
    """Quantized GEMV. x is float16, or fp32 for q4_0/q6_K."""
    return _ext.qgemv(wq, x, format)


def qgemv_q4_0_f32_up_gate_gelu(up, gate, x):
    """Fused Q4_0 up+gate decode GEMV with a gated-GELU epilogue in one launch:
    out = gelu_tanh(gate @ x) * (up @ x). up/gate uint8 Q4_0 (N, K/32, 18), x fp32 (K, 1).
    Returns (N, 1) fp32. MPS."""
    return _ext.qgemv_q4_0_f32_up_gate_gelu(up, gate, x)


def qgemv_q4_0_f32_up_gate(up, gate, x):
    """Fused Q4_0 up+gate decode GEMV in one launch. up/gate uint8 Q4_0 (N, K/32, 18),
    x fp32 (K, 1). Returns (up_out, gate_out), each (N, 1) fp32. MPS."""
    return tuple(_ext.qgemv_q4_0_f32_up_gate(up, gate, x))


def qgemv_q4_0_f32_qkv(qw, kw, vw, x):
    """Fused Q4_0 Q/K/V decode GEMV in one launch (GQA-friendly, Nkv may differ from Nq).
    qw/kw/vw uint8 Q4_0 (N, K/32, 18), x fp32 (K, 1). Returns (q, k, v). MPS."""
    return tuple(_ext.qgemv_q4_0_f32_qkv(qw, kw, vw, x))


def dequant_gather(table, ids, format, scale=1.0):
    """Gather packed q4_0/q8_0/q6_K rows directly to fp16."""
    return _ext.dequant_gather(table, ids, str(format), float(scale))


def quantized_embedding(table, ids, add, format, scale=1.0, use_add=False,
                        output_dtype="float16"):
    """Packed embedding lookup with optional additive epilogue."""
    return _ext.quantized_embedding(
        table, ids, add, str(format), float(scale), bool(use_add), str(output_dtype))


def quantized_embedding_bag(table, ids, offsets, sample_weights, format,
                            scale=1.0, use_weights=False, mean_mode=False,
                            output_dtype="float16"):
    """CSR sum/mean embedding bag over packed rows."""
    return _ext.quantized_embedding_bag(
        table, ids, offsets, sample_weights, str(format), float(scale),
        bool(use_weights), bool(mean_mode), str(output_dtype))


def qflux_gelu(wq: torch.Tensor, x: torch.Tensor, bias: torch.Tensor, format: str = "q8_0"):
    """Quantized fused GEMM+GELU: gelu(dequantize(wq) @ x + bias). x (K,M) f16; bias (M,) f16. MPS."""
    return _ext.qflux_gelu(wq, x, bias, format)


def attn_q(q, kq, vq, format="q8_0", causal=False, multiwarp=False):
    """Quantized-KV flash attention over q8_0/q4_0/fp8_e4m3/mxfp8 K/V blocks. MPS."""
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


def qgemv_w2a8(wq, xq, a_scale, version=2):
    """BitNet W2A8 decode GEMV. version=2 selects the newer decode kernel. MPS."""
    return _ext.qgemv_w2a8(wq, xq, a_scale, int(version))


def qgemv_w2a8_v2(wq, xq, a_scale):
    """BitNet W2A8 decode GEMV v2. MPS."""
    return _ext.qgemv_w2a8_v2(wq, xq, a_scale)
