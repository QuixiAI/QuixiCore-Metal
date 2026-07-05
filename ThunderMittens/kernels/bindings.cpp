//// Copyright © 2023-2024 Apple Inc.
//
//#include <nanobind/nanobind.h>
//#include <nanobind/stl/variant.h>
//
//#include "add_rt/add_rt.h"
//#include "attn_fwd/attn_fwd.h"
//#include "matmul_custom/matmul_custom.h"
//
//namespace nb = nanobind;
//using namespace nb::literals;
//
//using namespace mlx::core;
//
//NB_MODULE(_ext, m) {
//  m.doc() = "TK extension for MLX";
//      m.def(
//      "add_rt",
//      &add_rt,
//      "x"_a,
//      "y"_a,
//      nb::kw_only(),
//      "stream"_a = nb::none(),
//      R"(
//        adds
//      )");
//
//    m.def(
//      "attn_fwd",
//      &attn_fwd,
//      "q"_a,
//      "k"_a,
//      "v"_a,
//      nb::kw_only(),
//      "stream"_a = nb::none(),
//      R"(
//        attn fwd
//      )");
//
//    m.def(
//      "matmul_custom",
//      &matmul_custom,
//      "x"_a,
//      "y"_a,
//      nb::kw_only(),
//      "stream"_a = nb::none(),
//      R"(
//        gemm
//      )");
//}

// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "add_rt/add_rt.h"
#include "attn_fwd/attn_fwd.h"
#include "matmul_custom/matmul_custom.h"
#include "layernorm/layernorm.h"
#include "rms_norm/rms_norm.h"
#include "add_norm/add_norm.h"
#include "rope_kv/rope_kv.h"
#include "qk_norm_rope/qk_norm_rope.h"
#include "mla/mla.h"
#include "paged_attn_v2/paged_attn_v2.h"
#include "quant_rt/quant_rt.h"
#include "sampling/sampling.h"
#include "moe/moe.h"
#include "softmax/softmax.h"
#include "rotary/rotary.h"
#include "gelu/gelu.h"
#include "dropout/dropout.h"
#include "optim/adamw.h"
#include "embedding/embedding.h"
#include "glu/glu.h"
#include "hadamard/hadamard.h"
#include "kv_cache/kv_cache.h"
#include "attn_causal/attn_causal.h"
#include "attn_varlen/attn_varlen.h"
#include "lm_head/lm_head.h"
#include "cross_entropy/cross_entropy.h"
#include "flux/flux.h"
#include "gemm_staged/gemm_staged.h"
#include "attn_multiwarp/attn_multiwarp.h"
#include "linear_attn/linear_attn.h"
#include "hedgehog/hedgehog.h"
#include "lin_attn_causal/lin_attn_causal.h"
#include "mamba2/mamba2.h"
#include "lin_attn_decay/lin_attn_decay.h"
#include "based/based.h"
#include "attn_bwd/attn_bwd.h"
#include "cmplx_matmul/cmplx_matmul.h"
#include "fftconv/fftconv.h"
#include "qgemm/qgemm.h"
#include "qgemv/qgemv.h"
#include "qflux/qflux.h"
#include "qgemv_int/qgemv_int.h"
#include "attn_q/attn_q.h"
#include "qgemm_int/qgemm_int.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlx::core;

NB_MODULE(_ext, m) {
  m.doc() = "TK extension for MLX";
      m.def(
      "add_rt",
      &add_rt,
      "x"_a,
      "y"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        adds
      )");

    m.def(
      "attn_fwd",
      &attn_fwd,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "softcap"_a = 0.0f,
      "sinks"_a = nb::none(),
      "stream"_a = nb::none(),
      R"(
        Dense attention forward (B,H,N,D bf16). softcap > 0 applies Gemma-style logit
        soft-capping (softcap*tanh(s/softcap)); sinks is an optional per-head (H,) fp32
        attention-sink logit added to the softmax denominator (gpt-oss).
      )");

    m.def(
      "matmul_custom",
      &matmul_custom,
      "x"_a,
      "y"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        gemm
      )");

    m.def(
      "layernorm",
      &layernorm,
      "x"_a,
      "weight"_a,
      "bias"_a,
      nb::kw_only(),
      "eps"_a = 1e-5f,
      "stream"_a = nb::none(),
      R"(
        layernorm over the last axis: (x - mean) * rsqrt(var + eps) * weight + bias
      )");

    m.def(
      "layernorm_bwd_dx",
      &layernorm_bwd_dx,
      "x"_a,
      "weight"_a,
      "dy"_a,
      "mean"_a,
      "rstd"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        LayerNorm backward dX: rstd*(g - mean(g) - x_hat*mean(g*x_hat)), g=dY*W. mean/rstd (rows,).
      )");

    m.def(
      "layernorm_bwd_fused",
      &layernorm_bwd_fused,
      "x"_a,
      "weight"_a,
      "dy"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused LayerNorm backward: returns [dX (rows,D), dweight (D,) fp32, dbias (D,) fp32].
      )");

    m.def(
      "rms_norm",
      &rms_norm,
      "x"_a,
      "weight"_a,
      nb::kw_only(),
      "eps"_a = 1e-5f,
      "stream"_a = nb::none(),
      R"(
        rms_norm over the last axis: x * rsqrt(mean(x^2) + eps) * weight
      )");

    m.def(
      "rms_norm_bwd_dx",
      &rms_norm_bwd_dx,
      "x"_a,
      "weight"_a,
      "dy"_a,
      "rstd"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        RMSNorm backward dX: rstd*(dY*W) - (rstd^3 * rowsum(dY*W*x)/D) * x. rstd (rows,) precomputed.
      )");

    m.def(
      "rms_norm_bwd_fused",
      &rms_norm_bwd_fused,
      "x"_a,
      "weight"_a,
      "dy"_a,
      "eps"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Fused RMSNorm backward: returns [dX (rows,D), dweight (D,) fp32], rstd computed in-kernel.
      )");

    m.def(
      "rms_norm_add",
      &rms_norm_add,
      "x"_a,
      "residual"_a,
      "weight"_a,
      nb::kw_only(),
      "eps"_a = 1e-5f,
      "stream"_a = nb::none(),
      R"(
        fused residual-add + rms_norm. Returns (out, x+residual):
        out = rms_norm(x + residual) * weight
      )");

    m.def(
      "layernorm_add",
      &layernorm_add,
      "x"_a,
      "residual"_a,
      "weight"_a,
      "bias"_a,
      nb::kw_only(),
      "eps"_a = 1e-5f,
      "stream"_a = nb::none(),
      R"(
        fused residual-add + layernorm. Returns (out, x+residual):
        out = layernorm(x + residual) * weight + bias
      )");

    m.def(
      "rms_norm_add_fp8", &rms_norm_add_fp8,
      "x"_a, "residual"_a, "weight"_a, "eps"_a, "scale"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + rms_norm + static-scale fp8. Returns (codes uint8, x+residual).)");
    m.def(
      "rms_norm_add_fp8_dyn", &rms_norm_add_fp8_dyn,
      "x"_a, "residual"_a, "weight"_a, "eps"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + rms_norm + dynamic per-row fp8. Returns (codes, x+residual, scale).)");
    m.def(
      "layernorm_add_fp8", &layernorm_add_fp8,
      "x"_a, "residual"_a, "weight"_a, "bias"_a, "eps"_a, "scale"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + layernorm + static-scale fp8. Returns (codes uint8, x+residual).)");
    m.def(
      "layernorm_add_fp8_dyn", &layernorm_add_fp8_dyn,
      "x"_a, "residual"_a, "weight"_a, "bias"_a, "eps"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + layernorm + dynamic per-row fp8. Returns (codes, x+residual, scale).)");

    m.def(
      "rope_kv_insert",
      &rope_kv_insert,
      "k"_a,
      "v"_a,
      "cos"_a,
      "sin"_a,
      "positions"_a,
      "slot_mapping"_a,
      "key_cache"_a,
      "value_cache"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        fused RoPE (split-half) on K + paged-KV insert. Returns (key_cache, value_cache).
      )");

    m.def(
      "rope_kv_insert_norm", &rope_kv_insert_norm,
      "k"_a, "v"_a, "cos"_a, "sin"_a, "positions"_a, "slot_mapping"_a,
      "key_cache"_a, "value_cache"_a, "norm_weight"_a, "eps"_a, "gemma"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused K RMSNorm + RoPE + paged-KV insert. gemma=True uses (1+weight). Returns (kc, vc).)");

    m.def(
      "rope_q", &rope_q,
      "q"_a, "cos"_a, "sin"_a, "positions"_a, "norm_weight"_a, "do_norm"_a, "gemma"_a,
      "eps"_a = 1e-6f, nb::kw_only(), "stream"_a = nb::none(),
      R"(rotate (+optional weighted RMSNorm) Q into a contiguous q_out (split-half RoPE). q
         (num_tokens, num_q_heads, D); cos/sin (P, D/2). {f16,bf16,f32}.)");

    m.def(
      "mla_q_norm_rope", &mla_q_norm_rope,
      "q"_a, "cos"_a, "sin"_a, "positions"_a, "norm_weight"_a,
      "num_heads"_a, "nope_dim"_a, "rope_dim"_a, "norm_mode"_a, "eps"_a = 1e-6f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(DeepSeek MLA Q-path: optional RMSNorm (mode 0/1/2) + GPT-J interleaved RoPE on the last
         rope_dim dims. head_dim=nope_dim+rope_dim, %64==0; cos/sin (max_pos, rope_dim/2).)");

    m.def(
      "mla_kv_insert", &mla_kv_insert,
      "kv_c"_a, "k_pe"_a, "cos"_a, "sin"_a, "positions"_a, "slot_mapping"_a, "kv_cache"_a,
      "norm_weight"_a, "rope_dim"_a, "norm_mode"_a, "eps"_a = 1e-6f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(DeepSeek MLA classic KV-insert: (optionally kv_a-normed) latent + interleaved-RoPE k_pe into
         a paged bf16 cache (nb, bs, LATENT+rope_dim). Returns the updated kv_cache.)");

    m.def(
      "mla_decode", &mla_decode,
      "q"_a, "kv_cache"_a, "block_table"_a, "context_lens"_a, "scale"_a = 0.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(DeepSeek MLA absorb-path latent decode (MQA). q (B,N,576)=[ql_nope(512)|q_pe(64)] attends
         a shared latent cache (nb,bs,576); value over the 512 latent only. Returns o (B,N,512).)");

    m.def(
      "mla_decode_fp8", &mla_decode_fp8,
      "q"_a, "data_cache"_a, "scale_cache"_a, "block_table"_a, "context_lens"_a, "scale"_a = 0.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(DeepSeek-V4 dense latent decode over the UE8M0-packed cache: q (B,N,512) attends the packed
         (data (…,576) uint8, scale (…,8) uint8) cache, score/value over 512, scale 512^-0.5. -> o (B,N,512).)");

    m.def(
      "mla_decode_fp8_sparse", &mla_decode_fp8_sparse,
      "q"_a, "data_cache"_a, "scale_cache"_a, "block_table"_a, "indices"_a, "topk_length"_a,
      "scale"_a = 0.0f, nb::kw_only(), "stream"_a = nb::none(),
      R"(DeepSeek-V4 sparse latent decode: like mla_decode_fp8 but each query attends only
         indices[b, 0:topk_length[b]] (the indexer top-k set). -> o (B,N,512).)");

    m.def(
      "mla_kv_insert_fp8", &mla_kv_insert_fp8,
      "kv"_a, "cos"_a, "sin"_a, "positions"_a, "slot_mapping"_a, "data_cache"_a, "scale_cache"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(DeepSeek-V4 packed MLA KV-insert: 448 NoPE -> e4m3 fp8 with per-64-block UE8M0 scales,
         64 RoPE -> interleaved bf16. Returns (data_cache (…,576) uint8, scale_cache (…,8) uint8).)");

    m.def(
      "paged_attention_v2",
      &paged_attention_v2,
      "q"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      nb::kw_only(),
      "scale"_a = 0.0f,
      "partition_size"_a = 512,
      "window"_a = 0,
      "softcap"_a = 0.0f,
      "sinks"_a = nb::none(),
      "stream"_a = nb::none(),
      R"(
        long-context paged decode attention (partition/reduce). GQA/MQA aware.
        window > 0 restricts to the `window` most recent keys. softcap > 0 = Gemma-style
        logit capping; sinks (H,) = gpt-oss attention sinks (merged once, in the reduce).
      )");

  m.def(
      "cascade_attention",
      &cascade_attention,
      "q"_a,
      "prefix_k"_a,
      "prefix_v"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      nb::kw_only(),
      "scale"_a = 0.0f,
      "partition_size"_a = 512,
      "stream"_a = nb::none(),
      R"(
        cascade / shared-prefix attention: a shared contiguous prefix KV plus each request's paged
        suffix, merged via the shared log-sum-exp reduce == full softmax over [prefix ++ suffix].
      )");

  m.def(
      "cascade_attention_multi",
      &cascade_attention_multi,
      "q"_a,
      "prefix_ks"_a,
      "prefix_vs"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      nb::kw_only(),
      "scale"_a = 0.0f,
      "partition_size"_a = 512,
      "stream"_a = nb::none(),
      R"(
        N-level cascade attention: a list of shared-prefix levels + the paged suffix, all partials
        concatenated + one log-sum-exp reduce == full softmax over [level0 ++ ... ++ suffix].
      )");

  m.def(
      "cascade_attention_fp8",
      &cascade_attention_fp8,
      "q"_a,
      "prefix_k"_a,
      "prefix_v"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "k_scale"_a,
      "v_scale"_a,
      nb::kw_only(),
      "scale"_a = 0.0f,
      "partition_size"_a = 512,
      "fmt"_a = 0,
      "stream"_a = nb::none(),
      R"(
        Cascade over a uint8 fp8 (e4m3/e5m2) shared prefix (per-kv-head dequant on read) + the
        regular paged suffix, merged in one reduce. k_scale/v_scale are (num_kv_heads,) float.
      )");

    m.def(
      "paged_attention_v2_fp8",
      &paged_attention_v2_fp8,
      "q"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "k_scale"_a,
      "v_scale"_a,
      nb::kw_only(),
      "scale"_a = 0.0f,
      "partition_size"_a = 512,
      "fmt"_a = 0,
      "window"_a = 0,
      "softcap"_a = 0.0f,
      "sinks"_a = nb::none(),
      "stream"_a = nb::none(),
      R"(
        long-context paged decode over an fp8 (uint8) cache, per-head scales; fmt 0=e4m3, 1=e5m2. GQA aware.
        window > 0 restricts to the `window` most recent keys; softcap/sinks as paged_attention_v2.
      )");

    m.def(
      "moe_route_topk",
      &moe_route_topk,
      "logits"_a,
      "k"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        MoE routing: top-k experts + renormalized softmax weights. Returns (ids int32, weights f32).
      )");

    m.def(
      "moe_permute",
      &moe_permute,
      "topk_ids"_a,
      "num_experts"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        MoE permute: group T*k routing rows by expert. Returns
        [sorted_row_idx, offsets, inv_idx, counts(scratch), cursor(scratch)] (int32).
      )");

    m.def(
      "moe_pad_schedule",
      &moe_pad_schedule,
      "sorted_row_idx"_a,
      "offsets"_a,
      "k"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        MoE padded schedule: 32-row-padded per-expert segments for the grouped GEMMs.
        Returns [expert_of_tile (max_tiles, -1 = pad tile), gather_idx (total_pad_max,
        -1 = pad row), inv_pad (T*k), off_pad (E+1)] (int32). Worst-case static sizing.
      )");

    m.def(
      "moe_gather",
      &moe_gather,
      "x"_a,
      "gather_idx"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        MoE gather: out[p, :] = x[gather_idx[p], :] (zeros where gather_idx[p] < 0).
      )");

    m.def(
      "moe_grouped_gemm",
      &moe_grouped_gemm,
      "permuted_input"_a, "W"_a, "expert_of_tile"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused grouped expert GEMM: out = permuted_input @ W[expert]. Returns (total_rows, H).)");

    m.def(
      "moe_grouped_gemm_rect", &moe_grouped_gemm_rect,
      "A"_a, "W"_a, "expert_of_tile"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(rectangular grouped expert GEMM: out(rows,N_out) = A(rows,K_dim) @ W[e](K_dim,N_out).)");

    m.def(
      "moe_grouped_gemm_swiglu", &moe_grouped_gemm_swiglu,
      "A"_a, "W1"_a, "expert_of_tile"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(fused SiLU-GLU GEMM1: out(rows,inter) = silu(A@W1_gate)*(A@W1_up); W1[e] is (H,2*inter).)");

    m.def(
      "qk_norm_rope", &qk_norm_rope,
      "qkv"_a, "q_weight"_a, "k_weight"_a, "cos"_a, "sin"_a, "positions"_a,
      "num_heads_q"_a, "num_heads_k"_a, "num_heads_v"_a,
      nb::kw_only(), "eps"_a = 1e-6f, "interleaved"_a = false, "gemma"_a = false,
      "stream"_a = nb::none(),
      R"(fused per-head QK-RMSNorm + RoPE over packed QKV (V heads copied through).
         interleaved: False = NeoX split-half, True = GPT-J pairs. Returns the new qkv.)");

    m.def(
      "moe_route_grouped", &moe_route_grouped,
      "logits"_a, "bias"_a, "has_bias"_a, "k"_a, "n_group"_a, "topk_group"_a,
      "renormalize"_a, "routed_scaling_factor"_a, "scoring_func"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(DeepSeek-style grouped routing (noaux_tc): sigmoid/softmax/sqrt-softplus scoring,
         bias-corrected selection, group top-2 ranking, weights from unbiased scores.
         Returns (ids int32, weights f32).)");

    m.def(
      "moe_grouped_gemm_rect_q", &moe_grouped_gemm_rect_q,
      "A"_a, "Wq"_a, "expert_of_tile"_a, "bias"_a, "has_bias"_a, "K_dim"_a, "N_out"_a,
      "format"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(quantized grouped expert GEMM: out(rows,N_out) = A @ dequant(Wq[e])^T [+ bias[e]].
         Wq (E, N_out, row_bytes) uint8, quant groups along K (quantize_expert_stack layout).)");

    m.def(
      "moe_grouped_gemm_swiglu_q", &moe_grouped_gemm_swiglu_q,
      "A"_a, "W1q"_a, "expert_of_tile"_a, "bias"_a, "has_bias"_a, "inter"_a,
      "act_mode"_a, "alpha"_a, "limit"_a, "format"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(quantized fused SwiGLU GEMM1 from a packed [gate|up] expert stack (E, 2*inter, row_bytes).
         act_mode 0 = silu(gate)*up; 1 = swiglu_oai (clamped, alpha-sigmoid, (1+up)) for gpt-oss.)");

    m.def(
      "moe_finalize",
      &moe_finalize,
      "expert_out"_a,
      "inv_idx"_a,
      "topk_weights"_a,
      "k"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        MoE finalize: out[t] = sum_k weight[t,k] * expert_out[inv_idx[t*k+k]]. Returns (T, Hdim).
      )");

    m.def(
      "argmax_sample",
      &argmax_sample,
      "logits"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        greedy sampling: argmax token index over the last (vocab) axis. Returns int32.
      )");

    m.def(
      "sample_categorical",
      &sample_categorical,
      "logits"_a,
      nb::kw_only(),
      "temperature"_a = 1.0f,
      "seed"_a = 0u,
      "stream"_a = nb::none(),
      R"(
        Gumbel-max categorical sampling from softmax(logits/temperature). Returns int32.
      )");

    m.def(
      "top_k_sample",
      &top_k_sample,
      "logits"_a,
      "k"_a,
      nb::kw_only(),
      "temperature"_a = 1.0f,
      "seed"_a = 0u,
      "stream"_a = nb::none(),
      R"(
        top-k sampling: Gumbel-max from softmax over the k highest logits. Returns int32.
      )");

    m.def(
      "top_p_sample",
      &top_p_sample,
      "logits"_a,
      "p"_a,
      nb::kw_only(),
      "temperature"_a = 1.0f,
      "seed"_a = 0u,
      "stream"_a = nb::none(),
      R"(
        top-p (nucleus) sampling: Gumbel-max from the smallest top-prob set with mass >= p. Returns int32.
      )");

  m.def(
      "min_p_sample",
      &min_p_sample,
      "logits"_a,
      "min_p"_a,
      nb::kw_only(),
      "temperature"_a = 1.0f,
      "seed"_a = 0u,
      "stream"_a = nb::none(),
      R"(
        min-p sampling: Gumbel-max over tokens with prob >= min_p * max_prob. Returns int32.
      )");

  m.def(
      "typical_p_sample",
      &typical_p_sample,
      "logits"_a,
      "typical_p"_a,
      nb::kw_only(),
      "temperature"_a = 1.0f,
      "seed"_a = 0u,
      "stream"_a = nb::none(),
      R"(
        typical-p sampling: Gumbel-max over the smallest-surprise |(-log p)-H| mass typical_p. int32.
      )");

  m.def(
      "apply_token_bitmask",
      &apply_token_bitmask,
      "logits"_a,
      "bitmask"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Grammar / structured-output masking: logits[v] = -inf where the packed allow-bitmask bit
        (bitmask (T, ceil(V/32)) uint32) for token v is 0. Returns masked logits, same dtype.
      )");

  m.def(
      "apply_bad_words",
      &apply_bad_words,
      "logits"_a,
      "bad_ids"_a,
      "bad_lens"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Bad / stop-word masking: logits[t, bad_ids[t,j]] = -inf for j < bad_lens[t]. Returns masked
        logits, same dtype. bad_ids (T, maxbad) int, bad_lens (T,) int.
      )");

    m.def(
      "beam_advance",
      &beam_advance,
      "logits"_a,
      "cum_log_probs"_a,
      "beam_width"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        beam-search advance: one fused log-softmax + cumulative-score + top-beam_width step.
        Returns [next_token (B,BM), parent_beam (B,BM), cum_log_probs' (B,BM)].
      )");

  m.def(
      "spec_verify_linear",
      &spec_verify_linear,
      "draft_tokens"_a,
      "draft_probs"_a,
      "target_probs"_a,
      "bonus_tokens"_a,
      "accept_u"_a,
      "seed"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Speculative decoding linear rejection-sampling verification (vLLM contract). Returns
        [out_tokens (B,S+1) int32 (-1 = placeholder), accepted_cnt (B,) int32].
      )");

  m.def(
      "spec_verify_tree",
      &spec_verify_tree,
      "draft_tokens"_a,
      "target_probs"_a,
      "retrieve_next_token"_a,
      "retrieve_next_sibling"_a,
      "tree_valid"_a,
      "seed"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Speculative tree verification (target-only rejection). Returns [accept_index, accept_token,
        accept_num] (int32; -1-padded).
      )");

  m.def(
      "build_dynamic_tree",
      &build_dynamic_tree,
      "parents"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Build draft-tree pointers on device from a (B, N) parent list. Returns
        [retrieve_next_token, retrieve_next_sibling, positions] (all int32).
      )");

  m.def(
      "spec_compact",
      &spec_compact,
      "out_tokens"_a,
      "accepted_cnt"_a,
      "seq_lens"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Compact accepted spec tokens: returns [packed_tokens, packed_pos, cu_accepted] (all int32).
      )");

  m.def(
      "spec_update_kv_meta",
      &spec_update_kv_meta,
      "seq_lens"_a,
      "accepted_cnt"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        new_seq_lens[b] = seq_lens[b] + accepted_cnt[b] + 1. Returns (B,) int32.
      )");

    m.def(
      "apply_penalty",
      &apply_penalty,
      "logits"_a,
      "prev_tokens"_a,
      "bias"_a,
      "parent_ids"_a,
      nb::kw_only(),
      "temperature"_a = 1.0f,
      "repetition_penalty"_a = 1.0f,
      "presence_penalty"_a = 0.0f,
      "frequency_penalty"_a = 0.0f,
      "eos_id"_a = -1,
      "min_length"_a = 0,
      "gen_len"_a = 0,
      "stream"_a = nb::none(),
      R"(
        temperature + repetition/presence/frequency penalties + logit bias + min-length EOS mask.
        parent_ids (num_tokens,) redirects each row's occurrence history (beam search: beam inherits
        its parent beam's history). Returns [penalized, counts]; use [0].
      )");

    m.def(
      "quantize_per_tensor_fp8", &quantize_per_tensor_fp8,
      "x"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(per-tensor fp8 e4m3 quant (global absmax/448 via atomic-max). Returns [codes, scale, scratch].)");
    m.def(
      "quantize_per_tensor_int8", &quantize_per_tensor_int8,
      "x"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(per-tensor symmetric int8 quant (global absmax/127). Returns [codes, scale, scratch].)");

    m.def(
      "quantize_per_token_fp8",
      &quantize_per_token_fp8,
      "x"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        per-row fp8 e4m3 quantization. Returns (codes uint8, scale f32) with scale=absmax/448.
      )");

    m.def(
      "quantize_per_token_int8",
      &quantize_per_token_int8,
      "x"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        per-row symmetric int8 quantization. Returns (codes int8, scale f32) with scale=absmax/127.
      )");

    m.def(
      "softmax",
      &softmax_tk,
      "x"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        softmax over the last axis
      )");

    m.def(
      "rotary",
      &rotary,
      "x"_a,
      "cos"_a,
      "sin"_a,
      "interleaved"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        rotary positional embedding; x is (B,H,N,D), cos/sin are (N,D/2).
        interleaved=False: split-half (GPT-NeoX); interleaved=True: GPT-J adjacent pairs.
      )");

    m.def(
      "gelu",
      &gelu,
      "x"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        GELU activation (tanh approximation), over the last axis
      )");

    m.def(
      "gelu_bwd",
      &gelu_bwd,
      "x"_a,
      "dy"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        GELU backward (tanh approximation): dx = dy * gelu'(x). Elementwise.
      )");

    m.def(
      "dropout",
      &dropout,
      "x"_a,
      "p"_a,
      "seed"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Inverted dropout: out = keep ? x/(1-p) : 0, keep_i = rng_uniform(seed, i) >= p.
      )");

    m.def(
      "dropout_backward",
      &dropout_backward,
      "dy"_a,
      "p"_a,
      "seed"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Dropout backward: dx = keep ? dy/(1-p) : 0 (same mask recomputed from seed).
      )");

    m.def(
      "adamw",
      &adamw,
      "param"_a,
      "grad"_a,
      "m"_a,
      "v"_a,
      "lr"_a,
      "beta1"_a,
      "beta2"_a,
      "eps"_a,
      "weight_decay"_a,
      "step"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        AdamW step: returns (param', m', v') from (param, grad, m, v) and step t.
      )");

    m.def(
      "embedding_lookup",
      &embedding_lookup,
      "token_ids"_a,
      "table"_a,
      "pos_table"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Token embedding lookup: out[t] = scale*table[token_ids[t]] (+ pos_table[t] if size>1).
      )");

    m.def(
      "embedding_backward",
      &embedding_backward,
      "token_ids"_a,
      "dY"_a,
      "vocab"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Embedding backward: scatter-add dY rows into a (vocab, D) fp32 grad table by token id.
      )");

    m.def(
      "embedding_backward_sorted",
      &embedding_backward_sorted,
      "sorted_ids"_a,
      "perm"_a,
      "dY"_a,
      "vocab"_a,
      "scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Embedding backward (sorted-segment, atomic-free): accumulate dY into a (vocab, D) fp32 grad
        table from host-presorted (sorted_ids, perm). Same result as embedding_backward.
      )");

    m.def(
      "merge_multimodal_spans",
      &merge_multimodal_spans,
      "text"_a,
      "modal"_a,
      "src"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Multimodal span merge: out[t] = src[t]>=0 ? modal[src[t]] : text[t].
      )");

    m.def(
      "build_multimodal_src",
      &build_multimodal_src,
      "span_offsets"_a,
      "span_lengths"_a,
      "modal_starts"_a,
      "num_tok"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Build the multimodal src map on-device: src[t] = modal_starts[k]+off in span k, else -1.
      )");

    m.def(
      "glu",
      &glu,
      "x"_a,
      "gate"_a,
      nb::kw_only(),
      "mode"_a = "swiglu",
      "alpha"_a = 1.0f,
      "limit"_a = 1.0e20f,
      "stream"_a = nb::none(),
      R"(
        GLU-family activation: reglu, geglu, swiglu, swiglu_oai, geglu_erf, or geglu_quick
      )");

    m.def(
      "glu_backward",
      &glu_backward,
      "x"_a,
      "gate"_a,
      "dc"_a,
      nb::kw_only(),
      "mode"_a = "swiglu",
      "alpha"_a = 1.0f,
      "limit"_a = 1.0e20f,
      "stream"_a = nb::none(),
      R"(
        GLU-family backward: returns (da, db) = grads wrt x, gate given upstream grad dc.
      )");

    m.def(
      "hadamard",
      &hadamard,
      "x"_a,
      "scale"_a = 0.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Walsh-Hadamard transform over the final axis; default scale is 1/sqrt(D)
      )");

    m.def(
      "kv_cache_scatter",
      &kv_cache_scatter,
      "key"_a,
      "value"_a,
      "slot_mapping"_a,
      "num_blocks"_a,
      "block_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        scatter packed key/value rows (T,H,D) into paged KV caches (num_blocks, block_size, H, D)
      )");

    m.def(
      "kv_cache_gather",
      &kv_cache_gather,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "cu_seq_lens"_a,
      "num_tokens"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        gather paged KV caches back into contiguous key/value tensors
      )");

    m.def(
      "kv_cache_copy_blocks",
      &kv_cache_copy_blocks,
      "key_cache"_a,
      "value_cache"_a,
      "block_mapping"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        copy KV cache blocks according to (src, dst) block pairs
      )");

  m.def(
      "beam_build_copy_pairs",
      &beam_build_copy_pairs,
      "parent_beam"_a,
      "block_table"_a,
      "seq_lens"_a,
      "block_size"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Build the (src,dst) block-copy pairs for a beam KV reorder on-device (no host readback).
        Returns a fixed (B*BM*max_blocks, 2) int64 buffer for kv_cache_copy_blocks.
      )");

  m.def(
      "beam_remap_block_table",
      &beam_remap_block_table,
      "block_table"_a,
      "parent_beam"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Zero-copy beam reorder: new_block_table[b*BM+k] = block_table[b*BM+parent_beam[b,k]].
      )");

    m.def(
      "kv_cache_scales",
      &kv_cache_scales,
      "key"_a,
      "value"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        compute fp8 KV-cache scales as absmax(key/value) / 240
      )");

    m.def(
      "paged_attention",
      &paged_attention,
      "q"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "scale"_a = 0.0f,
      nb::kw_only(),
      "window"_a = 0,
      "stream"_a = nb::none(),
      R"(
        decode paged attention over caches shaped (num_blocks, block_size, H, D);
        window > 0 restricts to the `window` most recent keys (Mistral sliding window)
      )");

    m.def(
      "paged_attention_alibi",
      &paged_attention_alibi,
      "q"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "alibi_slopes"_a,
      "scale"_a = 0.0f,
      nb::kw_only(),
      "window"_a = 0,
      "stream"_a = nb::none(),
      R"(
        paged decode with a per-head ALiBi linear position bias (alibi_slopes is (num_heads,)).
        window > 0 restricts to the `window` most recent keys.
      )");

    m.def(
      "paged_attention_block_sparse",
      &paged_attention_block_sparse,
      "q"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "block_mask"_a,
      "scale"_a = 0.0f,
      nb::kw_only(),
      "window"_a = 0,
      "stream"_a = nb::none(),
      R"(
        block-sparse paged decode; block_mask (batch, max_blocks) int32 (1=attend, 0=skip) per KV block.
        window > 0 restricts to the `window` most recent keys.
      )");

    m.def(
      "paged_attention_staged",
      &paged_attention_staged,
      "q"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "scale"_a = 0.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        GQA KV-reuse staged decode; bit-equivalent to paged_attention, staged via threadgroup memory.
      )");

    m.def(
      "paged_attention_xcache",
      &paged_attention_xcache,
      "q"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "scale"_a = 0.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        paged decode over a vLLM x-packed KV cache: key (nb, nkv, hd/x, bs, x), value (nb, nkv, hd, bs).
      )");

    m.def(
      "kv_cache_scatter_fp8",
      &kv_cache_scatter_fp8,
      "key"_a,
      "value"_a,
      "slot_mapping"_a,
      "num_blocks"_a,
      "block_size"_a,
      "k_scale"_a,
      "v_scale"_a,
      "fmt"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        scatter K/V into a uint8 paged cache with per-head (num_heads,) scales; fmt 0=e4m3, 1=e5m2. Returns (kc, vc).
      )");

    m.def(
      "paged_attention_fp8",
      &paged_attention_fp8,
      "q"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "k_scale"_a,
      "v_scale"_a,
      "scale"_a = 0.0f,
      "fmt"_a = 0,
      nb::kw_only(),
      "window"_a = 0,
      "stream"_a = nb::none(),
      R"(
        decode paged attention over fp8 (uint8) caches, dequantized on read; fmt 0=e4m3, 1=e5m2. GQA aware.
        window > 0 restricts to the `window` most recent keys.
      )");

    m.def(
      "attn_causal",
      &attn_causal,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "softcap"_a = 0.0f,
      "sinks"_a = nb::none(),
      "stream"_a = nb::none(),
      R"(
        causal (lower-triangular) attention forward. softcap > 0 = Gemma-style logit
        capping; sinks (H,) = gpt-oss attention-sink logits (denominator only).
      )");

    m.def(
      "attn_window",
      &attn_window,
      "q"_a,
      "k"_a,
      "v"_a,
      "window"_a,
      nb::kw_only(),
      "softcap"_a = 0.0f,
      "sinks"_a = nb::none(),
      "stream"_a = nb::none(),
      R"(
        sliding-window causal attention forward: query i attends keys
        [max(0, i-window+1), i]; window <= 0 disables the window
      )");

    m.def(
      "attn_varlen_prefill",
      &attn_varlen_prefill,
      "q_hm"_a,
      "key_cache"_a,
      "value_cache"_a,
      "block_table"_a,
      "context_lens"_a,
      "tile_seq"_a,
      "tile_local0"_a,
      "seq_qlen"_a,
      "scale"_a,
      nb::kw_only(),
      "softcap"_a = 0.0f,
      "sinks"_a = nb::none(),
      "stream"_a = nb::none(),
      R"(
        varlen / paged-prefill causal attention: ragged packed queries (head-major,
        padded per sequence to a multiple of 8) reading K/V from the paged cache.
        softcap > 0 = Gemma-style logit capping; sinks (H,) = gpt-oss attention sinks.
      )");

  m.def(
      "varlen_build_worklist",
      &varlen_build_worklist,
      "cu_seqlens"_a,
      "max_tiles"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        On-device varlen prefill scheduler: cu_seqlens (B+1) -> [qlens, pad_off, tile_seq,
        tile_local0, n_tiles] (tile_seq is -1 past n_tiles). max_tiles is a host upper bound.
      )");

  m.def(
      "varlen_pad_q",
      &varlen_pad_q,
      "q_packed"_a,
      "cu_seqlens"_a,
      "pad_off"_a,
      "total_padded"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Device varlen Q pad/gather: packed (total_q,H,D) bf16 -> padded head-major (H,total_padded,D).
      )");

  m.def(
      "varlen_regather_o",
      &varlen_regather_o,
      "o_hm"_a,
      "cu_seqlens"_a,
      "pad_off"_a,
      "total_q"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Device varlen output re-gather: head-major (H,total_padded,D) bf16 -> packed (total_q,H,D).
      )");

    m.def(
      "lm_head_sample",
      &lm_head_sample,
      "h"_a,
      "w"_a,
      "bias"_a,
      "mode"_a,
      "k"_a,
      "temperature"_a,
      "seed"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        fused LM-head + sampling: token id per row without materializing (T,V) logits.
        mode 0=argmax, 1=categorical, 2=top-k
      )");

    m.def(
      "lm_head_sample_q",
      &lm_head_sample_q,
      "h"_a,
      "wq"_a,
      "bias"_a,
      "v"_a,
      "k"_a,
      "fmt"_a,
      "mode"_a,
      "topk"_a,
      "temperature"_a,
      "seed"_a,
      "top_p"_a = 0.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        fused LM-head + sampling over quantized (q8_0/q4_0) weights; mode 0=argmax, 1=categorical,
        2=topk, 3=topp (nucleus over the over-selected top-k candidate pool; top_p in (0,1], k = the
        candidate cap). No (T,V) logits materialization.
      )");

    m.def(
      "cross_entropy_fwd",
      &cross_entropy_fwd,
      "logits"_a,
      "targets"_a,
      "ignore_index"_a,
      "label_smoothing"_a,
      "z_loss"_a,
      "softcap"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        fused cross-entropy forward: per-row [loss, lse] without storing probabilities
      )");

    m.def(
      "cross_entropy_bwd",
      &cross_entropy_bwd,
      "logits"_a,
      "targets"_a,
      "lse"_a,
      "grad_out"_a,
      "ignore_index"_a,
      "label_smoothing"_a,
      "z_loss"_a,
      "softcap"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        fused cross-entropy backward: grad_logits (T,V), out-of-place
      )");

    m.def(
      "flux_gelu",
      &flux_gelu,
      "x"_a,
      "w"_a,
      "bias"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        fused GEMM + GELU: gelu(x @ w + bias)
      )");

    m.def(
      "flux_gate",
      &flux_gate,
      "x"_a,
      "w"_a,
      "bias"_a,
      "gate"_a,
      "residual"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        fused GEMM + gate + residual: (x @ w + bias) * gate + residual
      )");

    m.def(
      "gemm_staged",
      &gemm_staged,
      "x"_a,
      "y"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        multi-simdgroup threadgroup-staged GEMM: x @ y
      )");

    m.def(
      "attn_multiwarp",
      &attn_multiwarp,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        multi-warp flash attention forward (shared K/V across simdgroups)
      )");

    m.def(
      "linear_attn",
      &linear_attn,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        non-causal linear attention (identity feature map): Q @ (K^T @ V)
      )");

    m.def(
      "hedgehog",
      &hedgehog,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        hedgehog linear attention: phi(Q) @ (phi(K)^T @ V), phi(x)=exp(x-rowmax(x))
      )");

    m.def(
      "lin_attn_causal",
      &lin_attn_causal,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        causal linear attention (identity feature map), chunked running-KV scan
      )");

    m.def(
      "mamba2",
      &mamba2,
      "C"_a,
      "B"_a,
      "X"_a,
      "cumlog"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Mamba-2 / SSD forward: Y_t = sum_{j<=t} (C_t.B_j) exp(cumlog_t-cumlog_j) X_j
      )");

    m.def(
      "mamba2_bwd",
      &mamba2_bwd,
      "C"_a,
      "B"_a,
      "X"_a,
      "cumlog"_a,
      "dY"_a,
      nb::kw_only(),
      "force_quadratic"_a = false,
      "stream"_a = nb::none(),
      R"(
        Mamba-2 / SSD backward: returns [dC, dB, dX, dcumlog] (dcumlog = rowsum(M) - colsum(M)).
        force_quadratic bypasses the chunked linear-time route (for testing route agreement).
      )");

    m.def(
      "mamba2_chunked",
      &ssd_chunked,
      "C"_a,
      "B"_a,
      "X"_a,
      "cumlog"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Mamba-2 / SSD forward, forced chunked linear-time route (tests/benchmarks). N%64==0, N>=128.
      )");

    m.def(
      "mamba2_bwd_chunked",
      &ssd_chunked_bwd,
      "C"_a,
      "B"_a,
      "X"_a,
      "cumlog"_a,
      "dY"_a,
      "stream"_a = nb::none(),
      R"(
        Mamba-2 / SSD backward, forced chunked linear-time route (tests/benchmarks). N%64==0, N>=128.
      )");

    m.def(
      "ssd_decode",
      &ssd_decode,
      "S"_a,
      "alpha"_a,
      "x"_a,
      "k"_a,
      "q"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Single-token SSD decode step: S' = alpha*S + x(outer)k ; y = S'.q. Returns [y, S'].
      )");

    m.def(
      "lin_attn_decay",
      &lin_attn_decay,
      "q"_a, "k"_a, "v"_a, "cl"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        decay/retention linear attention: out_i = sum_{j<=i} exp(cl_i-cl_j) (q_i.k_j) v_j; cl=-slope*pos
      )");

    m.def(
      "based",
      &based,
      "q"_a, "k"_a, "v"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Based Taylor-map linear attention: out_i = sum_{j<=i} (1 + x + x^2/2) v_j, x=(q.k)/sqrt(D_QK)
      )");

    m.def(
      "attn_fwd_l", &attn_fwd_l,
      "q"_a, "k"_a, "v"_a, "causal"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(flash-attention forward returning (o, L) where L is the log2-domain logsumexp per query row)");

    m.def(
      "attn_bwd_prep", &attn_bwd_prep,
      "o"_a, "do_"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(backward prep: delta = rowsum(dO . O) (B,H,N) fp32)");

    m.def(
      "attn_bwd_dq", &attn_bwd_dq,
      "q"_a, "k"_a, "v"_a, "do_"_a, "L"_a, "delta"_a, "causal"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(flash-attention backward dQ)");

    m.def(
      "attn_bwd_dkv", &attn_bwd_dkv,
      "q"_a, "k"_a, "v"_a, "do_"_a, "L"_a, "delta"_a, "causal"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(flash-attention backward returning (dK, dV))");

    m.def(
      "cmplx_matmul",
      &cmplx_matmul,
      "a"_a,
      "b"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        complex GEMM D = A @ B; operands carry a leading size-2 (real,imag) axis
      )");

    m.def(
      "fftconv",
      &fftconv,
      "x"_a,
      "fmat"_a,
      "twf"_a,
      "finv"_a,
      "twi"_a,
      "kf"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Monarch FFT convolution (N=S*S); complex inputs with leading size-2 axis, real output
      )");

    m.def(
      "qgemm",
      &qgemm,
      "wq"_a,
      "x"_a,
      "format"_a = "q8_0",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        quantized GEMM (Marlin's method): out = dequantize(wq) @ x; wq packed weight blocks
      )");

    m.def(
      "qgemm_blockscale", &qgemm_blockscale,
      "wq"_a, "x"_a, "scale2d"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fp8_block2d: codes-only fp8 weights + separate (N/128,K/128) tile scale -> dequant @ x)");

    m.def(
      "qgemm_fp8_scaled", &qgemm_fp8_scaled,
      "wq"_a, "xq"_a, "w_scale"_a, "a_scale"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fp8 rank-1 scaled GEMM: both operands fp8 e4m3 -> dequant @ dequant, *w_scale[n]*a_scale[m])");

    m.def(
      "qgemm_actorder_k", &qgemm_actorder_k,
      "wq"_a, "x"_a, "perm"_a, "format"_a = "kU4B8",
      nb::kw_only(), "stream"_a = nb::none(),
      R"(GPTQ act-order qgemm with an in-kernel g_idx gather (no materialized permuted X))");

    m.def(
      "qgemm_direct",
      &qgemm_direct,
      "wq"_a,
      "x"_a,
      "format"_a = "q8_0",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        quantized GEMM, dequant-direct-to-fragment (Marlin zero-shuffle; no threadgroup staging)
      )");

    m.def(
      "qgemv",
      &qgemv,
      "wq"_a,
      "x"_a,
      "format"_a = "q8_0",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        quantized GEMV (batch-1 decode): out = dequantize(wq) @ x; x is (K,1)
      )");

    m.def(
      "qflux_gelu",
      &qflux_gelu,
      "wq"_a,
      "x"_a,
      "bias"_a,
      "format"_a = "q8_0",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        quantized fused GEMM+GELU: gelu(dequantize(wq) @ x + bias)
      )");

    m.def(
      "qgemv_w8a8",
      &qgemv_w8a8,
      "wq"_a, "xq"_a, "w_scale"_a, "a_scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        W8A8 decode GEMV: int8 weight x int8 activation -> int32, then *w_scale[n]*a_scale
      )");

    m.def(
      "attn_q",
      &attn_q,
      "q"_a, "kq"_a, "vq"_a,
      "format"_a = "q8_0",
      "causal"_a = false,
      "multiwarp"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        quantized-KV flash attention: softmax(QK^T)V with K,V dequantized from blocks
      )");

    m.def(
      "qgemv_w2a8",
      &qgemv_w2a8,
      "wq"_a, "xq"_a, "a_scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        BitNet W2A8 decode GEMV: ternary 2-bit weight x int8 activation -> int32, per-group scale
      )");

    m.def(
      "qgemm_w8a8", &qgemm_w8a8,
      "wq"_a, "xq"_a, "w_scale"_a, "a_scale"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(W8A8 prefill GEMM (int8 x int8 -> int32, bit-exact, then scale))");

    m.def(
      "qgemm_w2a8", &qgemm_w2a8,
      "wq"_a, "xq"_a, "a_scale"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(BitNet W2A8 prefill GEMM (ternary 2-bit x int8 -> int32))");
}
