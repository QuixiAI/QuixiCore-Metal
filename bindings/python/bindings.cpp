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
#include "mean_pool_rms_l2/mean_pool_rms_l2.h"
#include "add_norm/add_norm.h"
#include "rope_kv/rope_kv.h"
#include "qk_norm_rope/qk_norm_rope.h"
#include "selective_scan/selective_scan.h"
#include "gdn/gdn.h"
#include "act_quant/act_quant.h"
#include "fake_quant/fake_quant.h"
#include "fake_quant_fp8/fake_quant_fp8.h"
#include "weight_quant_ternary/weight_quant_ternary.h"
#include "quantize_tq2_0/quantize_tq2_0.h"
#include "ternary_stats/ternary_stats.h"
#include "minference/minference.h"
#include "turboquant/turboquant.h"
#include "marginal/marginal.h"
#include "indexer/indexer.h"
#include "mla/mla.h"
#include "paged_attn_v2/paged_attn_v2.h"
#include "quant_rt/quant_rt.h"
#include "sampling/sampling.h"
#include "sampling/transforms.h"
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
#include "kd_kl_topk/kd_kl_topk.h"
#include "kd_kl_dense/kd_kl_dense.h"
#include "flux/flux.h"
#include "gemm_staged/gemm_staged.h"
#include "gemm_v3/gemm_v3.h"
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
#include "qgemm_bwd/qgemm_bwd.h"
#include "qgemm_fused/qgemm_fused.h"
#include "qgemv/qgemv.h"
#include "qflux/qflux.h"
#include "qgemv_int/qgemv_int.h"
#include "attn_q/attn_q.h"
#include "attn_decode/attn_decode.h"
#include "qgemm_int/qgemm_int.h"
#include "swin_attn/swin_attn.h"
#include "decode_linear/decode_linear.h"
#include "dequant_gather/dequant_gather.h"
#include "edge_mlp/edge_mlp.h"
#include "patch_merge/patch_merge.h"

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
      "mean_pool_rms_l2",
      &mean_pool_rms_l2,
      "x"_a,
      "weight"_a,
      nb::kw_only(),
      "eps"_a = 1e-5f,
      "stream"_a = nb::none(),
      R"(
        mean-pool an (M, D) block into one (D,) embedding, then RMSNorm(weight) + L2-normalize
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
      "decode_layernorm_add", &decode_layernorm_add,
      "x"_a, "residual"_a, "weight"_a, "bias"_a,
      "eps"_a = 1e-5f, nb::kw_only(), "stream"_a = nb::none(),
      R"(decode-compatible residual-add + LayerNorm with materialized rounding semantics.)");

    m.def(
      "rms_norm_add_fp8", &rms_norm_add_fp8,
      "x"_a, "residual"_a, "weight"_a, "eps"_a, "scale"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + rms_norm + static-scale fp8. Returns (codes uint8, x+residual).)");
    m.def(
      "rms_norm_add_int8_dyn", &rms_norm_add_int8_dyn,
      "x"_a, "residual"_a, "weight"_a, "eps"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + rms_norm + dynamic per-row int8. Returns (codes, x+residual, scale).)");
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
      "layernorm_add_int8_dyn", &layernorm_add_int8_dyn,
      "x"_a, "residual"_a, "weight"_a, "bias"_a, "eps"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + layernorm + dynamic per-row int8. Returns (codes, x+residual, scale).)");
    m.def(
      "rms_norm_add_per_block_fp8", &rms_norm_add_per_block_fp8,
      "x"_a, "residual"_a, "weight"_a, "eps"_a, "ue8m0"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + rms_norm + per-128-block fp8. Returns (codes, x+residual, scale (rows,D/128)).)");
    m.def(
      "rms_norm_add_per_block_int8", &rms_norm_add_per_block_int8,
      "x"_a, "residual"_a, "weight"_a, "eps"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + rms_norm + per-128-block int8. Returns (codes, x+residual, scale (rows,D/128)).)");
    m.def(
      "layernorm_add_per_block_fp8", &layernorm_add_per_block_fp8,
      "x"_a, "residual"_a, "weight"_a, "bias"_a, "eps"_a, "ue8m0"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + layernorm + per-128-block fp8. Returns (codes, x+residual, scale (rows,D/128)).)");
    m.def(
      "layernorm_add_per_block_int8", &layernorm_add_per_block_int8,
      "x"_a, "residual"_a, "weight"_a, "bias"_a, "eps"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused residual-add + layernorm + per-128-block int8. Returns (codes, x+residual, scale (rows,D/128)).)");

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

    m.def("indexer_k_quant_and_cache", &indexer_k_quant_and_cache,
      "k"_a, "slot_mapping"_a, "code_cache"_a, "scale_cache"_a, "quant_block_size"_a = 128,
      "ue8m0"_a = false, nb::kw_only(), "stream"_a = nb::none(),
      R"(DeepSeek-V3.2 indexer K quant -> [code_cache u8, scale_cache f32] (functional).)");
    m.def("indexer_k_gather", &indexer_k_gather,
      "code_cache"_a, "scale_cache"_a, "slots"_a, "head_dim"_a, "quant_block_size"_a = 128,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(gather + dequantize the indexer cache to bf16 K for a slot list.)");

    m.def("tau_tail", &tau_tail,
      "qkv"_a, "tok_qv_lin"_a, "tau_pos_table"_a, "positions"_a, "n_heads"_a, "head_dim"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(scale Q and V slices of a packed (T, 3*q_dim) QKV by tanh(gate)+tau_pos.)");
    m.def("packbits", &packbits,
      "x"_a, "bit_order_big"_a = true, nb::kw_only(), "stream"_a = nb::none(),
      R"(pack bool/uint8 into bits (np.packbits); big or little bit order.)");
    m.def("segment_packbits", &segment_packbits,
      "x"_a, "input_indptr"_a, "output_indptr"_a, "total_output_bytes"_a,
      "bit_order_big"_a = true,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(ragged per-row packbits (output_indptr = host cumsum of ceil(len/8)).)");
    m.def("permute_cols", &permute_cols,
      "x"_a, "perm"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(16-bit column gather x[:, perm] (dtype-agnostic on 2-byte elements).)");

    m.def("tq_encode", &tq_encode,
      "key"_a, "value"_a, "key_cache"_a, "value_cache"_a, "key_scale"_a, "value_scale"_a,
      "key_zero"_a, "slot_mapping"_a, "v_centroids"_a, "signs"_a, "block_size"_a,
      "k_bits"_a, "k_signed"_a, "v_bits"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(TurboQuant KV encode -> [key_cache, value_cache, key_scale, value_scale, key_zero].)");
    m.def("tq_decode", &tq_decode,
      "key_cache"_a, "value_cache"_a, "key_scale"_a, "value_scale"_a, "key_zero"_a, "slots"_a,
      "v_centroids"_a, "signs"_a, "num_kv_heads"_a, "head_size"_a, "block_size"_a,
      "k_bits"_a, "k_signed"_a, "v_bits"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(TurboQuant KV decode (gather + dequantize) -> [k_out, v_out] float32.)");

    m.def("minference_block_mask", &minference_block_mask,
      "vertical_indexes"_a, "slash_indexes"_a, "context_lens"_a, "max_blocks"_a,
      "block_size"_a, "vertical_topk"_a = 1 << 30, "slash_topk"_a = 1 << 30,
      "last_n_blocks"_a = 1,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(MInference decode block-mask builder: per-head vertical cols + slash offsets ->
         (batch, num_heads, max_blocks) int32 mask for paged_attention_block_sparse.)");

    m.def("quadratic_transform", &quadratic_transform,
      "logits"_a, "factor"_a, "curve"_a = 1.0f, "temperature"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(quadratic/smoothing logit transform (factor 0 = identity).)");
    m.def("top_nsigma_mask", &top_nsigma_mask,
      "logits"_a, "nsigma"_a, "temperature"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(top-nsigma: mask logits below max - nsigma*std.)");
    m.def("top_a_mask", &top_a_mask,
      "logits"_a, "top_a"_a, "temperature"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(top-A: mask probs below top_a * pmax^2 (log-space exact).)");
    m.def("epsilon_cutoff_mask", &epsilon_cutoff_mask,
      "logits"_a, "epsilon"_a, "temperature"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(epsilon cutoff: mask probs below epsilon (argmax survives).)");
    m.def("eta_cutoff_mask", &eta_cutoff_mask,
      "logits"_a, "eta"_a, "temperature"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(eta sampling: mask probs below min(eta, sqrt(eta)*exp(-entropy)).)");
    m.def("xtc_mask", &xtc_mask,
      "logits"_a, "threshold"_a, "probability"_a, "seed"_a = 0, "temperature"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(XTC: with on-device coin `probability`, remove all probs >= threshold except the
         least likely of them.)");
    m.def("skew_transform", &skew_transform,
      "probs"_a, "skew"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(skew: pow(index-order CDF, exp(skew)) reshaping over probability rows.)");
    m.def("top_k_renorm", &top_k_renorm,
      "probs"_a, "k"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(keep the top-k probs, renormalize to 1, zero elsewhere (k <= 64).)");
    m.def("top_p_renorm", &top_p_renorm,
      "probs"_a, "p"_a, nb::kw_only(), "stream"_a = nb::none(),
      R"(keep the smallest prob set with mass >= p (bisection, no sort), renormalize.)");
    m.def("no_repeat_ngram_mask", &no_repeat_ngram_mask,
      "logits"_a, "prev_tokens"_a, "lens"_a, "ngram_size"_a, "temperature"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(ban tokens completing an already-seen ngram_size-gram (n >= 2).)");
    m.def("dry_penalty", &dry_penalty,
      "logits"_a, "prev_tokens"_a, "lens"_a, "breakers"_a, "multiplier"_a,
      "base"_a = 1.75f, "allowed_length"_a = 2, "range"_a = 0, "max_ngram"_a = 64,
      "max_occurrences"_a = 64, "early_exit_match_len"_a = 64, "temperature"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(DRY repetition penalty: penalize tokens extending repeated suffixes by
         multiplier*base^(match_len+1-allowed_length). breakers (NB,) int32, pad -1.)");

    m.def(
      "silu_mul_quant_fp8", &silu_mul_quant_fp8,
      "x"_a, "gate"_a, "mode"_a = 0, "alpha"_a = 1.702f, "limit"_a = 7.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused gated-activation -> dynamic per-token fp8: returns (codes u8, scale (rows,)).
         mode 0 swiglu, 1 swiglu_oai (gpt-oss).)");

    m.def(
      "silu_mul_quant_int8", &silu_mul_quant_int8,
      "x"_a, "gate"_a, "mode"_a = 0, "alpha"_a = 1.702f, "limit"_a = 7.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused gated-activation -> dynamic per-token int8 (feeds qgemm_w8a8).)");

    m.def(
      "silu_mul_quant_fp8_group", &silu_mul_quant_fp8_group,
      "x"_a, "gate"_a, "group_size"_a = 128, "ue8m0"_a = false, "mode"_a = 0,
      "alpha"_a = 1.702f, "limit"_a = 7.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused gated-activation -> per-group fp8 (scale (rows, D/G); ue8m0 = 2^k scales).)");

    m.def(
      "fake_quant_int8", &fake_quant_int8,
      "x"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(one-pass per-token int8 fake quant: returns (x_q bf16, codes i8, scale f32).)");

    m.def(
      "fake_quant_fp8", &fake_quant_fp8,
      "x"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(per-tensor e4m3 fake quant: returns (x_fq, scale, scratch).)");

    m.def(
      "silu_mul_fake_quant_int8", &silu_mul_fake_quant_int8,
      "x"_a, "gate"_a, "mode"_a = 0, "alpha"_a = 1.702f, "limit"_a = 7.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused gated activation plus one-pass int8 fake quant: returns (x_q, codes, scale).)");

    m.def(
      "weight_quant_ternary", &weight_quant_ternary,
      "w"_a, "group_k"_a = 32,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(BitNet ternary weight quantization: returns (packed u8 bitnet blocks, w_deq bf16).)");

    m.def(
      "weight_quant_ternary_pt", &weight_quant_ternary_pt,
      "w"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(BitNet per-tensor ternary weight quantization: returns (packed, w_deq, scratch).)");

    m.def(
      "quantize_tq2_0", &quantize_tq2_0,
      "w"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(TQ2_0 GGUF ternary weight quantization: returns (packed block_tq2_0 bytes, w_deq bf16).)");

    m.def(
      "ternary_stats", &ternary_stats,
      "wq"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(packed BitNet ternary blocks -> per-row {-1,0,+1} counts int32 (rows,3).)");

    m.def(
      "code_flip_count", &code_flip_count,
      "a"_a, "b"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(packed BitNet ternary blocks -> per-row count of code changes between a and b.)");

    m.def(
      "quantize_per_group_fp8", &quantize_per_group_fp8,
      "x"_a, "group_size"_a = 128, "ue8m0"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(per-group dynamic fp8 e4m3: returns (codes u8, scale (rows, D/G) f32).
         ue8m0 rounds scales up to powers of two (MX convention).)");

    m.def(
      "quantize_per_group_int8", &quantize_per_group_int8,
      "x"_a, "group_size"_a = 128,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(per-group dynamic symmetric int8: returns (codes i8, scale (rows, D/G) f32).)");

    m.def(
      "quantize_per_token_int8_azp", &quantize_per_token_int8_azp,
      "x"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(asymmetric per-token int8 (vLLM azp): returns (codes, scale (rows,), azp (rows,) i32).)");

    m.def(
      "qgemm_w8a8_azp", &qgemm_w8a8_azp,
      "wq"_a, "xq"_a, "w_scale"_a, "a_scale"_a, "w_rowsum"_a, "azp"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(azp-corrected W8A8 GEMM: y = s_w*s_a*(W@Xq^T - azp*rowsum(W)).)");

    m.def(
      "gdn_recur", &gdn_recur,
      "q"_a, "k"_a, "v"_a, "g"_a, "beta"_a, "state_pool"_a, "cu_seqlens"_a, "slot_mapping"_a,
      "load_initial"_a = true,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(GatedDeltaNet delta-rule linear attention over varlen packed sequences with a
         persistent per-request fp32 state pool. Returns [y, new_state_pool].)");

    m.def(
      "selective_scan", &selective_scan,
      "u"_a, "delta"_a, "A"_a, "B"_a, "C"_a, "state"_a, "D"_a = nb::none(),
      "delta_bias"_a = nb::none(), "z"_a = nb::none(), "delta_softplus"_a = true,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(Mamba-1 (S6) selective scan, dense batch (channel-major). Returns [out, new_state].)");

    m.def(
      "selective_scan_varlen", &selective_scan_varlen,
      "u"_a, "delta"_a, "A"_a, "B"_a, "C"_a, "query_start_loc"_a, "state"_a,
      "D"_a = nb::none(), "delta_bias"_a = nb::none(), "z"_a = nb::none(),
      "cache_indices"_a = nb::none(), "has_initial_state"_a = nb::none(),
      "delta_softplus"_a = true, "null_block_id"_a = -1,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(Varlen S6 scan over flattened tokens with a per-request paged state pool.
         Returns [out, new_state_pool] (untouched slots preserved).)");

    m.def(
      "selective_scan_varlen_apc", &selective_scan_varlen_apc,
      "u"_a, "delta"_a, "A"_a, "B"_a, "C"_a, "query_start_loc"_a, "cache_indices"_a,
      "has_initial_state"_a, "state"_a, "block_idx_first_scheduled_token"_a,
      "block_idx_last_scheduled_token"_a, "initial_state_idx"_a, "cu_chunk_seqlen"_a,
      "last_chunk_indices"_a, "block_size"_a, "cache_indices_stride"_a, "use_chunk_metadata"_a,
      "D"_a = nb::none(), "delta_bias"_a = nb::none(), "z"_a = nb::none(),
      "delta_softplus"_a = true, "null_block_id"_a = -1,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(Varlen S6 scan with automatic-prefix-caching paged state checkpointing.
         Returns [out, new_state_pool].)");

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
      "moe_grouped_gemm_bwd_dx", &moe_grouped_gemm_bwd_dx,
      "dy"_a, "W"_a, "expert_of_tile"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(MoE backward grouped GEMM: dx = dy @ W[e]^T over the padded schedule.)");

    m.def(
      "moe_grouped_gemm_bwd_dw", &moe_grouped_gemm_bwd_dw,
      "A"_a, "dy"_a, "off_pad"_a, "num_experts"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(MoE backward grouped GEMM: dW[e] = A^T @ dy per padded expert segment.)");

    m.def(
      "moe_finalize_bwd", &moe_finalize_bwd,
      "grad_out"_a, "expert_out"_a, "inv_idx"_a, "topk_weights"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(MoE finalize backward: returns (grad_expert_out zero-padded, grad_weights).)");

    m.def(
      "moe_gather_bwd", &moe_gather_bwd,
      "dA"_a, "inv_idx"_a, "k"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(MoE gather backward: dx[t] = sum_k dA[inv_idx[t*k+k]].)");

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

  m.def("rejection_greedy_sample", &rejection_greedy_sample,
      "cu_num_draft_tokens"_a, "draft_token_ids"_a, "target_argmax"_a, "bonus_token_ids"_a,
      "max_draft"_a, "is_greedy"_a = nb::none(),
      nb::kw_only(), "stream"_a = nb::none(),
      R"(vLLM greedy rejection verify. Returns out (B, max_draft+1) int32.)");
  m.def("rejection_random_sample", &rejection_random_sample,
      "cu_num_draft_tokens"_a, "draft_token_ids"_a, "target_probs"_a, "bonus_token_ids"_a,
      "recovered_token_ids"_a, "uniform_probs"_a, "max_draft"_a, "draft_probs"_a = nb::none(),
      "is_greedy"_a = nb::none(),
      nb::kw_only(), "stream"_a = nb::none(),
      R"(vLLM stochastic rejection verify (u <= p/q). Returns out (B, max_draft+1) int32.)");
  m.def("sample_recovered_tokens", &sample_recovered_tokens,
      "cu_num_draft_tokens"_a, "draft_token_ids"_a, "target_probs"_a, "inv_q"_a,
      "draft_probs"_a = nb::none(),
      nb::kw_only(), "stream"_a = nb::none(),
      R"(recovered token per draft position: argmax(max(0,p-q) * inv_q). Returns (total,) int32.)");
  m.def("eagle_prepare_inputs_padded", &eagle_prepare_inputs_padded,
      "cu_num_draft_tokens"_a, "valid_sampled_tokens_count"_a, "query_start_loc"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(EAGLE prep: [token_indices_to_sample, num_rejected] int32.)");
  m.def("eagle_prepare_next_token_padded", &eagle_prepare_next_token_padded,
      "sampled_token_ids"_a, "discard_request_mask"_a, "backup_next_token_ids"_a, "vocab_size"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(EAGLE next-seed token: [next_token_ids, valid_sampled_tokens_count] int32.)");
  m.def("eagle_step_slot_mapping_metadata", &eagle_step_slot_mapping_metadata,
      "positions"_a, "block_table"_a, "seq_lens"_a, "block_size"_a, "max_model_len"_a, "pad_id"_a,
      "input_batch_size"_a = -1,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(EAGLE step slot: [out_clamped_positions, out_slot_mapping, new_seq_lens] int32.)");
  m.def("eagle_expand_int32", &eagle_expand_int32,
      "input"_a, "cu_num_tokens"_a, "total"_a, "replace_from"_a, "replace_to"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(broadcast input[r] across [cu[r],cu[r+1]) with replace_from->replace_to. (total,) int32.)");

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
      "adamw_masked",
      &adamw_masked,
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
      "mask"_a,
      "seg_size"_a,
      "mask_mode"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        Masked AdamW step for segmented/cold parameters. mask_mode 0 skips inactive
        segment updates; mask_mode 1 skips only decoupled decay on inactive segments.
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
      "kv_cache_gather_fp8", &kv_cache_gather_fp8,
      "key_cache"_a, "value_cache"_a, "block_table"_a, "cu_seq_lens"_a, "k_scale"_a,
      "v_scale"_a, "num_tokens"_a, "fmt"_a = 0,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fp8 KV gather + upconvert: dequantize e4m3/e5m2 codes to bf16 via per-kv_head scales.
         Returns [key_out, value_out] bf16.)");

    m.def(
      "kv_cache_scale_update", &kv_cache_scale_update,
      "key"_a, "value"_a, "old_key_scale"_a, "old_value_scale"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(incremental per-tensor KV scale running-max: new = max(old, absmax/240).)");

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
      "lm_head_beam_advance",
      &lm_head_beam_advance,
      "h"_a,
      "wq"_a,
      "bias"_a,
      "cum_log_probs"_a,
      "beam_width"_a,
      "format"_a = "q4_0",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        quantized LM-head + exact beam-search advance without materializing full logits.
        Returns [next_token, parent_beam, updated cumulative log-probability].
      )");

    m.def(
      "lm_head_constrained", &lm_head_constrained,
      "h"_a, "w"_a, "bias"_a, "forbidden"_a, "previous"_a,
      "eos_id"_a = -1, "forbid_eos"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(dense projection + row-conditioned grammar mask + greedy token and log-probability.)");

    m.def(
      "lm_head_masked", &lm_head_masked,
      "h"_a, "w"_a, "bias"_a, "allow_mask"_a,
      "format"_a = "", "topk"_a = 1, "normalize_allowed"_a = true,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(masked dense/packed LM head returning top-k ids and log-probabilities.)");

    m.def(
      "lm_head_candidates", &lm_head_candidates,
      "h"_a, "w"_a, "bias"_a, "candidate_ids"_a, "offsets"_a,
      "format"_a = "", "topk"_a = 1,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(CSR candidate dense/packed LM head returning top-k ids and log-probabilities.)");

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
      "kd_kl_topk_fwd",
      &kd_kl_topk_fwd,
      "logits"_a,
      "t_idx"_a,
      "t_prob"_a,
      "invtemp"_a = 1.0f,
      "tail_mode"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(sparse-teacher KD-KL forward over top-k teacher probabilities. Returns (loss, lse).)");

    m.def(
      "kd_kl_topk_bwd",
      &kd_kl_topk_bwd,
      "logits"_a,
      "t_idx"_a,
      "t_prob"_a,
      "lse"_a,
      "grad_out"_a,
      "invtemp"_a = 1.0f,
      "tail_mode"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(sparse-teacher KD-KL backward. Returns grad_logits.)");

    m.def(
      "kd_kl_dense_fwd",
      &kd_kl_dense_fwd,
      "t_logits"_a,
      "s_logits"_a,
      "invtemp"_a = 1.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(dense-teacher KD-KL forward. Returns (loss, lse_t, lse_s).)");

    m.def(
      "kd_kl_dense_bwd",
      &kd_kl_dense_bwd,
      "t_logits"_a,
      "s_logits"_a,
      "lse_t"_a,
      "lse_s"_a,
      "grad_out"_a,
      "invtemp"_a = 1.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(dense-teacher KD-KL backward. Returns grad_s.)");

    m.def(
      "kd_ce_fused_fwd",
      &kd_ce_fused_fwd,
      "t_logits"_a,
      "s_logits"_a,
      "targets"_a,
      "invtemp"_a = 1.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(fused CE + dense-KD forward. Returns (ce, kd, lse_sr, lse_st, lse_t).)");

    m.def(
      "kd_ce_fused_bwd",
      &kd_ce_fused_bwd,
      "t_logits"_a,
      "s_logits"_a,
      "targets"_a,
      "lse_sr"_a,
      "lse_st"_a,
      "lse_t"_a,
      "go_ce"_a,
      "go_kd"_a,
      "invtemp"_a = 1.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(fused CE + dense-KD backward. Returns combined grad_s.)");

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
      "flux_gelu_erf", &flux_gelu_erf,
      "x"_a, "w"_a, "bias"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused GEMM + erf GELU.)");

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
      "gemm_v3",
      &gemm_v3,
      "x"_a,
      "y"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        academic 2x2-warp staged GEMM: x @ y
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
      "qgemm_bwd",
      &qgemm_bwd,
      "grad_y"_a,
      "wq"_a,
      "format"_a = "bitnet",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        BitNet packed-weight backward GEMM: grad_x = grad_y @ dequant(wq)
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
      "dequant_gather", &dequant_gather,
      "table"_a, "ids"_a, "format"_a, "scale"_a = 1.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(gather packed GGUF rows and dequantize directly to fp16.)");

    m.def(
      "quantized_embedding", &quantized_embedding,
      "table"_a, "ids"_a, "add"_a, "format"_a,
      "scale"_a = 1.0f, "use_add"_a = false,
      "output_dtype"_a = "float16",
      nb::kw_only(), "stream"_a = nb::none(),
      R"(packed embedding lookup with optional additive epilogue and selectable output dtype.)");

    m.def(
      "quantized_embedding_bag", &quantized_embedding_bag,
      "table"_a, "ids"_a, "offsets"_a, "sample_weights"_a, "format"_a,
      "scale"_a = 1.0f, "use_weights"_a = false, "mean_mode"_a = false,
      "output_dtype"_a = "float16",
      nb::kw_only(), "stream"_a = nb::none(),
      R"(CSR embedding bag over packed rows; sum or valid-id mean.)");

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
      "attn_decode",
      &attn_decode,
      "q"_a, "k"_a, "v"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        batch-1 GQA decode attention over dense KV: q (Hq,D), k/v (Tk,Hkv,D)
      )");

    m.def(
      "attn_decode_bh", &attn_decode_bh,
      "q"_a, "k"_a, "v"_a, "context_length"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(batched head-major GQA decode over a preallocated dense KV cache.)");

    m.def(
      "decode_cache_attention", &decode_cache_attention,
      "q"_a, "new_k"_a, "new_v"_a, "cos"_a, "sin"_a,
      "positions"_a, "context_lengths"_a, "q_weight"_a, "k_weight"_a,
      "key_cache"_a, "value_cache"_a, "eps"_a = 1e-6f,
      "do_q_norm"_a = false, "do_k_norm"_a = false, "gemma"_a = false,
      "softmax_scale"_a = 0.0f,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused Q/K norm, RoPE, dense-cache append, and multi-simdgroup decode attention.)");

    m.def(
      "swin_attn_d32", &swin_attn_d32,
      "qkv"_a, "relative_bias"_a, "mask"_a, "windows_per_image"_a = 0,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(Swin window attention for packed qkv with head dimension 32.)");

    m.def(
      "patch_merge_layernorm", &patch_merge_layernorm,
      "input"_a, "weight"_a, "bias"_a, "height"_a, "width"_a,
      "eps"_a = 1e-5f, nb::kw_only(), "stream"_a = nb::none(),
      R"(fused Swin 2x2 patch gather and LayerNorm.)");

    m.def(
      "space_to_depth_norm_linear", &space_to_depth_norm_linear,
      "input"_a, "norm_weight"_a, "norm_bias"_a,
      "projection_weight"_a, "projection_bias"_a,
      "height"_a, "width"_a, "block_size"_a = 2, "eps"_a = 1e-5f,
      "use_norm_bias"_a = true, "use_projection_bias"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fused space-to-depth, LayerNorm, and dense projection.)");

    m.def(
      "edge_mlp_256x7", &edge_mlp_256x7,
      "hidden"_a, "first_weight"_a, "first_bias"_a,
      "second_weight"_a, "second_bias"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(fixed 512-to-256-to-7 pairwise edge MLP.)");

    m.def(
      "decode_linear", &decode_linear,
      "x"_a, "weight"_a, "bias"_a, "gelu"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(latency-oriented decode linear with optional erf GELU.)");

    m.def(
      "decode_linear_residual", &decode_linear_residual,
      "x"_a, "weight"_a, "bias"_a, "residual"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(decode linear plus materialized-order residual addition.)");

    m.def(
      "decode_linear_q8", &decode_linear_q8,
      "x"_a, "weight"_a, "bias"_a, "residual"_a,
      "gelu"_a = false, "use_residual"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(q8_0 decode linear with optional erf GELU and residual.)");

    m.def(
      "decode_linear_epilogue", &decode_linear_epilogue,
      "x"_a, "weight"_a, "bias"_a, "residual"_a,
      "format"_a = "", "activation"_a = 0,
      "use_bias"_a = false, "use_residual"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(dense/packed decode linear with fused bias, activation, and residual.)");

    m.def(
      "decode_swiglu", &decode_swiglu,
      "x"_a, "gate_weight"_a, "up_weight"_a,
      "gate_bias"_a, "up_bias"_a,
      "format"_a = "", "use_bias"_a = false,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(dense/packed pair of decode projections with fused SwiGLU.)");

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
      "qgemv_w2a8_v2",
      &qgemv_w2a8_v2,
      "wq"_a, "xq"_a, "a_scale"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        BitNet W2A8 decode GEMV v2: one packed block per lane
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

    m.def(
      "qgemm_w2a8_fused", &qgemm_w2a8_fused,
      "wq"_a, "x"_a,
      nb::kw_only(), "stream"_a = nb::none(),
      R"(BitNet fused per-token int8 activation quant + W2A8 GEMM.)");
}
