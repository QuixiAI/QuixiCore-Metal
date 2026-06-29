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

#include "add_rt/add_rt.h"
#include "attn_fwd/attn_fwd.h"
#include "matmul_custom/matmul_custom.h"
#include "layernorm/layernorm.h"
#include "rms_norm/rms_norm.h"
#include "softmax/softmax.h"
#include "rotary/rotary.h"
#include "gelu/gelu.h"
#include "attn_causal/attn_causal.h"
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
      "stream"_a = nb::none(),
      R"(
        attn fwd
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
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        rotary positional embedding (split-half / GPT-NeoX); x is (B,H,N,D), cos/sin are (N,D/2)
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
      "attn_causal",
      &attn_causal,
      "q"_a,
      "k"_a,
      "v"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"(
        causal (lower-triangular) attention forward
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
