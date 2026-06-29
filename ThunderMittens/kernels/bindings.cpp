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
#include "cmplx_matmul/cmplx_matmul.h"
#include "fftconv/fftconv.h"
#include "qgemm/qgemm.h"
#include "qgemv/qgemv.h"
#include "qflux/qflux.h"

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
}
