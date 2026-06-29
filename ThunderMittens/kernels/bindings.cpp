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

#include "add_rt/add_rt.h"
#include "attn_fwd/attn_fwd.h"
#include "matmul_custom/matmul_custom.h"
#include "layernorm/layernorm.h"
#include "rms_norm/rms_norm.h"
#include "softmax/softmax.h"
#include "rotary/rotary.h"
#include "gelu/gelu.h"
#include "attn_causal/attn_causal.h"

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
}
