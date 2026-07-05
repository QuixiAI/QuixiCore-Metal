// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operations
///////////////////////////////////////////////////////////////////////////////

/**
 *  Fused residual-add + RMSNorm over the last axis. Returns two arrays:
 *      out     = (x + residual) * rsqrt(mean((x+residual)^2) + eps) * weight
 *      res_out = x + residual   (the summed residual the next block consumes)
 *
 *  x and residual are (..., D); weight is (D,). bf16 in/out, fp32 compute.
 *  D must be one of {256, 512, 768, 1024}.
 **/
std::vector<array> rms_norm_add(
    const array& x,
    const array& residual,
    const array& weight,
    float eps = 1e-5f,
    StreamOrDevice s = {});

/**
 *  Fused residual-add + LayerNorm over the last axis. Returns two arrays:
 *      out     = ((x+residual) - mean) * rsqrt(var + eps) * weight + bias
 *      res_out = x + residual
 *
 *  x and residual are (..., D); weight and bias are (D,). bf16 in/out.
 **/
std::vector<array> layernorm_add(
    const array& x,
    const array& residual,
    const array& weight,
    const array& bias,
    float eps = 1e-5f,
    StreamOrDevice s = {});

/**
 *  fp8 e4m3 epilogue variants: out = e4m3(norm(x+residual)*weight[+bias] / scale) as uint8
 *  codes, plus res_out = x+residual (bf16). Static returns [codes, res_out]; dynamic (per-row
 *  absmax/448) returns [codes, res_out, scale (per row)].
 **/
std::vector<array> rms_norm_add_fp8(
    const array& x, const array& residual, const array& weight, float eps, float scale,
    StreamOrDevice s = {});
std::vector<array> rms_norm_add_fp8_dyn(
    const array& x, const array& residual, const array& weight, float eps, StreamOrDevice s = {});
/** int8 sibling of rms_norm_add_fp8_dyn (dynamic per-row symmetric int8, feeds qgemm_w8a8).
 *  Returns (codes int8, x+residual, scale (rows,) f32). */
std::vector<array> rms_norm_add_int8_dyn(
    const array& x, const array& residual, const array& weight, float eps, StreamOrDevice s = {});
std::vector<array> layernorm_add_fp8(
    const array& x, const array& residual, const array& weight, const array& bias, float eps,
    float scale, StreamOrDevice s = {});
std::vector<array> layernorm_add_fp8_dyn(
    const array& x, const array& residual, const array& weight, const array& bias, float eps,
    StreamOrDevice s = {});

///////////////////////////////////////////////////////////////////////////////
// Primitives
///////////////////////////////////////////////////////////////////////////////

class AddNormFp8 : public Primitive {
 public:
  AddNormFp8(Stream stream, bool layernorm, bool dynamic, float eps, float inv_scale,
             bool int8q = false)
    : Primitive(stream), layernorm_(layernorm), dynamic_(dynamic), eps_(eps),
      inv_scale_(inv_scale), int8_(int8q) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AddNormFp8"; }
  void print(std::ostream& os) override { os << "AddNormFp8"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const AddNormFp8&>(other);
    return layernorm_ == o.layernorm_ && dynamic_ == o.dynamic_ && eps_ == o.eps_ &&
        inv_scale_ == o.inv_scale_ && int8_ == o.int8_;
  }

 private:
  bool layernorm_, dynamic_;
  float eps_, inv_scale_;
  bool int8_;
};

class RMSNormAdd : public Primitive {
 public:
  explicit RMSNormAdd(Stream stream, float eps)
    : Primitive(stream), eps_(eps) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const { return "RMSNormAdd"; }
  void print(std::ostream& os) override { os << "RMSNormAdd"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  float eps_;
};

class LayerNormAdd : public Primitive {
 public:
  explicit LayerNormAdd(Stream stream, float eps)
    : Primitive(stream), eps_(eps) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  const char* name() const { return "LayerNormAdd"; }
  void print(std::ostream& os) override { os << "LayerNormAdd"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  float eps_;
};

} // namespace mlx::core
