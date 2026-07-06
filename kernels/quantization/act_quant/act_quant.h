// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Fused gated-activation -> dynamic quant epilogues. x is the ACTIVATED operand, gate the
 *  multiplier (tk.glu order); act "swiglu" (mode 0) or gpt-oss "swiglu_oai" (mode 1, alpha +
 *  limit). Per-token variants return [codes, scale (rows,)]; the fp8 per-group variant
 *  returns [codes, scale (rows, D/G)] with optional ue8m0 power-of-two scales.
 **/
std::vector<array> silu_mul_quant_fp8(
    const array& x, const array& gate, int mode = 0, float alpha = 1.702f,
    float limit = 7.0f, StreamOrDevice s = {});
std::vector<array> silu_mul_quant_int8(
    const array& x, const array& gate, int mode = 0, float alpha = 1.702f,
    float limit = 7.0f, StreamOrDevice s = {});
std::vector<array> silu_mul_quant_fp8_group(
    const array& x, const array& gate, int group_size = 128, bool ue8m0 = false,
    int mode = 0, float alpha = 1.702f, float limit = 7.0f, StreamOrDevice s = {});

class ActQuant : public Primitive {
 public:
  // kind: 0 = fp8 per-token, 1 = int8 per-token, 2 = fp8 per-group
  ActQuant(Stream stream, int kind, int mode, float alpha, float limit, int group_size,
           bool ue8m0)
      : Primitive(stream), kind_(kind), mode_(mode), alpha_(alpha), limit_(limit),
        group_size_(group_size), ue8m0_(ue8m0) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "ActQuant"; }
  void print(std::ostream& os) override { os << "ActQuant"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const ActQuant&>(other);
    return kind_ == o.kind_ && mode_ == o.mode_ && alpha_ == o.alpha_ &&
           limit_ == o.limit_ && group_size_ == o.group_size_ && ue8m0_ == o.ue8m0_;
  }

 private:
  int kind_;
  int mode_;
  float alpha_;
  float limit_;
  int group_size_;
  bool ue8m0_;
};

} // namespace mlx::core
