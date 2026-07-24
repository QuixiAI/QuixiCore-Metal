// Copyright © 2026 QuixiCore contributors.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Direct F16-adapter LoRA delta:
 *
 *   low = f16(f16(x) @ A^T)
 *   delta = f16(low @ B^T)
 *   out = cast_x((has_base ? base : 0) + scale * delta)
 *
 * x is (M,K), A is F16 (R,K), B is F16 (N,R), and optional base is
 * (M,N). The F16 (M,R) intermediate stays in threadgroup memory.
 */
array lora_apply_direct(
    const array& x, const array& A, const array& B, const array& base,
    float scale, bool has_base, StreamOrDevice s = {});

class LoraApplyDirect : public Primitive {
 public:
  LoraApplyDirect(Stream stream, float scale, bool has_base)
      : Primitive(stream), scale_(scale), has_base_(has_base) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&,
      const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&,
      const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LoraApplyDirect"; }
  void print(std::ostream& os) override { os << "LoraApplyDirect"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const LoraApplyDirect&>(other);
    return scale_ == rhs.scale_ && has_base_ == rhs.has_base_;
  }

 private:
  float scale_;
  bool has_base_;
};

}  // namespace mlx::core
