#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> weight_quant_ternary(
    const array& w,
    int group_k = 32,
    StreamOrDevice s = {});

std::vector<array> weight_quant_ternary_pt(
    const array& w,
    StreamOrDevice s = {});

class WeightQuantTernary : public Primitive {
 public:
  WeightQuantTernary(Stream stream, int group_k)
      : Primitive(stream), group_k_(group_k) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "WeightQuantTernary"; }
  void print(std::ostream& os) override { os << "WeightQuantTernary"; }
  bool is_equivalent(const Primitive& other) const override {
    return group_k_ == static_cast<const WeightQuantTernary&>(other).group_k_;
  }

 private:
  int group_k_;
};

class WeightQuantTernaryPt : public Primitive {
 public:
  explicit WeightQuantTernaryPt(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "WeightQuantTernaryPt"; }
  void print(std::ostream& os) override { os << "WeightQuantTernaryPt"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
