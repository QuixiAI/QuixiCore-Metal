#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array edge_mlp_256x7(
    const array& hidden,
    const array& first_weight,
    const array& first_bias,
    const array& second_weight,
    const array& second_bias,
    StreamOrDevice s = {});

class EdgeMlpProject256 : public Primitive {
 public:
  explicit EdgeMlpProject256(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "EdgeMlpProject256"; }
  void print(std::ostream& os) override { os << "EdgeMlpProject256"; }
  bool is_equivalent(const Primitive& other) const override {
    return typeid(*this) == typeid(other);
  }
};

class EdgeMlpCombine256x7 : public Primitive {
 public:
  explicit EdgeMlpCombine256x7(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "EdgeMlpCombine256x7"; }
  void print(std::ostream& os) override { os << "EdgeMlpCombine256x7"; }
  bool is_equivalent(const Primitive& other) const override {
    return typeid(*this) == typeid(other);
  }
};

} // namespace mlx::core
