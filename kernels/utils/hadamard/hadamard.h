// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array hadamard(const array& x, float scale = 0.0f, StreamOrDevice s = {});

class HadamardTK : public Primitive {
 public:
  HadamardTK(Stream stream, float scale) : Primitive(stream), scale_(scale) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&,
      const std::vector<array>&,
      const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&,
      const std::vector<array>&,
      const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&,
      const std::vector<int>&) override;
  const char* name() const { return "HadamardTK"; }

  void print(std::ostream& os) override { os << "HadamardTK"; }
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const HadamardTK&>(other).scale_;
  }

 private:
  float scale_;
};

} // namespace mlx::core
