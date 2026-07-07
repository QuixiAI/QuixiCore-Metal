// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> quantize_tq2_0(const array& w, StreamOrDevice s = {});

class QuantizeTQ20 : public Primitive {
 public:
  explicit QuantizeTQ20(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizeTQ20"; }
  void print(std::ostream& os) override { os << "QuantizeTQ20"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
