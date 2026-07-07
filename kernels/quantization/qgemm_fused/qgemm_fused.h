// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array qgemm_w2a8_fused(const array& wq, const array& x, StreamOrDevice s = {});

class QGemmW2A8Fused : public Primitive {
 public:
  explicit QGemmW2A8Fused(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QGemmW2A8Fused"; }
  void print(std::ostream& os) override { os << "QGemmW2A8Fused"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
