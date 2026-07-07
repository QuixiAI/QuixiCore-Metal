// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array ternary_stats(const array& wq, StreamOrDevice s = {});
array code_flip_count(const array& a, const array& b, StreamOrDevice s = {});

class TernaryStats : public Primitive {
 public:
  explicit TernaryStats(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "TernaryStats"; }
  void print(std::ostream& os) override { os << "TernaryStats"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class CodeFlipCount : public Primitive {
 public:
  explicit CodeFlipCount(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "CodeFlipCount"; }
  void print(std::ostream& os) override { os << "CodeFlipCount"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
