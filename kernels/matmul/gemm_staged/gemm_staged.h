// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Multi-simdgroup, threadgroup-staged GEMM: out = x @ y.
 *  x (N,K), y (K,M); f32/bf16. N%32, M%32, K%16. */
array gemm_staged(const array& x, const array& y, StreamOrDevice s = {});

class GemmStaged : public Primitive {
 public:
  explicit GemmStaged(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "GemmStaged"; }

  void print(std::ostream& os) override { os << "GemmStaged"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
