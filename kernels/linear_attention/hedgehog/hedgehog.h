// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Hedgehog-style linear attention (non-causal): out = phi(Q) @ (phi(K)^T @ V),
 *  phi(x) = exp(x - rowmax(x)). q,k,v are (B,H,N,D), bf16; D=64, N a multiple of 8. */
array hedgehog(const array& q, const array& k, const array& v, StreamOrDevice s = {});

class Hedgehog : public Primitive {
 public:
  explicit Hedgehog(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "Hedgehog"; }

  void print(std::ostream& os) override { os << "Hedgehog"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
