// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Based: 2nd-order Taylor feature-map linear attention (causal):
 *  out_i = sum_{j<=i} (1 + x + x^2/2) * v_j, x = (q_i . k_j)/sqrt(D_QK), phi = exp Taylor map.
 *  q,k are (B,H,N,16) bf16; v is (B,H,N,64) bf16; out (B,H,N,64). D_QK=16, D_VO=64, N a multiple of 8. */
array based(const array& q, const array& k, const array& v, StreamOrDevice s = {});

class Based : public Primitive {
 public:
  explicit Based(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "Based"; }

  void print(std::ostream& os) override { os << "Based"; }
  bool is_equivalent(const Primitive&) const override { return true; }
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
