// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Mamba-2 / SSD forward (materialized chunked form):
 *  Y_t = sum_{j<=t} (C_t . B_j) * exp(cumlog_t - cumlog_j) * X_j.
 *  C,B,X are (B,H,N,D) bf16; cumlog (B,H,N) fp32 = cumsum(log a); D=64, N a multiple of 8. */
array mamba2(const array& C, const array& B, const array& X, const array& cumlog,
             StreamOrDevice s = {});

class Mamba2 : public Primitive {
 public:
  explicit Mamba2(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  void print(std::ostream& os) override { os << "Mamba2"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
