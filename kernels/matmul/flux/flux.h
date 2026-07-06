// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operations
///////////////////////////////////////////////////////////////////////////////

/** Fused GEMM + GELU:  out = gelu(x @ w + bias).
 *  x (N,K), w (K,M), bias (M,); bf16/f32. N%32, M%32, K%16. */
array flux_gelu(const array& x, const array& w, const array& bias, StreamOrDevice s = {});

/** Fused GEMM + gate + residual:  out = (x @ w + bias) * gate + residual.
 *  x (N,K), w (K,M), bias (M,), gate (M,), residual (N,M); bf16/f32. */
array flux_gate(const array& x, const array& w, const array& bias, const array& gate,
                const array& residual, StreamOrDevice s = {});

///////////////////////////////////////////////////////////////////////////////
// Primitives
///////////////////////////////////////////////////////////////////////////////

class FluxGelu : public Primitive {
 public:
  explicit FluxGelu(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "FluxGelu"; }

  void print(std::ostream& os) override { os << "FluxGelu"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

class FluxGate : public Primitive {
 public:
  explicit FluxGate(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "FluxGate"; }

  void print(std::ostream& os) override { os << "FluxGate"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
