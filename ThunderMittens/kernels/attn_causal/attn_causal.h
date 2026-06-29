// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Causal (lower-triangular) flash-attention forward.
 *  q, k, v are (B, H, N, D), bf16; D in {64, 128}, N a multiple of 8.
 **/
array attn_causal(
    const array& q,
    const array& k,
    const array& v,
    StreamOrDevice s = {}
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class AttnCausal : public Primitive {
 public:
  explicit AttnCausal(Stream stream) : Primitive(stream) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  void print(std::ostream& os) override {
    os << "AttnCausal";
  }

  bool is_equivalent(const Primitive& other) const override;

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

} // namespace mlx::core
