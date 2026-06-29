// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  GELU activation (tanh approximation), matching mx.nn.gelu_approx.
 *  Elementwise over the last axis; bf16 in/out, fp32 compute. D in {256,512,768,1024}.
 **/
array gelu(
    const array& x,
    StreamOrDevice s = {}
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class Gelu : public Primitive {
 public:
  explicit Gelu(Stream stream) : Primitive(stream) {};

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
    os << "Gelu";
  }

  bool is_equivalent(const Primitive& other) const override;

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

} // namespace mlx::core
