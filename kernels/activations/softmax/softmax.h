// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  TKSoftmax over the last axis: y = exp(x - max(x)) / sum(exp(x - max(x))).
 *  x is (..., D), bf16 in/out, fp32 compute. D in {256,512,768,1024}.
 **/
array softmax_tk(
    const array& x,
    StreamOrDevice s = {}
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class TKSoftmax : public Primitive {
 public:
  explicit TKSoftmax(Stream stream) : Primitive(stream) {};

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
  const char* name() const { return "TKSoftmax"; }


  void print(std::ostream& os) override {
    os << "TKSoftmax";
  }

  bool is_equivalent(const Primitive& other) const override;

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

} // namespace mlx::core
