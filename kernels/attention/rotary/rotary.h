// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Rotary positional embedding (RoPE), split-half / GPT-NeoX convention
 *  (matches mx.fast.rope(..., traditional=False)).
 *
 *  x is (B, H, N, D); cos and sin are precomputed (N, D/2). bf16 in/out.
 *  D in {64, 128}.
 **/
array rotary(
    const array& x,   // (B, H, N, D)
    const array& cos, // (N, D/2)
    const array& sin, // (N, D/2)
    bool interleaved = false,  // false = split-half (NeoX); true = GPT-J interleaved
    StreamOrDevice s = {}
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class Rotary : public Primitive {
 public:
  explicit Rotary(Stream stream, bool interleaved = false)
      : Primitive(stream), interleaved_(interleaved) {};

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
  const char* name() const { return "Rotary"; }


  void print(std::ostream& os) override {
    os << "Rotary";
  }

  bool is_equivalent(const Primitive& other) const override;

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

 private:
  bool interleaved_;
};

} // namespace mlx::core
