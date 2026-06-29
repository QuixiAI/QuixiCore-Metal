// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  LayerNorm (forward), normalized over the last axis:
 *      y = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias
 *
 *  x is (..., D); weight and bias are (D,). bf16 in/out, fp32 compute.
 **/
array layernorm(
    const array& x,      // Input array, normalized over the last axis
    const array& weight, // Per-channel scale, shape (D,)
    const array& bias,   // Per-channel shift, shape (D,)
    float eps = 1e-5f,   // Numerical stability epsilon
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class LayerNorm : public Primitive {
 public:
  explicit LayerNorm(Stream stream, float eps)
    : Primitive(stream), eps_(eps) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  /** The Jacobian-vector product. */
  std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  /** The vector-Jacobian product. */
  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  /** Vectorize the primitive across the given axes. */
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  /** Print the primitive. */
  void print(std::ostream& os) override {
    os << "LayerNorm";
  }

  /** Equivalence check **/
  bool is_equivalent(const Primitive& other) const override;

  /** Fall back implementation for evaluation on CPU */
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

 private:
  float eps_;
};

} // namespace mlx::core
