// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 **/
/** Dense attention forward (B,H,N,D bf16). softcap <= 0 disables Gemma-style logit
 *  soft-capping. sinks: optional per-head fp32 (H,) attention-sink logits (gpt-oss) —
 *  added to the softmax denominator only. */
array attn_fwd(
    const array& q, // Input array q
    const array& k, // Input array k
    const array& v, // Input array q
    float softcap = 0.0f,
    const std::optional<array>& sinks = std::nullopt,
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class AttnFwd : public Primitive {
 public:
  explicit AttnFwd(Stream stream, float softcap = 0.0f, bool has_sink = false)
    : Primitive(stream), softcap_(softcap), has_sink_(has_sink) {};

  /**
   * A primitive must know how to evaluate itself on the CPU/GPU
   * for the given inputs and populate the output array.
   *
   * To avoid unnecessary allocations, the evaluation function
   * is responsible for allocating space for the array.
   */
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

  /**
   * The primitive must know how to vectorize itself across
   * the given axes. The output is a pair containing the array
   * representing the vectorized computation and the axis which
   * corresponds to the output vectorized dimension.
   */
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  /** Print the primitive. */
  const char* name() const { return "AttnFwd"; }

  void print(std::ostream& os) override {
    os << "AttnFwd";
  }

  /** Equivalence check **/
  bool is_equivalent(const Primitive& other) const override;

  /** Fall back implementation for evaluation on CPU */
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

 private:
  float softcap_;
  bool has_sink_;
};

} // namespace mlx::core
