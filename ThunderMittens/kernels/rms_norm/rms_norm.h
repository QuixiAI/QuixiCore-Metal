// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  RMSNorm (forward), normalized over the last axis:
 *      y = x * rsqrt(mean(x^2) + eps) * weight
 *
 *  x is (..., D); weight is (D,). bf16 in/out, fp32 compute. No mean-subtraction,
 *  no bias (cf. LayerNorm).
 **/
array rms_norm(
    const array& x,      // Input array, normalized over the last axis
    const array& weight, // Per-channel scale, shape (D,)
    float eps = 1e-5f,   // Numerical stability epsilon
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

/**
 *  RMSNorm backward, dX only: dX_i = rstd*(dY_i*W_i) - (rstd^3 * sum_j(dY_j*W_j*x_j) / D) * x_i.
 *  x/dy (rows, D); w (D,); rstd (rows,) fp32 precomputed. Returns dX (rows, D). dW (= sum over rows
 *  of dY*x*rstd) and dbias are cheap framework reductions done by the router.
 **/
array rms_norm_bwd_dx(
    const array& x,
    const array& weight,
    const array& dy,
    const array& rstd,
    StreamOrDevice s = {});

class RMSNormBwdDx : public Primitive {
 public:
  explicit RMSNormBwdDx(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "RMSNormBwdDx"; }
  void print(std::ostream& os) override { os << "RMSNormBwdDx"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class RMSNorm : public Primitive {
 public:
  explicit RMSNorm(Stream stream, float eps)
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
  const char* name() const { return "RMSNorm"; }

  void print(std::ostream& os) override {
    os << "RMSNorm";
  }

  /** Equivalence check **/
  bool is_equivalent(const Primitive& other) const override;

  /** Fall back implementation for evaluation on CPU */
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

 private:
  float eps_;
};

} // namespace mlx::core
