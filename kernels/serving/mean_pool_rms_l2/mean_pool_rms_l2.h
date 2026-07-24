// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  mean_pool_rms_l2: pool an (M, D) block of token states into one (D,) embedding.
 *
 *      p = (1/M) * sum_r x[r]                    // mean pool over the M rows
 *      n = p * rsqrt(mean(p^2) + eps) * weight   // RMSNorm (no bias, no mean-sub)
 *      y = n * rsqrt(sum(n^2) + tiny)            // L2-normalize
 *
 *  x is (M, D); weight is (D,). bf16 in/out, fp32 compute. D in {256,512,768,1024}.
 *  Shape-keyed by the hidden width D; any model with that width uses it.
 **/
array mean_pool_rms_l2(
    const array& x,       // Input token states, shape (M, D)
    const array& weight,  // Per-channel RMSNorm scale, shape (D,)
    float eps = 1e-5f,    // RMSNorm epsilon
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

/** Batched mask-aware pooling over x(B,T,D); mask(B,T) nonzero keeps a row. */
array masked_mean_pool_rms_l2(
    const array& x, const array& mask, const array& weight,
    float eps = 1e-5f, StreamOrDevice s = {});

class MeanPoolRmsL2 : public Primitive {
 public:
  explicit MeanPoolRmsL2(Stream stream, float eps)
    : Primitive(stream), eps_(eps) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  /** Print the primitive. */
  const char* name() const { return "MeanPoolRmsL2"; }

  void print(std::ostream& os) override {
    os << "MeanPoolRmsL2";
  }

  /** Equivalence check **/
  bool is_equivalent(const Primitive& other) const override;

  /** Fall back implementation for evaluation on CPU */
  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);

 private:
  float eps_;
};

class MaskedMeanPoolRmsL2 : public Primitive {
 public:
  MaskedMeanPoolRmsL2(Stream stream, float eps) : Primitive(stream), eps_(eps) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  const char* name() const { return "MaskedMeanPoolRmsL2"; }
  void print(std::ostream& os) override { os << "MaskedMeanPoolRmsL2"; }
  bool is_equivalent(const Primitive& other) const override {
    return eps_ == static_cast<const MaskedMeanPoolRmsL2&>(other).eps_;
  }
 private:
  float eps_;
};

}  // namespace mlx::core
