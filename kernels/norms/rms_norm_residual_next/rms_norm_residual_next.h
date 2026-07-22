// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  rms_norm_residual_next: the residual-stream seam between two transformer
 *  sub-blocks, fused into one pass. Returns two arrays:
 *
 *      res_out  = residual + rms_norm(x) * post_weight   // post-norm + residual add
 *      next_out = rms_norm(res_out) * next_weight        // next block's pre-norm input
 *
 *  where rms_norm(v) = v * rsqrt(mean(v^2) + eps). x and residual are (..., D);
 *  post_weight and next_weight are (D,). bf16 in/out, fp32 compute. D must be one
 *  of {256, 512, 768, 1024}. Shape-keyed by the hidden width D.
 **/
std::vector<array> rms_norm_residual_next(
    const array& x,
    const array& post_weight,
    const array& residual,
    const array& next_weight,
    float eps = 1e-5f,
    StreamOrDevice s = {});

class RMSNormResidualNext : public Primitive {
 public:
  explicit RMSNormResidualNext(Stream stream, float eps)
    : Primitive(stream), eps_(eps) {};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  const char* name() const { return "RMSNormResidualNext"; }
  void print(std::ostream& os) override { os << "RMSNormResidualNext"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  float eps_;
};

}  // namespace mlx::core
