// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Fused cross-entropy forward over the vocab axis. logits (T, V) f32/f16/bf16, targets (T,) int32.
 *  Returns [loss (T,) f32, lse (T,) f32] — per-row loss (unreduced) and log-sum-exp. Rows whose
 *  target == ignore_index get loss 0. Supports label smoothing and a z-loss (z_loss * lse^2).
 **/
std::vector<array> cross_entropy_fwd(
    const array& logits,
    const array& targets,
    int ignore_index,
    float label_smoothing,
    float z_loss,
    float softcap,
    StreamOrDevice s = {});

/**
 *  Fused cross-entropy backward. Returns grad_logits (T, V) in logits' dtype (out-of-place).
 *  grad_out is the per-row upstream gradient (T,) f32 (e.g. 1/n_non_ignore for a mean reduction).
 **/
array cross_entropy_bwd(
    const array& logits,
    const array& targets,
    const array& lse,
    const array& grad_out,
    int ignore_index,
    float label_smoothing,
    float z_loss,
    float softcap,
    StreamOrDevice s = {});

class CrossEntropyFwd : public Primitive {
 public:
  CrossEntropyFwd(Stream stream, int ignore_index, float label_smoothing, float z_loss,
                  float softcap)
      : Primitive(stream), ignore_index_(ignore_index), label_smoothing_(label_smoothing),
        z_loss_(z_loss), softcap_(softcap) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "CrossEntropyFwd"; }
  void print(std::ostream& os) override { os << "CrossEntropyFwd"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const CrossEntropyFwd&>(other);
    return ignore_index_ == o.ignore_index_ && label_smoothing_ == o.label_smoothing_ &&
           z_loss_ == o.z_loss_ && softcap_ == o.softcap_;
  }

 private:
  int ignore_index_;
  float label_smoothing_;
  float z_loss_;
  float softcap_;
};

class CrossEntropyBwd : public Primitive {
 public:
  CrossEntropyBwd(Stream stream, int ignore_index, float label_smoothing, float z_loss,
                  float softcap)
      : Primitive(stream), ignore_index_(ignore_index), label_smoothing_(label_smoothing),
        z_loss_(z_loss), softcap_(softcap) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "CrossEntropyBwd"; }
  void print(std::ostream& os) override { os << "CrossEntropyBwd"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const CrossEntropyBwd&>(other);
    return ignore_index_ == o.ignore_index_ && label_smoothing_ == o.label_smoothing_ &&
           z_loss_ == o.z_loss_ && softcap_ == o.softcap_;
  }

 private:
  int ignore_index_;
  float label_smoothing_;
  float z_loss_;
  float softcap_;
};

} // namespace mlx::core
