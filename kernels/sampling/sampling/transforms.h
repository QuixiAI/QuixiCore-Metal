// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

// ---- the "sampler zoo" logit/prob transforms (see sampling_transforms.metal). Every logit
// transform takes `temperature` and returns TEMPERED logits (the apply_penalty contract);
// masked tokens are -inf. probs-domain utilities (skew/renorms) take probability rows. ----

array quadratic_transform(const array& logits, float factor, float curve = 1.0f,
                          float temperature = 1.0f, StreamOrDevice s = {});
array logits_softcap(const array& logits, float cap, StreamOrDevice s = {});
array value_clip(const array& x, float min_value, float max_value,
                 StreamOrDevice s = {});
array top_nsigma_mask(const array& logits, float nsigma, float temperature = 1.0f,
                      StreamOrDevice s = {});
array top_a_mask(const array& logits, float top_a, float temperature = 1.0f,
                 StreamOrDevice s = {});
array epsilon_cutoff_mask(const array& logits, float epsilon, float temperature = 1.0f,
                          StreamOrDevice s = {});
array eta_cutoff_mask(const array& logits, float eta, float temperature = 1.0f,
                      StreamOrDevice s = {});
array xtc_mask(const array& logits, float threshold, float probability, int seed = 0,
               float temperature = 1.0f, StreamOrDevice s = {});
array skew_transform(const array& probs, float skew, StreamOrDevice s = {});
array top_k_renorm(const array& probs, int k, StreamOrDevice s = {});
array top_p_renorm(const array& probs, float p, StreamOrDevice s = {});
array no_repeat_ngram_mask(const array& logits, const array& prev_tokens, const array& lens,
                           int ngram_size, float temperature = 1.0f, StreamOrDevice s = {});
array dry_penalty(const array& logits, const array& prev_tokens, const array& lens,
                  const array& breakers, float multiplier, float base = 1.75f,
                  int allowed_length = 2, int range = 0, int max_ngram = 64,
                  int max_occurrences = 64, int early_exit_match_len = 64,
                  float temperature = 1.0f, StreamOrDevice s = {});

class SamplerTransform : public Primitive {
 public:
  // kind: 0 quadratic, 1 nsigma, 2 top_a, 3 eps, 4 eta, 5 xtc, 6 skew, 7 topk_renorm,
  // 8 topp_renorm, 9 ngram, 10 dry, 11 final-logit softcap, 12 generic value clip.
  // f0..f3 / i0..i3 / seed are kind-specific params.
  SamplerTransform(Stream stream, int kind, float f0, float f1, float f2, float f3,
                   int i0, int i1, int i2, int i3, uint32_t seed)
      : Primitive(stream), kind_(kind), f0_(f0), f1_(f1), f2_(f2), f3_(f3),
        i0_(i0), i1_(i1), i2_(i2), i3_(i3), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SamplerTransform"; }
  void print(std::ostream& os) override { os << "SamplerTransform[" << kind_ << "]"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const SamplerTransform&>(other);
    return kind_ == o.kind_ && f0_ == o.f0_ && f1_ == o.f1_ && f2_ == o.f2_ && f3_ == o.f3_ &&
           i0_ == o.i0_ && i1_ == o.i1_ && i2_ == o.i2_ && i3_ == o.i3_ && seed_ == o.seed_;
  }

 private:
  int kind_;
  float f0_, f1_, f2_, f3_;
  int i0_, i1_, i2_, i3_;
  uint32_t seed_;
};

} // namespace mlx::core
