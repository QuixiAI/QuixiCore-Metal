// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Greedy sampling: argmax over the last (vocab) axis. logits is (..., V),
 *  float32/float16/bfloat16. Returns int32 token indices of shape logits.shape[:-1].
 *  Named *_sample to avoid collision with mlx::core::argmax.
 **/
array argmax_sample(const array& logits, StreamOrDevice s = {});

/**
 *  Stochastic categorical sampling (Gumbel-max) from softmax(logits/temperature).
 *  logits (..., V); returns int32 token indices of shape logits.shape[:-1]. The draw
 *  is fully determined by (seed, row) so it is exactly reproducible.
 **/
array sample_categorical(
    const array& logits, float temperature = 1.0f, uint32_t seed = 0,
    StreamOrDevice s = {});

/**
 *  Top-k sampling: restrict to the k highest-logit tokens, then sample (Gumbel-max)
 *  from softmax over them with temperature. logits (..., V); returns int32 token
 *  indices of shape logits.shape[:-1]. Reproducible given (seed, row). k <= 64.
 **/
array top_k_sample(
    const array& logits, int k, float temperature = 1.0f, uint32_t seed = 0,
    StreamOrDevice s = {});

/**
 *  Top-p (nucleus) sampling: sample (Gumbel-max) from the smallest set of highest-prob
 *  tokens whose cumulative softmax(logits/temperature) mass >= p. logits (..., V);
 *  returns int32 token indices of shape logits.shape[:-1]. Reproducible from (seed, row).
 **/
array top_p_sample(
    const array& logits, float p, float temperature = 1.0f, uint32_t seed = 0,
    StreamOrDevice s = {});

/** min-p sampling: keep tokens with (tempered) prob >= min_p * max_prob, Gumbel-max sample among
 *  them. logits (..., V); returns the token index per row (..., ). min_p in (0, 1]. */
array min_p_sample(
    const array& logits, float min_p, float temperature = 1.0f, uint32_t seed = 0,
    StreamOrDevice s = {});

/** Grammar / structured-output masking: logits[v] = -inf where the packed allow-bitmask bit for v
 *  is 0. logits (T, V); bitmask (T, ceil(V/32)) uint32. Returns masked logits (T, V), same dtype. */
array apply_token_bitmask(
    const array& logits, const array& bitmask, StreamOrDevice s = {});

/** Bad / stop-word masking: logits[t, bad_ids[t,j]] = -inf for j < bad_lens[t]. logits (T, V);
 *  bad_ids (T, maxbad) int; bad_lens (T,) int. Returns masked logits (T, V), same dtype. */
array apply_bad_words(
    const array& logits, const array& bad_ids, const array& bad_lens, StreamOrDevice s = {});

/**
 *  Apply temperature + repetition/presence/frequency penalties to logits given the
 *  generated token history. logits (T, V); prev_tokens (T, L) int (out-of-range entries,
 *  e.g. -1, are ignored padding). Returns the penalized logits (T, V), same dtype.
 *  Order (vLLM): logit*=1/temp; if seen: logit = logit<0 ? logit*rep : logit/rep;
 *  logit -= presence; logit -= frequency*count.
 *  Returns [penalized (T,V), counts (T,V) int32 scratch]; callers use the first.
 **/
std::vector<array> apply_penalty(
    const array& logits, const array& prev_tokens, const array& bias, const array& parent_ids,
    float temperature = 1.0f,
    float repetition_penalty = 1.0f, float presence_penalty = 0.0f,
    float frequency_penalty = 0.0f, int eos_id = -1, int min_length = 0, int gen_len = 0,
    StreamOrDevice s = {});

class ArgmaxSample : public Primitive {
 public:
  explicit ArgmaxSample(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "ArgmaxSample"; }
  void print(std::ostream& os) override { os << "ArgmaxSample"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class SampleCategorical : public Primitive {
 public:
  SampleCategorical(Stream stream, float invtemp, uint32_t seed)
      : Primitive(stream), invtemp_(invtemp), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SampleCategorical"; }
  void print(std::ostream& os) override { os << "SampleCategorical"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const SampleCategorical&>(other);
    return invtemp_ == o.invtemp_ && seed_ == o.seed_;
  }

 private:
  float invtemp_;
  uint32_t seed_;
};

class TopKSample : public Primitive {
 public:
  TopKSample(Stream stream, int k, float invtemp, uint32_t seed)
      : Primitive(stream), k_(k), invtemp_(invtemp), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "TopKSample"; }
  void print(std::ostream& os) override { os << "TopKSample"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const TopKSample&>(other);
    return k_ == o.k_ && invtemp_ == o.invtemp_ && seed_ == o.seed_;
  }

 private:
  int k_;
  float invtemp_;
  uint32_t seed_;
};

class TopPSample : public Primitive {
 public:
  TopPSample(Stream stream, float p, float invtemp, uint32_t seed)
      : Primitive(stream), p_(p), invtemp_(invtemp), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "TopPSample"; }
  void print(std::ostream& os) override { os << "TopPSample"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const TopPSample&>(other);
    return p_ == o.p_ && invtemp_ == o.invtemp_ && seed_ == o.seed_;
  }

 private:
  float p_;
  float invtemp_;
  uint32_t seed_;
};

class MinPSample : public Primitive {
 public:
  MinPSample(Stream stream, float min_p, float invtemp, uint32_t seed)
      : Primitive(stream), min_p_(min_p), invtemp_(invtemp), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MinPSample"; }
  void print(std::ostream& os) override { os << "MinPSample"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MinPSample&>(other);
    return min_p_ == o.min_p_ && invtemp_ == o.invtemp_ && seed_ == o.seed_;
  }

 private:
  float min_p_;
  float invtemp_;
  uint32_t seed_;
};

class ApplyTokenBitmask : public Primitive {
 public:
  explicit ApplyTokenBitmask(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "ApplyTokenBitmask"; }
  void print(std::ostream& os) override { os << "ApplyTokenBitmask"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class ApplyBadWords : public Primitive {
 public:
  explicit ApplyBadWords(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "ApplyBadWords"; }
  void print(std::ostream& os) override { os << "ApplyBadWords"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class ApplyPenalty : public Primitive {
 public:
  ApplyPenalty(Stream stream, float invtemp, float rep, float presence, float freq,
               int eos_id, int min_length, int gen_len)
      : Primitive(stream), invtemp_(invtemp), rep_(rep), presence_(presence), freq_(freq),
        eos_id_(eos_id), min_length_(min_length), gen_len_(gen_len) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "ApplyPenalty"; }
  void print(std::ostream& os) override { os << "ApplyPenalty"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const ApplyPenalty&>(other);
    return invtemp_ == o.invtemp_ && rep_ == o.rep_ && presence_ == o.presence_ &&
        freq_ == o.freq_ && eos_id_ == o.eos_id_ && min_length_ == o.min_length_ &&
        gen_len_ == o.gen_len_;
  }

 private:
  float invtemp_, rep_, presence_, freq_;
  int eos_id_, min_length_, gen_len_;
};

/**
 *  Beam-search advance: one fused step of log-softmax + cumulative score + top-beam_width selection
 *  with parent tracking. logits (B*BM, V) f32/f16/bf16, cum_log_probs (B, BM) f32. Returns
 *  [next_token (B, BM) int32, parent_beam (B, BM) int32, new_cum_log_probs (B, BM) f32].
 *  beam_width (BM) <= 16. Step-0 convention: the caller sets cum_log_probs[:, 1:] = -inf.
 **/
std::vector<array> beam_advance(
    const array& logits,
    const array& cum_log_probs,
    int beam_width,
    StreamOrDevice s = {});

/**
 *  Speculative decoding: linear (non-tree) rejection-sampling verification (the vLLM contract).
 *  Given draft_tokens (B,S) int, draft_probs (B,S,V) f32, target_probs (B,S+1,V) f32, bonus_tokens
 *  (B,) int, and accept_u (B,S) f32 uniforms, returns [out_tokens (B,S+1) int32, accepted_cnt (B,)
 *  int32]. Draft dt is accepted iff u <= p_target/p_draft; the first rejection emits a token sampled
 *  from the residual (p_target-p_draft)+, and all-accept appends the bonus token. Positions after
 *  the recovered token are -1 (PLACEHOLDER). seed drives the residual Gumbel-max resample.
 **/
std::vector<array> spec_verify_linear(
    const array& draft_tokens,
    const array& draft_probs,
    const array& target_probs,
    const array& bonus_tokens,
    const array& accept_u,
    uint32_t seed,
    StreamOrDevice s = {});

class SpecVerifyLinear : public Primitive {
 public:
  SpecVerifyLinear(Stream stream, uint32_t seed) : Primitive(stream), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SpecVerifyLinear"; }
  void print(std::ostream& os) override { os << "SpecVerifyLinear"; }
  bool is_equivalent(const Primitive& other) const override {
    return seed_ == static_cast<const SpecVerifyLinear&>(other).seed_;
  }

 private:
  uint32_t seed_;
};

class BeamTopkPartials : public Primitive {
 public:
  BeamTopkPartials(Stream stream, int two_bm) : Primitive(stream), two_bm_(two_bm) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BeamTopkPartials"; }
  void print(std::ostream& os) override { os << "BeamTopkPartials"; }
  bool is_equivalent(const Primitive& other) const override {
    return two_bm_ == static_cast<const BeamTopkPartials&>(other).two_bm_;
  }

 private:
  int two_bm_;
};

class BeamSelect : public Primitive {
 public:
  BeamSelect(Stream stream, int beam_width, int two_bm)
      : Primitive(stream), beam_width_(beam_width), two_bm_(two_bm) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BeamSelect"; }
  void print(std::ostream& os) override { os << "BeamSelect"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const BeamSelect&>(other);
    return beam_width_ == o.beam_width_ && two_bm_ == o.two_bm_;
  }

 private:
  int beam_width_;
  int two_bm_;
};

} // namespace mlx::core
