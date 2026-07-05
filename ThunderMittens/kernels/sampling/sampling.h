// Copyright © 2023 Apple Inc.

#pragma once

#include <optional>

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

/** typical-p (locally-typical) sampling: keep the smallest-surprise tokens |(-log p)-H| until their
 *  cumulative prob reaches typical_p, Gumbel-max sample among them. typical_p in (0, 1]. */
array typical_p_sample(
    const array& logits, float typical_p, float temperature = 1.0f, uint32_t seed = 0,
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
  float min_p_, invtemp_;
  uint32_t seed_;
};

class TypicalPSample : public Primitive {
 public:
  TypicalPSample(Stream stream, float typ_p, float invtemp, uint32_t seed)
      : Primitive(stream), typ_p_(typ_p), invtemp_(invtemp), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "TypicalPSample"; }
  void print(std::ostream& os) override { os << "TypicalPSample"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const TypicalPSample&>(other);
    return typ_p_ == o.typ_p_ && invtemp_ == o.invtemp_ && seed_ == o.seed_;
  }

 private:
  float typ_p_, invtemp_;
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

/** Speculative TREE verification (target-only rejection, TRT-LLM dynamicTree). draft_tokens (B,N-1)
 *  int; target_probs (B,N,V) f32 (dist at each node's position); retrieve_next_token/-sibling (B,N)
 *  int (first-child / next-sibling pointers, -1 = none), node 0 = root; tree_valid (B,) int (0 =
 *  no tree exists this step -> sample the target root token, accept_num=0). Returns [accept_index
 *  (B,N) int32 tree positions, accept_token (B,N) int32 token ids, accept_num (B,) int32], -1-padded. */
std::vector<array> spec_verify_tree(
    const array& draft_tokens, const array& target_probs, const array& retrieve_next_token,
    const array& retrieve_next_sibling, const array& tree_valid, uint32_t seed, StreamOrDevice s = {});

class SpecVerifyTree : public Primitive {
 public:
  SpecVerifyTree(Stream stream, uint32_t seed) : Primitive(stream), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SpecVerifyTree"; }
  void print(std::ostream& os) override { os << "SpecVerifyTree"; }
  bool is_equivalent(const Primitive& other) const override {
    return seed_ == static_cast<const SpecVerifyTree&>(other).seed_;
  }

 private:
  uint32_t seed_;
};

/** build_dynamic_tree: device-resident construction of the draft-tree pointers from a per-node parent
 *  list `parents` (B, N) int (parents[b,0] = -1 root, parents[c] < c). Returns [retrieve_next_token
 *  (B,N), retrieve_next_sibling (B,N), positions (B,N)] int32 (first-child / next-sibling pointers,
 *  -1 = none; positions[c] = depth from root). The device analogue of spec_build_tree_pointers. */
std::vector<array> build_dynamic_tree(const array& parents, StreamOrDevice s = {});

class BuildDynamicTree : public Primitive {
 public:
  explicit BuildDynamicTree(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BuildDynamicTree"; }
  void print(std::ostream& os) override { os << "BuildDynamicTree"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

/** spec_compact: gather each request's valid tokens (accepted + recovered/bonus, vlen=accepted_cnt+1)
 *  from out_tokens (B, S+1) into a packed buffer with cu_accepted offsets. Returns [packed_tokens
 *  (B*(S+1),) int32, packed_pos (B*(S+1),) int32, cu_accepted (B+1,) int32]; packed_pos[k] =
 *  seq_lens[b]+j is the absolute KV position; cu_accepted[B] is the real total, unused tail is -1. */
std::vector<array> spec_compact(
    const array& out_tokens, const array& accepted_cnt, const array& seq_lens,
    StreamOrDevice s = {});

/** spec_update_kv_meta: new_seq_lens[b] = seq_lens[b] + accepted_cnt[b] + 1. Returns (B,) int32. */
array spec_update_kv_meta(
    const array& seq_lens, const array& accepted_cnt, StreamOrDevice s = {});

class SpecCompact : public Primitive {
 public:
  explicit SpecCompact(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SpecCompact"; }
  void print(std::ostream& os) override { os << "SpecCompact"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class SpecUpdateKvMeta : public Primitive {
 public:
  explicit SpecUpdateKvMeta(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SpecUpdateKvMeta"; }
  void print(std::ostream& os) override { os << "SpecUpdateKvMeta"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

// ---- vLLM v1 ragged rejection samplers (spec_decode.metal). cu_num_draft_tokens (B+1,) int32
// with a leading 0; all ids int32; external-noise buffers (uniform_probs, inv_q). ----

/** Greedy verify (argmax match): accept while draft_id == target_argmax, else stop; all-accept
 *  appends bonus. Returns out (B, max_draft+1) int32 (cleared to -1). */
array rejection_greedy_sample(
    const array& cu_num_draft_tokens, const array& draft_token_ids, const array& target_argmax,
    const array& bonus_token_ids, int max_draft,
    const std::optional<array>& is_greedy = std::nullopt, StreamOrDevice s = {});

/** Stochastic verify: accept iff uniform <= p_target/q_draft; on reject emit the precomputed
 *  recovered_token_ids and stop; all-accept appends bonus. draft_probs optional (no_draft_probs
 *  -> q=1). Returns out (B, max_draft+1) int32. */
array rejection_random_sample(
    const array& cu_num_draft_tokens, const array& draft_token_ids, const array& target_probs,
    const array& bonus_token_ids, const array& recovered_token_ids, const array& uniform_probs,
    int max_draft, const std::optional<array>& draft_probs = std::nullopt,
    const std::optional<array>& is_greedy = std::nullopt, StreamOrDevice s = {});

/** Recovered token per draft position: argmax_v (max(0, p_target - q_draft) * inv_q[req, v]).
 *  inv_q (B, V) is the per-request exponential-race noise. Returns out (total_draft,) int32. */
array sample_recovered_tokens(
    const array& cu_num_draft_tokens, const array& draft_token_ids, const array& target_probs,
    const array& inv_q, const std::optional<array>& draft_probs = std::nullopt,
    StreamOrDevice s = {});

class RejectionSampler : public Primitive {
 public:
  // kind: 0 greedy, 1 random, 2 recovered.
  RejectionSampler(Stream stream, int kind, int s1, int no_draft_probs, int has_is_greedy)
      : Primitive(stream), kind_(kind), s1_(s1), no_draft_probs_(no_draft_probs),
        has_is_greedy_(has_is_greedy) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "RejectionSampler"; }
  void print(std::ostream& os) override { os << "RejectionSampler[" << kind_ << "]"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const RejectionSampler&>(other);
    return kind_ == o.kind_ && s1_ == o.s1_ && no_draft_probs_ == o.no_draft_probs_ &&
           has_is_greedy_ == o.has_is_greedy_;
  }

 private:
  int kind_, s1_, no_draft_probs_, has_is_greedy_;
};

// ---- EAGLE spec-decode input-prep metadata builders (spec_decode.metal). Integer, per request. ----

/** eagle_prepare_inputs_padded: rejected = num_draft>0 ? num_draft+1-valid_count : 0;
 *  token_indices_to_sample[r] = query_start_loc[r+1]-1 - rejected; num_rejected[r] = rejected.
 *  Returns [token_indices_to_sample (B,), num_rejected (B,)] int32. cu/qsl are (B+1,). */
std::vector<array> eagle_prepare_inputs_padded(
    const array& cu_num_draft_tokens, const array& valid_sampled_tokens_count,
    const array& query_start_loc, StreamOrDevice s = {});

/** eagle_prepare_next_token_padded: next seed token = last valid sampled (or backup if none /
 *  discarded). Returns [next_token_ids (B,), valid_sampled_tokens_count (B,)] int32. */
std::vector<array> eagle_prepare_next_token_padded(
    const array& sampled_token_ids, const array& discard_request_mask,
    const array& backup_next_token_ids, int vocab_size, StreamOrDevice s = {});

/** eagle_step_slot_mapping_metadata: build the paged-KV write slot for the next draft step.
 *  Returns [out_clamped_positions (input_bs,), out_slot_mapping (input_bs,), new_seq_lens (B,)]
 *  int32. positions/block_table/seq_lens are (B, ...); input_batch_size defaults to B. */
std::vector<array> eagle_step_slot_mapping_metadata(
    const array& positions, const array& block_table, const array& seq_lens, int block_size,
    int max_model_len, int pad_id, int input_batch_size = -1, StreamOrDevice s = {});

/** eagle_expand_int32: broadcast input[r] across [cu[r], cu[r+1]) with replace_from->replace_to.
 *  Returns output (total,) int32. cu is (B+1,); total is the output length. */
array eagle_expand_int32(
    const array& input, const array& cu_num_tokens, int total, int replace_from, int replace_to,
    StreamOrDevice s = {});

class EagleMeta : public Primitive {
 public:
  EagleMeta(Stream stream, int kind, int p0, int p1, int p2, int p3, int p4)
      : Primitive(stream), kind_(kind), p0_(p0), p1_(p1), p2_(p2), p3_(p3), p4_(p4) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "EagleMeta"; }
  void print(std::ostream& os) override { os << "EagleMeta[" << kind_ << "]"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const EagleMeta&>(other);
    return kind_ == o.kind_ && p0_ == o.p0_ && p1_ == o.p1_ && p2_ == o.p2_ && p3_ == o.p3_ &&
           p4_ == o.p4_;
  }

 private:
  int kind_, p0_, p1_, p2_, p3_, p4_;
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
