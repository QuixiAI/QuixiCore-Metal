// Copyright © 2023 Apple Inc.

#pragma once

#include <string>
#include <utility>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Fused LM-head + sampling: select a decode token per row of h WITHOUT materializing the (T, V)
 *  logits. h is (T, K), W is (V, K) row-major, both fp16/bf16/f32 (same dtype). mode:
 *    0 = argmax (greedy), 1 = categorical (Gumbel-max over softmax(logits/temperature)),
 *    2 = top-k (Gumbel-max over the k highest logits). bias is (V,) or empty. temperature > 0.
 *  Returns (T,) int32 token ids. The Gumbel noise is indexed by the GLOBAL vocab id so the fused
 *  draw equals the unfused sampler on the same logits + seed.
 **/
array lm_head_sample(
    const array& h,
    const array& W,
    const array& bias,
    int mode,
    int k,
    float temperature,
    uint32_t seed,
    StreamOrDevice s = {});

/**
 *  Fused LM-head + sampling over QUANTIZED weights (dequantized on read). Wq is the packed weight
 *  tensor for format `fmt` (q8_0/q4_0/q6_K/nvfp4), K is the full hidden dim.
 *  Returns (T,) int32. h is fp16/bf16/f32; q6_K is fp32-only.
 **/
array lm_head_sample_q(
    const array& h,
    const array& Wq,
    const array& bias,
    int V,
    int K,
    const std::string& fmt,
    int mode,
    int k,
    float temperature,
    uint32_t seed,
    float top_p = 0.0f,   // mode 3 (topp): nucleus threshold; k = over-selected candidate cap
    StreamOrDevice s = {});

/** Quantized LM-head + exact beam-search advance without materializing (B*BM,V)
 *  logits. Wq is q4_0/q8_0/nvfp4 packed (V,K/block_k,block_bytes); h is (B*BM,K),
 *  cum_log_probs is (B,BM), and BM <= 16. Returns next token, parent beam,
 *  and updated cumulative log-probability, each (B,BM). */
std::vector<array> lm_head_beam_advance(
    const array& h,
    const array& Wq,
    const array& bias,
    const array& cum_log_probs,
    int beam_width,
    const std::string& format = "q4_0",
    StreamOrDevice s = {});

/** Dense LM head with a row-conditioned grammar mask. Returns
 *  [selected_token (T,) int32, selected_logprob (T,) fp32]. `forbidden` is a
 *  uint8 (V,V) matrix indexed by (previous_token, candidate_token). */
std::vector<array> lm_head_constrained(
    const array& h,
    const array& W,
    const array& bias,
    const array& forbidden,
    const array& previous,
    int eos_id = -1,
    bool forbid_eos = false,
    StreamOrDevice s = {});

/** Fused dense or q4_0/q8_0/q6_K/nvfp4 LM-head projection with a packed
 *  allow bitmask. Returns top-k ids and
 *  log-probabilities, both (T, topk).  `normalize_allowed=true` normalizes over
 *  legal tokens; false normalizes over the full vocabulary before masking. */
std::vector<array> lm_head_masked(
    const array& h,
    const array& W,
    const array& bias,
    const array& allow_mask,
    const std::string& format = "",
    int topk = 1,
    bool normalize_allowed = true,
    StreamOrDevice s = {});

/** Sparse dense or q4_0/q8_0/q6_K/nvfp4 candidate LM head. `candidate_ids`
 *  is flat and `offsets` has T+1
 *  entries. Candidate ids within each row must be unique. */
std::vector<array> lm_head_candidates(
    const array& h,
    const array& W,
    const array& bias,
    const array& candidate_ids,
    const array& offsets,
    const std::string& format = "",
    int topk = 1,
    StreamOrDevice s = {});

class LmHeadMaskedPartials : public Primitive {
 public:
  LmHeadMaskedPartials(Stream stream, std::string format, int vocab, int hidden,
                       int tile_v, int topk, bool use_bias, bool normalize_allowed)
      : Primitive(stream), format_(std::move(format)), vocab_(vocab), hidden_(hidden),
        tile_v_(tile_v), topk_(topk), use_bias_(use_bias),
        normalize_allowed_(normalize_allowed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadMaskedPartials"; }
  void print(std::ostream& os) override { os << "LmHeadMaskedPartials"; }
  bool is_equivalent(const Primitive& other) const override;
 private:
  std::string format_;
  int vocab_;
  int hidden_;
  int tile_v_;
  int topk_;
  bool use_bias_;
  bool normalize_allowed_;
};

class LmHeadMaskedReduce : public Primitive {
 public:
  LmHeadMaskedReduce(Stream stream, int topk) : Primitive(stream), topk_(topk) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadMaskedReduce"; }
  void print(std::ostream& os) override { os << "LmHeadMaskedReduce"; }
  bool is_equivalent(const Primitive& other) const override;
 private:
  int topk_;
};

class LmHeadCandidates : public Primitive {
 public:
  LmHeadCandidates(Stream stream, std::string format, int vocab, int hidden,
                   int topk, bool use_bias)
      : Primitive(stream), format_(std::move(format)), vocab_(vocab), hidden_(hidden),
        topk_(topk), use_bias_(use_bias) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadCandidates"; }
  void print(std::ostream& os) override { os << "LmHeadCandidates"; }
  bool is_equivalent(const Primitive& other) const override;
 private:
  std::string format_;
  int vocab_;
  int hidden_;
  int topk_;
  bool use_bias_;
};

class LmHeadConstrainedPartials : public Primitive {
 public:
  LmHeadConstrainedPartials(
      Stream stream, int vocab, int hidden, int use_bias, int eos_id, bool forbid_eos)
      : Primitive(stream), vocab_(vocab), hidden_(hidden), use_bias_(use_bias),
        eos_id_(eos_id), forbid_eos_(forbid_eos) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadConstrainedPartials"; }
  void print(std::ostream& os) override { os << "LmHeadConstrainedPartials"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  int vocab_;
  int hidden_;
  int use_bias_;
  int eos_id_;
  bool forbid_eos_;
};

class LmHeadConstrainedReduce : public Primitive {
 public:
  explicit LmHeadConstrainedReduce(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadConstrainedReduce"; }
  void print(std::ostream& os) override { os << "LmHeadConstrainedReduce"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class LmHeadArgcatPartialsQ : public Primitive {
 public:
  LmHeadArgcatPartialsQ(Stream stream, int use_gumbel, float invtemp, uint32_t seed, int use_bias,
                        int tile_v, int V, int K, const std::string& fmt)
      : Primitive(stream), use_gumbel_(use_gumbel), invtemp_(invtemp), seed_(seed),
        use_bias_(use_bias), tile_v_(tile_v), V_(V), K_(K), fmt_(fmt) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadArgcatPartialsQ"; }
  void print(std::ostream& os) override { os << "LmHeadArgcatPartialsQ"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const LmHeadArgcatPartialsQ&>(other);
    return use_gumbel_ == o.use_gumbel_ && invtemp_ == o.invtemp_ && seed_ == o.seed_ &&
           use_bias_ == o.use_bias_ && tile_v_ == o.tile_v_ && V_ == o.V_ && K_ == o.K_ &&
           fmt_ == o.fmt_;
  }

 private:
  int use_gumbel_;
  float invtemp_;
  uint32_t seed_;
  int use_bias_;
  int tile_v_;
  int V_;
  int K_;
  std::string fmt_;
};

class LmHeadTopkPartialsQ : public Primitive {
 public:
  LmHeadTopkPartialsQ(Stream stream, int topk, int use_bias, int tile_v, int V, int K,
                      const std::string& fmt)
      : Primitive(stream), topk_(topk), use_bias_(use_bias), tile_v_(tile_v), V_(V), K_(K),
        fmt_(fmt) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadTopkPartialsQ"; }
  void print(std::ostream& os) override { os << "LmHeadTopkPartialsQ"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const LmHeadTopkPartialsQ&>(other);
    return topk_ == o.topk_ && use_bias_ == o.use_bias_ && tile_v_ == o.tile_v_ && V_ == o.V_ &&
           K_ == o.K_ && fmt_ == o.fmt_;
  }

 private:
  int topk_;
  int use_bias_;
  int tile_v_;
  int V_;
  int K_;
  std::string fmt_;
};

// Like LmHeadTopkPartialsQ but ALSO emits a per-tile tempered logsumexp (3rd output) so the top-p
// reduce can build the true full-vocab normalizer. Carries invtemp (the softmax normalizer is
// temperature-dependent).
class LmHeadToppPartialsQ : public Primitive {
 public:
  LmHeadToppPartialsQ(Stream stream, int topk, int use_bias, int tile_v, int V, int K, float invtemp,
                      const std::string& fmt, int rows_per_tg = 1)
      : Primitive(stream), topk_(topk), use_bias_(use_bias), tile_v_(tile_v), V_(V), K_(K),
        invtemp_(invtemp), fmt_(fmt), rows_per_tg_(rows_per_tg) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadToppPartialsQ"; }
  void print(std::ostream& os) override { os << "LmHeadToppPartialsQ"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const LmHeadToppPartialsQ&>(other);
    return topk_ == o.topk_ && use_bias_ == o.use_bias_ && tile_v_ == o.tile_v_ && V_ == o.V_ &&
           K_ == o.K_ && invtemp_ == o.invtemp_ && fmt_ == o.fmt_ &&
           rows_per_tg_ == o.rows_per_tg_;
  }

 private:
  int topk_;
  int use_bias_;
  int tile_v_;
  int V_;
  int K_;
  float invtemp_;
  std::string fmt_;
  int rows_per_tg_;
};

class LmHeadArgcatPartials : public Primitive {
 public:
  LmHeadArgcatPartials(Stream stream, int use_gumbel, float invtemp, uint32_t seed, int use_bias,
                       int tile_v)
      : Primitive(stream), use_gumbel_(use_gumbel), invtemp_(invtemp), seed_(seed),
        use_bias_(use_bias), tile_v_(tile_v) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadArgcatPartials"; }
  void print(std::ostream& os) override { os << "LmHeadArgcatPartials"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const LmHeadArgcatPartials&>(other);
    return use_gumbel_ == o.use_gumbel_ && invtemp_ == o.invtemp_ && seed_ == o.seed_ &&
           use_bias_ == o.use_bias_ && tile_v_ == o.tile_v_;
  }

 private:
  int use_gumbel_;
  float invtemp_;
  uint32_t seed_;
  int use_bias_;
  int tile_v_;
};

class LmHeadArgcatReduce : public Primitive {
 public:
  explicit LmHeadArgcatReduce(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadArgcatReduce"; }
  void print(std::ostream& os) override { os << "LmHeadArgcatReduce"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class LmHeadTopkPartials : public Primitive {
 public:
  LmHeadTopkPartials(Stream stream, int topk, int use_bias, int tile_v)
      : Primitive(stream), topk_(topk), use_bias_(use_bias), tile_v_(tile_v) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadTopkPartials"; }
  void print(std::ostream& os) override { os << "LmHeadTopkPartials"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const LmHeadTopkPartials&>(other);
    return topk_ == o.topk_ && use_bias_ == o.use_bias_ && tile_v_ == o.tile_v_;
  }

 private:
  int topk_;
  int use_bias_;
  int tile_v_;
};

class LmHeadTopkReduce : public Primitive {
 public:
  LmHeadTopkReduce(Stream stream, int topk, float invtemp, uint32_t seed)
      : Primitive(stream), topk_(topk), invtemp_(invtemp), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadTopkReduce"; }
  void print(std::ostream& os) override { os << "LmHeadTopkReduce"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const LmHeadTopkReduce&>(other);
    return topk_ == o.topk_ && invtemp_ == o.invtemp_ && seed_ == o.seed_;
  }

 private:
  int topk_;
  float invtemp_;
  uint32_t seed_;
};

// Top-p (nucleus) reduce over the over-selected candidate pool (feeds off LmHeadTopkPartials(Q)).
class LmHeadToppReduce : public Primitive {
 public:
  LmHeadToppReduce(Stream stream, int topk, float p, float invtemp, uint32_t seed)
      : Primitive(stream), topk_(topk), p_(p), invtemp_(invtemp), seed_(seed) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadToppReduce"; }
  void print(std::ostream& os) override { os << "LmHeadToppReduce"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const LmHeadToppReduce&>(other);
    return topk_ == o.topk_ && p_ == o.p_ && invtemp_ == o.invtemp_ && seed_ == o.seed_;
  }

 private:
  int topk_;
  float p_, invtemp_;
  uint32_t seed_;
};

class LmHeadBeamReduce : public Primitive {
 public:
  LmHeadBeamReduce(Stream stream, int two_bm)
      : Primitive(stream), two_bm_(two_bm) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LmHeadBeamReduce"; }
  void print(std::ostream& os) override { os << "LmHeadBeamReduce"; }
  bool is_equivalent(const Primitive& other) const override {
    return typeid(*this) == typeid(other) &&
        two_bm_ == static_cast<const LmHeadBeamReduce&>(other).two_bm_;
  }

 private:
  int two_bm_;
};

} // namespace mlx::core
