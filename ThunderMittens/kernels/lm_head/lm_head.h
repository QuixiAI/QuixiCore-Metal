// Copyright © 2023 Apple Inc.

#pragma once

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
 *  tensor for format `fmt` (q8_0/q4_0), K is the full hidden dim. mode 0 = argmax, 1 = categorical
 *  (topk not supported for the quant path). Returns (T,) int32. h fp16/bf16/f32.
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

} // namespace mlx::core
