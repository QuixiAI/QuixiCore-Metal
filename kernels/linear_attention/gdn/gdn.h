// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  GDN / GatedDeltaNet linear attention (delta-rule recurrence). Varlen packed inputs:
 *  q/k (total_tokens, Hk, Dk), v (total_tokens, Hv, Dv), g/beta (total_tokens, Hv)
 *  (g is the decay MULTIPLIER in (0,1], not a log); cu_seqlens (R+1,) int32;
 *  state_pool (num_slots, Hv, Dv, Dk) fp32 indexed by slot_mapping (R,) int32.
 *  Dk in {64,128}; Hv % Hk == 0 (GQA). load_initial: read the pool (decode/continuation)
 *  vs start at 0 (fresh prefill). Returns [y (total_tokens, Hv, Dv), new_state_pool]
 *  (functional: the pool is clone-updated, untouched slots preserved).
 **/
std::vector<array> gdn_recur(
    const array& q, const array& k, const array& v, const array& g, const array& beta,
    const array& state_pool, const array& cu_seqlens, const array& slot_mapping,
    bool load_initial, StreamOrDevice s = {});

/**
 *  Varlen causal depthwise convolution over projected Q/K/V channels.
 *  x (total_tokens, channels), weight (channels, kernel_size), and an explicit
 *  fp32 state pool (num_slots, channels, kernel_size - 1). cu_seqlens and
 *  slot_mapping follow gdn_recur. The state stores raw projected values in
 *  oldest-to-newest order. Returns [out, new_state_pool]; out optionally has
 *  SiLU applied, and the input pool is never mutated.
 **/
std::vector<array> gdn_short_conv(
    const array& x, const array& weight, const array& state_pool,
    const array& cu_seqlens, const array& slot_mapping,
    bool load_initial = true, bool apply_silu = true, StreamOrDevice s = {});

/**
 *  Split an activated mixed Q/K/V tensor and normalize Q/K for Gated DeltaNet.
 *  mixed is an already-activated (total_tokens, 2 * Hk * Dk + Hv * Dv)
 *  tensor. The operation RMS-normalizes Q and K in fp32 and multiplies them
 *  by explicit q_scale/k_scale; V is copied while splitting. Returns [q, k, v].
 **/
std::vector<array> gdn_qkv_prepare(
    const array& mixed, int num_k_heads, int num_v_heads,
    int key_head_dim, int value_head_dim, float eps,
    float q_scale, float k_scale, StreamOrDevice s = {});

/**
 *  Convert projection logits to recurrence controls in fp32:
 *    decay = exp(-exp(A_log) * softplus(a + dt_bias))
 *    beta  = sigmoid(b)
 *  a/b are (total_tokens, Hv), A_log/dt_bias are (Hv,). Returns [decay, beta].
 **/
std::vector<array> gdn_gate_beta(
    const array& a, const array& b, const array& A_log,
    const array& dt_bias, StreamOrDevice s = {});

/**
 *  Per-value-head gated RMSNorm:
 *    out = rms_norm(y, weight, eps) * silu(z)
 *  y/z are (total_tokens, Hv, Dv), weight is (Dv,); fp32 reduction and gated
 *  product are rounded once to the input dtype.
 **/
array gdn_gated_rmsnorm(
    const array& y, const array& z, const array& weight,
    float eps = 1.0e-6f, StreamOrDevice s = {});

class GdnRecur : public Primitive {
 public:
  GdnRecur(Stream stream, bool load_initial)
      : Primitive(stream), load_initial_(load_initial) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "GdnRecur"; }
  void print(std::ostream& os) override { os << "GdnRecur"; }
  bool is_equivalent(const Primitive& other) const override {
    return load_initial_ == static_cast<const GdnRecur&>(other).load_initial_;
  }

 private:
  bool load_initial_;
};

class GdnShortConv : public Primitive {
 public:
  GdnShortConv(Stream stream, bool load_initial, bool apply_silu, int kernel_size)
      : Primitive(stream), load_initial_(load_initial), apply_silu_(apply_silu),
        kernel_size_(kernel_size) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "GdnShortConv"; }
  void print(std::ostream& os) override { os << "GdnShortConv"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const GdnShortConv&>(other);
    return load_initial_ == o.load_initial_ && apply_silu_ == o.apply_silu_ &&
           kernel_size_ == o.kernel_size_;
  }

 private:
  bool load_initial_;
  bool apply_silu_;
  int kernel_size_;
};

class GdnQkvPrepare : public Primitive {
 public:
  GdnQkvPrepare(Stream stream, int Hk, int Hv, int Dk, int Dv,
                float eps, float q_scale, float k_scale)
      : Primitive(stream), Hk_(Hk), Hv_(Hv), Dk_(Dk), Dv_(Dv), eps_(eps),
        q_scale_(q_scale), k_scale_(k_scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "GdnQkvPrepare"; }
  void print(std::ostream& os) override { os << "GdnQkvPrepare"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const GdnQkvPrepare&>(other);
    return Hk_ == o.Hk_ && Hv_ == o.Hv_ && Dk_ == o.Dk_ && Dv_ == o.Dv_ &&
           eps_ == o.eps_ && q_scale_ == o.q_scale_ && k_scale_ == o.k_scale_;
  }

 private:
  int Hk_, Hv_, Dk_, Dv_;
  float eps_, q_scale_, k_scale_;
};

class GdnGateBeta : public Primitive {
 public:
  explicit GdnGateBeta(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "GdnGateBeta"; }
  void print(std::ostream& os) override { os << "GdnGateBeta"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class GdnGatedRmsNorm : public Primitive {
 public:
  GdnGatedRmsNorm(Stream stream, int dim, float eps)
      : Primitive(stream), dim_(dim), eps_(eps) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "GdnGatedRmsNorm"; }
  void print(std::ostream& os) override { os << "GdnGatedRmsNorm"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const GdnGatedRmsNorm&>(other);
    return dim_ == o.dim_ && eps_ == o.eps_;
  }

 private:
  int dim_;
  float eps_;
};

} // namespace mlx::core
