// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Fused per-head QK-RMSNorm + RoPE over a packed QKV buffer (Qwen3 / gpt-oss pattern).
 *  qkv (T, (Hq+Hk+Hv)*D) bf16; every Q/K head is RMSNormed over D (q_weight/k_weight (D,)),
 *  rotated at positions[token] (cos/sin (max_pos, D/2)); V heads copied through.
 *  interleaved: false = NeoX split-half, true = GPT-J pairs. gemma: weight as (1+w).
 *  Full rotary only (rotary_dim == D); D in {64, 128, 256}. Returns the new qkv.
 **/
array qk_norm_rope(
    const array& qkv,
    const array& q_weight,
    const array& k_weight,
    const array& cosb,
    const array& sinb,
    const array& positions,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    float eps = 1e-6f,
    bool interleaved = false,
    bool gemma = false,
    StreamOrDevice s = {});

/**
 *  qk_norm_rope with a fused f16 KV split-store. Same fused per-head QK-RMSNorm + RoPE, but
 *  the normed+roped result is split into the three tensors attention consumes, casting the KV
 *  halves to f16 in the same pass: returns [q_out (T, Hq*D) bf16, k_out (T, Hk*D) f16,
 *  v_out (T, Hv*D) f16]. V heads are copied through. Same D/interleaved/gemma semantics.
 **/
std::vector<array> qk_norm_rope_kv_f16(
    const array& qkv,
    const array& q_weight,
    const array& k_weight,
    const array& cosb,
    const array& sinb,
    const array& positions,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    float eps = 1e-6f,
    bool interleaved = false,
    bool gemma = false,
    StreamOrDevice s = {});

class QkNormRope : public Primitive {
 public:
  QkNormRope(Stream stream, int hq, int hk, int hv, float eps, bool interleaved, bool gemma)
      : Primitive(stream), hq_(hq), hk_(hk), hv_(hv), eps_(eps),
        interleaved_(interleaved), gemma_(gemma) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QkNormRope"; }
  void print(std::ostream& os) override { os << "QkNormRope"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const QkNormRope&>(other);
    return hq_ == o.hq_ && hk_ == o.hk_ && hv_ == o.hv_ && eps_ == o.eps_ &&
           interleaved_ == o.interleaved_ && gemma_ == o.gemma_;
  }

 private:
  int hq_;
  int hk_;
  int hv_;
  float eps_;
  bool interleaved_;
  bool gemma_;
};

class QkNormRopeKvF16 : public Primitive {
 public:
  QkNormRopeKvF16(Stream stream, int hq, int hk, int hv, float eps, bool interleaved, bool gemma)
      : Primitive(stream), hq_(hq), hk_(hk), hv_(hv), eps_(eps),
        interleaved_(interleaved), gemma_(gemma) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  const char* name() const { return "QkNormRopeKvF16"; }
  void print(std::ostream& os) override { os << "QkNormRopeKvF16"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const QkNormRopeKvF16&>(other);
    return hq_ == o.hq_ && hk_ == o.hk_ && hv_ == o.hv_ && eps_ == o.eps_ &&
           interleaved_ == o.interleaved_ && gemma_ == o.gemma_;
  }

 private:
  int hq_;
  int hk_;
  int hv_;
  float eps_;
  bool interleaved_;
  bool gemma_;
};

} // namespace mlx::core
