// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Fused RoPE (split-half / GPT-NeoX) + paged-KV insert. Rotates K, then writes
 *  rotated K and (unrotated) V into the paged caches at slot_mapping[token].
 *  Returns the two updated caches (the un-inserted slots are copied through).
 *
 *  k, v      : (num_tokens, num_kv_heads, D)  bf16
 *  cos, sin  : (P, D/2)                        bf16 (precomputed)
 *  positions : (num_tokens,)                   int  (per-token RoPE position)
 *  slot_mapping : (num_tokens,)                int  (paged slot; < 0 skips)
 *  key_cache, value_cache : (num_blocks, block_size, num_kv_heads, D) bf16
 *  D must be 64 or 128.
 **/
std::vector<array> rope_kv_insert(
    const array& k,
    const array& v,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& slot_mapping,
    const array& key_cache,
    const array& value_cache,
    StreamOrDevice s = {});

/**
 *  Same as rope_kv_insert but RMSNorms K over the head dim before RoPE (fused Q/K-norm +
 *  RoPE + insert). norm_weight is (D,) bf16; gemma=true uses (1+weight). Returns updated caches.
 **/
std::vector<array> rope_kv_insert_norm(
    const array& k, const array& v, const array& cos, const array& sin, const array& positions,
    const array& slot_mapping, const array& key_cache, const array& value_cache,
    const array& norm_weight, float eps, bool gemma, StreamOrDevice s = {});

/**
 *  Q-path: rotate (split-half) and optionally weighted-RMSNorm Q into a contiguous q_out (Q is
 *  not paged). q (num_tokens, num_q_heads, D); cos/sin (P, D/2); positions (num_tokens,).
 *  do_norm=true applies RMSNorm over D with norm_weight (D,) (gemma uses 1+weight). {f16,bf16,f32}.
 **/
array rope_q(
    const array& q,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& norm_weight,
    bool do_norm,
    bool gemma,
    float eps = 1e-6f,
    StreamOrDevice s = {});

class RopeQ : public Primitive {
 public:
  RopeQ(Stream stream, bool do_norm, bool gemma, float eps)
      : Primitive(stream), do_norm_(do_norm), gemma_(gemma), eps_(eps) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "RopeQ"; }
  void print(std::ostream& os) override { os << "RopeQ"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const RopeQ&>(other);
    return do_norm_ == o.do_norm_ && gemma_ == o.gemma_ && eps_ == o.eps_;
  }

 private:
  bool do_norm_, gemma_;
  float eps_;
};

class RopeKvInsertNorm : public Primitive {
 public:
  RopeKvInsertNorm(Stream stream, float eps, bool gemma)
      : Primitive(stream), eps_(eps), gemma_(gemma) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "RopeKvInsertNorm"; }
  void print(std::ostream& os) override { os << "RopeKvInsertNorm"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const RopeKvInsertNorm&>(other);
    return eps_ == o.eps_ && gemma_ == o.gemma_;
  }

 private:
  float eps_;
  bool gemma_;
};

class RopeKvInsert : public Primitive {
 public:
  explicit RopeKvInsert(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&,
      const std::vector<array>&,
      const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&,
      const std::vector<array>&,
      const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&,
      const std::vector<int>&) override;
  const char* name() const { return "RopeKvInsert"; }
  void print(std::ostream& os) override { os << "RopeKvInsert"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
