// Copyright © 2024 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  DeepSeek MLA Q-path preprocessing (P1). Per (token, head): optional RMSNorm over the full
 *  head dim (norm_mode 0=none, 1=rms no-weight, 2=rms + weight), then GPT-J *interleaved* RoPE on
 *  the last `rope_dim` dims (the `nope` prefix passes through), bf16 out.
 *
 *  q : (num_tokens, num_heads, head_dim) — head_dim = nope_dim + rope_dim, head_dim % 64 == 0.
 *  cos/sin : (max_pos, rope_dim/2) bf16.  positions : (num_tokens,) int32.
 *  norm_weight : (head_dim,) bf16 (read only when norm_mode == 2; pass a dummy otherwise).
 *  Returns q_out, same shape as q.
 **/
array mla_q_norm_rope(
    const array& q,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& norm_weight,
    int num_heads,
    int nope_dim,
    int rope_dim,
    int norm_mode,
    float eps = 1e-6f,
    StreamOrDevice s = {});

/**
 *  DeepSeek MLA classic KV-insert (P2). Writes the compressed latent kv_c (optionally
 *  kv_a-RMSNormed, norm_mode 0=none/2=weighted) + interleaved-RoPE'd k_pe into a paged bf16 cache
 *  kv_cache[num_blocks, block_size, LATENT + rope_dim] (MQA — one latent per token). Clone-then-
 *  insert: the mapped slots are overwritten; the rest keep the input cache's values.
 *
 *  kv_c : (num_tokens, LATENT), LATENT % 64 == 0.  k_pe : (num_tokens, rope_dim), rope_dim/2 <= 32.
 *  cos/sin : (max_pos, rope_dim/2).  positions : (num_tokens,) int32.  slot_mapping : (num_tokens,) int64.
 *  norm_weight : (LATENT,) bf16 (read only when norm_mode == 2). Returns the updated kv_cache.
 **/
array mla_kv_insert(
    const array& kv_c,
    const array& k_pe,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& slot_mapping,
    const array& kv_cache,
    const array& norm_weight,
    int rope_dim,
    int norm_mode,
    float eps = 1e-6f,
    StreamOrDevice s = {});

class MlaQNormRope : public Primitive {
 public:
  MlaQNormRope(Stream stream, int num_heads, int nope_dim, int rope_dim, int norm_mode, float eps)
      : Primitive(stream),
        num_heads_(num_heads),
        nope_dim_(nope_dim),
        rope_dim_(rope_dim),
        norm_mode_(norm_mode),
        eps_(eps) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MlaQNormRope"; }
  void print(std::ostream& os) override { os << "MlaQNormRope"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MlaQNormRope&>(other);
    return num_heads_ == o.num_heads_ && nope_dim_ == o.nope_dim_ && rope_dim_ == o.rope_dim_ &&
        norm_mode_ == o.norm_mode_ && eps_ == o.eps_;
  }

 private:
  int num_heads_, nope_dim_, rope_dim_, norm_mode_;
  float eps_;
};

class MlaKvInsert : public Primitive {
 public:
  MlaKvInsert(Stream stream, int rope_dim, int norm_mode, float eps)
      : Primitive(stream), rope_dim_(rope_dim), norm_mode_(norm_mode), eps_(eps) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MlaKvInsert"; }
  void print(std::ostream& os) override { os << "MlaKvInsert"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MlaKvInsert&>(other);
    return rope_dim_ == o.rope_dim_ && norm_mode_ == o.norm_mode_ && eps_ == o.eps_;
  }

 private:
  int rope_dim_, norm_mode_;
  float eps_;
};

} // namespace mlx::core
