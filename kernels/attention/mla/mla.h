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

/**
 *  DeepSeek-V4/V3.2 packed MLA KV-insert (P3). The 512-wide latent [448 NoPE | 64 RoPE] is packed:
 *  NoPE → e4m3 fp8 with a per-64-block UE8M0 (power-of-2) scale, RoPE → interleaved RoPE bf16.
 *  Returns (data_cache uint8 (…, 576) = 448 codes ‖ 128 rope bytes, scale_cache uint8 (…, 8) = 7
 *  UE8M0 bytes + pad). Clone-then-insert. Dequant: e4m3_decode(code) * 2^(scale_byte-127).
 *
 *  kv : (num_tokens, 512).  cos/sin : (max_pos, 32).  positions/slot_mapping : (num_tokens,).
 *  data_cache/scale_cache : the paged caches to update (num_blocks, block_size, {576, 8}) uint8.
 **/
std::vector<array> mla_kv_insert_fp8(
    const array& kv,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& slot_mapping,
    const array& data_cache,
    const array& scale_cache,
    StreamOrDevice s = {});

/**
 *  DeepSeek MLA absorb-path latent flash-decode (P4). The absorbed query q = [ql_nope ‖ q_pe]
 *  (LATENT+rope wide; ql_nope = q_nope @ W_UK_T applied by the caller) attends against the shared
 *  paged latent cache kv_cache[nb, bs, LATENT+rope] (MQA): score over the full width, value
 *  accumulate over the LATENT part only. Returns o (batch, num_heads, LATENT); the caller then
 *  up-projects with W_UV. Algebraically equal to the MHA (up-projected) attention.
 *
 *  q : (batch, num_heads, LATENT+rope) bf16.  kv_cache : (num_blocks, block_size, LATENT+rope) bf16.
 *  block_table : (batch, max_blocks) int32.  context_lens : (batch,) int32.
 **/
array mla_decode(
    const array& q,
    const array& kv_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    StreamOrDevice s = {});

/**
 *  DeepSeek-V4 dense latent decode over the UE8M0-packed cache (P4a). q (B, N, 512) [the absorbed
 *  512-wide query] attends the packed cache (data_cache (nb,bs,576) uint8 + scale_cache (nb,bs,8)
 *  uint8) with dequant-on-read; score and value both over the full 512, scale = 512^-0.5. Returns
 *  o (B, N, 512) bf16 (the inverse-RoPE of o[448:512] and wo_a/wo_b are caller-applied).
 **/
array mla_decode_fp8(
    const array& q,
    const array& data_cache,
    const array& scale_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    StreamOrDevice s = {});

/**
 *  DeepSeek-V4 sparse latent decode (P4b). Same as mla_decode_fp8 but each query attends only the
 *  caller-provided token positions indices[b, 0:topk_length[b]] (the indexer's top-k set).
 *  indices (batch, max_topk) int32; topk_length (batch,) int32. Returns o (B, N, 512).
 **/
array mla_decode_fp8_sparse(
    const array& q,
    const array& data_cache,
    const array& scale_cache,
    const array& block_table,
    const array& indices,
    const array& topk_length,
    float scale = 0.0f,
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

class MlaDecode : public Primitive {
 public:
  MlaDecode(Stream stream, float scale) : Primitive(stream), scale_(scale) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MlaDecode"; }
  void print(std::ostream& os) override { os << "MlaDecode"; }
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const MlaDecode&>(other).scale_;
  }

 private:
  float scale_;
};

// Partitioned multi-head MLA decode (P4v2): emits paged-v2-style partials
// (tmp_out, max_logits, exp_sums); combined by PagedAttentionV2Reduce.
class MlaDecodePartition : public Primitive {
 public:
  MlaDecodePartition(Stream stream, float scale, int num_partitions, int partition_size)
      : Primitive(stream), scale_(scale), num_partitions_(num_partitions),
        partition_size_(partition_size) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MlaDecodePartition"; }
  void print(std::ostream& os) override { os << "MlaDecodePartition"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MlaDecodePartition&>(other);
    return scale_ == o.scale_ && num_partitions_ == o.num_partitions_ &&
           partition_size_ == o.partition_size_;
  }

 private:
  float scale_;
  int num_partitions_;
  int partition_size_;
};

// Partitioned fp8 MLA decode (dense OR sparse, distinguished by input count: 5 = dense
// (q, data, scale, table, context_lens), 6 = sparse (…, indices, topk_length)). Emits
// paged-v2-style partials combined by PagedAttentionV2Reduce.
class MlaDecodeFp8Partition : public Primitive {
 public:
  MlaDecodeFp8Partition(Stream stream, float scale, int num_partitions, int partition_size)
      : Primitive(stream), scale_(scale), num_partitions_(num_partitions),
        partition_size_(partition_size) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MlaDecodeFp8Partition"; }
  void print(std::ostream& os) override { os << "MlaDecodeFp8Partition"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MlaDecodeFp8Partition&>(other);
    return scale_ == o.scale_ && num_partitions_ == o.num_partitions_ &&
           partition_size_ == o.partition_size_;
  }

 private:
  float scale_;
  int num_partitions_;
  int partition_size_;
};

class MlaDecodeFp8 : public Primitive {
 public:
  MlaDecodeFp8(Stream stream, float scale) : Primitive(stream), scale_(scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MlaDecodeFp8"; }
  void print(std::ostream& os) override { os << "MlaDecodeFp8"; }
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const MlaDecodeFp8&>(other).scale_;
  }

 private:
  float scale_;
};

class MlaDecodeFp8Sparse : public Primitive {
 public:
  MlaDecodeFp8Sparse(Stream stream, float scale) : Primitive(stream), scale_(scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MlaDecodeFp8Sparse"; }
  void print(std::ostream& os) override { os << "MlaDecodeFp8Sparse"; }
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const MlaDecodeFp8Sparse&>(other).scale_;
  }

 private:
  float scale_;
};

class MlaKvInsertFp8 : public Primitive {
 public:
  explicit MlaKvInsertFp8(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MlaKvInsertFp8"; }
  void print(std::ostream& os) override { os << "MlaKvInsertFp8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
