// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Long-context paged decode attention (vLLM-v2 partition/reduce). Splits each
 *  (head, batch) query across KV-sequence partitions, computes a local softmax
 *  per partition, then merges via the log-sum-exp rescaling trick. GQA/MQA aware
 *  (num_q_heads may be a multiple of the cache's num_kv_heads).
 *
 *  q : (batch, num_heads, D); caches : (num_blocks, block_size, num_kv_heads, D).
 *  D ∈ {64,128}. Returns out : (batch, num_heads, D). partition_size must be a
 *  positive multiple of block_size.
 **/
array paged_attention_v2(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    int partition_size = 512,
    int window = 0,
    float softcap = 0.0f,
    const std::optional<array>& sinks = std::nullopt,
    StreamOrDevice s = {});

/**
 *  Long-context paged decode attention over an fp8 (uint8 e4m3/e5m2) paged cache, dequantized
 *  on read with per-head k_scale/v_scale ((num_kv_heads,) arrays). Same partition/reduce
 *  structure as paged_attention_v2 (the reduce kernel is shared). fmt: 0=e4m3, 1=e5m2.
 **/
array paged_attention_v2_fp8(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& k_scale,
    const array& v_scale,
    float scale = 0.0f,
    int partition_size = 512,
    int fmt = 0,
    int window = 0,
    float softcap = 0.0f,
    const std::optional<array>& sinks = std::nullopt,
    StreamOrDevice s = {});

/**
 *  Cascade / shared-prefix attention. All B requests attend a single SHARED, CONTIGUOUS prefix KV
 *  (prefix_k/prefix_v (prefix_len, H_KV, D)) — e.g. a common system prompt — plus each request's own
 *  paged suffix (key_cache/value_cache + block_table + context_lens, as in paged_attention_v2). The
 *  prefix and suffix are attended independently, their (m,l,o) states concatenated along the
 *  partition axis, and folded by the shared paged_attention_reduce log-sum-exp merge — exactly the
 *  full softmax over [prefix ++ suffix]. q/out (B,H,D); D in {64,128}; GQA/MQA supported.
 **/
array cascade_attention(
    const array& q,
    const array& prefix_k,
    const array& prefix_v,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    int partition_size = 512,
    StreamOrDevice s = {});

// N-level cascade: a list of shared-prefix levels + the paged suffix, merged in one reduce.
array cascade_attention_multi(
    const array& q,
    const std::vector<array>& prefix_ks,
    const std::vector<array>& prefix_vs,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    int partition_size = 512,
    StreamOrDevice s = {});

// --- internal primitives (not bound directly) ---

// fp8 cascade over a uint8 shared prefix (fp8 prefix partition + regular paged suffix).
array cascade_attention_fp8(
    const array& q,
    const array& prefix_k,
    const array& prefix_v,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& k_scale,
    const array& v_scale,
    float scale = 0.0f,
    int partition_size = 512,
    int fmt = 0,
    StreamOrDevice s = {});

class CascadePrefixPartitionFp8 : public Primitive {
 public:
  CascadePrefixPartitionFp8(Stream stream, float scale, int num_partitions, int partition_size,
                            int fmt)
      : Primitive(stream), scale_(scale), num_partitions_(num_partitions),
        partition_size_(partition_size), fmt_(fmt) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "CascadePrefixPartitionFp8"; }
  void print(std::ostream& os) override { os << "CascadePrefixPartitionFp8"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const CascadePrefixPartitionFp8&>(other);
    return scale_ == o.scale_ && num_partitions_ == o.num_partitions_ &&
        partition_size_ == o.partition_size_ && fmt_ == o.fmt_;
  }

 private:
  float scale_;
  int num_partitions_, partition_size_, fmt_;
};

class CascadePrefixPartition : public Primitive {
 public:
  CascadePrefixPartition(Stream stream, float scale, int num_partitions, int partition_size)
      : Primitive(stream),
        scale_(scale),
        num_partitions_(num_partitions),
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
  const char* name() const { return "CascadePrefixPartition"; }
  void print(std::ostream& os) override { os << "CascadePrefixPartition"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const CascadePrefixPartition&>(other);
    return scale_ == o.scale_ && num_partitions_ == o.num_partitions_ &&
        partition_size_ == o.partition_size_;
  }

 private:
  float scale_;
  int num_partitions_;
  int partition_size_;
};

class PagedAttentionV2Partition : public Primitive {
 public:
  PagedAttentionV2Partition(Stream stream, float scale, int num_partitions, int partition_size,
                            int window = 0, float softcap = 0.0f)
      : Primitive(stream),
        scale_(scale),
        num_partitions_(num_partitions),
        partition_size_(partition_size),
        window_(window),
        softcap_(softcap) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PagedAttentionV2Partition"; }
  void print(std::ostream& os) override { os << "PagedAttentionV2Partition"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PagedAttentionV2Partition&>(other);
    return scale_ == o.scale_ && num_partitions_ == o.num_partitions_ &&
        partition_size_ == o.partition_size_ && window_ == o.window_ &&
        softcap_ == o.softcap_;
  }

 private:
  float scale_;
  int num_partitions_;
  int partition_size_;
  int window_;
  float softcap_;
};

class PagedAttentionV2PartitionFp8 : public Primitive {
 public:
  PagedAttentionV2PartitionFp8(
      Stream stream, float scale, int num_partitions, int partition_size, int fmt, int window = 0,
      float softcap = 0.0f)
      : Primitive(stream),
        scale_(scale),
        num_partitions_(num_partitions),
        partition_size_(partition_size),
        fmt_(fmt),
        window_(window),
        softcap_(softcap) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PagedAttentionV2PartitionFp8"; }
  void print(std::ostream& os) override { os << "PagedAttentionV2PartitionFp8"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PagedAttentionV2PartitionFp8&>(other);
    return scale_ == o.scale_ && num_partitions_ == o.num_partitions_ &&
        partition_size_ == o.partition_size_ && fmt_ == o.fmt_ && window_ == o.window_ &&
        softcap_ == o.softcap_;
  }

 private:
  float scale_;
  int num_partitions_;
  int partition_size_;
  int fmt_;
  int window_;
  float softcap_;
};

class PagedAttentionV2Reduce : public Primitive {
 public:
  // has_sink: inputs[3] is the per-head (H,) fp32 gpt-oss attention-sink buffer (else a
  // placeholder). The sink enters the merged softmax denominator here, exactly once.
  explicit PagedAttentionV2Reduce(Stream stream, bool has_sink = false)
      : Primitive(stream), has_sink_(has_sink) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PagedAttentionV2Reduce"; }
  void print(std::ostream& os) override { os << "PagedAttentionV2Reduce"; }
  bool is_equivalent(const Primitive& other) const override {
    return has_sink_ == static_cast<const PagedAttentionV2Reduce&>(other).has_sink_;
  }

 private:
  bool has_sink_;
};

} // namespace mlx::core
