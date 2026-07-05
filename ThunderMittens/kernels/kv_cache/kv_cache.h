// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> kv_cache_scatter(
    const array& key,
    const array& value,
    const array& slot_mapping,
    int num_blocks,
    int block_size,
    StreamOrDevice s = {});

std::vector<array> kv_cache_gather(
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& cu_seq_lens,
    int num_tokens,
    StreamOrDevice s = {});

std::vector<array> kv_cache_copy_blocks(
    const array& key_cache,
    const array& value_cache,
    const array& block_mapping,
    StreamOrDevice s = {});

// Build the (src,dst) block-copy pairs for a beam KV reorder on-device (no host readback). Returns a
// fixed (B*BM*max_blocks, 2) int64 buffer of pairs (sentinel (-1,-1) for empty slots) that feeds
// kv_cache_copy_blocks. parent_beam (B,BM) int32, block_table (B*BM,max_blocks) int32, seq_lens
// (B*BM,) int32.
array beam_build_copy_pairs(
    const array& parent_beam,
    const array& block_table,
    const array& seq_lens,
    int block_size,
    StreamOrDevice s = {});

/** Zero-copy beam KV reorder: returns a new block table (B*BM, max_blocks) int32 where each child
 *  beam's rows point at its parent beam's physical blocks (new[b*BM+k] = block_table[b*BM+
 *  parent_beam[b,k]]) — no KV copy. Children share physical blocks: the cache manager must
 *  refcount / copy-on-write before a beam mutates a block (out of scope). */
array beam_remap_block_table(
    const array& block_table,
    const array& parent_beam,
    StreamOrDevice s = {});

std::vector<array> kv_cache_scales(
    const array& key,
    const array& value,
    StreamOrDevice s = {});

// window > 0 restricts the decode query to the `window` most recent keys (Mistral sliding window);
// window <= 0 attends the full context. Composes with ALiBi / block-sparse / GQA.
array paged_attention(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    int window = 0,
    StreamOrDevice s = {});

// Paged decode with ALiBi: adds a per-head linear position bias slope[h]*(t-ctx+1) to each score.
// alibi_slopes is (num_heads,). (Runs the same kernel as paged_attention with use_alibi=1.)
array paged_attention_alibi(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& alibi_slopes,
    float scale = 0.0f,
    int window = 0,
    StreamOrDevice s = {});

// Block-sparse paged decode: a query skips entire KV blocks it doesn't attend to. block_mask is
// (batch, max_blocks) int32 (1 = attend, 0 = skip), sharing the block_table's layout.
array paged_attention_block_sparse(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& block_mask,
    float scale = 0.0f,
    int window = 0,
    StreamOrDevice s = {});

// vLLM x-packed cache decode: reads caches in vLLM's memory order so a ThunderMittens decode can
// consume a vLLM KV cache directly. key_cache (num_blocks, num_kv_heads, head_size/x, block_size, x);
// value_cache (num_blocks, num_kv_heads, head_size, block_size). x = 16/sizeof(dtype).
array paged_attention_xcache(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    StreamOrDevice s = {});

// GQA KV-reuse staged decode: bit-equivalent to paged_attention but stages each KV vector
// once into threadgroup memory and reuses it across the query heads sharing that kv_head.
array paged_attention_staged(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    float scale = 0.0f,
    StreamOrDevice s = {});

// fp8 KV cache: scatter K/V into a uint8 (e4m3) paged cache with per-head scales
// (k_scale/v_scale are (num_heads,)/(num_kv_heads,) arrays; a per-tensor caller passes a
// broadcast array), and decode-paged-attention that dequantizes on read. GQA/MQA aware.
std::vector<array> kv_cache_scatter_fp8(
    const array& key,
    const array& value,
    const array& slot_mapping,
    int num_blocks,
    int block_size,
    const array& k_scale,
    const array& v_scale,
    int fmt = 0,   // 0 = e4m3, 1 = e5m2
    StreamOrDevice s = {});

array paged_attention_fp8(
    const array& q,
    const array& key_cache,
    const array& value_cache,
    const array& block_table,
    const array& context_lens,
    const array& k_scale,
    const array& v_scale,
    float scale = 0.0f,
    int fmt = 0,   // 0 = e4m3, 1 = e5m2
    int window = 0,
    StreamOrDevice s = {});

class KvCacheScatter : public Primitive {
 public:
  KvCacheScatter(Stream stream, int block_size)
      : Primitive(stream), block_size_(block_size) {}

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
  const char* name() const { return "KvCacheScatter"; }

  void print(std::ostream& os) override { os << "KvCacheScatter"; }
  bool is_equivalent(const Primitive& other) const override {
    return block_size_ == static_cast<const KvCacheScatter&>(other).block_size_;
  }

 private:
  int block_size_;
};

class KvCacheGather : public Primitive {
 public:
  KvCacheGather(Stream stream, int num_tokens)
      : Primitive(stream), num_tokens_(num_tokens) {}

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
  const char* name() const { return "KvCacheGather"; }

  void print(std::ostream& os) override { os << "KvCacheGather"; }
  bool is_equivalent(const Primitive& other) const override {
    return num_tokens_ == static_cast<const KvCacheGather&>(other).num_tokens_;
  }

 private:
  int num_tokens_;
};

class KvCacheCopyBlocks : public Primitive {
 public:
  explicit KvCacheCopyBlocks(Stream stream) : Primitive(stream) {}

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
  const char* name() const { return "KvCacheCopyBlocks"; }

  void print(std::ostream& os) override { os << "KvCacheCopyBlocks"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class BeamRemapBlockTable : public Primitive {
 public:
  explicit BeamRemapBlockTable(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "BeamRemapBlockTable"; }
  void print(std::ostream& os) override { os << "BeamRemapBlockTable"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class BeamBuildCopyPairs : public Primitive {
 public:
  BeamBuildCopyPairs(Stream stream, int block_size)
      : Primitive(stream), block_size_(block_size) {}

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
  const char* name() const { return "BeamBuildCopyPairs"; }

  void print(std::ostream& os) override { os << "BeamBuildCopyPairs"; }
  bool is_equivalent(const Primitive& other) const override {
    return block_size_ == static_cast<const BeamBuildCopyPairs&>(other).block_size_;
  }

 private:
  int block_size_;
};

class KvCacheScales : public Primitive {
 public:
  explicit KvCacheScales(Stream stream) : Primitive(stream) {}

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
  const char* name() const { return "KvCacheScales"; }

  void print(std::ostream& os) override { os << "KvCacheScales"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class KvCacheScatterFp8 : public Primitive {
 public:
  KvCacheScatterFp8(Stream stream, int block_size, int fmt)
      : Primitive(stream), block_size_(block_size), fmt_(fmt) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "KvCacheScatterFp8"; }
  void print(std::ostream& os) override { os << "KvCacheScatterFp8"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const KvCacheScatterFp8&>(other);
    return block_size_ == o.block_size_ && fmt_ == o.fmt_;
  }

 private:
  int block_size_;
  int fmt_;
};

class PagedAttentionFp8 : public Primitive {
 public:
  PagedAttentionFp8(Stream stream, float scale, int fmt, int window = 0)
      : Primitive(stream), scale_(scale), fmt_(fmt), window_(window) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PagedAttentionFp8"; }
  void print(std::ostream& os) override { os << "PagedAttentionFp8"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PagedAttentionFp8&>(other);
    return scale_ == o.scale_ && fmt_ == o.fmt_ && window_ == o.window_;
  }

 private:
  float scale_;
  int fmt_;
  int window_;
};

class PagedAttention : public Primitive {
 public:
  PagedAttention(Stream stream, float scale, bool use_alibi = false, bool use_mask = false,
                 int window = 0, int mask_heads = 1)
      : Primitive(stream), scale_(scale), use_alibi_(use_alibi), use_mask_(use_mask),
        window_(window) , mask_heads_(mask_heads) {}

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
  const char* name() const { return "PagedAttention"; }

  void print(std::ostream& os) override { os << "PagedAttention"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const PagedAttention&>(other);
    return scale_ == o.scale_ && use_alibi_ == o.use_alibi_ && use_mask_ == o.use_mask_ &&
           window_ == o.window_ && mask_heads_ == o.mask_heads_;
  }

 private:
  float scale_;
  bool use_alibi_;
  bool use_mask_;
  int window_;
  int mask_heads_;
};

class PagedAttentionStaged : public Primitive {
 public:
  PagedAttentionStaged(Stream stream, float scale) : Primitive(stream), scale_(scale) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PagedAttentionStaged"; }
  void print(std::ostream& os) override { os << "PagedAttentionStaged"; }
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const PagedAttentionStaged&>(other).scale_;
  }

 private:
  float scale_;
};

class PagedAttentionXcache : public Primitive {
 public:
  PagedAttentionXcache(Stream stream, float scale) : Primitive(stream), scale_(scale) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PagedAttentionXcache"; }
  void print(std::ostream& os) override { os << "PagedAttentionXcache"; }
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const PagedAttentionXcache&>(other).scale_;
  }

 private:
  float scale_;
};

} // namespace mlx::core
