// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** DeepSeek-V3.2 (DSA/NSA) indexer K quant-and-cache: quantize the indexer K per
 *  quant_block_size (canonical 128) into an e4m3 code cache + fp32 scale cache the sparse-
 *  attention selector reads. k (tokens, head_dim); slot_mapping (tokens,) int (<0 skips);
 *  caches (num_slots, head_dim) uint8 and (num_slots, head_dim/qbs) f32. use_ue8m0 rounds
 *  scales to powers of two. Functional — returns [code_cache, scale_cache] (untouched slots
 *  preserved). */
std::vector<array> indexer_k_quant_and_cache(
    const array& k, const array& slot_mapping, const array& code_cache,
    const array& scale_cache, int quant_block_size, bool ue8m0, StreamOrDevice s = {});

/** Gather + dequantize the indexer cache back to bf16 K for a slot list: k_out[row] =
 *  decode(code_cache[slot]) * scale_cache[slot, qblock]. Returns k_out bf16 (n, head_dim). */
array indexer_k_gather(
    const array& code_cache, const array& scale_cache, const array& slots, int head_dim,
    int quant_block_size, StreamOrDevice s = {});

class IndexerKQuant : public Primitive {
 public:
  IndexerKQuant(Stream stream, int quant_block_size, bool ue8m0)
      : Primitive(stream), qbs_(quant_block_size), ue8m0_(ue8m0) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "IndexerKQuant"; }
  void print(std::ostream& os) override { os << "IndexerKQuant"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const IndexerKQuant&>(other);
    return qbs_ == o.qbs_ && ue8m0_ == o.ue8m0_;
  }

 private:
  int qbs_;
  bool ue8m0_;
};

class IndexerKGather : public Primitive {
 public:
  IndexerKGather(Stream stream, int head_dim, int quant_block_size)
      : Primitive(stream), head_dim_(head_dim), qbs_(quant_block_size) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "IndexerKGather"; }
  void print(std::ostream& os) override { os << "IndexerKGather"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const IndexerKGather&>(other);
    return head_dim_ == o.head_dim_ && qbs_ == o.qbs_;
  }

 private:
  int head_dim_;
  int qbs_;
};

} // namespace mlx::core
