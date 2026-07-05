// Copyright © 2023 Apple Inc.

#pragma once

#include <optional>
#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Mamba-1 (S6) selective scan forward, dense batch. Channel-major layouts:
 *  u/delta (batch, dim, seqlen) io dtype (f32/f16/bf16); A (dim, dstate) f32;
 *  B/C (batch, n_groups, dstate, seqlen) io dtype; optional D/delta_bias (dim,) f32 and
 *  gate z (batch, dim, seqlen). state (batch, dim, dstate) f32 carries the recurrence.
 *  Returns [out (batch, dim, seqlen), new_state]. dstate <= 256.
 **/
std::vector<array> selective_scan(
    const array& u, const array& delta, const array& A, const array& B, const array& C,
    const array& state, const std::optional<array>& D = std::nullopt,
    const std::optional<array>& delta_bias = std::nullopt,
    const std::optional<array>& z = std::nullopt, bool delta_softplus = true,
    StreamOrDevice s = {});

/**
 *  Varlen selective scan over a flattened token axis: u/delta/z/out (dim, total_tokens);
 *  B/C (n_groups, dstate, total_tokens); query_start_loc (B+1,) int32 marks each request's
 *  token range. state is a persistent (num_slots, dim, dstate) f32 pool indexed per request
 *  by cache_indices (optional; identity when absent; slots == null_block_id are skipped);
 *  has_initial_state (B,) uint8 (optional) gates the state load (fresh prefill starts at 0).
 *  Returns [out, new_state_pool] (the pool is clone-updated — untouched slots preserved).
 **/
std::vector<array> selective_scan_varlen(
    const array& u, const array& delta, const array& A, const array& B, const array& C,
    const array& query_start_loc, const array& state,
    const std::optional<array>& D = std::nullopt,
    const std::optional<array>& delta_bias = std::nullopt,
    const std::optional<array>& z = std::nullopt,
    const std::optional<array>& cache_indices = std::nullopt,
    const std::optional<array>& has_initial_state = std::nullopt,
    bool delta_softplus = true, int null_block_id = -1, StreamOrDevice s = {});

/**
 *  Varlen + automatic-prefix-caching (APC) selective scan. Same S6 recurrence as the varlen
 *  path, but the running state is checkpointed into PAGED state blocks at chunk boundaries
 *  and the initial state is read from a (possibly cached) prefix block. Buffer table mirrors
 *  vLLM's mamba paged scan. u/delta/z/out (dim, total_tokens); B/C (n_groups, dstate,
 *  total_tokens); state a (num_slots, dim, dstate) f32 pool. The chunk metadata
 *  (block_idx_first/last_scheduled_token, initial_state_idx, cu_chunk_seqlen,
 *  last_chunk_indices) drives the block indexing; use_chunk_metadata=false chunks by
 *  block_size. Returns [out, new_state_pool]. has_initial_state (B,) uint8, required.
 **/
std::vector<array> selective_scan_varlen_apc(
    const array& u, const array& delta, const array& A, const array& B, const array& C,
    const array& query_start_loc, const array& cache_indices, const array& has_initial_state,
    const array& state, const array& block_idx_first_scheduled_token,
    const array& block_idx_last_scheduled_token, const array& initial_state_idx,
    const array& cu_chunk_seqlen, const array& last_chunk_indices, int block_size,
    int cache_indices_stride, bool use_chunk_metadata, const std::optional<array>& D = std::nullopt,
    const std::optional<array>& delta_bias = std::nullopt,
    const std::optional<array>& z = std::nullopt, bool delta_softplus = true,
    int null_block_id = -1, StreamOrDevice s = {});

class SelectiveScanApc : public Primitive {
 public:
  SelectiveScanApc(Stream stream, bool has_d, bool has_bias, bool has_z, bool delta_softplus,
                   int null_block_id, int batch, int block_size, int cache_indices_stride,
                   bool use_chunk_metadata)
      : Primitive(stream), has_d_(has_d), has_bias_(has_bias), has_z_(has_z),
        delta_softplus_(delta_softplus), null_block_id_(null_block_id), batch_(batch),
        block_size_(block_size), cache_indices_stride_(cache_indices_stride),
        use_chunk_metadata_(use_chunk_metadata) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SelectiveScanApc"; }
  void print(std::ostream& os) override { os << "SelectiveScanApc"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const SelectiveScanApc&>(other);
    return has_d_ == o.has_d_ && has_bias_ == o.has_bias_ && has_z_ == o.has_z_ &&
           delta_softplus_ == o.delta_softplus_ && null_block_id_ == o.null_block_id_ &&
           batch_ == o.batch_ && block_size_ == o.block_size_ &&
           cache_indices_stride_ == o.cache_indices_stride_ &&
           use_chunk_metadata_ == o.use_chunk_metadata_;
  }

 private:
  bool has_d_, has_bias_, has_z_, delta_softplus_;
  int null_block_id_, batch_, block_size_, cache_indices_stride_;
  bool use_chunk_metadata_;
};

class SelectiveScan : public Primitive {
 public:
  SelectiveScan(Stream stream, bool varlen, bool has_d, bool has_bias, bool has_z,
                bool delta_softplus, bool use_cache_indices, bool use_has_initial_state,
                int null_block_id, int batch)
      : Primitive(stream), varlen_(varlen), has_d_(has_d), has_bias_(has_bias), has_z_(has_z),
        delta_softplus_(delta_softplus), use_cache_indices_(use_cache_indices),
        use_has_initial_state_(use_has_initial_state), null_block_id_(null_block_id),
        batch_(batch) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SelectiveScan"; }
  void print(std::ostream& os) override { os << "SelectiveScan"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const SelectiveScan&>(other);
    return varlen_ == o.varlen_ && has_d_ == o.has_d_ && has_bias_ == o.has_bias_ &&
           has_z_ == o.has_z_ && delta_softplus_ == o.delta_softplus_ &&
           use_cache_indices_ == o.use_cache_indices_ &&
           use_has_initial_state_ == o.use_has_initial_state_ &&
           null_block_id_ == o.null_block_id_ && batch_ == o.batch_;
  }

 private:
  bool varlen_;
  bool has_d_;
  bool has_bias_;
  bool has_z_;
  bool delta_softplus_;
  bool use_cache_indices_;
  bool use_has_initial_state_;
  int null_block_id_;
  int batch_;
};

} // namespace mlx::core
