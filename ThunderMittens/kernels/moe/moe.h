// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  MoE routing: top-k expert selection with renormalized softmax weights.
 *  logits is (num_tokens, num_experts), float32/float16/bfloat16. Returns
 *  (topk_ids int32, topk_weights float32), both (num_tokens, k). The weights are
 *  softmax over the k selected logits (Mixtral renormalized top-k rule). k <= 16.
 **/
std::vector<array> moe_route_topk(const array& logits, int k, StreamOrDevice s = {});

/**
 *  MoE permute: group the T*K routing rows by expert id. topk_ids is (num_tokens, k)
 *  int32. Returns 5 int32 arrays [sorted_row_idx (T*K), offsets (E+1), inv_idx (T*K),
 *  counts (E, scratch), cursor (E, scratch)] — callers use the first three. A flat
 *  routing row r maps to token r/k, slot r%k; offsets[e] is expert e's start.
 **/
std::vector<array> moe_permute(const array& topk_ids, int num_experts, StreamOrDevice s = {});

/**
 *  MoE finalize: out[t] = sum_k topk_weights[t,k] * expert_out[inv_idx[t*k+k]].
 *  expert_out is (T*K, Hdim) in permuted order; topk_weights (T, k) f32. Returns (T, Hdim).
 **/
array moe_finalize(
    const array& expert_out, const array& inv_idx, const array& topk_weights, int k,
    StreamOrDevice s = {});

/**
 *  Fused grouped expert GEMM: out = permuted_input @ W[expert]. permuted_input (total_rows, H)
 *  with rows grouped by expert, each segment padded to a 32-multiple; W (E, H, H);
 *  expert_of_tile (total_rows/32,) int32 gives the expert of each 32-row tile. Returns
 *  (total_rows, H). float32/bfloat16; requires total_rows % 32 == 0 and H % 32 == 0.
 **/
array moe_grouped_gemm(
    const array& permuted_input, const array& W, const array& expert_of_tile, StreamOrDevice s = {});

class MoeRouteTopk : public Primitive {
 public:
  MoeRouteTopk(Stream stream, int k) : Primitive(stream), k_(k) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeRouteTopk"; }
  void print(std::ostream& os) override { os << "MoeRouteTopk"; }
  bool is_equivalent(const Primitive& other) const override {
    return k_ == static_cast<const MoeRouteTopk&>(other).k_;
  }

 private:
  int k_;
};

class MoePermute : public Primitive {
 public:
  MoePermute(Stream stream, int num_experts) : Primitive(stream), num_experts_(num_experts) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoePermute"; }
  void print(std::ostream& os) override { os << "MoePermute"; }
  bool is_equivalent(const Primitive& other) const override {
    return num_experts_ == static_cast<const MoePermute&>(other).num_experts_;
  }

 private:
  int num_experts_;
};

class MoeGroupedGemm : public Primitive {
 public:
  explicit MoeGroupedGemm(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemm"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemm"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoeFinalize : public Primitive {
 public:
  MoeFinalize(Stream stream, int k) : Primitive(stream), k_(k) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeFinalize"; }
  void print(std::ostream& os) override { os << "MoeFinalize"; }
  bool is_equivalent(const Primitive& other) const override {
    return k_ == static_cast<const MoeFinalize&>(other).k_;
  }

 private:
  int k_;
};

} // namespace mlx::core
