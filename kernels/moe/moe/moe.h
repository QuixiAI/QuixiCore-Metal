// Copyright © 2023 Apple Inc.

#pragma once

#include <string>
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
 *  DeepSeek-style grouped (node-limited) routing with bias-corrected selection (HF "noaux_tc").
 *  score = scoring(logit) (0 softmax / 1 sigmoid / 2 sqrt-softplus); selection uses
 *  score + bias[e] (bias optional); groups ranked by their top-2 biased sum, top `topk_group`
 *  survive; top-k experts among survivors. Emitted weight = UNBIASED score, optionally
 *  renormalized over the selected set, x routed_scaling_factor. Returns (ids int32,
 *  weights float32), both (num_tokens, k). E <= 512, E % n_group == 0, n_group <= 32, k <= 16.
 **/
std::vector<array> moe_route_grouped(
    const array& logits, const array& bias, bool has_bias, int k, int n_group, int topk_group,
    bool renormalize, float routed_scaling_factor, int scoring_func, StreamOrDevice s = {});

/**
 *  MoE permute: group the T*K routing rows by expert id. topk_ids is (num_tokens, k)
 *  int32. Returns 5 int32 arrays [sorted_row_idx (T*K), offsets (E+1), inv_idx (T*K),
 *  counts (E, scratch), cursor (E, scratch)] — callers use the first three. A flat
 *  routing row r maps to token r/k, slot r%k; offsets[e] is expert e's start.
 **/
std::vector<array> moe_permute(const array& topk_ids, int num_experts, StreamOrDevice s = {});

/**
 *  MoE padded schedule (GPU replacement for the host glue): turns moe_permute's compact
 *  layout into 32-row-padded per-expert segments for the grouped GEMMs. Static worst-case
 *  sizing: total_pad_max = ceil32(T*K + 31*E), max_tiles = total_pad_max/32; -1 sentinels
 *  mark tiles/rows beyond the real (data-dependent) total. Returns int32 arrays
 *  [expert_of_tile (max_tiles), gather_idx (total_pad_max), inv_pad (T*K), off_pad (E+1)].
 *  inv_pad[r] is the padded row finalize must read for routing row r.
 **/
std::vector<array> moe_pad_schedule(
    const array& sorted_row_idx, const array& offsets, int k, StreamOrDevice s = {});

/**
 *  MoE gather: permuted_input[p, :] = x[gather_idx[p], :] (zeros where gather_idx[p] < 0).
 *  x (T, H) float32/bfloat16; returns (len(gather_idx), H).
 **/
array moe_gather(const array& x, const array& gather_idx, StreamOrDevice s = {});

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

/** Rectangular grouped GEMM: out (total_rows, N_out) = A (total_rows, K_dim) @ W[e] (K_dim, N_out).
 *  W is (E, K_dim, N_out); K_dim % 16 == 0, N_out % 32 == 0, total_rows % 32 == 0. **/
array moe_grouped_gemm_rect(
    const array& A, const array& W, const array& expert_of_tile, StreamOrDevice s = {});

/** Fused SiLU-GLU GEMM1: out (total_rows, inter) = silu(A @ W1_gate) * (A @ W1_up).
 *  A (total_rows, H); W1 (E, H, 2*inter) laid out [gate | up]. H % 16 == 0, inter % 32 == 0. **/
array moe_grouped_gemm_swiglu(
    const array& A, const array& W1, const array& expert_of_tile, StreamOrDevice s = {});

/** Quantized rectangular grouped GEMM: out (total_rows, N_out) = A @ dequant(Wq[e])^T.
 *  A (total_rows, K_dim) bfloat16. Wq (E, N_out, row_bytes) uint8 — experts packed row-major
 *  over (N_out, K_dim) with quant groups along K_dim (tk.quant.quantize_expert_stack layout;
 *  row_bytes = K_dim/block_k * block_bytes). bias (E, N_out) bfloat16 added per output column
 *  when has_bias (pass a 1-element dummy otherwise). total_rows % 32 == 0, K_dim % 32 == 0
 *  (and % block_k), N_out % 32 == 0. format in {mxfp4,kU4,fp8_e4m3,q8_0,nvfp4,q4_K}. **/
array moe_grouped_gemm_rect_q(
    const array& A, const array& Wq, const array& expert_of_tile, const array& bias,
    bool has_bias, int K_dim, int N_out, const std::string& format, StreamOrDevice s = {});

/** Quantized fused SwiGLU GEMM1: out (total_rows, inter). W1q (E, 2*inter, row_bytes) uint8,
 *  packed rows [gate(inter) | up(inter)] over K = H. act_mode 0: silu(gate)*up;
 *  act_mode 1: gpt-oss swiglu_oai — min/clamp by `limit`, gate*sigmoid(alpha*gate)*(1+up).
 *  bias (E, 2*inter) bfloat16 pre-activation when has_bias. **/
array moe_grouped_gemm_swiglu_q(
    const array& A, const array& W1q, const array& expert_of_tile, const array& bias,
    bool has_bias, int inter, int act_mode, float alpha, float limit,
    const std::string& format, StreamOrDevice s = {});

array moe_grouped_gemm_bwd_dx(
    const array& dy, const array& W, const array& expert_of_tile, StreamOrDevice s = {});
array moe_grouped_gemm_bwd_dw(
    const array& A, const array& dy, const array& off_pad, int num_experts,
    StreamOrDevice s = {});
std::vector<array> moe_finalize_bwd(
    const array& grad_out, const array& expert_out, const array& inv_idx,
    const array& topk_weights, StreamOrDevice s = {});
array moe_gather_bwd(const array& dA, const array& inv_idx, int k, StreamOrDevice s = {});

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

class MoeRouteGrouped : public Primitive {
 public:
  MoeRouteGrouped(Stream stream, int k, int n_group, int topk_group, bool renormalize,
                  float routed_scaling_factor, int scoring_func, bool has_bias)
      : Primitive(stream), k_(k), n_group_(n_group), topk_group_(topk_group),
        renormalize_(renormalize), routed_scaling_factor_(routed_scaling_factor),
        scoring_func_(scoring_func), has_bias_(has_bias) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeRouteGrouped"; }
  void print(std::ostream& os) override { os << "MoeRouteGrouped"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MoeRouteGrouped&>(other);
    return k_ == o.k_ && n_group_ == o.n_group_ && topk_group_ == o.topk_group_ &&
           renormalize_ == o.renormalize_ &&
           routed_scaling_factor_ == o.routed_scaling_factor_ &&
           scoring_func_ == o.scoring_func_ && has_bias_ == o.has_bias_;
  }

 private:
  int k_;
  int n_group_;
  int topk_group_;
  bool renormalize_;
  float routed_scaling_factor_;
  int scoring_func_;
  bool has_bias_;
};

class MoePadSchedule : public Primitive {
 public:
  MoePadSchedule(Stream stream, int num_experts, int k)
      : Primitive(stream), num_experts_(num_experts), k_(k) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoePadSchedule"; }
  void print(std::ostream& os) override { os << "MoePadSchedule"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MoePadSchedule&>(other);
    return num_experts_ == o.num_experts_ && k_ == o.k_;
  }

 private:
  int num_experts_;
  int k_;
};

class MoeGather : public Primitive {
 public:
  explicit MoeGather(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGather"; }
  void print(std::ostream& os) override { os << "MoeGather"; }
  bool is_equivalent(const Primitive&) const override { return true; }
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

class MoeGroupedGemmRect : public Primitive {
 public:
  explicit MoeGroupedGemmRect(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemmRect"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemmRect"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoeGroupedGemmSwiglu : public Primitive {
 public:
  explicit MoeGroupedGemmSwiglu(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemmSwiglu"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemmSwiglu"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoeGroupedGemmRectQ : public Primitive {
 public:
  MoeGroupedGemmRectQ(Stream stream, std::string format, bool has_bias, int K_dim, int N_out)
      : Primitive(stream), format_(std::move(format)), has_bias_(has_bias),
        K_dim_(K_dim), N_out_(N_out) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemmRectQ"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemmRectQ"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MoeGroupedGemmRectQ&>(other);
    return format_ == o.format_ && has_bias_ == o.has_bias_ &&
           K_dim_ == o.K_dim_ && N_out_ == o.N_out_;
  }

 private:
  std::string format_;
  bool has_bias_;
  int K_dim_;
  int N_out_;
};

class MoeGroupedGemmSwigluQ : public Primitive {
 public:
  MoeGroupedGemmSwigluQ(Stream stream, std::string format, bool has_bias, int inter,
                        int act_mode, float alpha, float limit)
      : Primitive(stream), format_(std::move(format)), has_bias_(has_bias), inter_(inter),
        act_mode_(act_mode), alpha_(alpha), limit_(limit) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemmSwigluQ"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemmSwigluQ"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const MoeGroupedGemmSwigluQ&>(other);
    return format_ == o.format_ && has_bias_ == o.has_bias_ && inter_ == o.inter_ &&
           act_mode_ == o.act_mode_ && alpha_ == o.alpha_ && limit_ == o.limit_;
  }

 private:
  std::string format_;
  bool has_bias_;
  int inter_;
  int act_mode_;
  float alpha_;
  float limit_;
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

class MoeGroupedGemmBwdDx : public Primitive {
 public:
  explicit MoeGroupedGemmBwdDx(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemmBwdDx"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemmBwdDx"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoeGroupedGemmBwdDw : public Primitive {
 public:
  explicit MoeGroupedGemmBwdDw(Stream stream, int num_experts)
      : Primitive(stream), num_experts_(num_experts) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGroupedGemmBwdDw"; }
  void print(std::ostream& os) override { os << "MoeGroupedGemmBwdDw"; }
  bool is_equivalent(const Primitive& other) const override {
    return num_experts_ == static_cast<const MoeGroupedGemmBwdDw&>(other).num_experts_;
  }

 private:
  int num_experts_;
};

class MoeFinalizeBwd : public Primitive {
 public:
  explicit MoeFinalizeBwd(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeFinalizeBwd"; }
  void print(std::ostream& os) override { os << "MoeFinalizeBwd"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class MoeGatherBwd : public Primitive {
 public:
  explicit MoeGatherBwd(Stream stream, int k) : Primitive(stream), k_(k) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "MoeGatherBwd"; }
  void print(std::ostream& os) override { os << "MoeGatherBwd"; }
  bool is_equivalent(const Primitive& other) const override {
    return k_ == static_cast<const MoeGatherBwd&>(other).k_;
  }

 private:
  int k_;
};

} // namespace mlx::core
