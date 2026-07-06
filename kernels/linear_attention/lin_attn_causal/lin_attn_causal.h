// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Causal linear attention (identity feature map): out_i = sum_{j<=i} (q_i.k_j) v_j,
 *  via a chunked running-KV scan. q,k,v are (B,H,N,D), bf16; D=64, N a multiple of 8. */
array lin_attn_causal(const array& q, const array& k, const array& v, StreamOrDevice s = {});

// Chunked-parallel pipeline primitives (used automatically for N % 64 == 0, N >= 128):
// LinChunkKV (per-chunk KV states) -> LinChunkScan (exclusive chunk prefix) -> LinChunkOut.
class LinChunkKV : public Primitive {
 public:
  explicit LinChunkKV(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LinChunkKV"; }
  void print(std::ostream& os) override { os << "LinChunkKV"; }
  bool is_equivalent(const Primitive& other) const override;
};

class LinChunkScan : public Primitive {
 public:
  explicit LinChunkScan(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LinChunkScan"; }
  void print(std::ostream& os) override { os << "LinChunkScan"; }
  bool is_equivalent(const Primitive& other) const override;
};

class LinChunkOut : public Primitive {
 public:
  explicit LinChunkOut(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LinChunkOut"; }
  void print(std::ostream& os) override { os << "LinChunkOut"; }
  bool is_equivalent(const Primitive& other) const override;
};

class LinAttnCausal : public Primitive {
 public:
  explicit LinAttnCausal(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "LinAttnCausal"; }

  void print(std::ostream& os) override { os << "LinAttnCausal"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
