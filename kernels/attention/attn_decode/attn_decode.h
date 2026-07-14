// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array attn_decode(const array& q, const array& k, const array& v, StreamOrDevice s = {});

array attn_decode_bh(
    const array& q,
    const array& k,
    const array& v,
    int context_length,
    StreamOrDevice s = {});

std::vector<array> decode_cache_attention(
    const array& q,
    const array& new_k,
    const array& new_v,
    const array& cos,
    const array& sin,
    const array& positions,
    const array& context_lengths,
    const array& q_weight,
    const array& k_weight,
    const array& key_cache,
    const array& value_cache,
    float eps = 1e-6f,
    bool do_q_norm = false,
    bool do_k_norm = false,
    bool gemma = false,
    float softmax_scale = 0.0f,
    StreamOrDevice s = {});

class AttnDecode : public Primitive {
 public:
  explicit AttnDecode(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnDecode"; }
  void print(std::ostream& os) override { os << "AttnDecode"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class AttnDecodeBh : public Primitive {
 public:
  AttnDecodeBh(Stream stream, int context_length)
      : Primitive(stream), context_length_(context_length) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnDecodeBh"; }
  void print(std::ostream& os) override { os << "AttnDecodeBh"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  int context_length_;
};

class DecodeCacheAttention : public Primitive {
 public:
  DecodeCacheAttention(Stream stream, float eps, bool do_q_norm,
                       bool do_k_norm, bool gemma, float softmax_scale)
      : Primitive(stream), eps_(eps), do_q_norm_(do_q_norm),
        do_k_norm_(do_k_norm), gemma_(gemma), softmax_scale_(softmax_scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "DecodeCacheAttention"; }
  void print(std::ostream& os) override { os << "DecodeCacheAttention"; }
  bool is_equivalent(const Primitive& other) const override;
 private:
  float eps_;
  bool do_q_norm_;
  bool do_k_norm_;
  bool gemma_;
  float softmax_scale_;
};

} // namespace mlx::core
