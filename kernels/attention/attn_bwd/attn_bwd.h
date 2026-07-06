// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

// FlashAttention-2 backward, decomposed into one-simdgroup-per-block kernels (no atomics).
// q,k,v,o,do are (B,H,N,D) bf16; L,delta are (B,H,N) fp32; D in {64,128}, N%8==0.

/** Forward + logsumexp: returns {o (B,H,N,D) bf16, L (B,H,N) fp32}. causal masks future positions. */
std::vector<array> attn_fwd_l(const array& q, const array& k, const array& v, bool causal = false,
                              StreamOrDevice s = {});

/** D_i = rowsum(dO_i ∘ O_i) -> delta (B,H,N) fp32. */
array attn_bwd_prep(const array& o, const array& do_, StreamOrDevice s = {});

/** dQ (B,H,N,D) bf16. */
array attn_bwd_dq(const array& q, const array& k, const array& v, const array& do_,
                  const array& L, const array& delta, bool causal, StreamOrDevice s = {});

/** {dK, dV} (B,H,N,D) bf16. */
std::vector<array> attn_bwd_dkv(const array& q, const array& k, const array& v, const array& do_,
                                const array& L, const array& delta, bool causal, StreamOrDevice s = {});

class AttnFwdL : public Primitive {
 public:
  explicit AttnFwdL(Stream stream, bool causal) : Primitive(stream), causal_(causal) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnFwdL"; }

  void print(std::ostream& os) override { os << "AttnFwdL"; }
  bool is_equivalent(const Primitive& o) const override {
    return typeid(*this) == typeid(o) && causal_ == static_cast<const AttnFwdL&>(o).causal_;
  }
 private:
  bool causal_;
};

class AttnBwdPrep : public Primitive {
 public:
  explicit AttnBwdPrep(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnBwdPrep"; }

  void print(std::ostream& os) override { os << "AttnBwdPrep"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class AttnBwdDQ : public Primitive {
 public:
  explicit AttnBwdDQ(Stream stream, bool causal) : Primitive(stream), causal_(causal) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnBwdDQ"; }

  void print(std::ostream& os) override { os << "AttnBwdDQ"; }
  bool is_equivalent(const Primitive& o) const override {
    return typeid(*this) == typeid(o) && causal_ == static_cast<const AttnBwdDQ&>(o).causal_;
  }
 private:
  bool causal_;
};

class AttnBwdDKV : public Primitive {
 public:
  explicit AttnBwdDKV(Stream stream, bool causal) : Primitive(stream), causal_(causal) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AttnBwdDKV"; }

  void print(std::ostream& os) override { os << "AttnBwdDKV"; }
  bool is_equivalent(const Primitive& o) const override {
    return typeid(*this) == typeid(o) && causal_ == static_cast<const AttnBwdDKV&>(o).causal_;
  }
 private:
  bool causal_;
};

} // namespace mlx::core
