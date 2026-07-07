// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> kd_kl_dense_fwd(
    const array& t_logits, const array& s_logits, float invtemp = 1.0f, StreamOrDevice s = {});
array kd_kl_dense_bwd(
    const array& t_logits, const array& s_logits, const array& lse_t, const array& lse_s,
    const array& grad_out, float invtemp = 1.0f, StreamOrDevice s = {});
std::vector<array> kd_ce_fused_fwd(
    const array& t_logits, const array& s_logits, const array& targets,
    float invtemp = 1.0f, StreamOrDevice s = {});
array kd_ce_fused_bwd(
    const array& t_logits, const array& s_logits, const array& targets, const array& lse_sr,
    const array& lse_st, const array& lse_t, const array& go_ce, const array& go_kd,
    float invtemp = 1.0f, StreamOrDevice s = {});

class KdKlDenseFwd : public Primitive {
 public:
  KdKlDenseFwd(Stream stream, float invtemp) : Primitive(stream), invtemp_(invtemp) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "KdKlDenseFwd"; }
  void print(std::ostream& os) override { os << "KdKlDenseFwd"; }
  bool is_equivalent(const Primitive& other) const override {
    return invtemp_ == static_cast<const KdKlDenseFwd&>(other).invtemp_;
  }
  float invtemp_;
};

class KdKlDenseBwd : public Primitive {
 public:
  KdKlDenseBwd(Stream stream, float invtemp) : Primitive(stream), invtemp_(invtemp) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "KdKlDenseBwd"; }
  void print(std::ostream& os) override { os << "KdKlDenseBwd"; }
  bool is_equivalent(const Primitive& other) const override {
    return invtemp_ == static_cast<const KdKlDenseBwd&>(other).invtemp_;
  }
  float invtemp_;
};

class KdCeFusedFwd : public Primitive {
 public:
  KdCeFusedFwd(Stream stream, float invtemp) : Primitive(stream), invtemp_(invtemp) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "KdCeFusedFwd"; }
  void print(std::ostream& os) override { os << "KdCeFusedFwd"; }
  bool is_equivalent(const Primitive& other) const override {
    return invtemp_ == static_cast<const KdCeFusedFwd&>(other).invtemp_;
  }
  float invtemp_;
};

class KdCeFusedBwd : public Primitive {
 public:
  KdCeFusedBwd(Stream stream, float invtemp) : Primitive(stream), invtemp_(invtemp) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "KdCeFusedBwd"; }
  void print(std::ostream& os) override { os << "KdCeFusedBwd"; }
  bool is_equivalent(const Primitive& other) const override {
    return invtemp_ == static_cast<const KdCeFusedBwd&>(other).invtemp_;
  }
  float invtemp_;
};

} // namespace mlx::core
