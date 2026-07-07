#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> kd_kl_topk_fwd(
    const array& logits,
    const array& t_idx,
    const array& t_prob,
    float invtemp = 1.0f,
    int tail_mode = 0,
    StreamOrDevice s = {});

array kd_kl_topk_bwd(
    const array& logits,
    const array& t_idx,
    const array& t_prob,
    const array& lse,
    const array& grad_out,
    float invtemp = 1.0f,
    int tail_mode = 0,
    StreamOrDevice s = {});

class KdKlTopkFwd : public Primitive {
 public:
  KdKlTopkFwd(Stream stream, float invtemp, int tail_mode)
      : Primitive(stream), invtemp_(invtemp), tail_mode_(tail_mode) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "KdKlTopkFwd"; }
  void print(std::ostream& os) override { os << "KdKlTopkFwd"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const KdKlTopkFwd&>(other);
    return invtemp_ == o.invtemp_ && tail_mode_ == o.tail_mode_;
  }

 private:
  float invtemp_;
  int tail_mode_;
};

class KdKlTopkBwd : public Primitive {
 public:
  KdKlTopkBwd(Stream stream, float invtemp, int tail_mode)
      : Primitive(stream), invtemp_(invtemp), tail_mode_(tail_mode) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "KdKlTopkBwd"; }
  void print(std::ostream& os) override { os << "KdKlTopkBwd"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const KdKlTopkBwd&>(other);
    return invtemp_ == o.invtemp_ && tail_mode_ == o.tail_mode_;
  }

 private:
  float invtemp_;
  int tail_mode_;
};

} // namespace mlx::core
