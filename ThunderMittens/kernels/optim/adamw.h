#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  AdamW optimizer step (decoupled weight decay). Given param/grad (T) and fp32 moments m, v and the
 *  current step t (>=1), returns {param', m', v'}:
 *    m' = b1 m + (1-b1) g;  v' = b2 v + (1-b2) g^2;  mhat = m'/(1-b1^t);  vhat = v'/(1-b2^t)
 *    param' = param - lr*( mhat/(sqrt(vhat)+eps) + wd*param )
 **/
std::vector<array> adamw(
    const array& param,
    const array& grad,
    const array& m,
    const array& v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int step,
    StreamOrDevice s = {});

class AdamW : public Primitive {
 public:
  AdamW(Stream stream, float lr, float b1, float b2, float eps, float wd, int step)
      : Primitive(stream), lr_(lr), b1_(b1), b2_(b2), eps_(eps), wd_(wd), step_(step) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AdamW"; }
  void print(std::ostream& os) override { os << "AdamW"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const AdamW&>(other);
    return lr_ == o.lr_ && b1_ == o.b1_ && b2_ == o.b2_ && eps_ == o.eps_ &&
           wd_ == o.wd_ && step_ == o.step_;
  }

 private:
  float lr_, b1_, b2_, eps_, wd_;
  int step_;
};

} // namespace mlx::core
