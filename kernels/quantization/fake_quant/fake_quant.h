#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> fake_quant_int8(const array& x, StreamOrDevice s = {});
std::vector<array> silu_mul_fake_quant_int8(
    const array& x,
    const array& gate,
    int mode = 0,
    float alpha = 1.702f,
    float limit = 7.0f,
    StreamOrDevice s = {});

class FakeQuantInt8 : public Primitive {
 public:
  FakeQuantInt8(Stream stream, bool silu_mul, int mode, float alpha, float limit)
      : Primitive(stream), silu_mul_(silu_mul), mode_(mode), alpha_(alpha), limit_(limit) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "FakeQuantInt8"; }
  void print(std::ostream& os) override { os << "FakeQuantInt8"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& o = static_cast<const FakeQuantInt8&>(other);
    return silu_mul_ == o.silu_mul_ && mode_ == o.mode_ && alpha_ == o.alpha_ &&
           limit_ == o.limit_;
  }

 private:
  bool silu_mul_;
  int mode_;
  float alpha_;
  float limit_;
};

} // namespace mlx::core
