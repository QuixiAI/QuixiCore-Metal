#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array cross_attention(const array& q, const array& k, const array& v,
                      const array& key_lengths, const array& bias,
                      float scale, float softcap, bool has_bias,
                      StreamOrDevice s = {});

class CrossAttention : public Primitive {
 public:
  CrossAttention(Stream stream, float scale, float softcap, bool has_bias)
      : Primitive(stream), scale_(scale), softcap_(softcap), has_bias_(has_bias) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "CrossAttention"; }
  void print(std::ostream& os) override { os << "CrossAttention"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const CrossAttention&>(other);
    return scale_ == o.scale_ && softcap_ == o.softcap_ && has_bias_ == o.has_bias_;
  }
 private:
  float scale_, softcap_;
  bool has_bias_;
};

}  // namespace mlx::core
