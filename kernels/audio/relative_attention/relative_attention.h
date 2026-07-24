#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array audio_relative_attention(
    const array& q, const array& k, const array& v,
    const array& relative_k, const array& per_dim_scale,
    const array& lengths, int chunk_size, int left_context, int right_context,
    float q_scale, float k_scale, float softcap, StreamOrDevice s = {});

class AudioRelativeAttention : public Primitive {
 public:
  AudioRelativeAttention(Stream stream, int chunk_size, int left_context,
                         int right_context, float q_scale, float k_scale,
                         float softcap)
      : Primitive(stream), chunk_size_(chunk_size), left_context_(left_context),
        right_context_(right_context), q_scale_(q_scale), k_scale_(k_scale),
        softcap_(softcap) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AudioRelativeAttention"; }
  void print(std::ostream& os) override { os << "AudioRelativeAttention"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const AudioRelativeAttention&>(other);
    return chunk_size_ == o.chunk_size_ && left_context_ == o.left_context_ &&
           right_context_ == o.right_context_ && q_scale_ == o.q_scale_ &&
           k_scale_ == o.k_scale_ && softcap_ == o.softcap_;
  }
 private:
  int chunk_size_, left_context_, right_context_;
  float q_scale_, k_scale_, softcap_;
};

}  // namespace mlx::core
