#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array patch_merge_layernorm(
    const array& input,
    const array& weight,
    const array& bias,
    int height,
    int width,
    float eps = 1e-5f,
    StreamOrDevice s = {});

array space_to_depth_norm_linear(
    const array& input,
    const array& norm_weight,
    const array& norm_bias,
    const array& projection_weight,
    const array& projection_bias,
    int height,
    int width,
    int block_size = 2,
    float eps = 1e-5f,
    bool use_norm_bias = true,
    bool use_projection_bias = false,
    StreamOrDevice s = {});

class PatchMergeLayerNorm : public Primitive {
 public:
  PatchMergeLayerNorm(Stream stream, int height, int width, float eps)
      : Primitive(stream), height_(height), width_(width), eps_(eps) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PatchMergeLayerNorm"; }
  void print(std::ostream& os) override { os << "PatchMergeLayerNorm"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  int height_;
  int width_;
  float eps_;
};

class SpaceToDepthNormLinear : public Primitive {
 public:
  SpaceToDepthNormLinear(Stream stream, int height, int width, int block_size,
                         float eps, bool use_norm_bias,
                         bool use_projection_bias)
      : Primitive(stream), height_(height), width_(width), block_size_(block_size),
        eps_(eps), use_norm_bias_(use_norm_bias),
        use_projection_bias_(use_projection_bias) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SpaceToDepthNormLinear"; }
  void print(std::ostream& os) override { os << "SpaceToDepthNormLinear"; }
  bool is_equivalent(const Primitive& other) const override;

 private:
  int height_;
  int width_;
  int block_size_;
  float eps_;
  bool use_norm_bias_;
  bool use_projection_bias_;
};

} // namespace mlx::core
