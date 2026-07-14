#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "patch_merge/patch_merge.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool patch_float(Dtype dtype) {
  return dtype == float32 || dtype == float16 || dtype == bfloat16;
}
}

array patch_merge_layernorm(
    const array& input, const array& weight, const array& bias,
    int height, int width, float eps, StreamOrDevice s) {
  if (input.dtype() != bfloat16 || input.ndim() != 3 || height <= 0 || width <= 0) {
    throw std::invalid_argument(
        "patch_merge_layernorm: input must be bfloat16 (B,H*W,C), with positive H/W");
  }
  const int channels = input.shape(2);
  const int dimension = 4 * channels;
  if (input.shape(0) <= 0 || channels <= 0 || input.shape(1) != height * width ||
      weight.dtype() != bfloat16 ||
      bias.dtype() != bfloat16 || weight.ndim() != 1 || bias.ndim() != 1 ||
      weight.shape(0) != dimension || bias.shape(0) != dimension) {
    throw std::invalid_argument(
        "patch_merge_layernorm: spatial shape or bfloat16 weight/bias (4*C,) mismatch");
  }
  const int patches = ((height + 1) / 2) * ((width + 1) / 2);
  return array(
      {input.shape(0), patches, dimension}, bfloat16,
      std::make_shared<PatchMergeLayerNorm>(to_stream(s), height, width, eps),
      {contiguous(input, false, s), contiguous(weight, false, s),
       contiguous(bias, false, s)});
}

array space_to_depth_norm_linear(
    const array& input, const array& norm_weight, const array& norm_bias,
    const array& projection_weight, const array& projection_bias,
    int height, int width, int block_size, float eps,
    bool use_norm_bias, bool use_projection_bias, StreamOrDevice s) {
  if (!patch_float(input.dtype()) || input.ndim() != 3 ||
      input.shape(0) <= 0 || input.shape(2) <= 0 || height <= 0 || width <= 0 ||
      input.shape(1) != height * width || (block_size != 2 && block_size != 4)) {
    throw std::invalid_argument(
        "space_to_depth_norm_linear: input must be (B,H*W,C) fp32/fp16/bf16; block_size is 2 or 4");
  }
  const int dimension = block_size * block_size * input.shape(2);
  if (dimension > 4096 || norm_weight.dtype() != input.dtype() ||
      norm_weight.ndim() != 1 || norm_weight.shape(0) != dimension) {
    throw std::invalid_argument(
        "space_to_depth_norm_linear: norm_weight must be (block_size^2*C), dimension <= 4096");
  }
  if (use_norm_bias &&
      (norm_bias.dtype() != input.dtype() || norm_bias.ndim() != 1 ||
       norm_bias.shape(0) != dimension)) {
    throw std::invalid_argument(
        "space_to_depth_norm_linear: norm_bias must match norm_weight");
  }
  if (projection_weight.dtype() != input.dtype() ||
      projection_weight.ndim() != 2 || projection_weight.shape(0) <= 0 ||
      projection_weight.shape(1) != dimension) {
    throw std::invalid_argument(
        "space_to_depth_norm_linear: projection_weight must be (O,block_size^2*C)");
  }
  const int out_channels = projection_weight.shape(0);
  if (use_projection_bias &&
      (projection_bias.dtype() != input.dtype() || projection_bias.ndim() != 1 ||
       projection_bias.shape(0) != out_channels)) {
    throw std::invalid_argument(
        "space_to_depth_norm_linear: projection_bias must be (O,)");
  }
  const int patches = ((height + block_size - 1) / block_size) *
      ((width + block_size - 1) / block_size);
  return array(
      {input.shape(0), patches, out_channels}, input.dtype(),
      std::make_shared<SpaceToDepthNormLinear>(
          to_stream(s), height, width, block_size, eps,
          use_norm_bias, use_projection_bias),
      {contiguous(input, false, s), contiguous(norm_weight, false, s),
       contiguous(astype(norm_bias, input.dtype(), s), false, s),
       contiguous(projection_weight, false, s),
       contiguous(astype(projection_bias, input.dtype(), s), false, s)});
}

void PatchMergeLayerNorm::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PatchMergeLayerNorm has no CPU implementation.");
}
void SpaceToDepthNormLinear::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SpaceToDepthNormLinear has no CPU implementation.");
}

void PatchMergeLayerNorm::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_patch_merge_layernorm(
      encoder, inputs[0], inputs[1], inputs[2], output,
      inputs[0].shape(0), height_, width_, inputs[0].shape(2), eps_);
}

void SpaceToDepthNormLinear::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_space_to_depth_norm_linear(
      encoder, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], output,
      inputs[0].shape(0), height_, width_, inputs[0].shape(2), output.shape(2),
      block_size_, eps_, use_norm_bias_, use_projection_bias_,
      type_to_name(inputs[0]));
}

std::vector<array> PatchMergeLayerNorm::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("PatchMergeLayerNorm has no jvp implementation.");
}
std::vector<array> PatchMergeLayerNorm::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("PatchMergeLayerNorm has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> PatchMergeLayerNorm::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("PatchMergeLayerNorm has no vmap implementation.");
}
bool PatchMergeLayerNorm::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const PatchMergeLayerNorm&>(other);
  return height_ == o.height_ && width_ == o.width_ && eps_ == o.eps_;
}

std::vector<array> SpaceToDepthNormLinear::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SpaceToDepthNormLinear has no jvp implementation.");
}
std::vector<array> SpaceToDepthNormLinear::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("SpaceToDepthNormLinear has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SpaceToDepthNormLinear::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SpaceToDepthNormLinear has no vmap implementation.");
}
bool SpaceToDepthNormLinear::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const SpaceToDepthNormLinear&>(other);
  return height_ == o.height_ && width_ == o.width_ &&
      block_size_ == o.block_size_ && eps_ == o.eps_ &&
      use_norm_bias_ == o.use_norm_bias_ &&
      use_projection_bias_ == o.use_projection_bias_;
}

} // namespace mlx::core
