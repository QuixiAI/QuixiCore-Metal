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

void PatchMergeLayerNorm::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PatchMergeLayerNorm has no CPU implementation.");
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

} // namespace mlx::core
