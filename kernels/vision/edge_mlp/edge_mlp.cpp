#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "edge_mlp/edge_mlp.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array edge_mlp_256x7(
    const array& hidden, const array& first_weight, const array& first_bias,
    const array& second_weight, const array& second_bias, StreamOrDevice s) {
  const auto dtype = hidden.dtype();
  if ((dtype != float32 && dtype != bfloat16) || hidden.ndim() != 3 ||
      hidden.shape(2) != 256 || first_weight.shape() != std::vector<int>{256, 512} ||
      first_bias.shape() != std::vector<int>{256} ||
      second_weight.shape() != std::vector<int>{7, 256} ||
      second_bias.shape() != std::vector<int>{7} || first_weight.dtype() != dtype ||
      first_bias.dtype() != dtype || second_weight.dtype() != dtype ||
      second_bias.dtype() != dtype) {
    throw std::invalid_argument(
        "edge_mlp_256x7: expected hidden (B,L,256), weights (256,512)/(7,256), "
        "biases (256,)/(7,), all fp32 or bf16");
  }
  const int batch = hidden.shape(0), length = hidden.shape(1);
  if (batch <= 0 || length <= 0) {
    throw std::invalid_argument("edge_mlp_256x7: B and L must be positive");
  }
  auto partials = array::make_arrays(
      {{batch, length, 256}, {batch, length, 256}}, {dtype, dtype},
      std::make_shared<EdgeMlpProject256>(to_stream(s)),
      {contiguous(hidden, false, s), contiguous(first_weight, false, s),
       contiguous(first_bias, false, s)});
  return array(
      {batch, 7, length, length}, dtype,
      std::make_shared<EdgeMlpCombine256x7>(to_stream(s)),
      {partials[0], partials[1], contiguous(second_weight, false, s),
       contiguous(second_bias, false, s)});
}

void EdgeMlpProject256::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("EdgeMlpProject256 has no CPU implementation.");
}
void EdgeMlpProject256::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& left = outputs[0];
  auto& right = outputs[1];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  left.set_data(allocator::malloc_or_wait(left.nbytes()));
  right.set_data(allocator::malloc_or_wait(right.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_edge_mlp_project_256(
      encoder, inputs[0], inputs[1], inputs[2], left, right,
      inputs[0].shape(0), inputs[0].shape(1), type_to_name(inputs[0]));
}

void EdgeMlpCombine256x7::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("EdgeMlpCombine256x7 has no CPU implementation.");
}
void EdgeMlpCombine256x7::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_edge_mlp_combine_256x7(
      encoder, inputs[0], inputs[1], inputs[2], inputs[3], output,
      inputs[0].shape(0), inputs[0].shape(1), type_to_name(inputs[0]));
}

#define EDGE_MLP_NO_AUTODIFF(CLASS, LABEL)                                  \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&, const std::vector<array>&) {                  \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&, const std::vector<int>&) {                  \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

EDGE_MLP_NO_AUTODIFF(EdgeMlpProject256, "EdgeMlpProject256")
EDGE_MLP_NO_AUTODIFF(EdgeMlpCombine256x7, "EdgeMlpCombine256x7")

} // namespace mlx::core
