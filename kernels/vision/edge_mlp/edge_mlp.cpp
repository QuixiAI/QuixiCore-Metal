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
  return array(
      {batch, 7, length, length}, dtype,
      std::make_shared<EdgeMlp256x7>(to_stream(s)),
      {contiguous(hidden, false, s), contiguous(first_weight, false, s),
       contiguous(first_bias, false, s), contiguous(second_weight, false, s),
       contiguous(second_bias, false, s)});
}

void EdgeMlp256x7::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("EdgeMlp256x7 has no CPU implementation.");
}
void EdgeMlp256x7::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_edge_mlp_256x7(
      encoder, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], output,
      inputs[0].shape(0), inputs[0].shape(1), type_to_name(inputs[0]));
}

std::vector<array> EdgeMlp256x7::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("EdgeMlp256x7 has no jvp implementation.");
}
std::vector<array> EdgeMlp256x7::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("EdgeMlp256x7 has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> EdgeMlp256x7::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("EdgeMlp256x7 has no vmap implementation.");
}

} // namespace mlx::core
