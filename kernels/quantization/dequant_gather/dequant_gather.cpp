#include <stdexcept>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "dequant_gather/dequant_gather.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
std::pair<int, int> format_layout(const std::string& format) {
  if (format == "q4_0") return {32, 18};
  if (format == "q8_0") return {32, 34};
  if (format == "q6_K") return {256, 210};
  throw std::invalid_argument("dequant_gather: format must be q4_0, q8_0, or q6_K");
}
}

array dequant_gather(
    const array& table, const array& ids, const std::string& format,
    float scale, StreamOrDevice s) {
  const auto [block_k, block_bytes] = format_layout(format);
  if (table.dtype() != uint8 || table.ndim() != 3 || table.shape(0) <= 0 ||
      table.shape(1) <= 0 || table.shape(2) != block_bytes) {
    throw std::invalid_argument(
        "dequant_gather: table must be packed uint8 (rows, blocks, block_bytes)");
  }
  if (ids.size() == 0) {
    throw std::invalid_argument("dequant_gather: ids must not be empty");
  }
  const int columns = table.shape(1) * block_k;
  std::vector<int> output_shape(ids.shape().begin(), ids.shape().end());
  output_shape.push_back(columns);
  return array(
      output_shape, float16,
      std::make_shared<DequantGather>(to_stream(s), format, table.shape(0), columns, scale),
      {contiguous(table, false, s), contiguous(astype(ids, int32, s), false, s)});
}

void DequantGather::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DequantGather has no CPU implementation.");
}
void DequantGather::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_dequant_gather(
      encoder, inputs[0], inputs[1], output, rows_, columns_, inputs[1].size(),
      scale_, format_);
}
std::vector<array> DequantGather::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("DequantGather has no jvp implementation.");
}
std::vector<array> DequantGather::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("DequantGather has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> DequantGather::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("DequantGather has no vmap implementation.");
}
bool DequantGather::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const DequantGather&>(other);
  return format_ == o.format_ && rows_ == o.rows_ && columns_ == o.columns_ &&
      scale_ == o.scale_;
}

} // namespace mlx::core
