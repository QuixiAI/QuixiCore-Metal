#include <stdexcept>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "base_q/base_q.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {
namespace {

struct BaseQShape {
  int rows;
  int columns;
  int groups_per_row;
};

struct BaseQExpertShape {
  int experts;
  int output_rows;
  int columns;
  int groups_per_row;
};

Dtype base_q_output_dtype(const std::string& name) {
  if (name == "float16" || name == "f16") return float16;
  if (name == "bfloat16" || name == "bf16") return bfloat16;
  if (name == "float32" || name == "f32") return float32;
  throw std::invalid_argument(
      "BaseQN: output_dtype must be float16, bfloat16, or float32");
}

const char* base_q_output_dtype_name(Dtype dtype) {
  if (dtype == float16) return "float16";
  if (dtype == bfloat16) return "bfloat16";
  if (dtype == float32) return "float32";
  throw std::invalid_argument(
      "BaseQN matmul: x must be float16, bfloat16, or float32");
}

Dtype base_q_scale_dtype(tk::BaseQScaleType type) {
  switch (type) {
    case tk::BaseQScaleType::BF16: return bfloat16;
    case tk::BaseQScaleType::F16: return float16;
    case tk::BaseQScaleType::E8M0:
    case tk::BaseQScaleType::E4M3: return uint8;
  }
  throw std::invalid_argument("BaseQN: invalid scale type");
}

BaseQShape check_base_q_storage(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const tk::BaseQDescriptor& descriptor,
    const char* operation) {
  const std::string prefix = std::string(operation) + ": ";
  if (codes.dtype() != uint8 || codes.ndim() != 2) {
    throw std::invalid_argument(prefix + "codes must be uint8 (rows, packed_bytes)");
  }
  if (scales.dtype() != base_q_scale_dtype(descriptor.scale_type) ||
      scales.ndim() != 2) {
    throw std::invalid_argument(
        prefix + "scales must be a rank-2 tensor with the declared scale_dtype");
  }
  const int rows = codes.shape(0);
  const int groups = scales.shape(1);
  if (rows <= 0 || groups <= 0 || scales.shape(0) != rows) {
    throw std::invalid_argument(prefix + "codes/scales rows and group count must be positive and match");
  }
  const int columns = groups * descriptor.group_size;
  const long packed_bits = static_cast<long>(columns) * descriptor.bits;
  if ((packed_bits & 7) != 0 || codes.shape(1) != packed_bits / 8) {
    throw std::invalid_argument(
        prefix + "codes.shape[1] must equal columns * bits / 8");
  }
  if (!descriptor.symmetric && !biases.has_value()) {
    throw std::invalid_argument(prefix + "biases are required for asymmetric BaseQN");
  }
  if (biases.has_value() &&
      (biases->dtype() != scales.dtype() || biases->shape() != scales.shape())) {
    throw std::invalid_argument(prefix + "biases must match scales shape and dtype");
  }
  return {rows, columns, groups};
}

BaseQExpertShape check_base_q_expert_storage(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const tk::BaseQDescriptor& descriptor,
    const char* operation) {
  const std::string prefix = std::string(operation) + ": ";
  if (codes.dtype() != uint8 || codes.ndim() != 3) {
    throw std::invalid_argument(
        prefix + "codes must be uint8 (experts, output_rows, packed_bytes)");
  }
  if (scales.dtype() != base_q_scale_dtype(descriptor.scale_type) ||
      scales.ndim() != 3) {
    throw std::invalid_argument(
        prefix + "scales must be rank 3 with the declared scale_dtype");
  }
  const int experts = codes.shape(0);
  const int output_rows = codes.shape(1);
  const int groups = scales.shape(2);
  if (experts <= 0 || output_rows <= 0 || groups <= 0 ||
      scales.shape(0) != experts || scales.shape(1) != output_rows) {
    throw std::invalid_argument(
        prefix + "codes/scales expert, row, and group dimensions must be positive and match");
  }
  const int columns = groups * descriptor.group_size;
  const long packed_bits = static_cast<long>(columns) * descriptor.bits;
  if ((packed_bits & 7) != 0 || codes.shape(2) != packed_bits / 8) {
    throw std::invalid_argument(
        prefix + "codes.shape[2] must equal columns * bits / 8");
  }
  if (!descriptor.symmetric && !biases.has_value()) {
    throw std::invalid_argument(prefix + "biases are required for asymmetric BaseQN");
  }
  if (biases.has_value() &&
      (biases->dtype() != scales.dtype() || biases->shape() != scales.shape())) {
    throw std::invalid_argument(prefix + "biases must match scales shape and dtype");
  }
  return {experts, output_rows, columns, groups};
}

array base_q_bias_input(
    const array& scales, const std::optional<array>& biases, StreamOrDevice s) {
  return contiguous(biases.has_value() ? *biases : scales, false, s);
}

array base_qmatmul_impl(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& x, int bits,
    int group_size, const std::string& scale_dtype, bool symmetric,
    const std::string& layout, bool gemv_only, StreamOrDevice s) {
  const auto descriptor = tk::make_base_q_descriptor(
      bits, group_size, scale_dtype, symmetric, layout);
  const auto shape = check_base_q_storage(
      codes, scales, biases, descriptor, gemv_only ? "base_qgemv" : "base_qgemm");
  if (x.ndim() != 2 || x.shape(0) != shape.columns) {
    throw std::invalid_argument(
        std::string(gemv_only ? "base_qgemv" : "base_qgemm") +
        ": x must be (K, M) with K matching the packed weights");
  }
  if (gemv_only && x.shape(1) != 1) {
    throw std::invalid_argument("base_qgemv: x must be (K, 1)");
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16 && x.dtype() != float32) {
    throw std::invalid_argument("BaseQN matmul: x must be float16, bfloat16, or float32");
  }
  if (x.shape(1) <= 0) {
    throw std::invalid_argument("BaseQN matmul: M must be positive");
  }
  // The direct decode kernel is strongly decode/GEMV-shaped. Measurements at
  // M=2 and M=8 show that materializing once and using MLX GEMM is 6.8-7.0x
  // faster, so reserve the direct primitive for the M=1 specialization.
  if (!gemv_only && x.shape(1) > 1) {
    auto weights = base_qdequant(
        codes, scales, biases, bits, group_size, scale_dtype, symmetric,
        layout, base_q_output_dtype_name(x.dtype()), s);
    return matmul(weights, x, s);
  }
  return array(
      {shape.rows, x.shape(1)}, x.dtype(),
      std::make_shared<BaseQMatmul>(
          to_stream(s), descriptor, shape.rows, shape.columns, x.shape(1)),
      {contiguous(codes, false, s), contiguous(scales, false, s),
       base_q_bias_input(scales, biases, s), contiguous(x, false, s)});
}

} // namespace

array base_qdequant(
    const array& codes, const array& scales,
    const std::optional<array>& biases, int bits, int group_size,
    const std::string& scale_dtype, bool symmetric, const std::string& layout,
    const std::string& output_dtype, StreamOrDevice s) {
  const auto descriptor = tk::make_base_q_descriptor(
      bits, group_size, scale_dtype, symmetric, layout);
  const auto shape = check_base_q_storage(
      codes, scales, biases, descriptor, "base_qdequant");
  return array(
      {shape.rows, shape.columns}, base_q_output_dtype(output_dtype),
      std::make_shared<BaseQDequant>(
          to_stream(s), descriptor, shape.rows, shape.columns),
      {contiguous(codes, false, s), contiguous(scales, false, s),
       base_q_bias_input(scales, biases, s)});
}

array base_qgemv(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& x, int bits,
    int group_size, const std::string& scale_dtype, bool symmetric,
    const std::string& layout, StreamOrDevice s) {
  return base_qmatmul_impl(
      codes, scales, biases, x, bits, group_size, scale_dtype, symmetric,
      layout, true, s);
}

array base_qgemm(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& x, int bits,
    int group_size, const std::string& scale_dtype, bool symmetric,
    const std::string& layout, StreamOrDevice s) {
  return base_qmatmul_impl(
      codes, scales, biases, x, bits, group_size, scale_dtype, symmetric,
      layout, false, s);
}

std::vector<array> base_qgemv_qkv(
    const array& q_codes, const array& q_scales,
    const std::optional<array>& q_biases, const array& k_codes,
    const array& k_scales, const std::optional<array>& k_biases,
    const array& v_codes, const array& v_scales,
    const std::optional<array>& v_biases, const array& x, int bits,
    int group_size, const std::string& scale_dtype, bool symmetric,
    const std::string& layout, StreamOrDevice s) {
  const auto descriptor = tk::make_base_q_descriptor(
      bits, group_size, scale_dtype, symmetric, layout);
  const auto q_shape = check_base_q_storage(
      q_codes, q_scales, q_biases, descriptor, "base_qgemv_qkv(q)");
  const auto k_shape = check_base_q_storage(
      k_codes, k_scales, k_biases, descriptor, "base_qgemv_qkv(k)");
  const auto v_shape = check_base_q_storage(
      v_codes, v_scales, v_biases, descriptor, "base_qgemv_qkv(v)");
  if (q_shape.columns != k_shape.columns || q_shape.columns != v_shape.columns) {
    throw std::invalid_argument("base_qgemv_qkv: Q/K/V inner dimensions must match");
  }
  if (x.ndim() != 2 || x.shape(0) != q_shape.columns || x.shape(1) != 1 ||
      (x.dtype() != float16 && x.dtype() != bfloat16 && x.dtype() != float32)) {
    throw std::invalid_argument(
        "base_qgemv_qkv: x must be float16, bfloat16, or float32 (K, 1)");
  }
  return array::make_arrays(
      {{q_shape.rows, 1}, {k_shape.rows, 1}, {v_shape.rows, 1}},
      {x.dtype(), x.dtype(), x.dtype()},
      std::make_shared<BaseQGemvQKV>(
          to_stream(s), descriptor, q_shape.rows, k_shape.rows, v_shape.rows,
          q_shape.columns),
      {contiguous(q_codes, false, s), contiguous(q_scales, false, s),
       base_q_bias_input(q_scales, q_biases, s), contiguous(k_codes, false, s),
       contiguous(k_scales, false, s), base_q_bias_input(k_scales, k_biases, s),
       contiguous(v_codes, false, s), contiguous(v_scales, false, s),
       base_q_bias_input(v_scales, v_biases, s), contiguous(x, false, s)});
}

array base_qgemv_swiglu(
    const array& gate_codes, const array& gate_scales,
    const std::optional<array>& gate_biases, const array& up_codes,
    const array& up_scales, const std::optional<array>& up_biases,
    const array& x, int bits, int group_size, const std::string& scale_dtype,
    bool symmetric, const std::string& layout, StreamOrDevice s) {
  const auto descriptor = tk::make_base_q_descriptor(
      bits, group_size, scale_dtype, symmetric, layout);
  const auto gate_shape = check_base_q_storage(
      gate_codes, gate_scales, gate_biases, descriptor, "base_qgemv_swiglu(gate)");
  const auto up_shape = check_base_q_storage(
      up_codes, up_scales, up_biases, descriptor, "base_qgemv_swiglu(up)");
  if (gate_shape.rows != up_shape.rows || gate_shape.columns != up_shape.columns) {
    throw std::invalid_argument("base_qgemv_swiglu: gate/up shapes must match");
  }
  if (x.ndim() != 2 || x.shape(0) != gate_shape.columns || x.shape(1) != 1 ||
      (x.dtype() != float16 && x.dtype() != bfloat16 && x.dtype() != float32)) {
    throw std::invalid_argument(
        "base_qgemv_swiglu: x must be float16, bfloat16, or float32 (K, 1)");
  }
  return array(
      {gate_shape.rows, 1}, x.dtype(),
      std::make_shared<BaseQGemvSwiGLU>(
          to_stream(s), descriptor, gate_shape.rows, gate_shape.columns),
      {contiguous(gate_codes, false, s), contiguous(gate_scales, false, s),
       base_q_bias_input(gate_scales, gate_biases, s),
       contiguous(up_codes, false, s), contiguous(up_scales, false, s),
       base_q_bias_input(up_scales, up_biases, s), contiguous(x, false, s)});
}

array base_qembedding(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& ids, int bits,
    int group_size, const std::string& scale_dtype, bool symmetric,
    const std::string& layout, const std::string& output_dtype,
    StreamOrDevice s) {
  const auto descriptor = tk::make_base_q_descriptor(
      bits, group_size, scale_dtype, symmetric, layout);
  const auto shape = check_base_q_storage(
      codes, scales, biases, descriptor, "base_qembedding");
  if (ids.size() == 0) {
    throw std::invalid_argument("base_qembedding: ids must not be empty");
  }
  std::vector<int> output_shape(ids.shape().begin(), ids.shape().end());
  output_shape.push_back(shape.columns);
  return array(
      output_shape, base_q_output_dtype(output_dtype),
      std::make_shared<BaseQEmbedding>(
          to_stream(s), descriptor, shape.rows, shape.columns, ids.size()),
      {contiguous(codes, false, s), contiguous(scales, false, s),
       base_q_bias_input(scales, biases, s),
       contiguous(astype(ids, int32, s), false, s)});
}

array base_qmoe_gemm(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& input,
    const array& expert_of_tile, int bits, int group_size,
    const std::string& scale_dtype, bool symmetric,
    const std::string& layout, StreamOrDevice s) {
  const auto descriptor = tk::make_base_q_descriptor(
      bits, group_size, scale_dtype, symmetric, layout);
  const auto shape = check_base_q_expert_storage(
      codes, scales, biases, descriptor, "base_qmoe_gemm");
  if (input.ndim() != 2 || input.shape(1) != shape.columns ||
      (input.dtype() != float16 && input.dtype() != bfloat16 &&
       input.dtype() != float32)) {
    throw std::invalid_argument(
        "base_qmoe_gemm: input must be float16, bfloat16, or float32 (total_rows, K)");
  }
  const int total_rows = input.shape(0);
  if (total_rows <= 0 || total_rows % 32 != 0 || shape.columns % 32 != 0 ||
      shape.output_rows % 32 != 0) {
    throw std::invalid_argument(
        "base_qmoe_gemm: total_rows, K, and output_rows must be positive multiples of 32");
  }
  if (expert_of_tile.ndim() != 1 ||
      expert_of_tile.shape(0) != total_rows / 32) {
    throw std::invalid_argument(
        "base_qmoe_gemm: expert_of_tile must be (total_rows/32,)");
  }
  return array(
      {total_rows, shape.output_rows}, input.dtype(),
      std::make_shared<BaseQMoeGemm>(
          to_stream(s), descriptor, total_rows, shape.columns,
          shape.output_rows),
      {contiguous(input, false, s), contiguous(codes, false, s),
       contiguous(scales, false, s), base_q_bias_input(scales, biases, s),
       contiguous(astype(expert_of_tile, int32, s), false, s)});
}

array base_qmoe_swiglu(
    const array& codes, const array& scales,
    const std::optional<array>& biases, const array& input,
    const array& expert_of_tile, int bits, int group_size,
    const std::string& scale_dtype, bool symmetric,
    const std::string& layout, StreamOrDevice s) {
  const auto descriptor = tk::make_base_q_descriptor(
      bits, group_size, scale_dtype, symmetric, layout);
  const auto shape = check_base_q_expert_storage(
      codes, scales, biases, descriptor, "base_qmoe_swiglu");
  if (shape.output_rows % 2 != 0) {
    throw std::invalid_argument(
        "base_qmoe_swiglu: packed output-row dimension must be 2 * intermediate");
  }
  const int intermediate = shape.output_rows / 2;
  if (input.ndim() != 2 || input.shape(1) != shape.columns ||
      (input.dtype() != float16 && input.dtype() != bfloat16 &&
       input.dtype() != float32)) {
    throw std::invalid_argument(
        "base_qmoe_swiglu: input must be float16, bfloat16, or float32 (total_rows, K)");
  }
  const int total_rows = input.shape(0);
  if (total_rows <= 0 || total_rows % 32 != 0 || shape.columns % 32 != 0 ||
      intermediate <= 0 || intermediate % 32 != 0) {
    throw std::invalid_argument(
        "base_qmoe_swiglu: total_rows, K, and intermediate must be positive multiples of 32");
  }
  if (expert_of_tile.ndim() != 1 ||
      expert_of_tile.shape(0) != total_rows / 32) {
    throw std::invalid_argument(
        "base_qmoe_swiglu: expert_of_tile must be (total_rows/32,)");
  }
  return array(
      {total_rows, intermediate}, input.dtype(),
      std::make_shared<BaseQMoeSwiGLU>(
          to_stream(s), descriptor, total_rows, shape.columns, intermediate),
      {contiguous(input, false, s), contiguous(codes, false, s),
       contiguous(scales, false, s), base_q_bias_input(scales, biases, s),
       contiguous(astype(expert_of_tile, int32, s), false, s)});
}

void BaseQDequant::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BaseQDequant has no CPU implementation.");
}
void BaseQMatmul::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BaseQMatmul has no CPU implementation.");
}
void BaseQEmbedding::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BaseQEmbedding has no CPU implementation.");
}
void BaseQGemvQKV::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BaseQGemvQKV has no CPU implementation.");
}
void BaseQGemvSwiGLU::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BaseQGemvSwiGLU has no CPU implementation.");
}
void BaseQMoeGemm::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BaseQMoeGemm has no CPU implementation.");
}
void BaseQMoeSwiGLU::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("BaseQMoeSwiGLU has no CPU implementation.");
}

void BaseQDequant::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_base_qdequant(
      encoder, output, inputs[0], inputs[1], inputs[2], rows_, columns_,
      descriptor_, type_to_name(output));
}

void BaseQMatmul::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_base_qmatmul(
      encoder, output, inputs[0], inputs[1], inputs[2], inputs[3], rows_,
      inner_, columns_, descriptor_, type_to_name(output));
}

void BaseQEmbedding::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_base_qembedding(
      encoder, output, inputs[0], inputs[1], inputs[2], inputs[3], rows_,
      columns_, tokens_, descriptor_, type_to_name(output));
}

void BaseQGemvQKV::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q_output = outputs[0];
  auto& k_output = outputs[1];
  auto& v_output = outputs[2];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  q_output.set_data(allocator::malloc_or_wait(q_output.nbytes()));
  k_output.set_data(allocator::malloc_or_wait(k_output.nbytes()));
  v_output.set_data(allocator::malloc_or_wait(v_output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_base_qgemv_qkv(
      encoder, q_output, k_output, v_output, inputs[0], inputs[1], inputs[2],
      inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8],
      inputs[9], q_rows_, k_rows_, v_rows_, inner_, descriptor_,
      type_to_name(q_output));
}

void BaseQGemvSwiGLU::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_base_qgemv_swiglu(
      encoder, output, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
      inputs[5], inputs[6], rows_, inner_, descriptor_, type_to_name(output));
}

void BaseQMoeGemm::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_base_qmoe_gemm(
      encoder, output, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
      total_rows_, inner_, output_rows_, descriptor_, type_to_name(output));
}

void BaseQMoeSwiGLU::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_base_qmoe_swiglu(
      encoder, output, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
      total_rows_, inner_, intermediate_, descriptor_, type_to_name(output));
}

#define BASE_Q_NO_AUTODIFF(CLASS)                                            \
std::vector<array> CLASS::jvp(                                               \
    const std::vector<array>&, const std::vector<array>&,                    \
    const std::vector<int>&) {                                               \
  throw std::runtime_error(#CLASS " has no jvp implementation.");            \
}                                                                            \
std::vector<array> CLASS::vjp(                                               \
    const std::vector<array>&, const std::vector<array>&,                    \
    const std::vector<int>&, const std::vector<array>&) {                    \
  throw std::runtime_error(#CLASS " has no vjp implementation.");            \
}                                                                            \
std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                 \
    const std::vector<array>&, const std::vector<int>&) {                    \
  throw std::runtime_error(#CLASS " has no vmap implementation.");           \
}

BASE_Q_NO_AUTODIFF(BaseQDequant)
BASE_Q_NO_AUTODIFF(BaseQMatmul)
BASE_Q_NO_AUTODIFF(BaseQEmbedding)
BASE_Q_NO_AUTODIFF(BaseQGemvQKV)
BASE_Q_NO_AUTODIFF(BaseQGemvSwiGLU)
BASE_Q_NO_AUTODIFF(BaseQMoeGemm)
BASE_Q_NO_AUTODIFF(BaseQMoeSwiGLU)

bool BaseQDequant::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  const auto& value = static_cast<const BaseQDequant&>(other);
  return descriptor_ == value.descriptor_ && rows_ == value.rows_ &&
      columns_ == value.columns_;
}

bool BaseQMatmul::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  const auto& value = static_cast<const BaseQMatmul&>(other);
  return descriptor_ == value.descriptor_ && rows_ == value.rows_ &&
      inner_ == value.inner_ && columns_ == value.columns_;
}

bool BaseQEmbedding::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  const auto& value = static_cast<const BaseQEmbedding&>(other);
  return descriptor_ == value.descriptor_ && rows_ == value.rows_ &&
      columns_ == value.columns_ && tokens_ == value.tokens_;
}

bool BaseQGemvQKV::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  const auto& value = static_cast<const BaseQGemvQKV&>(other);
  return descriptor_ == value.descriptor_ && q_rows_ == value.q_rows_ &&
      k_rows_ == value.k_rows_ && v_rows_ == value.v_rows_ &&
      inner_ == value.inner_;
}

bool BaseQGemvSwiGLU::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  const auto& value = static_cast<const BaseQGemvSwiGLU&>(other);
  return descriptor_ == value.descriptor_ && rows_ == value.rows_ &&
      inner_ == value.inner_;
}

bool BaseQMoeGemm::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  const auto& value = static_cast<const BaseQMoeGemm&>(other);
  return descriptor_ == value.descriptor_ && total_rows_ == value.total_rows_ &&
      inner_ == value.inner_ && output_rows_ == value.output_rows_;
}

bool BaseQMoeSwiGLU::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  const auto& value = static_cast<const BaseQMoeSwiGLU&>(other);
  return descriptor_ == value.descriptor_ && total_rows_ == value.total_rows_ &&
      inner_ == value.inner_ && intermediate_ == value.intermediate_;
}

} // namespace mlx::core
