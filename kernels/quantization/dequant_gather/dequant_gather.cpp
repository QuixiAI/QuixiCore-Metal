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
  if (format == "q4_K") return {256, 144};
  if (format == "q5_K") return {256, 176};
  if (format == "q6_K") return {256, 210};
  if (format == "q2_K") return {256, 84};
  if (format == "q3_K") return {256, 110};
  if (format == "iq4_nl") return {32, 18};
  if (format == "iq4_xs") return {256, 136};
  if (format == "kU4B8") return {128, 66};
  if (format == "kU4") return {128, 68};
  if (format == "hqq") return {64, 36};
  if (format == "fp8_e4m3") return {32, 34};
  if (format == "nvfp4") return {16, 9};
  if (format == "mxfp4") return {32, 17};
  throw std::invalid_argument(
      "quantized_embedding: unsupported packed format '" + format + "'");
}

Dtype embedding_dtype(const std::string& name) {
  if (name == "float16") return float16;
  if (name == "bfloat16") return bfloat16;
  if (name == "float32") return float32;
  throw std::invalid_argument(
      "quantized_embedding: output_dtype must be float16, bfloat16, or float32");
}

void check_packed_table(const array& table, int block_bytes, const char* name) {
  if (table.dtype() != uint8 || table.ndim() != 3 || table.shape(0) <= 0 ||
      table.shape(1) <= 0 || table.shape(2) != block_bytes) {
    throw std::invalid_argument(std::string(name) +
        ": table must be packed uint8 (rows, blocks, block_bytes)");
  }
}
}

array dequant_gather(
    const array& table, const array& ids, const std::string& format,
    float scale, StreamOrDevice s) {
  const auto [block_k, block_bytes] = format_layout(format);
  if (format != "q4_0" && format != "q8_0" && format != "q6_K") {
    throw std::invalid_argument(
        "dequant_gather: format must be q4_0, q8_0, or q6_K");
  }
  check_packed_table(table, block_bytes, "dequant_gather");
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

array quantized_embedding(
    const array& table, const array& ids, const array& add,
    const std::string& format, float scale, bool use_add,
    const std::string& output_dtype, StreamOrDevice s) {
  const auto [block_k, block_bytes] = format_layout(format);
  check_packed_table(table, block_bytes, "quantized_embedding");
  if (ids.size() == 0) {
    throw std::invalid_argument("quantized_embedding: ids must not be empty");
  }
  const Dtype dtype = embedding_dtype(output_dtype);
  const int columns = table.shape(1) * block_k;
  std::vector<int> shape(ids.shape().begin(), ids.shape().end());
  shape.push_back(columns);
  if (use_add && (add.dtype() != dtype || add.shape() != shape)) {
    throw std::invalid_argument(
        "quantized_embedding: add must match output shape and output_dtype");
  }
  return array(
      shape, dtype,
      std::make_shared<QuantizedEmbedding>(
          to_stream(s), format, table.shape(0), columns, scale, use_add,
          output_dtype),
      {contiguous(table, false, s),
       contiguous(astype(ids, int32, s), false, s),
       contiguous(astype(add, dtype, s), false, s)});
}

array quantized_embedding_bag(
    const array& table, const array& ids, const array& offsets,
    const array& sample_weights, const std::string& format, float scale,
    bool use_weights, bool mean_mode, const std::string& output_dtype,
    StreamOrDevice s) {
  const auto [block_k, block_bytes] = format_layout(format);
  check_packed_table(table, block_bytes, "quantized_embedding_bag");
  if (ids.ndim() != 1 || offsets.ndim() != 1 || offsets.shape(0) < 2) {
    throw std::invalid_argument(
        "quantized_embedding_bag: ids and offsets must be 1D; offsets needs bags + 1 entries");
  }
  if (use_weights &&
      (sample_weights.ndim() != 1 || sample_weights.size() != ids.size())) {
    throw std::invalid_argument(
        "quantized_embedding_bag: sample_weights must have one value per id");
  }
  const Dtype dtype = embedding_dtype(output_dtype);
  const int columns = table.shape(1) * block_k;
  const int bags = offsets.shape(0) - 1;
  return array(
      {bags, columns}, dtype,
      std::make_shared<QuantizedEmbeddingBag>(
          to_stream(s), format, table.shape(0), columns, scale,
          use_weights, mean_mode, output_dtype),
      {contiguous(table, false, s),
       contiguous(astype(ids, int32, s), false, s),
       contiguous(astype(offsets, int32, s), false, s),
       contiguous(astype(sample_weights, float32, s), false, s)});
}

void DequantGather::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DequantGather has no CPU implementation.");
}
void QuantizedEmbedding::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizedEmbedding has no CPU implementation.");
}
void QuantizedEmbeddingBag::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QuantizedEmbeddingBag has no CPU implementation.");
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
void QuantizedEmbedding::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_quantized_embedding(
      encoder, inputs[0], inputs[1], inputs[2], output, rows_, columns_,
      inputs[1].size(), scale_, use_add_, format_, type_to_name(output));
}
void QuantizedEmbeddingBag::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& output = outputs[0];
  auto& stream = this->stream();
  auto& device = metal::device(stream.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& command_encoder = device.get_command_encoder(stream.index);
  MLXEncoder encoder(device, command_encoder);
  tk::launch_quantized_embedding_bag(
      encoder, inputs[0], inputs[1], inputs[2], inputs[3], output,
      rows_, columns_, inputs[1].size(), inputs[2].shape(0) - 1, scale_,
      use_weights_, mean_mode_, format_, type_to_name(output));
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

#define TK_EMBEDDING_NO_AUTODIFF(CLASS, LABEL)                               \
std::vector<array> CLASS::jvp(                                                \
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) { \
  throw std::runtime_error(LABEL " has no jvp implementation.");             \
}                                                                             \
std::vector<array> CLASS::vjp(                                                \
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, \
    const std::vector<array>&) {                                               \
  throw std::runtime_error(LABEL " has no vjp implementation.");             \
}                                                                             \
std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                  \
    const std::vector<array>&, const std::vector<int>&) {                      \
  throw std::runtime_error(LABEL " has no vmap implementation.");            \
}

TK_EMBEDDING_NO_AUTODIFF(QuantizedEmbedding, "QuantizedEmbedding")
TK_EMBEDDING_NO_AUTODIFF(QuantizedEmbeddingBag, "QuantizedEmbeddingBag")

bool QuantizedEmbedding::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const QuantizedEmbedding&>(other);
  return format_ == o.format_ && rows_ == o.rows_ && columns_ == o.columns_ &&
      scale_ == o.scale_ && use_add_ == o.use_add_ &&
      output_dtype_ == o.output_dtype_;
}
bool QuantizedEmbeddingBag::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const QuantizedEmbeddingBag&>(other);
  return format_ == o.format_ && rows_ == o.rows_ && columns_ == o.columns_ &&
      scale_ == o.scale_ && use_weights_ == o.use_weights_ &&
      mean_mode_ == o.mean_mode_ && output_dtype_ == o.output_dtype_;
}

} // namespace mlx::core
