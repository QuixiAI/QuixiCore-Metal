#include <stdexcept>
#include <string>
#include <utility>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "decode_linear/decode_linear.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool decode_float(Dtype d) { return d == float32 || d == bfloat16; }
bool decode_epilogue_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}

std::pair<int, int> decode_format_layout(const std::string& format) {
  if (format == "q4_0") return {32, 18};
  if (format == "q8_0") return {32, 34};
  if (format == "q6_K") return {256, 210};
  if (format == "nvfp4") return {16, 9};
  throw std::invalid_argument(
      "decode_linear_epilogue: format must be empty/dense, q4_0, q8_0, q6_K, or nvfp4");
}

void check_dense(const array& x, const array& weight, const array& bias, const char* name) {
  if (!decode_float(x.dtype()) || x.ndim() != 2 || weight.ndim() != 2 ||
      weight.dtype() != x.dtype() || bias.dtype() != x.dtype() ||
      x.shape(0) <= 0 || x.shape(1) <= 0 || weight.shape(0) <= 0 ||
      x.shape(1) != weight.shape(1) || bias.ndim() != 1 || bias.shape(0) != weight.shape(0)) {
    throw std::invalid_argument(std::string(name) +
        ": x (B,K), weight (N,K), and bias (N,) must share fp32/bf16 dtype");
  }
}

int check_epilogue_weight(
    const array& x, const array& weight, const std::string& format,
    const char* name) {
  if (!decode_epilogue_float(x.dtype()) || x.ndim() != 2 ||
      x.shape(0) <= 0 || x.shape(1) <= 0) {
    throw std::invalid_argument(std::string(name) +
        ": x must be non-empty (B,K) fp32/fp16/bf16");
  }
  const int K = x.shape(1);
  if (format.empty()) {
    if (weight.dtype() != x.dtype() || weight.ndim() != 2 ||
        weight.shape(0) <= 0 || weight.shape(1) != K) {
      throw std::invalid_argument(std::string(name) +
          ": dense weight must be (N,K) with x dtype");
    }
    return weight.shape(0);
  }
  const auto [block_k, block_bytes] = decode_format_layout(format);
  if (weight.dtype() != uint8 || weight.ndim() != 3 || K % block_k != 0 ||
      weight.shape(0) <= 0 || weight.shape(1) != K / block_k ||
      weight.shape(2) != block_bytes) {
    throw std::invalid_argument(std::string(name) +
        ": packed weight must be uint8 (N,K/block_k,block_bytes)");
  }
  return weight.shape(0);
}
}

array decode_linear(const array& x, const array& weight, const array& bias,
                    bool gelu, StreamOrDevice s) {
  check_dense(x, weight, bias, "decode_linear");
  return array({x.shape(0), weight.shape(0)}, x.dtype(),
               std::make_shared<DecodeLinear>(to_stream(s), gelu),
               {contiguous(x, false, s), contiguous(weight, false, s),
                contiguous(bias, false, s)});
}

array decode_linear_residual(const array& x, const array& weight, const array& bias,
                             const array& residual, StreamOrDevice s) {
  check_dense(x, weight, bias, "decode_linear_residual");
  if (residual.dtype() != x.dtype() || residual.ndim() != 2 ||
      residual.shape(0) != x.shape(0) || residual.shape(1) != weight.shape(0)) {
    throw std::invalid_argument("decode_linear_residual: residual must be (B,N) with x dtype");
  }
  return array(residual.shape(), x.dtype(),
               std::make_shared<DecodeLinearResidual>(to_stream(s)),
               {contiguous(x, false, s), contiguous(weight, false, s),
                contiguous(bias, false, s), contiguous(residual, false, s)});
}

array decode_linear_q8(const array& x, const array& packed_weight, const array& bias,
                       const array& residual, bool gelu, bool use_residual,
                       StreamOrDevice s) {
  if (!decode_float(x.dtype()) || x.ndim() != 2 || packed_weight.dtype() != uint8 ||
      packed_weight.ndim() != 3 || packed_weight.shape(2) != 34) {
    throw std::invalid_argument(
        "decode_linear_q8: x must be (B,K) fp32/bf16 and weight uint8 (N,K/32,34)");
  }
  const int B = x.shape(0), K = x.shape(1), N = packed_weight.shape(0);
  if (B <= 0 || K <= 0 || N <= 0 || K % 32 != 0 ||
      packed_weight.shape(1) != K / 32 || bias.ndim() != 1 ||
      bias.shape(0) != N || bias.dtype() != x.dtype()) {
    throw std::invalid_argument("decode_linear_q8: packed K or bias mismatch");
  }
  if (use_residual && (residual.dtype() != x.dtype() || residual.ndim() != 2 ||
                       residual.shape(0) != B || residual.shape(1) != N)) {
    throw std::invalid_argument("decode_linear_q8: residual must be (B,N) with x dtype");
  }
  return array({B, N}, x.dtype(),
               std::make_shared<DecodeLinearQ8>(to_stream(s), gelu, use_residual),
               {contiguous(x, false, s), contiguous(packed_weight, false, s),
                contiguous(bias, false, s), contiguous(residual, false, s)});
}

array decode_linear_epilogue(
    const array& x, const array& weight, const array& bias,
    const array& residual, const std::string& format, int activation,
    bool use_bias, bool use_residual, StreamOrDevice s) {
  if (activation < 0 || activation > 2) {
    throw std::invalid_argument(
        "decode_linear_epilogue: activation must be 0 (none), 1 (gelu), or 2 (silu)");
  }
  const int N = check_epilogue_weight(x, weight, format, "decode_linear_epilogue");
  const int B = x.shape(0);
  if (use_bias &&
      (bias.dtype() != x.dtype() || bias.ndim() != 1 || bias.shape(0) != N)) {
    throw std::invalid_argument(
        "decode_linear_epilogue: bias must be (N,) with x dtype");
  }
  if (use_residual &&
      (residual.dtype() != x.dtype() || residual.ndim() != 2 ||
       residual.shape(0) != B || residual.shape(1) != N)) {
    throw std::invalid_argument(
        "decode_linear_epilogue: residual must be (B,N) with x dtype");
  }
  return array(
      {B, N}, x.dtype(),
      std::make_shared<DecodeLinearEpilogue>(
          to_stream(s), format, activation, use_bias, use_residual),
      {contiguous(x, false, s), contiguous(weight, false, s),
       contiguous(astype(bias, x.dtype(), s), false, s),
       contiguous(astype(residual, x.dtype(), s), false, s)});
}

array decode_swiglu(
    const array& x, const array& gate_weight, const array& up_weight,
    const array& gate_bias, const array& up_bias, const std::string& format,
    bool use_bias, StreamOrDevice s) {
  const int N = check_epilogue_weight(x, gate_weight, format, "decode_swiglu");
  const int up_n = check_epilogue_weight(x, up_weight, format, "decode_swiglu");
  if (up_n != N || up_weight.shape() != gate_weight.shape()) {
    throw std::invalid_argument(
        "decode_swiglu: gate and up weights must have identical shapes");
  }
  if (use_bias &&
      (gate_bias.dtype() != x.dtype() || up_bias.dtype() != x.dtype() ||
       gate_bias.ndim() != 1 || up_bias.ndim() != 1 ||
       gate_bias.shape(0) != N || up_bias.shape(0) != N)) {
    throw std::invalid_argument(
        "decode_swiglu: both biases must be (N,) with x dtype");
  }
  return array(
      {x.shape(0), N}, x.dtype(),
      std::make_shared<DecodeSwiGLU>(to_stream(s), format, use_bias),
      {contiguous(x, false, s), contiguous(gate_weight, false, s),
       contiguous(up_weight, false, s),
       contiguous(astype(gate_bias, x.dtype(), s), false, s),
       contiguous(astype(up_bias, x.dtype(), s), false, s)});
}

void DecodeLinear::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeLinear has no CPU implementation.");
}
void DecodeLinearResidual::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeLinearResidual has no CPU implementation.");
}
void DecodeLinearQ8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeLinearQ8 has no CPU implementation.");
}
void DecodeLinearEpilogue::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeLinearEpilogue has no CPU implementation.");
}
void DecodeSwiGLU::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeSwiGLU has no CPU implementation.");
}

void DecodeLinear::eval_gpu(const std::vector<array>& in, std::vector<array>& out) {
  auto& y = out[0]; auto& s = stream(); auto& d = metal::device(s.device);
  y.set_data(allocator::malloc_or_wait(y.nbytes()));
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_decode_linear(enc, in[0], in[1], in[2], y, in[0].shape(0), in[0].shape(1),
                           in[1].shape(0), gelu_, type_to_name(in[0]));
}

void DecodeLinearResidual::eval_gpu(const std::vector<array>& in, std::vector<array>& out) {
  auto& y = out[0]; auto& s = stream(); auto& d = metal::device(s.device);
  y.set_data(allocator::malloc_or_wait(y.nbytes()));
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_decode_linear_residual(enc, in[0], in[1], in[2], in[3], y,
      in[0].shape(0), in[0].shape(1), in[1].shape(0), type_to_name(in[0]));
}

void DecodeLinearQ8::eval_gpu(const std::vector<array>& in, std::vector<array>& out) {
  auto& y = out[0]; auto& s = stream(); auto& d = metal::device(s.device);
  y.set_data(allocator::malloc_or_wait(y.nbytes()));
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_decode_linear_q8(enc, in[0], in[1], in[2], in[3], y,
      in[0].shape(0), in[0].shape(1), in[1].shape(0), gelu_, use_residual_,
      type_to_name(in[0]));
}

void DecodeLinearEpilogue::eval_gpu(
    const std::vector<array>& in, std::vector<array>& out) {
  auto& y = out[0]; auto& s = stream(); auto& d = metal::device(s.device);
  y.set_data(allocator::malloc_or_wait(y.nbytes()));
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_decode_linear_epilogue(
      enc, in[0], in[1], in[2], in[3], y, in[0].shape(0), in[0].shape(1),
      y.shape(1), activation_, use_bias_, use_residual_, format_,
      type_to_name(in[0]));
}

void DecodeSwiGLU::eval_gpu(
    const std::vector<array>& in, std::vector<array>& out) {
  auto& y = out[0]; auto& s = stream(); auto& d = metal::device(s.device);
  y.set_data(allocator::malloc_or_wait(y.nbytes()));
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  tk::launch_decode_swiglu(
      enc, in[0], in[1], in[2], in[3], in[4], y,
      in[0].shape(0), in[0].shape(1), y.shape(1), use_bias_, format_,
      type_to_name(in[0]));
}

#define TK_DECODE_NO_AUTODIFF(CLASS, LABEL)                                  \
  std::vector<array> CLASS::jvp(const std::vector<array>&, const std::vector<array>&, \
                                const std::vector<int>&) { throw std::runtime_error(LABEL " has no jvp implementation."); } \
  std::vector<array> CLASS::vjp(const std::vector<array>&, const std::vector<array>&, \
                                const std::vector<int>&, const std::vector<array>&) { throw std::runtime_error(LABEL " has no vjp implementation."); } \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                \
      const std::vector<array>&, const std::vector<int>&) { throw std::runtime_error(LABEL " has no vmap implementation."); }

TK_DECODE_NO_AUTODIFF(DecodeLinear, "DecodeLinear")
TK_DECODE_NO_AUTODIFF(DecodeLinearResidual, "DecodeLinearResidual")
TK_DECODE_NO_AUTODIFF(DecodeLinearQ8, "DecodeLinearQ8")
TK_DECODE_NO_AUTODIFF(DecodeLinearEpilogue, "DecodeLinearEpilogue")
TK_DECODE_NO_AUTODIFF(DecodeSwiGLU, "DecodeSwiGLU")

bool DecodeLinear::is_equivalent(const Primitive& other) const {
  return typeid(*this) == typeid(other) && gelu_ == static_cast<const DecodeLinear&>(other).gelu_;
}
bool DecodeLinearQ8::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const DecodeLinearQ8&>(other);
  return gelu_ == o.gelu_ && use_residual_ == o.use_residual_;
}
bool DecodeLinearEpilogue::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const DecodeLinearEpilogue&>(other);
  return format_ == o.format_ && activation_ == o.activation_ &&
      use_bias_ == o.use_bias_ && use_residual_ == o.use_residual_;
}
bool DecodeSwiGLU::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const DecodeSwiGLU&>(other);
  return format_ == o.format_ && use_bias_ == o.use_bias_;
}

} // namespace mlx::core
