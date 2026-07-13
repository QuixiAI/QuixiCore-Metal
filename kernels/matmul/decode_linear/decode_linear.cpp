#include <stdexcept>

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

void check_dense(const array& x, const array& weight, const array& bias, const char* name) {
  if (!decode_float(x.dtype()) || x.ndim() != 2 || weight.ndim() != 2 ||
      weight.dtype() != x.dtype() || bias.dtype() != x.dtype() ||
      x.shape(0) <= 0 || x.shape(1) <= 0 || weight.shape(0) <= 0 ||
      x.shape(1) != weight.shape(1) || bias.ndim() != 1 || bias.shape(0) != weight.shape(0)) {
    throw std::invalid_argument(std::string(name) +
        ": x (B,K), weight (N,K), and bias (N,) must share fp32/bf16 dtype");
  }
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

void DecodeLinear::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeLinear has no CPU implementation.");
}
void DecodeLinearResidual::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeLinearResidual has no CPU implementation.");
}
void DecodeLinearQ8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeLinearQ8 has no CPU implementation.");
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

bool DecodeLinear::is_equivalent(const Primitive& other) const {
  return typeid(*this) == typeid(other) && gelu_ == static_cast<const DecodeLinear&>(other).gelu_;
}
bool DecodeLinearQ8::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const DecodeLinearQ8&>(other);
  return gelu_ == o.gelu_ && use_residual_ == o.use_residual_;
}

} // namespace mlx::core
