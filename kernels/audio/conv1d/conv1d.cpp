#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"
#include "conv1d/conv1d.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {
namespace {
bool audio_float(Dtype d) { return d == float32 || d == float16 || d == bfloat16; }
int conv_out(int length, int kernel, int stride, int pad_left, int pad_right, int dilation) {
  return (length + pad_left + pad_right - dilation * (kernel - 1) - 1) / stride + 1;
}
void check_common(const array& x, int stride, int padding, int dilation) {
  if (x.ndim() != 3 || !audio_float(x.dtype()) || stride <= 0 || padding < 0 || dilation <= 0)
    throw std::invalid_argument("audio_conv1d: need float (B,T,C), positive stride/dilation");
}
}

array audio_conv1d_direct(const array& x, const array& weight, const array& bias,
                          int stride, int padding, int dilation, bool has_bias,
                          StreamOrDevice s) {
  check_common(x, stride, padding, dilation);
  if (weight.ndim() != 3 || weight.dtype() != x.dtype() || weight.shape(2) != x.shape(2))
    throw std::invalid_argument("audio_conv1d_direct: weight must be same-dtype (O,K,C)");
  if (has_bias && (bias.ndim() != 1 || bias.shape(0) != weight.shape(0)))
    throw std::invalid_argument("audio_conv1d_direct: bias must be (O)");
  const int ot = conv_out(x.shape(1), weight.shape(1), stride, padding, padding, dilation);
  if (ot <= 0) throw std::invalid_argument("audio_conv1d_direct: empty output");
  return array({x.shape(0), ot, weight.shape(0)}, x.dtype(),
               std::make_shared<AudioConv1D>(to_stream(s), false, stride, padding, padding, dilation,
                                             has_bias, 0),
               {contiguous(x, false, s), contiguous(weight, false, s),
                contiguous(astype(bias, x.dtype(), s), false, s)});
}

array audio_depthwise_conv1d(const array& x, const array& weight, const array& bias,
                             int stride, int padding, int dilation, bool has_bias,
                             int activation, StreamOrDevice s) {
  return audio_depthwise_conv1d_asymmetric(
      x, weight, bias, stride, padding, padding, dilation, has_bias, activation, s);
}

array audio_depthwise_conv1d_asymmetric(
    const array& x, const array& weight, const array& bias,
    int stride, int pad_left, int pad_right, int dilation, bool has_bias,
    int activation, StreamOrDevice s) {
  check_common(x, stride, pad_left, dilation);
  if (pad_right < 0)
    throw std::invalid_argument("audio_depthwise_conv1d_asymmetric: padding must be nonnegative");
  if (weight.ndim() != 2 || weight.dtype() != x.dtype() || weight.shape(0) != x.shape(2) ||
      activation < 0 || activation > 1)
    throw std::invalid_argument("audio_depthwise_conv1d: weight must be same-dtype (C,K); activation 0/1");
  if (has_bias && (bias.ndim() != 1 || bias.shape(0) != x.shape(2)))
    throw std::invalid_argument("audio_depthwise_conv1d: bias must be (C)");
  const int ot = conv_out(x.shape(1), weight.shape(1), stride, pad_left, pad_right, dilation);
  if (ot <= 0) throw std::invalid_argument("audio_depthwise_conv1d: empty output");
  return array({x.shape(0), ot, x.shape(2)}, x.dtype(),
               std::make_shared<AudioConv1D>(to_stream(s), true, stride, pad_left, pad_right, dilation,
                                             has_bias, activation),
               {contiguous(x, false, s), contiguous(weight, false, s),
                contiguous(astype(bias, x.dtype(), s), false, s)});
}

void AudioConv1D::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AudioConv1D has no CPU implementation.");
}
void AudioConv1D::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0]; auto& weight = inputs[1]; auto& bias = inputs[2]; auto& out = outputs[0];
  auto& s = stream(); auto& dev = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = dev.get_command_encoder(s.index); MLXEncoder enc(dev, ce);
  if (depthwise_) {
    tk::launch_audio_depthwise_conv1d(enc, x, weight, bias, out, x.shape(0), x.shape(1),
        x.shape(2), out.shape(1), weight.shape(1), stride_, pad_left_, dilation_,
        has_bias_ ? 1 : 0, activation_, type_to_name(x));
  } else {
    tk::launch_audio_conv1d(enc, x, weight, bias, out, x.shape(0), x.shape(1), x.shape(2),
        out.shape(1), out.shape(2), weight.shape(1), stride_, pad_left_, dilation_,
        has_bias_ ? 1 : 0, type_to_name(x));
  }
}

#define AUDIO_NO_AUTODIFF(NAME)                                                  \
std::vector<array> NAME::jvp(const std::vector<array>&, const std::vector<array>&,\
                             const std::vector<int>&) { throw std::runtime_error("AudioConv1D has no jvp"); }\
std::vector<array> NAME::vjp(const std::vector<array>&, const std::vector<array>&,\
                             const std::vector<int>&, const std::vector<array>&) { throw std::runtime_error("AudioConv1D has no vjp"); }\
std::pair<std::vector<array>, std::vector<int>> NAME::vmap(                      \
    const std::vector<array>&, const std::vector<int>&) { throw std::runtime_error("AudioConv1D has no vmap"); }
AUDIO_NO_AUTODIFF(AudioConv1D)

}  // namespace mlx::core
