#include <cmath>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"
#include "cross_attn/cross_attn.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {
namespace {
bool ca_float(Dtype d) { return d == float32 || d == float16 || d == bfloat16; }
}

array cross_attention(const array& q, const array& k, const array& v,
                      const array& key_lengths, const array& bias,
                      float scale, float softcap, bool has_bias, StreamOrDevice s) {
  if (q.ndim() != 4 || k.ndim() != 4 || v.shape() != k.shape() || !ca_float(q.dtype()) ||
      k.dtype() != q.dtype() || q.shape(0) != k.shape(0) || q.shape(3) != k.shape(3) ||
      !(q.shape(3) == 64 || q.shape(3) == 128 || q.shape(3) == 256) ||
      q.shape(1) % k.shape(1) != 0) {
    throw std::invalid_argument("cross_attention: q(B,Hq,Tq,D), k/v(B,Hkv,Tk,D), D=64/128/256, Hq%Hkv=0");
  }
  if (key_lengths.ndim() != 1 || key_lengths.shape(0) != q.shape(0))
    throw std::invalid_argument("cross_attention: key_lengths must be (B)");
  if (has_bias && (bias.ndim() != 4 || bias.shape(0) != q.shape(0) ||
      bias.shape(1) != q.shape(1) || bias.shape(2) != q.shape(2) || bias.shape(3) != k.shape(2)))
    throw std::invalid_argument("cross_attention: bias must be (B,Hq,Tq,Tk)");
  const float used_scale = scale > 0.0f ? scale : 1.0f / std::sqrt(float(q.shape(3)));
  if (!std::isfinite(used_scale) || softcap < 0.0f || !std::isfinite(softcap))
    throw std::invalid_argument("cross_attention: scale must be positive/auto and softcap finite >=0");
  return array(q.shape(), q.dtype(),
               std::make_shared<CrossAttention>(to_stream(s), used_scale, softcap, has_bias),
               {contiguous(q, false, s), contiguous(k, false, s), contiguous(v, false, s),
                contiguous(astype(key_lengths, int32, s), false, s),
                contiguous(astype(bias, float32, s), false, s)});
}

void CrossAttention::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("CrossAttention has no CPU implementation.");
}
void CrossAttention::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
  auto& lengths = inputs[3]; auto& bias = inputs[4]; auto& out = outputs[0];
  auto& s = stream(); auto& dev = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = dev.get_command_encoder(s.index); MLXEncoder enc(dev, ce);
  tk::launch_cross_attention(enc, q, k, v, lengths, bias, out,
      q.shape(0), q.shape(1), q.shape(2), k.shape(1), k.shape(2), q.shape(3),
      scale_, softcap_, has_bias_ ? 1 : 0, type_to_name(q));
}

#define CA_NO_AUTODIFF(NAME)                                                     \
std::vector<array> NAME::jvp(const std::vector<array>&, const std::vector<array>&,\
                             const std::vector<int>&) { throw std::runtime_error("CrossAttention has no jvp"); }\
std::vector<array> NAME::vjp(const std::vector<array>&, const std::vector<array>&,\
                             const std::vector<int>&, const std::vector<array>&) { throw std::runtime_error("CrossAttention has no vjp"); }\
std::pair<std::vector<array>, std::vector<int>> NAME::vmap(                      \
    const std::vector<array>&, const std::vector<int>&) { throw std::runtime_error("CrossAttention has no vmap"); }
CA_NO_AUTODIFF(CrossAttention)

}  // namespace mlx::core
