#include <cmath>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"
#include "relative_attention/relative_attention.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {
namespace {
bool ar_float(Dtype d) { return d == float32 || d == float16 || d == bfloat16; }
}

array audio_relative_attention(
    const array& q, const array& k, const array& v,
    const array& relative_k, const array& per_dim_scale,
    const array& lengths, int chunk_size, int left_context, int right_context,
    float q_scale, float k_scale, float softcap, StreamOrDevice s) {
  if (q.ndim() != 4 || q.shape() != k.shape() || q.shape() != v.shape() ||
      !ar_float(q.dtype()) || k.dtype() != q.dtype() || v.dtype() != q.dtype() ||
      !(q.shape(3) == 64 || q.shape(3) == 128 || q.shape(3) == 256)) {
    throw std::invalid_argument(
        "audio_relative_attention: q/k/v must be same-shape float (B,T,H,D), D=64/128/256");
  }
  if (relative_k.ndim() != 3 || relative_k.dtype() != q.dtype() ||
      relative_k.shape(1) != q.shape(2) || relative_k.shape(2) != q.shape(3) ||
      relative_k.shape(0) <= 0 || per_dim_scale.ndim() != 1 ||
      per_dim_scale.shape(0) != q.shape(3) || lengths.ndim() != 1 ||
      lengths.shape(0) != q.shape(0)) {
    throw std::invalid_argument(
        "audio_relative_attention: need relative_k(P,H,D), per_dim_scale(D), lengths(B)");
  }
  if (chunk_size <= 0 || left_context <= 0 || right_context < 0 || softcap < 0.0f ||
      !std::isfinite(softcap)) {
    throw std::invalid_argument("audio_relative_attention: invalid context geometry or softcap");
  }
  const float ln2 = std::log(2.0f);
  const float used_q_scale = q_scale > 0.0f
      ? q_scale : 1.0f / (std::sqrt(float(q.shape(3))) * ln2);
  const float used_k_scale = k_scale > 0.0f ? k_scale : 1.0f / ln2;
  if (!std::isfinite(used_q_scale) || !std::isfinite(used_k_scale))
    throw std::invalid_argument("audio_relative_attention: scales must be finite and positive/auto");
  return array(
      q.shape(), q.dtype(),
      std::make_shared<AudioRelativeAttention>(
          to_stream(s), chunk_size, left_context, right_context,
          used_q_scale, used_k_scale, softcap),
      {contiguous(q, false, s), contiguous(k, false, s), contiguous(v, false, s),
       contiguous(relative_k, false, s),
       contiguous(astype(per_dim_scale, float32, s), false, s),
       contiguous(astype(lengths, int32, s), false, s)});
}

void AudioRelativeAttention::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AudioRelativeAttention has no CPU implementation.");
}
void AudioRelativeAttention::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0]; auto& out = outputs[0]; auto& s = stream();
  auto& dev = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = dev.get_command_encoder(s.index); MLXEncoder enc(dev, ce);
  tk::launch_audio_relative_attention(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], out,
      q.shape(0), q.shape(1), q.shape(2), q.shape(3), inputs[3].shape(0),
      chunk_size_, left_context_, right_context_, q_scale_, k_scale_, softcap_,
      type_to_name(q));
}

#define AR_NO_AUTODIFF(NAME)                                                     \
std::vector<array> NAME::jvp(const std::vector<array>&, const std::vector<array>&,\
                             const std::vector<int>&) { throw std::runtime_error("AudioRelativeAttention has no jvp"); }\
std::vector<array> NAME::vjp(const std::vector<array>&, const std::vector<array>&,\
                             const std::vector<int>&, const std::vector<array>&) { throw std::runtime_error("AudioRelativeAttention has no vjp"); }\
std::pair<std::vector<array>, std::vector<int>> NAME::vmap(                      \
    const std::vector<array>&, const std::vector<int>&) { throw std::runtime_error("AudioRelativeAttention has no vmap"); }
AR_NO_AUTODIFF(AudioRelativeAttention)

}  // namespace mlx::core
