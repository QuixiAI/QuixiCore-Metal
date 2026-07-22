// Copyright © 2024 Apple Inc.

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "attn_fwd_sg/attn_fwd_sg.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array attn_fwd_sg_d256(
    const array& q,
    const array& k,
    const array& v,
    float scale /* = 0.0f */,
    int window /* = 0 */,
    StreamOrDevice s /* = {} */) {
  if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3 ||
      q.shape(2) != 256 || k.shape(2) != 256 || v.shape(2) != 256) {
    throw std::invalid_argument("attn_fwd_sg_d256: q/k/v must be (T, H, 256)");
  }
  const int T = q.shape(0), Hq = q.shape(1), Hkv = k.shape(0) == T ? k.shape(1) : -1;
  if (k.shape(0) != T || v.shape(0) != T || v.shape(1) != Hkv || Hkv <= 0 || Hq % Hkv != 0) {
    throw std::invalid_argument(
        "attn_fwd_sg_d256: k/v must share T and Hkv, and Hq must be a multiple of Hkv");
  }
  const float scale_f = scale > 0.0f ? scale : 1.0f / std::sqrt(256.0f);
  // Pad K/V token count up to the 32-key block so the tail block's simdgroup_load is in-bounds.
  const int T_pad = ((T + 31) / 32) * 32;
  array kf = astype(k, float16, s);
  array vf = astype(v, float16, s);
  if (T_pad != T) {
    kf = pad(kf, {0}, Shape{0}, Shape{T_pad - T}, array(0, float16), "constant", s);
    vf = pad(vf, {0}, Shape{0}, Shape{T_pad - T}, array(0, float16), "constant", s);
  }
  return array(
      {T, Hq, 256}, float32,
      std::make_shared<AttnFwdSgD256>(to_stream(s), scale_f, window, T),
      {contiguous(astype(q, float32, s), false, s),
       contiguous(kf, false, s), contiguous(vf, false, s)});
}

void AttnFwdSgD256::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AttnFwdSgD256 has no CPU implementation.");
}

void AttnFwdSgD256::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int Hq = q.shape(1);
  const int Hkv = k.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_fwd_sg_d256(enc, q, k, v, out, static_cast<uint32_t>(n_tokens_),
                              static_cast<uint32_t>(window_), scale_,
                              static_cast<uint32_t>(Hq), static_cast<uint32_t>(Hkv));
}

}  // namespace mlx::core
