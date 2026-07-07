// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "attn_decode/attn_decode.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool ad_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}
}

array attn_decode(const array& q, const array& k, const array& v, StreamOrDevice s) {
  if (!ad_float(q.dtype()) || q.dtype() != k.dtype() || k.dtype() != v.dtype()) {
    throw std::invalid_argument("attn_decode: q/k/v must be same floating dtype");
  }
  if (q.ndim() != 2 || k.ndim() != 3 || v.ndim() != 3) {
    throw std::invalid_argument("attn_decode: expected q (Hq,D), k/v (Tk,Hkv,D)");
  }
  const int Hq = q.shape(0), D = q.shape(1);
  const int Tk = k.shape(0), Hkv = k.shape(1);
  if (k.shape(2) != D || v.shape() != k.shape() || D > 128 || Hq % Hkv != 0) {
    throw std::invalid_argument("attn_decode: shape mismatch, D<=128, Hq%Hkv required");
  }
  return array({Hq, D}, q.dtype(), std::make_shared<AttnDecode>(to_stream(s)),
               {contiguous(q, false, s), contiguous(k, false, s), contiguous(v, false, s)});
}

void AttnDecode::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AttnDecode has no CPU implementation.");
}

void AttnDecode::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_decode(enc, q, k, v, out, k.shape(0), q.shape(0), k.shape(1), q.shape(1),
                         type_to_name(q));
}

#define TK_ATTND_NO_AUTODIFF(CLASS, LABEL)                                    \
  std::vector<array> CLASS::jvp(                                               \
      const std::vector<array>&, const std::vector<array>&,                    \
      const std::vector<int>&) {                                               \
    throw std::runtime_error(LABEL " has no jvp implementation.");             \
  }                                                                            \
  std::vector<array> CLASS::vjp(                                               \
      const std::vector<array>&, const std::vector<array>&,                    \
      const std::vector<int>&, const std::vector<array>&) {                    \
    throw std::runtime_error(LABEL " has no vjp implementation.");             \
  }                                                                            \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                 \
      const std::vector<array>&, const std::vector<int>&) {                    \
    throw std::runtime_error(LABEL " has no vmap implementation.");            \
  }

TK_ATTND_NO_AUTODIFF(AttnDecode, "AttnDecode")

} // namespace mlx::core
