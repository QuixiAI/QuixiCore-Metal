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

array attn_decode_bh(
    const array& q, const array& k, const array& v,
    int context_length, StreamOrDevice s) {
  if (!ad_float(q.dtype()) || q.dtype() != k.dtype() || k.dtype() != v.dtype()) {
    throw std::invalid_argument("attn_decode_bh: q/k/v must be same floating dtype");
  }
  if (q.ndim() != 3 || k.ndim() != 4 || v.shape() != k.shape()) {
    throw std::invalid_argument(
        "attn_decode_bh: expected q (B,Hq,D), k/v (B,Hkv,cache_T,D)");
  }
  const int batch = q.shape(0), heads_q = q.shape(1), dimension = q.shape(2);
  const int heads_kv = k.shape(1), cache_length = k.shape(2);
  if (batch <= 0 || heads_q <= 0 || dimension <= 0 || cache_length <= 0 ||
      k.shape(0) != batch || k.shape(3) != dimension || dimension > 128 ||
      heads_kv <= 0 || heads_q % heads_kv != 0 || context_length <= 0 ||
      context_length > cache_length) {
    throw std::invalid_argument(
        "attn_decode_bh: shape mismatch, D<=128, Hq%Hkv, and 0<context<=cache_T required");
  }
  return array(
      q.shape(), q.dtype(),
      std::make_shared<AttnDecodeBh>(to_stream(s), context_length),
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

void AttnDecodeBh::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("AttnDecodeBh has no CPU implementation.");
}

void AttnDecodeBh::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& output = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  output.set_data(allocator::malloc_or_wait(output.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_decode_bh(
      enc, q, k, inputs[2], output, q.shape(0), context_length_, k.shape(2),
      q.shape(1), k.shape(1), q.shape(2), type_to_name(q));
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
TK_ATTND_NO_AUTODIFF(AttnDecodeBh, "AttnDecodeBh")

bool AttnDecodeBh::is_equivalent(const Primitive& other) const {
  return typeid(*this) == typeid(other) &&
      context_length_ == static_cast<const AttnDecodeBh&>(other).context_length_;
}

} // namespace mlx::core
