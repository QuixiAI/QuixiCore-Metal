// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>
#include <string>

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

std::vector<array> decode_cache_attention(
    const array& q, const array& new_k, const array& new_v,
    const array& cos, const array& sin, const array& positions,
    const array& context_lengths, const array& q_weight,
    const array& k_weight, const array& key_cache,
    const array& value_cache, float eps, bool do_q_norm,
    bool do_k_norm, bool gemma, float softmax_scale, StreamOrDevice s) {
  if (!ad_float(q.dtype()) || q.ndim() != 3 || q.shape(0) <= 0 ||
      q.shape(1) <= 0 || (q.shape(2) != 64 && q.shape(2) != 128)) {
    throw std::invalid_argument(
        "decode_cache_attention: q must be (B,Hq,D) fp32/fp16/bf16 with D=64 or 128");
  }
  const int batch = q.shape(0), heads_q = q.shape(1), dimension = q.shape(2);
  if (new_k.dtype() != q.dtype() || new_v.dtype() != q.dtype() ||
      new_k.ndim() != 3 || new_v.shape() != new_k.shape() ||
      new_k.shape(0) != batch || new_k.shape(1) <= 0 ||
      new_k.shape(2) != dimension || heads_q % new_k.shape(1) != 0) {
    throw std::invalid_argument(
        "decode_cache_attention: new_k/new_v must be (B,Hkv,D), Hq%Hkv=0");
  }
  const int heads_kv = new_k.shape(1);
  if (key_cache.dtype() != q.dtype() || value_cache.dtype() != q.dtype() ||
      key_cache.ndim() != 4 || value_cache.shape() != key_cache.shape() ||
      key_cache.shape(0) != batch || key_cache.shape(1) != heads_kv ||
      key_cache.shape(2) <= 0 || key_cache.shape(3) != dimension) {
    throw std::invalid_argument(
        "decode_cache_attention: caches must be (B,Hkv,cache_T,D) with q dtype");
  }
  if (cos.dtype() != q.dtype() || sin.dtype() != q.dtype() || cos.ndim() != 2 ||
      cos.shape() != sin.shape() || cos.shape(0) <= 0 ||
      cos.shape(1) != dimension / 2) {
    throw std::invalid_argument(
        "decode_cache_attention: cos/sin must be (P,D/2) with q dtype");
  }
  if (positions.ndim() != 1 || positions.shape(0) != batch ||
      context_lengths.ndim() != 1 || context_lengths.shape(0) != batch) {
    throw std::invalid_argument(
        "decode_cache_attention: positions/context_lengths must be (B,)");
  }
  if (do_q_norm &&
      (q_weight.dtype() != q.dtype() || q_weight.ndim() != 1 ||
       q_weight.shape(0) != dimension)) {
    throw std::invalid_argument("decode_cache_attention: q_weight must be (D,)");
  }
  if (do_k_norm &&
      (k_weight.dtype() != q.dtype() || k_weight.ndim() != 1 ||
       k_weight.shape(0) != dimension)) {
    throw std::invalid_argument("decode_cache_attention: k_weight must be (D,)");
  }
  if (!(eps > 0.0f) || softmax_scale < 0.0f) {
    throw std::invalid_argument(
        "decode_cache_attention: eps must be positive and softmax_scale non-negative");
  }
  auto dtype = q.dtype();
  return array::make_arrays(
      {q.shape(), key_cache.shape(), value_cache.shape()},
      {dtype, dtype, dtype},
      std::make_shared<DecodeCacheAttention>(
          to_stream(s), eps, do_q_norm, do_k_norm, gemma, softmax_scale),
      {contiguous(q, false, s), contiguous(new_k, false, s),
       contiguous(new_v, false, s), contiguous(cos, false, s),
       contiguous(sin, false, s),
       contiguous(astype(positions, int32, s), false, s),
       contiguous(astype(context_lengths, int32, s), false, s),
       contiguous(astype(q_weight, dtype, s), false, s),
       contiguous(astype(k_weight, dtype, s), false, s),
       contiguous(key_cache, false, s), contiguous(value_cache, false, s)});
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

void DecodeCacheAttention::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("DecodeCacheAttention has no CPU implementation.");
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

void DecodeCacheAttention::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  for (auto& output : outputs) {
    output.set_data(allocator::malloc_or_wait(output.nbytes()));
  }
  auto& s = stream(); auto& d = metal::device(s.device);
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  const std::string type_name = type_to_name(inputs[0]);
  tk::launch_kv_cache_clone(
      enc, inputs[9], inputs[10], outputs[1], outputs[2],
      static_cast<uint64_t>(inputs[9].size()), type_name);
  tk::launch_decode_cache_attention(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
      inputs[6], inputs[7], inputs[8], outputs[1], outputs[2], outputs[0],
      inputs[0].shape(0), inputs[0].shape(1), inputs[1].shape(1),
      inputs[9].shape(2), inputs[0].shape(2), eps_, do_q_norm_, do_k_norm_,
      gemma_, softmax_scale_, type_name);
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
TK_ATTND_NO_AUTODIFF(DecodeCacheAttention, "DecodeCacheAttention")

bool AttnDecodeBh::is_equivalent(const Primitive& other) const {
  return typeid(*this) == typeid(other) &&
      context_length_ == static_cast<const AttnDecodeBh&>(other).context_length_;
}
bool DecodeCacheAttention::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) return false;
  auto& o = static_cast<const DecodeCacheAttention&>(other);
  return eps_ == o.eps_ && do_q_norm_ == o.do_q_norm_ &&
      do_k_norm_ == o.do_k_norm_ && gemma_ == o.gemma_ &&
      softmax_scale_ == o.softmax_scale_;
}

} // namespace mlx::core
