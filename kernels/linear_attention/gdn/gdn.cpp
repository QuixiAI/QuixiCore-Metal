// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "gdn/gdn.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool gdn_is_float(Dtype t) { return t == float32 || t == float16 || t == bfloat16; }
}

std::vector<array> gdn_recur(
    const array& q, const array& k, const array& v, const array& g, const array& beta,
    const array& state_pool, const array& cu_seqlens, const array& slot_mapping,
    bool load_initial, StreamOrDevice s) {
  if (q.ndim() != 3 || k.shape() != q.shape() || v.ndim() != 3 ||
      v.shape(0) != q.shape(0)) {
    throw std::invalid_argument(
        "gdn_recur: q/k must be (total_tokens, Hk, Dk), v (total_tokens, Hv, Dv)");
  }
  const int Hk = q.shape(1), Dk = q.shape(2);
  const int Hv = v.shape(1), Dv = v.shape(2);
  if (!(Dk == 64 || Dk == 128)) {
    throw std::invalid_argument("gdn_recur: Dk must be 64 or 128");
  }
  if (Hv % Hk != 0) {
    throw std::invalid_argument("gdn_recur: Hv must be a multiple of Hk (GQA)");
  }
  if (g.ndim() != 2 || g.shape(0) != q.shape(0) || g.shape(1) != Hv ||
      beta.shape() != g.shape()) {
    throw std::invalid_argument("gdn_recur: g/beta must be (total_tokens, Hv)");
  }
  if (state_pool.ndim() != 4 || state_pool.shape(1) != Hv || state_pool.shape(2) != Dv ||
      state_pool.shape(3) != Dk) {
    throw std::invalid_argument("gdn_recur: state_pool must be (num_slots, Hv, Dv, Dk)");
  }
  if (cu_seqlens.ndim() != 1 || cu_seqlens.shape(0) < 2 || slot_mapping.ndim() != 1 ||
      slot_mapping.shape(0) != cu_seqlens.shape(0) - 1) {
    throw std::invalid_argument(
        "gdn_recur: cu_seqlens must be (R+1,), slot_mapping (R,)");
  }
  auto dt = q.dtype();
  if (!(dt == float32 || dt == float16 || dt == bfloat16)) {
    throw std::invalid_argument("gdn_recur: dtype must be float32/float16/bfloat16");
  }
  return array::make_arrays(
      {v.shape(), state_pool.shape()},
      {dt, float32},
      std::make_shared<GdnRecur>(to_stream(s), load_initial),
      {contiguous(q, false, s), contiguous(astype(k, dt, s), false, s),
       contiguous(astype(v, dt, s), false, s), contiguous(astype(g, dt, s), false, s),
       contiguous(astype(beta, dt, s), false, s),
       contiguous(astype(state_pool, float32, s), false, s),
       contiguous(astype(cu_seqlens, int32, s), false, s),
       contiguous(astype(slot_mapping, int32, s), false, s)});
}

std::vector<array> gdn_short_conv(
    const array& x, const array& weight, const array& state_pool,
    const array& cu_seqlens, const array& slot_mapping,
    bool load_initial, bool apply_silu, StreamOrDevice s) {
  if (x.ndim() != 2 || !gdn_is_float(x.dtype())) {
    throw std::invalid_argument(
        "gdn_short_conv: x must be (total_tokens, channels) float32/float16/bfloat16");
  }
  if (weight.ndim() != 2 || weight.shape(0) != x.shape(1)) {
    throw std::invalid_argument(
        "gdn_short_conv: weight must be (channels, kernel_size)");
  }
  const int kernel_size = weight.shape(1);
  if (kernel_size < 2 || kernel_size > 8) {
    throw std::invalid_argument("gdn_short_conv: kernel_size must be in [2, 8]");
  }
  if (state_pool.ndim() != 3 || state_pool.shape(1) != x.shape(1) ||
      state_pool.shape(2) != kernel_size - 1) {
    throw std::invalid_argument(
        "gdn_short_conv: state_pool must be (num_slots, channels, kernel_size - 1)");
  }
  if (cu_seqlens.ndim() != 1 || cu_seqlens.shape(0) < 2 ||
      slot_mapping.ndim() != 1 || slot_mapping.shape(0) != cu_seqlens.shape(0) - 1) {
    throw std::invalid_argument(
        "gdn_short_conv: cu_seqlens must be (R+1,), slot_mapping (R,)");
  }
  auto dt = x.dtype();
  return array::make_arrays(
      {x.shape(), state_pool.shape()}, {dt, float32},
      std::make_shared<GdnShortConv>(
          to_stream(s), load_initial, apply_silu, kernel_size),
      {contiguous(x, false, s), contiguous(astype(weight, dt, s), false, s),
       contiguous(astype(state_pool, float32, s), false, s),
       contiguous(astype(cu_seqlens, int32, s), false, s),
       contiguous(astype(slot_mapping, int32, s), false, s)});
}

std::vector<array> gdn_qkv_prepare(
    const array& mixed, int num_k_heads, int num_v_heads,
    int key_head_dim, int value_head_dim, float eps,
    float q_scale, float k_scale, StreamOrDevice s) {
  if (mixed.ndim() != 2 || !gdn_is_float(mixed.dtype())) {
    throw std::invalid_argument(
        "gdn_qkv_prepare: mixed must be (total_tokens, channels) float32/float16/bfloat16");
  }
  if (num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads % num_k_heads != 0) {
    throw std::invalid_argument(
        "gdn_qkv_prepare: num_v_heads must be a positive multiple of num_k_heads");
  }
  if (!((key_head_dim == 64 || key_head_dim == 128) &&
        (value_head_dim == 64 || value_head_dim == 128))) {
    throw std::invalid_argument(
        "gdn_qkv_prepare: key_head_dim/value_head_dim must be 64 or 128");
  }
  const int channels = 2 * num_k_heads * key_head_dim +
                       num_v_heads * value_head_dim;
  if (mixed.shape(1) != channels) {
    throw std::invalid_argument(
        "gdn_qkv_prepare: mixed last dimension does not match head configuration");
  }
  if (!(eps >= 0.0f) || !std::isfinite(q_scale) || !std::isfinite(k_scale)) {
    throw std::invalid_argument(
        "gdn_qkv_prepare: eps must be non-negative and scales finite");
  }
  const int tokens = mixed.shape(0);
  return array::make_arrays(
      {{tokens, num_k_heads, key_head_dim},
       {tokens, num_k_heads, key_head_dim},
       {tokens, num_v_heads, value_head_dim}},
      {mixed.dtype(), mixed.dtype(), mixed.dtype()},
      std::make_shared<GdnQkvPrepare>(
          to_stream(s), num_k_heads, num_v_heads, key_head_dim,
          value_head_dim, eps, q_scale, k_scale),
      {contiguous(mixed, false, s)});
}

std::vector<array> gdn_gate_beta(
    const array& a, const array& b, const array& A_log,
    const array& dt_bias, StreamOrDevice s) {
  if (a.ndim() != 2 || b.shape() != a.shape() || !gdn_is_float(a.dtype()) ||
      !gdn_is_float(b.dtype())) {
    throw std::invalid_argument(
        "gdn_gate_beta: a/b must be matching (total_tokens, value_heads) float tensors");
  }
  const int heads = a.shape(1);
  if (A_log.ndim() != 1 || A_log.shape(0) != heads ||
      dt_bias.ndim() != 1 || dt_bias.shape(0) != heads) {
    throw std::invalid_argument(
        "gdn_gate_beta: A_log/dt_bias must be (value_heads,)");
  }
  auto dt = promote_types(a.dtype(), b.dtype());
  return array::make_arrays(
      {a.shape(), a.shape()}, {float32, float32},
      std::make_shared<GdnGateBeta>(to_stream(s)),
      {contiguous(astype(a, dt, s), false, s),
       contiguous(astype(b, dt, s), false, s),
       contiguous(astype(A_log, float32, s), false, s),
       contiguous(astype(dt_bias, float32, s), false, s)});
}

array gdn_gated_rmsnorm(
    const array& y, const array& z, const array& weight,
    float eps, StreamOrDevice s) {
  if (y.ndim() != 3 || z.shape() != y.shape() || !gdn_is_float(y.dtype()) ||
      !gdn_is_float(z.dtype())) {
    throw std::invalid_argument(
        "gdn_gated_rmsnorm: y/z must be matching (total_tokens, heads, dim) float tensors");
  }
  const int dim = y.shape(2);
  if (!(dim == 64 || dim == 128)) {
    throw std::invalid_argument("gdn_gated_rmsnorm: dim must be 64 or 128");
  }
  if (weight.ndim() != 1 || weight.shape(0) != dim || !gdn_is_float(weight.dtype())) {
    throw std::invalid_argument("gdn_gated_rmsnorm: weight must be a float (dim,) tensor");
  }
  if (!(eps >= 0.0f)) {
    throw std::invalid_argument("gdn_gated_rmsnorm: eps must be non-negative");
  }
  auto dt = promote_types(y.dtype(), z.dtype());
  return array(
      y.shape(), dt,
      std::make_shared<GdnGatedRmsNorm>(to_stream(s), dim, eps),
      {contiguous(astype(y, dt, s), false, s),
       contiguous(astype(z, dt, s), false, s),
       contiguous(astype(weight, dt, s), false, s)});
}

void GdnRecur::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("GdnRecur has no CPU implementation.");
}

void GdnRecur::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& g = inputs[3];
  auto& beta = inputs[4];
  auto& pool_in = inputs[5];
  auto& cu = inputs[6];
  auto& slots = inputs[7];
  auto& y = outputs[0];
  auto& pool_out = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  y.set_data(allocator::malloc_or_wait(y.nbytes()));
  pool_out.set_data(allocator::malloc_or_wait(pool_out.nbytes()));
  const int R = cu.shape(0) - 1;
  const int Hk = q.shape(1), Dk = q.shape(2);
  const int Hv = v.shape(1), Dv = v.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  // functional pool: clone, then the kernel updates its request slots in place
  tk::launch_sscan_pool_clone(enc, pool_in, pool_out,
                              static_cast<uint32_t>(pool_in.size()));
  tk::launch_gdn_recur(enc, q, k, v, g, beta, pool_out, cu, slots, y, R, Hk, Hv, Dv, Dk,
                       load_initial_ ? 1 : 0, type_to_name(q));
}

void GdnShortConv::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("GdnShortConv has no CPU implementation.");
}

void GdnShortConv::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& weight = inputs[1];
  auto& pool_in = inputs[2];
  auto& cu = inputs[3];
  auto& slots = inputs[4];
  auto& out = outputs[0];
  auto& pool_out = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  pool_out.set_data(allocator::malloc_or_wait(pool_out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_sscan_pool_clone(
      enc, pool_in, pool_out, static_cast<uint32_t>(pool_in.size()));
  tk::launch_gdn_short_conv(
      enc, x, weight, pool_out, cu, slots, out,
      static_cast<int>(cu.shape(0) - 1), static_cast<int>(x.shape(1)),
      kernel_size_, load_initial_ ? 1 : 0, apply_silu_ ? 1 : 0,
      type_to_name(x));
}

void GdnQkvPrepare::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("GdnQkvPrepare has no CPU implementation.");
}

void GdnQkvPrepare::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& mixed = inputs[0];
  auto& q = outputs[0];
  auto& k = outputs[1];
  auto& v = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  q.set_data(allocator::malloc_or_wait(q.nbytes()));
  k.set_data(allocator::malloc_or_wait(k.nbytes()));
  v.set_data(allocator::malloc_or_wait(v.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_gdn_qkv_prepare(
      enc, mixed, q, k, v, static_cast<int>(mixed.shape(0)), Hk_, Hv_,
      Dk_, Dv_, eps_, q_scale_, k_scale_, type_to_name(mixed));
}

void GdnGateBeta::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("GdnGateBeta has no CPU implementation.");
}

void GdnGateBeta::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& decay = outputs[0];
  auto& beta = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  decay.set_data(allocator::malloc_or_wait(decay.nbytes()));
  beta.set_data(allocator::malloc_or_wait(beta.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_gdn_gate_beta(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], decay, beta,
      static_cast<uint32_t>(inputs[0].size()),
      static_cast<int>(inputs[0].shape(1)), type_to_name(inputs[0]));
}

void GdnGatedRmsNorm::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("GdnGatedRmsNorm has no CPU implementation.");
}

void GdnGatedRmsNorm::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_gdn_gated_rmsnorm(
      enc, inputs[0], inputs[1], inputs[2], out,
      static_cast<int>(inputs[0].size() / dim_), dim_, eps_,
      type_to_name(inputs[0]));
}

#define GDN_NO_AUTODIFF(CLASS, LABEL)                                             \
std::vector<array> CLASS::jvp(                                                    \
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) { \
  throw std::runtime_error(LABEL " has no jvp implementation.");                 \
}                                                                                 \
std::vector<array> CLASS::vjp(                                                    \
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, \
    const std::vector<array>&) {                                                  \
  throw std::runtime_error(LABEL " has no vjp implementation.");                 \
}                                                                                 \
std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                      \
    const std::vector<array>&, const std::vector<int>&) {                         \
  throw std::runtime_error(LABEL " has no vmap implementation.");                \
}

GDN_NO_AUTODIFF(GdnShortConv, "GdnShortConv")
GDN_NO_AUTODIFF(GdnQkvPrepare, "GdnQkvPrepare")
GDN_NO_AUTODIFF(GdnGateBeta, "GdnGateBeta")
GDN_NO_AUTODIFF(GdnGatedRmsNorm, "GdnGatedRmsNorm")

#undef GDN_NO_AUTODIFF

std::vector<array> GdnRecur::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("GdnRecur has no jvp implementation.");
}
std::vector<array> GdnRecur::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("GdnRecur has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> GdnRecur::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("GdnRecur has no vmap implementation.");
}

} // namespace mlx::core
