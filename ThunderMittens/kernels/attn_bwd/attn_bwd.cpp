// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "attn_bwd/attn_bwd.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

// ---- attn_fwd_l: {o, L} ----
std::vector<array> attn_fwd_l(const array& q, const array& k, const array& v, bool causal, StreamOrDevice s) {
  assert(q.dtype() == bfloat16 && k.dtype() == bfloat16 && v.dtype() == bfloat16);
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  assert((D == 64 || D == 128) && N % 8 == 0);
  (void)D;
  return array::make_arrays({{B, H, N, D}, {B, H, N}}, {bfloat16, float32},
                            std::make_shared<AttnFwdL>(to_stream(s), causal), {q, k, v});
}
void AttnFwdL::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void AttnFwdL::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
  auto& o = outputs[0]; auto& L = outputs[1];
  auto& s = stream(); auto& d = metal::device(s.device);
  o.set_data(allocator::malloc_or_wait(o.nbytes()));
  L.set_data(allocator::malloc_or_wait(L.nbytes()));
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_fwd_l(enc, q, k, v, o, L, (unsigned)N, (unsigned)H, B, D, causal_);
}
std::vector<array> AttnFwdL::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnFwdL has no jvp."); }
std::vector<array> AttnFwdL::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("AttnFwdL has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> AttnFwdL::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnFwdL has no vmap."); }

// ---- attn_bwd_prep: delta ----
array attn_bwd_prep(const array& o, const array& do_, StreamOrDevice s) {
  const int B = o.shape(0), H = o.shape(1), N = o.shape(2);
  return array({B, H, N}, float32, std::make_shared<AttnBwdPrep>(to_stream(s)), {o, do_});
}
void AttnBwdPrep::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void AttnBwdPrep::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& o = inputs[0]; auto& do_ = inputs[1]; auto& delta = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  delta.set_data(allocator::malloc_or_wait(delta.nbytes()));
  const int B = o.shape(0), H = o.shape(1), N = o.shape(2), D = o.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_bwd_prep(enc, o, do_, delta, (unsigned)N, (unsigned)H, B, D);
}
std::vector<array> AttnBwdPrep::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnBwdPrep has no jvp."); }
std::vector<array> AttnBwdPrep::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("AttnBwdPrep has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> AttnBwdPrep::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnBwdPrep has no vmap."); }

// ---- attn_bwd_dq: dq ----
array attn_bwd_dq(const array& q, const array& k, const array& v, const array& do_,
                  const array& L, const array& delta, bool causal, StreamOrDevice s) {
  return array(q.shape(), bfloat16,
               std::make_shared<AttnBwdDQ>(to_stream(s), causal), {q, k, v, do_, L, delta});
}
void AttnBwdDQ::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void AttnBwdDQ::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
  auto& do_ = inputs[3]; auto& L = inputs[4]; auto& delta = inputs[5];
  auto& dq = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  dq.set_data(allocator::malloc_or_wait(dq.nbytes()));
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  // causal_ is captured via the launch arg below
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_bwd_dq(enc, q, k, v, do_, L, delta, dq, (unsigned)N, (unsigned)H, B, D, causal_);
}
std::vector<array> AttnBwdDQ::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnBwdDQ has no jvp."); }
std::vector<array> AttnBwdDQ::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("AttnBwdDQ has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> AttnBwdDQ::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnBwdDQ has no vmap."); }

// ---- attn_bwd_dkv: {dk, dv} ----
std::vector<array> attn_bwd_dkv(const array& q, const array& k, const array& v, const array& do_,
                                const array& L, const array& delta, bool causal, StreamOrDevice s) {
  return array::make_arrays({k.shape(), v.shape()}, {bfloat16, bfloat16},
                            std::make_shared<AttnBwdDKV>(to_stream(s), causal), {q, k, v, do_, L, delta});
}
void AttnBwdDKV::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void AttnBwdDKV::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
  auto& do_ = inputs[3]; auto& L = inputs[4]; auto& delta = inputs[5];
  auto& dk = outputs[0]; auto& dv = outputs[1];
  auto& s = stream(); auto& d = metal::device(s.device);
  dk.set_data(allocator::malloc_or_wait(dk.nbytes()));
  dv.set_data(allocator::malloc_or_wait(dv.nbytes()));
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_attn_bwd_dkv(enc, q, k, v, do_, L, delta, dk, dv, (unsigned)N, (unsigned)H, B, D, causal_);
}
std::vector<array> AttnBwdDKV::jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnBwdDKV has no jvp."); }
std::vector<array> AttnBwdDKV::vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("AttnBwdDKV has no vjp."); }
std::pair<std::vector<array>, std::vector<int>> AttnBwdDKV::vmap(const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("AttnBwdDKV has no vmap."); }

} // namespace mlx::core
