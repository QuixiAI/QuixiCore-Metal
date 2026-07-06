// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "lin_attn_causal/lin_attn_causal.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array lin_attn_causal(const array& q, const array& k, const array& v, StreamOrDevice s) {
  assert(q.dtype() == bfloat16 && k.dtype() == bfloat16 && v.dtype() == bfloat16);
  assert(q.shape() == k.shape() && k.shape() == v.shape());
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  assert(D == 64 && "lin_attn_causal currently supports D=64");
  assert(N % 8 == 0 && "lin_attn_causal: N must be a multiple of 8");
  constexpr int L = 64;   // chunk rows (must match LIN_CHUNK_L in the metal)
  if (N % L != 0 || N < 2 * L) {
    // small/ragged N: the serial single-simdgroup scan
    return array(q.shape(), bfloat16,
                 std::make_shared<LinAttnCausal>(to_stream(s)), {q, k, v});
  }
  // chunked-parallel: per-chunk KV states -> exclusive chunk prefix -> seeded per-chunk scan
  const int C = N / L;
  auto kv = array({B, H, C, D, D}, float32,
                  std::make_shared<LinChunkKV>(to_stream(s)), {k, v});
  auto sex = array({B, H, C, D, D}, float32,
                   std::make_shared<LinChunkScan>(to_stream(s)), {kv});
  return array(q.shape(), bfloat16,
               std::make_shared<LinChunkOut>(to_stream(s)), {q, k, v, sex});
}

void LinChunkKV::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void LinChunkKV::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& k = inputs[0]; auto& v = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = k.shape(0), H = k.shape(1), N = k.shape(2), D = k.shape(3);
  const int C = out.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lin_chunk_kv(enc, k, v, out, static_cast<unsigned>(N),
                          static_cast<unsigned>(H), B, C, D);
}
std::vector<array> LinChunkKV::jvp(const std::vector<array>&, const std::vector<array>&,
                                   const std::vector<int>&) {
  throw std::runtime_error("LinChunkKV has no jvp implementation.");
}
std::vector<array> LinChunkKV::vjp(const std::vector<array>&, const std::vector<array>&,
                                   const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("LinChunkKV has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> LinChunkKV::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("LinChunkKV has no vmap implementation.");
}
bool LinChunkKV::is_equivalent(const Primitive&) const { return true; }

void LinChunkScan::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void LinChunkScan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& sin = inputs[0];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = sin.shape(0), H = sin.shape(1), C = sin.shape(2), D = sin.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lin_chunk_scan(enc, sin, out, static_cast<unsigned>(C), B * H, D);
}
std::vector<array> LinChunkScan::jvp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&) {
  throw std::runtime_error("LinChunkScan has no jvp implementation.");
}
std::vector<array> LinChunkScan::vjp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("LinChunkScan has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> LinChunkScan::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("LinChunkScan has no vmap implementation.");
}
bool LinChunkScan::is_equivalent(const Primitive&) const { return true; }

void LinChunkOut::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void LinChunkOut::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2]; auto& sex = inputs[3];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  const int C = sex.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lin_chunk_out(enc, q, k, v, sex, out, static_cast<unsigned>(N),
                           static_cast<unsigned>(H), B, C, D);
}
std::vector<array> LinChunkOut::jvp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&) {
  throw std::runtime_error("LinChunkOut has no jvp implementation.");
}
std::vector<array> LinChunkOut::vjp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("LinChunkOut has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> LinChunkOut::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("LinChunkOut has no vmap implementation.");
}
bool LinChunkOut::is_equivalent(const Primitive&) const { return true; }

void LinAttnCausal::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void LinAttnCausal::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void LinAttnCausal::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = q.shape(0), H = q.shape(1), N = q.shape(2), D = q.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lin_attn_causal(enc, q, k, v, out, static_cast<unsigned>(N),
                             static_cast<unsigned>(H), B, D);
}

std::vector<array> LinAttnCausal::jvp(const std::vector<array>&, const std::vector<array>&,
                                      const std::vector<int>&) {
  throw std::runtime_error("LinAttnCausal has no jvp implementation.");
}
std::vector<array> LinAttnCausal::vjp(const std::vector<array>&, const std::vector<array>&,
                                      const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("LinAttnCausal has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> LinAttnCausal::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("LinAttnCausal has no vmap implementation.");
}
bool LinAttnCausal::is_equivalent(const Primitive&) const { return true; }

} // namespace mlx::core
