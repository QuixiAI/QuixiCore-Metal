// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "flux/flux.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

static void check_gemm_shapes(const array& x, const array& w) {
  assert(x.dtype() == w.dtype() &&
         (x.dtype() == float32 || x.dtype() == bfloat16) &&
         "flux: dtype must be float32 or bfloat16");
  const int N = x.shape(0), K = x.shape(1), M = w.shape(1);
  assert(x.shape(1) == w.shape(0) && "flux: inner dims must match");
  assert(N % 32 == 0 && M % 32 == 0 && K % 16 == 0 &&
         "flux: requires N%32==0, M%32==0, K%16==0");
  (void)N; (void)K; (void)M;
}

///////////////////////////////////////////////////////////////////////////////
// flux_gelu
///////////////////////////////////////////////////////////////////////////////

array flux_gelu(const array& x, const array& w, const array& bias, StreamOrDevice s) {
  check_gemm_shapes(x, w);
  assert(bias.ndim() == 1 && bias.shape(0) == w.shape(1) && bias.dtype() == x.dtype());
  return array({x.shape(0), w.shape(1)}, x.dtype(),
               std::make_shared<FluxGelu>(to_stream(s)), {x, w, bias});
}

void FluxGelu::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void FluxGelu::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void FluxGelu::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& x = inputs[0]; auto& w = inputs[1]; auto& bias = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = x.shape(0), K = x.shape(1), M = w.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_flux_gelu(enc, out, x, w, bias, N, K, M, type_to_name(out));
}

std::vector<array> FluxGelu::jvp(const std::vector<array>&, const std::vector<array>&,
                                 const std::vector<int>&) {
  throw std::runtime_error("FluxGelu has no jvp implementation.");
}
std::vector<array> FluxGelu::vjp(const std::vector<array>&, const std::vector<array>&,
                                 const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("FluxGelu has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> FluxGelu::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("FluxGelu has no vmap implementation.");
}
bool FluxGelu::is_equivalent(const Primitive&) const { return true; }

///////////////////////////////////////////////////////////////////////////////
// flux_gate
///////////////////////////////////////////////////////////////////////////////

array flux_gate(const array& x, const array& w, const array& bias, const array& gate,
                const array& residual, StreamOrDevice s) {
  check_gemm_shapes(x, w);
  const int N = x.shape(0), M = w.shape(1);
  assert(bias.ndim() == 1 && bias.shape(0) == M);
  assert(gate.ndim() == 1 && gate.shape(0) == M);
  assert(residual.shape(0) == N && residual.shape(1) == M);
  return array({N, M}, x.dtype(),
               std::make_shared<FluxGate>(to_stream(s)), {x, w, bias, gate, residual});
}

void FluxGate::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void FluxGate::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void FluxGate::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 5);
  auto& x = inputs[0]; auto& w = inputs[1]; auto& bias = inputs[2];
  auto& gate = inputs[3]; auto& residual = inputs[4];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = x.shape(0), K = x.shape(1), M = w.shape(1);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_flux_gate(enc, out, x, w, bias, gate, residual, N, K, M, type_to_name(out));
}

std::vector<array> FluxGate::jvp(const std::vector<array>&, const std::vector<array>&,
                                 const std::vector<int>&) {
  throw std::runtime_error("FluxGate has no jvp implementation.");
}
std::vector<array> FluxGate::vjp(const std::vector<array>&, const std::vector<array>&,
                                 const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("FluxGate has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> FluxGate::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("FluxGate has no vmap implementation.");
}
bool FluxGate::is_equivalent(const Primitive&) const { return true; }

} // namespace mlx::core
