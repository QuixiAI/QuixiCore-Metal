// Copyright © 2024 Apple Inc.

#include <cassert>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "qgemv_fused/qgemv_fused.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {

// A Q4_0 packed weight is (N, K/32, 18) uint8; x is (K, 1) fp32.
void qgf_check(const array& w, const array& x, const char* who) {
  if (w.dtype() != uint8 || w.ndim() != 3 || w.shape(2) != 18) {
    throw std::invalid_argument(std::string(who) +
                                ": weight must be uint8 Q4_0 blocks (N, K/32, 18)");
  }
  if (x.dtype() != float32 || x.ndim() != 2 || x.shape(1) != 1) {
    throw std::invalid_argument(std::string(who) + ": x must be fp32 (K, 1)");
  }
  if (w.shape(1) * 32 != x.shape(0)) {
    throw std::invalid_argument(std::string(who) + ": weight K does not match x");
  }
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Operations
///////////////////////////////////////////////////////////////////////////////

array qgemv_q4_0_up_gate_gelu(
    const array& up, const array& gate, const array& x, StreamOrDevice s) {
  qgf_check(up, x, "qgemv_q4_0_up_gate_gelu");
  qgf_check(gate, x, "qgemv_q4_0_up_gate_gelu");
  const int N = up.shape(0);
  if (gate.shape(0) != N) {
    throw std::invalid_argument("qgemv_q4_0_up_gate_gelu: up/gate row counts differ");
  }
  return array({N, 1}, float32,
               std::make_shared<QGemvQ4_0UpGateGelu>(to_stream(s)), {up, gate, x});
}

std::vector<array> qgemv_q4_0_up_gate(
    const array& up, const array& gate, const array& x, StreamOrDevice s) {
  qgf_check(up, x, "qgemv_q4_0_up_gate");
  qgf_check(gate, x, "qgemv_q4_0_up_gate");
  const int N = up.shape(0);
  if (gate.shape(0) != N) {
    throw std::invalid_argument("qgemv_q4_0_up_gate: up/gate row counts differ");
  }
  return array::make_arrays(
      {{N, 1}, {N, 1}}, {float32, float32},
      std::make_shared<QGemvQ4_0UpGate>(to_stream(s)), {up, gate, x});
}

std::vector<array> qgemv_q4_0_qkv(
    const array& qw, const array& kw, const array& vw, const array& x,
    StreamOrDevice s) {
  qgf_check(qw, x, "qgemv_q4_0_qkv");
  qgf_check(kw, x, "qgemv_q4_0_qkv");
  qgf_check(vw, x, "qgemv_q4_0_qkv");
  const int Nq = qw.shape(0), Nkv = kw.shape(0);
  if (vw.shape(0) != Nkv) {
    throw std::invalid_argument("qgemv_q4_0_qkv: k/v row counts differ");
  }
  return array::make_arrays(
      {{Nq, 1}, {Nkv, 1}, {Nkv, 1}}, {float32, float32, float32},
      std::make_shared<QGemvQ4_0Qkv>(to_stream(s)), {qw, kw, vw, x});
}

///////////////////////////////////////////////////////////////////////////////
// CPU (no fallback — compose q4_0 qgemv + gelu in the framework for a reference)
///////////////////////////////////////////////////////////////////////////////

void QGemvQ4_0UpGateGelu::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QGemvQ4_0UpGateGelu has no CPU implementation.");
}
void QGemvQ4_0UpGate::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QGemvQ4_0UpGate has no CPU implementation.");
}
void QGemvQ4_0Qkv::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("QGemvQ4_0Qkv has no CPU implementation.");
}

///////////////////////////////////////////////////////////////////////////////
// Metal Backend
///////////////////////////////////////////////////////////////////////////////

void QGemvQ4_0UpGateGelu::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& up = inputs[0];
  auto& gate = inputs[1];
  auto& x = inputs[2];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int N = up.shape(0);
  const int K = static_cast<int>(x.shape(0));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemv_q4_0_f32_up_gate_gelu(enc, out, up, gate, x, N, K);
}

void QGemvQ4_0UpGate::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& up = inputs[0];
  auto& gate = inputs[1];
  auto& x = inputs[2];
  auto& up_out = outputs[0];
  auto& gate_out = outputs[1];
  auto& s = stream();
  auto& d = metal::device(s.device);
  up_out.set_data(allocator::malloc_or_wait(up_out.nbytes()));
  gate_out.set_data(allocator::malloc_or_wait(gate_out.nbytes()));
  const int N = up.shape(0);
  const int K = static_cast<int>(x.shape(0));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemv_q4_0_f32_up_gate(enc, up_out, gate_out, up, gate, x, N, K);
}

void QGemvQ4_0Qkv::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& qw = inputs[0];
  auto& kw = inputs[1];
  auto& vw = inputs[2];
  auto& x = inputs[3];
  auto& q_out = outputs[0];
  auto& k_out = outputs[1];
  auto& v_out = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  q_out.set_data(allocator::malloc_or_wait(q_out.nbytes()));
  k_out.set_data(allocator::malloc_or_wait(k_out.nbytes()));
  v_out.set_data(allocator::malloc_or_wait(v_out.nbytes()));
  const int Nq = qw.shape(0), Nkv = kw.shape(0);
  const int K = static_cast<int>(x.shape(0));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_qgemv_q4_0_f32_qkv(enc, q_out, k_out, v_out, qw, kw, vw, x, Nq, Nkv, K);
}

}  // namespace mlx::core
