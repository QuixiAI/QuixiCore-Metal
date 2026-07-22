// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operations
///////////////////////////////////////////////////////////////////////////////

// Fused packed-Q4_0 decode GEMVs (batch-1 decode), fp32 activation + output.
// Weights are GGUF Q4_0 packed blocks (N, K/32, 18) uint8; x is (K, 1) fp32.
// Shape-keyed by the fusion (up+gate+GELU / up+gate / Q+K+V), not by a model.

/** up+gate+GELU: out = gelu_tanh(gate @ x) * (up @ x). One (N, 1) fp32 array. */
array qgemv_q4_0_up_gate_gelu(
    const array& up, const array& gate, const array& x, StreamOrDevice s = {});

/** up+gate: returns [up @ x, gate @ x], each (N, 1) fp32. */
std::vector<array> qgemv_q4_0_up_gate(
    const array& up, const array& gate, const array& x, StreamOrDevice s = {});

/** Q/K/V: returns [Wq @ x, Wk @ x, Wv @ x]; q is (Nq,1), k/v (Nkv,1) fp32. */
std::vector<array> qgemv_q4_0_qkv(
    const array& qw, const array& kw, const array& vw, const array& x,
    StreamOrDevice s = {});

///////////////////////////////////////////////////////////////////////////////
// Primitives
///////////////////////////////////////////////////////////////////////////////

class QGemvQ4_0UpGateGelu : public Primitive {
 public:
  explicit QGemvQ4_0UpGateGelu(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  const char* name() const { return "QGemvQ4_0UpGateGelu"; }
  void print(std::ostream& os) override { os << "QGemvQ4_0UpGateGelu"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class QGemvQ4_0UpGate : public Primitive {
 public:
  explicit QGemvQ4_0UpGate(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  const char* name() const { return "QGemvQ4_0UpGate"; }
  void print(std::ostream& os) override { os << "QGemvQ4_0UpGate"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class QGemvQ4_0Qkv : public Primitive {
 public:
  explicit QGemvQ4_0Qkv(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  const char* name() const { return "QGemvQ4_0Qkv"; }
  void print(std::ostream& os) override { os << "QGemvQ4_0Qkv"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

}  // namespace mlx::core
