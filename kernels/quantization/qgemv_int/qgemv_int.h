// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** W8A8 / SmoothQuant decode GEMV (int8 x int8 -> int32, then scale):
 *  out[n,0] = w_scale[n] * a_scale * sum_k Wq[n,k] * Xq[k].
 *  wq int8 (N,K); xq int8 (K,1); w_scale half (N,); a_scale half (1,); out half (N,1). K%4==0. */
array qgemv_w8a8(const array& wq, const array& xq, const array& w_scale, const array& a_scale,
                 StreamOrDevice s = {});

/** BitNet W2A8 decode GEMV (ternary 2-bit x int8 -> int32, per-group absmean scale):
 *  wq uint8 bitnet blocks (N, K/32, 10); xq int8 (K,1); a_scale half (1,); out half (N,1). */
array qgemv_w2a8(const array& wq, const array& xq, const array& a_scale, StreamOrDevice s = {});

class QGemvW8A8 : public Primitive {
 public:
  explicit QGemvW8A8(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QGemvW8A8"; }

  void print(std::ostream& os) override { os << "QGemvW8A8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
  void eval(const std::vector<array>&, std::vector<array>&);
};

class QGemvW2A8 : public Primitive {
 public:
  explicit QGemvW2A8(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QGemvW2A8"; }

  void print(std::ostream& os) override { os << "QGemvW2A8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
