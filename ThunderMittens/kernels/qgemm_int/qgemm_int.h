// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** W8A8 prefill GEMM (int8 x int8 -> int32, then scale; M>1, bit-exact int32 accumulate):
 *  out[n,m] = w_scale[n] * a_scale[m] * sum_k Wq[n,k] * Xq[m,k].
 *  wq int8 (N,K); xq int8 (M,K) token-major; w_scale half (N,); a_scale half (M,); out half (N,M). K%4==0. */
/** azp-corrected W8A8: y = s_w*s_a*(W@Xq^T - azp*rowsum(W)). w_rowsum (N,) i32 host-side. */
array qgemm_w8a8_azp(const array& wq, const array& xq, const array& w_scale,
                     const array& a_scale, const array& w_rowsum, const array& azp,
                     StreamOrDevice s = {});

array qgemm_w8a8(const array& wq, const array& xq, const array& w_scale, const array& a_scale,
                 StreamOrDevice s = {});

/** BitNet W2A8 prefill GEMM (ternary 2-bit x int8 -> int32, per-group absmean scale):
 *  wq uint8 bitnet blocks (N, K/32, 10); xq int8 (M,K); a_scale half (M,); out half (N,M). */
array qgemm_w2a8(const array& wq, const array& xq, const array& a_scale, StreamOrDevice s = {});

class QGemmW8A8Azp : public Primitive {
 public:
  explicit QGemmW8A8Azp(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QGemmW8A8Azp"; }

  void print(std::ostream& os) override { os << "QGemmW8A8Azp"; }
  bool is_equivalent(const Primitive&) const override { return true; }
  void eval(const std::vector<array>&, std::vector<array>&);
};

class QGemmW8A8 : public Primitive {
 public:
  explicit QGemmW8A8(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QGemmW8A8"; }

  void print(std::ostream& os) override { os << "QGemmW8A8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
  void eval(const std::vector<array>&, std::vector<array>&);
};

class QGemmW2A8 : public Primitive {
 public:
  explicit QGemmW2A8(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QGemmW2A8"; }

  void print(std::ostream& os) override { os << "QGemmW2A8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
