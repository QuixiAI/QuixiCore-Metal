// Copyright © 2023 Apple Inc.

#pragma once

#include <string>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Quantized GEMM (Marlin's method): out = dequantize(wq) @ x.
 *  wq is packed weight blocks of shape (N, K/block_k, block_bytes) uint8 for the given
 *  `format` (e.g. "q8_0"); x is (K, M) float16; out is (N, M) float16. Dequant-to-shared
 *  then a standard simdgroup MMA. Shapes: N%32, M%32, K%block_k. */
array qgemm(const array& wq, const array& x, const std::string& format = "q8_0",
            StreamOrDevice s = {});

// Same op, but dequantizes weights straight into the simdgroup fragment (no threadgroup
// staging / barrier) — Marlin's zero-shuffle. Single simdgroup per 32x32 output tile.
array qgemm_direct(const array& wq, const array& x, const std::string& format = "q8_0",
                   StreamOrDevice s = {});

// GPTQ act-order with an in-kernel g_idx gather: wq quantized in permuted (K) order, x (K,M) f16,
// perm (K,) int32 (= argsort(g_idx)); X K-rows are gathered by perm during the load. out (N,M) f16.
array qgemm_actorder_k(const array& wq, const array& x, const array& perm,
                       const std::string& format = "kU4B8", StreamOrDevice s = {});

// fp8_block2d: codes-only fp8 weights (N, K/128, 128) + a separate (N/128, K/128) fp16 tile scale,
// x (K,M) f16 -> (N,M) f16. The storage-optimal fp8_block (no per-row scale replication).
array qgemm_blockscale(const array& wq, const array& x, const array& scale2d, StreamOrDevice s = {});

class QGemmBlockScale : public Primitive {
 public:
  explicit QGemmBlockScale(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  void print(std::ostream& os) override { os << "QGemmBlockScale"; }
  bool is_equivalent(const Primitive& other) const override { return typeid(*this) == typeid(other); }
  void eval(const std::vector<array>&, std::vector<array>&);
};

class QGemmActorder : public Primitive {
 public:
  explicit QGemmActorder(Stream stream, std::string format)
      : Primitive(stream), fmt_(std::move(format)) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&, const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(const std::vector<array>&, const std::vector<int>&) override;
  void print(std::ostream& os) override { os << "QGemmActorder[" << fmt_ << "]"; }
  bool is_equivalent(const Primitive& other) const override {
    return typeid(*this) == typeid(other) && fmt_ == static_cast<const QGemmActorder&>(other).fmt_;
  }
  void eval(const std::vector<array>&, std::vector<array>&);
 private:
  std::string fmt_;
};

class QGemm : public Primitive {
 public:
  explicit QGemm(Stream stream, std::string format, bool direct = false)
      : Primitive(stream), fmt_(std::move(format)), direct_(direct) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  void print(std::ostream& os) override { os << "QGemm[" << fmt_ << (direct_ ? ",direct]" : "]"); }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);

 private:
  std::string fmt_;
  bool direct_;
};

} // namespace mlx::core
