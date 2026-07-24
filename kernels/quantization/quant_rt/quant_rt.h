// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/**
 *  Runtime per-token (per-row) activation quantization on the GPU.
 *  Returns (codes, scale): scale[row] = absmax(row) / QMAX, codes = encode(x/scale).
 *  Reconstruct as scale[row] * decode(codes[row]).
 *
 *  x : (..., D), float32/float16/bfloat16. codes has x's shape; scale has x.shape[:-1].
 *   - fp8 : codes uint8 (e4m3, QMAX=448)
 *   - int8: codes int8  (symmetric, QMAX=127)
 **/
std::vector<array> quantize_per_token_fp8(const array& x, StreamOrDevice s = {});
std::vector<array> quantize_per_token_int8(const array& x, StreamOrDevice s = {});

/** Per-GROUP dynamic quant along the last axis (canonical group_size=128; the activation-side
 *  layout for block-quantized GEMMs). Returns [codes, scale (rows, D/G) f32]. ue8m0 rounds the
 *  fp8 scale up to a power of two (MX convention). D % group_size == 0, group_size % 4 == 0. */
std::vector<array> quantize_per_group_fp8(const array& x, int group_size = 128,
                                          bool ue8m0 = false, StreamOrDevice s = {});
std::vector<array> quantize_per_group_int8(const array& x, int group_size = 128,
                                           StreamOrDevice s = {});

/** ASYMMETRIC per-token int8 (vLLM azp): scale=(max-min)/255, azp=rint(-128-min/scale),
 *  q=clamp(rint(x/scale)+azp). Returns [codes i8, scale (rows,) f32, azp (rows,) i32]. */
std::vector<array> quantize_per_token_int8_azp(const array& x, StreamOrDevice s = {});

/**
 *  Per-tensor (global) dynamic quantization: one scale = global_absmax / QMAX (via a P3
 *  atomic-max reduction). Returns [codes, scale (scalar), scale_u (uint32 scratch)]; callers
 *  use the first two. fp8 e4m3 (QMAX=448) or symmetric int8 (QMAX=127).
 **/
std::vector<array> quantize_per_tensor_fp8(const array& x, StreamOrDevice s = {});
std::vector<array> quantize_per_tensor_int8(const array& x, StreamOrDevice s = {});

/** Per-input-channel calibration absmax over x(tokens, channels), returned as
 *  fp32 (channels,). If has_running is true, running(channels,) is merged with
 *  max semantics. NaN propagates deterministically; either infinity maps to
 *  +infinity. Repeated calls with running provide exact chunked accumulation. */
array calibration_absmax(const array& x, const array& running, bool has_running,
                         StreamOrDevice s = {});

class CalibrationAbsmax : public Primitive {
 public:
  explicit CalibrationAbsmax(Stream stream, bool has_running)
      : Primitive(stream), has_running_(has_running) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&,
      const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&,
      const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "CalibrationAbsmax"; }
  void print(std::ostream& os) override { os << "CalibrationAbsmax"; }
  bool is_equivalent(const Primitive& other) const override {
    return has_running_ ==
        static_cast<const CalibrationAbsmax&>(other).has_running_;
  }

 private:
  bool has_running_;
};

class QuantizePerTensor : public Primitive {
 public:
  QuantizePerTensor(Stream stream, bool is_int8) : Primitive(stream), is_int8_(is_int8) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerTensor"; }
  void print(std::ostream& os) override { os << "QuantizePerTensor"; }
  bool is_equivalent(const Primitive& other) const override {
    return is_int8_ == static_cast<const QuantizePerTensor&>(other).is_int8_;
  }

 private:
  bool is_int8_;
};

class QuantizePerTokenFp8 : public Primitive {
 public:
  explicit QuantizePerTokenFp8(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerTokenFp8"; }
  void print(std::ostream& os) override { os << "QuantizePerTokenFp8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class QuantizePerGroupFp8 : public Primitive {
 public:
  QuantizePerGroupFp8(Stream stream, int g, bool ue8m0)
      : Primitive(stream), g_(g), ue8m0_(ue8m0) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerGroupFp8"; }
  void print(std::ostream& os) override { os << "QuantizePerGroupFp8"; }
  bool is_equivalent(const Primitive& other) const override {
    auto& oo = static_cast<const QuantizePerGroupFp8&>(other);
    return g_ == oo.g_ && ue8m0_ == oo.ue8m0_;
  }

 private:
  int g_;
  bool ue8m0_;
};

class QuantizePerGroupInt8 : public Primitive {
 public:
  QuantizePerGroupInt8(Stream stream, int g) : Primitive(stream), g_(g) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerGroupInt8"; }
  void print(std::ostream& os) override { os << "QuantizePerGroupInt8"; }
  bool is_equivalent(const Primitive& other) const override {
    return g_ == static_cast<const QuantizePerGroupInt8&>(other).g_;
  }

 private:
  int g_;
};

class QuantizePerTokenInt8Azp : public Primitive {
 public:
  explicit QuantizePerTokenInt8Azp(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerTokenInt8Azp"; }
  void print(std::ostream& os) override { os << "QuantizePerTokenInt8Azp"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class QuantizePerTokenInt8 : public Primitive {
 public:
  explicit QuantizePerTokenInt8(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) override;
  std::vector<array> vjp(
      const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
      const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "QuantizePerTokenInt8"; }
  void print(std::ostream& os) override { os << "QuantizePerTokenInt8"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

} // namespace mlx::core
