// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Monarch FFT convolution forward (N = S*S, S in {16,32}). Complex operands carry a
 *  leading size-2 (real,imag) axis: x (2,B,H,S,S); F/twf/finv/twi (2,S,S); kf (2,H,S,S).
 *  All float32. Returns the real output (B,H,S,S). Exercises the complex MMA. */
array fftconv(const array& x, const array& F, const array& twf, const array& finv,
              const array& twi, const array& kf, StreamOrDevice s = {});

class FFTConv : public Primitive {
 public:
  explicit FFTConv(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "FFTConv"; }

  void print(std::ostream& os) override { os << "FFTConv"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
