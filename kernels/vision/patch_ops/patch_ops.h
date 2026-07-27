#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array extract_patches_2d(const array& x, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int pad_h, int pad_w,
                         StreamOrDevice s = {});
array extract_patches_3d(
    const array& x, int kernel_t, int kernel_h, int kernel_w,
    int stride_t, int stride_h, int stride_w,
    int pad_t, int pad_h, int pad_w, StreamOrDevice s = {});
array interpolate_position_2d(const array& table, int out_h, int out_w,
                              bool align_corners, StreamOrDevice s = {});
array avg_pool2d_tokens(const array& x, int kernel_h, int kernel_w,
                        int stride_h, int stride_w, bool ceil_mode,
                        StreamOrDevice s = {});
array factorized_position_2d(const array& position_ids, const array& table,
                             const array& valid_mask, StreamOrDevice s = {});
std::vector<array> pool_tokens_by_position(
    const array& x, const array& position_ids, const array& valid_mask,
    int output_length, int kernel_size, int source_width,
    StreamOrDevice s = {});

class PatchOps : public Primitive {
 public:
  PatchOps(Stream stream, int kind, int a, int b, int c, int d, int e, int f)
      : Primitive(stream), kind_(kind), a_(a), b_(b), c_(c), d_(d), e_(e), f_(f) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PatchOps"; }
  void print(std::ostream& os) override { os << "PatchOps(" << kind_ << ")"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const PatchOps&>(other);
    return kind_ == o.kind_ && a_ == o.a_ && b_ == o.b_ && c_ == o.c_ &&
           d_ == o.d_ && e_ == o.e_ && f_ == o.f_;
  }
 private:
  int kind_, a_, b_, c_, d_, e_, f_;
};

class FactorizedPosition2D : public Primitive {
 public:
  explicit FactorizedPosition2D(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "FactorizedPosition2D"; }
  void print(std::ostream& os) override { os << "FactorizedPosition2D"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class PatchOps3D : public Primitive {
 public:
  PatchOps3D(Stream stream, int kt, int kh, int kw, int st, int sh, int sw,
             int pt, int ph, int pw)
      : Primitive(stream), kt_(kt), kh_(kh), kw_(kw), st_(st), sh_(sh), sw_(sw),
        pt_(pt), ph_(ph), pw_(pw) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PatchOps3D"; }
  void print(std::ostream& os) override { os << "PatchOps3D"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const PatchOps3D&>(other);
    return kt_ == o.kt_ && kh_ == o.kh_ && kw_ == o.kw_ && st_ == o.st_ &&
           sh_ == o.sh_ && sw_ == o.sw_ && pt_ == o.pt_ && ph_ == o.ph_ && pw_ == o.pw_;
  }
 private:
  int kt_, kh_, kw_, st_, sh_, sw_, pt_, ph_, pw_;
};

class PoolTokensByPosition : public Primitive {
 public:
  PoolTokensByPosition(Stream stream, int output_length, int kernel_size,
                       int source_width)
      : Primitive(stream), output_length_(output_length), kernel_size_(kernel_size),
        source_width_(source_width) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "PoolTokensByPosition"; }
  void print(std::ostream& os) override { os << "PoolTokensByPosition"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const PoolTokensByPosition&>(other);
    return output_length_ == o.output_length_ && kernel_size_ == o.kernel_size_ &&
           source_width_ == o.source_width_;
  }
 private:
  int output_length_, kernel_size_, source_width_;
};

}  // namespace mlx::core
