#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

array audio_conv1d_direct(const array& x, const array& weight, const array& bias,
                          int stride, int padding, int dilation, bool has_bias,
                          StreamOrDevice s = {});
array audio_depthwise_conv1d(const array& x, const array& weight, const array& bias,
                             int stride, int padding, int dilation, bool has_bias,
                             int activation, StreamOrDevice s = {});
array audio_depthwise_conv1d_asymmetric(
    const array& x, const array& weight, const array& bias,
    int stride, int pad_left, int pad_right, int dilation, bool has_bias,
    int activation, StreamOrDevice s = {});

class AudioConv1D : public Primitive {
 public:
  AudioConv1D(Stream stream, bool depthwise, int stride, int pad_left, int pad_right, int dilation,
              bool has_bias, int activation)
      : Primitive(stream), depthwise_(depthwise), stride_(stride), pad_left_(pad_left),
        pad_right_(pad_right),
        dilation_(dilation), has_bias_(has_bias), activation_(activation) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "AudioConv1D"; }
  void print(std::ostream& os) override { os << "AudioConv1D"; }
  bool is_equivalent(const Primitive& other) const override {
    const auto& o = static_cast<const AudioConv1D&>(other);
    return depthwise_ == o.depthwise_ && stride_ == o.stride_ &&
           pad_left_ == o.pad_left_ && pad_right_ == o.pad_right_ &&
           dilation_ == o.dilation_ && has_bias_ == o.has_bias_ && activation_ == o.activation_;
  }
 private:
  bool depthwise_, has_bias_;
  int stride_, pad_left_, pad_right_, dilation_, activation_;
};

}  // namespace mlx::core
