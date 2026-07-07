#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "fake_quant/fake_quant.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {

bool fq_is_float(Dtype d) {
  return d == float32 || d == float16 || d == bfloat16;
}

std::vector<int> fq_scale_shape(const array& x) {
  std::vector<int> sh(x.shape().begin(), x.shape().end() - 1);
  if (sh.empty()) {
    sh.push_back(1);
  }
  return sh;
}

void fq_check(const array& x, const char* name) {
  if (x.ndim() < 1) {
    throw std::invalid_argument(std::string(name) + ": x must have at least 1 dimension");
  }
  if (!fq_is_float(x.dtype())) {
    throw std::invalid_argument(std::string(name) + ": x must be float32/float16/bfloat16");
  }
  if (x.shape(-1) % 4 != 0) {
    throw std::invalid_argument(std::string(name) + ": last dimension must be a multiple of 4");
  }
}

} // namespace

std::vector<array> fake_quant_int8(const array& x, StreamOrDevice s) {
  fq_check(x, "fake_quant_int8");
  return array::make_arrays(
      {x.shape(), x.shape(), fq_scale_shape(x)},
      {bfloat16, int8, float32},
      std::make_shared<FakeQuantInt8>(to_stream(s), false, 0, 1.702f, 7.0f),
      {contiguous(x, false, s)});
}

std::vector<array> silu_mul_fake_quant_int8(
    const array& x, const array& gate, int mode, float alpha, float limit, StreamOrDevice s) {
  fq_check(x, "silu_mul_fake_quant_int8");
  if (x.shape() != gate.shape()) {
    throw std::invalid_argument("silu_mul_fake_quant_int8: x and gate must have the same shape");
  }
  if (mode != 0 && mode != 1) {
    throw std::invalid_argument("silu_mul_fake_quant_int8: mode must be 0 or 1");
  }
  return array::make_arrays(
      {x.shape(), x.shape(), fq_scale_shape(x)},
      {bfloat16, int8, float32},
      std::make_shared<FakeQuantInt8>(to_stream(s), true, mode, alpha, limit),
      {contiguous(x, false, s), contiguous(astype(gate, x.dtype(), s), false, s)});
}

void FakeQuantInt8::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("FakeQuantInt8 has no CPU implementation.");
}

void FakeQuantInt8::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& x_q = outputs[0];
  auto& codes = outputs[1];
  auto& scale = outputs[2];
  auto& s = stream();
  auto& d = metal::device(s.device);
  x_q.set_data(allocator::malloc_or_wait(x_q.nbytes()));
  codes.set_data(allocator::malloc_or_wait(codes.nbytes()));
  scale.set_data(allocator::malloc_or_wait(scale.nbytes()));
  const int D = x.shape(-1);
  const int rows = static_cast<int>(x.size() / D);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  if (silu_mul_) {
    auto& gate = inputs[1];
    tk::launch_silu_mul_fake_quant_int8(
        enc, x, gate, x_q, codes, scale, rows, D, mode_, alpha_, limit_, type_to_name(x));
  } else {
    tk::launch_fake_quant_int8(enc, x, x_q, codes, scale, rows, D, type_to_name(x));
  }
}

#define TK_FQ_NO_AUTODIFF(CLASS, LABEL)                                       \
  std::vector<array> CLASS::jvp(                                              \
      const std::vector<array>&, const std::vector<array>&,                   \
      const std::vector<int>&) {                                              \
    throw std::runtime_error(LABEL " has no jvp implementation.");            \
  }                                                                           \
  std::vector<array> CLASS::vjp(                                              \
      const std::vector<array>&, const std::vector<array>&,                   \
      const std::vector<int>&, const std::vector<array>&) {                   \
    throw std::runtime_error(LABEL " has no vjp implementation.");            \
  }                                                                           \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                \
      const std::vector<array>&, const std::vector<int>&) {                   \
    throw std::runtime_error(LABEL " has no vmap implementation.");           \
  }

TK_FQ_NO_AUTODIFF(FakeQuantInt8, "FakeQuantInt8")

} // namespace mlx::core
