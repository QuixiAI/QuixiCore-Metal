// Copyright © 2026 QuixiCore contributors.

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "lora/lora.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

namespace {
bool lora_float(Dtype dtype) {
  return dtype == float32 || dtype == float16 || dtype == bfloat16;
}
}  // namespace

array lora_apply_direct(
    const array& x, const array& A, const array& B, const array& base,
    float scale, bool has_base, StreamOrDevice s) {
  if (x.ndim() != 2 || !lora_float(x.dtype())) {
    throw std::invalid_argument(
        "lora_apply_direct: x must be a float (rows, input_dim) tensor");
  }
  if (A.ndim() != 2 || B.ndim() != 2 || A.dtype() != float16 ||
      B.dtype() != float16) {
    throw std::invalid_argument(
        "lora_apply_direct: A(rank,input_dim) and B(output_dim,rank) must be float16");
  }
  const int rows = x.shape(0), input_dim = x.shape(1);
  const int rank = A.shape(0), output_dim = B.shape(0);
  if (A.shape(1) != input_dim || B.shape(1) != rank || rank < 1 || rank > 256 ||
      output_dim <= 0) {
    throw std::invalid_argument(
        "lora_apply_direct: incompatible shapes or rank outside [1, 256]");
  }
  if (has_base &&
      (base.ndim() != 2 || base.shape(0) != rows ||
       base.shape(1) != output_dim || !lora_float(base.dtype()))) {
    throw std::invalid_argument(
        "lora_apply_direct: base must be float (rows, output_dim) when provided");
  }
  if (!std::isfinite(scale)) {
    throw std::invalid_argument("lora_apply_direct: scale must be finite");
  }
  return array(
      {rows, output_dim}, x.dtype(),
      std::make_shared<LoraApplyDirect>(to_stream(s), scale, has_base),
      {contiguous(x, false, s), contiguous(A, false, s),
       contiguous(B, false, s),
       contiguous(astype(base, x.dtype(), s), false, s)});
}

void LoraApplyDirect::eval_cpu(
    const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("LoraApplyDirect has no CPU implementation.");
}

void LoraApplyDirect::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 4 && outputs.size() == 1);
  auto& x = inputs[0];
  auto& A = inputs[1];
  auto& B = inputs[2];
  auto& base = inputs[3];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_lora_apply_direct(
      enc, x, A, B, base, out, x.shape(0), x.shape(1),
      B.shape(0), A.shape(0), scale_, has_base_ ? 1 : 0,
      type_to_name(x));
}

std::vector<array> LoraApplyDirect::jvp(
    const std::vector<array>&, const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("LoraApplyDirect has no jvp implementation.");
}
std::vector<array> LoraApplyDirect::vjp(
    const std::vector<array>&, const std::vector<array>&,
    const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("LoraApplyDirect has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> LoraApplyDirect::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("LoraApplyDirect has no vmap implementation.");
}

}  // namespace mlx::core
