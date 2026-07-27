#include <cassert>
#include <stdexcept>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"
#include "patch_ops/patch_ops.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {
namespace {
bool patch_float(Dtype d) { return d == float32 || d == float16 || d == bfloat16; }
int out_floor(int n, int k, int s, int p) { return (n + 2 * p - k) / s + 1; }
int out_pool(int n, int k, int s, bool ceil) {
  return ceil ? (n - k + s - 1) / s + 1 : (n - k) / s + 1;
}
}

array extract_patches_2d(const array& x, int kh, int kw, int sh, int sw,
                         int ph, int pw, StreamOrDevice s) {
  if (x.ndim() != 4 || !patch_float(x.dtype()) || kh <= 0 || kw <= 0 ||
      sh <= 0 || sw <= 0 || ph < 0 || pw < 0) {
    throw std::invalid_argument("extract_patches_2d: need float NHWC x and positive kernel/stride");
  }
  const int oh = out_floor(x.shape(1), kh, sh, ph), ow = out_floor(x.shape(2), kw, sw, pw);
  if (oh <= 0 || ow <= 0) throw std::invalid_argument("extract_patches_2d: empty output");
  return array({x.shape(0), oh * ow, kh * kw * x.shape(3)}, x.dtype(),
               std::make_shared<PatchOps>(to_stream(s), 0, kh, kw, sh, sw, ph, pw),
               {contiguous(x, false, s)});
}

array extract_patches_3d(
    const array& x, int kt, int kh, int kw, int st, int sh, int sw,
    int pt, int ph, int pw, StreamOrDevice s) {
  if (x.ndim() != 5 || !patch_float(x.dtype()) || kt <= 0 || kh <= 0 || kw <= 0 ||
      st <= 0 || sh <= 0 || sw <= 0 || pt < 0 || ph < 0 || pw < 0) {
    throw std::invalid_argument(
        "extract_patches_3d: need float NTHWC x and positive kernel/stride");
  }
  const int ot = out_floor(x.shape(1), kt, st, pt);
  const int oh = out_floor(x.shape(2), kh, sh, ph);
  const int ow = out_floor(x.shape(3), kw, sw, pw);
  if (ot <= 0 || oh <= 0 || ow <= 0)
    throw std::invalid_argument("extract_patches_3d: empty output");
  return array(
      {x.shape(0), ot * oh * ow, kt * kh * kw * x.shape(4)}, x.dtype(),
      std::make_shared<PatchOps3D>(
          to_stream(s), kt, kh, kw, st, sh, sw, pt, ph, pw),
      {contiguous(x, false, s)});
}

array interpolate_position_2d(const array& table, int oh, int ow,
                              bool align, StreamOrDevice s) {
  if (table.ndim() != 3 || !patch_float(table.dtype()) || oh <= 0 || ow <= 0 ||
      table.shape(0) <= 0 || table.shape(1) <= 0) {
    throw std::invalid_argument("interpolate_position_2d: need float (H,W,D) and positive output");
  }
  return array({oh, ow, table.shape(2)}, table.dtype(),
               std::make_shared<PatchOps>(to_stream(s), 1, oh, ow, align ? 1 : 0, 0, 0, 0),
               {contiguous(table, false, s)});
}

array avg_pool2d_tokens(const array& x, int kh, int kw, int sh, int sw,
                        bool ceil, StreamOrDevice s) {
  if (x.ndim() != 4 || !patch_float(x.dtype()) || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0) {
    throw std::invalid_argument("avg_pool2d_tokens: need float NHWC x and positive kernel/stride");
  }
  const int oh = out_pool(x.shape(1), kh, sh, ceil), ow = out_pool(x.shape(2), kw, sw, ceil);
  if (oh <= 0 || ow <= 0) throw std::invalid_argument("avg_pool2d_tokens: empty output");
  return array({x.shape(0), oh, ow, x.shape(3)}, x.dtype(),
               std::make_shared<PatchOps>(to_stream(s), 2, kh, kw, sh, sw, ceil ? 1 : 0, 0),
               {contiguous(x, false, s)});
}

array factorized_position_2d(const array& position_ids, const array& table,
                             const array& valid_mask, StreamOrDevice s) {
  if (position_ids.ndim() != 3 || position_ids.shape(2) != 2 ||
      table.ndim() != 3 || table.shape(0) != 2 || !patch_float(table.dtype()) ||
      valid_mask.ndim() != 2 || valid_mask.shape(0) != position_ids.shape(0) ||
      valid_mask.shape(1) != position_ids.shape(1) || table.shape(1) <= 0) {
    throw std::invalid_argument(
        "factorized_position_2d: need ids(B,N,2), table(2,P,D), valid_mask(B,N)");
  }
  return array(
      {position_ids.shape(0), position_ids.shape(1), table.shape(2)}, table.dtype(),
      std::make_shared<FactorizedPosition2D>(to_stream(s)),
      {contiguous(astype(position_ids, int32, s), false, s),
       contiguous(table, false, s),
       contiguous(astype(valid_mask, int32, s), false, s)});
}

std::vector<array> pool_tokens_by_position(
    const array& x, const array& position_ids, const array& valid_mask,
    int output_length, int kernel_size, int source_width, StreamOrDevice s) {
  if (x.ndim() != 3 || !patch_float(x.dtype()) || position_ids.ndim() != 3 ||
      position_ids.shape(0) != x.shape(0) || position_ids.shape(1) != x.shape(1) ||
      position_ids.shape(2) != 2 || valid_mask.ndim() != 2 ||
      valid_mask.shape(0) != x.shape(0) || valid_mask.shape(1) != x.shape(1) ||
      output_length <= 0 || kernel_size <= 0 || source_width <= 0 ||
      source_width % kernel_size != 0) {
    throw std::invalid_argument(
        "pool_tokens_by_position: need x(B,N,D), ids(B,N,2), valid_mask(B,N), valid geometry");
  }
  return array::make_arrays(
      {{x.shape(0), output_length, x.shape(2)}, {x.shape(0), output_length}},
      {float32, int32},
      std::make_shared<PoolTokensByPosition>(
          to_stream(s), output_length, kernel_size, source_width),
      {contiguous(x, false, s),
       contiguous(astype(position_ids, int32, s), false, s),
       contiguous(astype(valid_mask, int32, s), false, s)});
}

void PatchOps::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PatchOps has no CPU implementation.");
}
void PatchOps::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0]; auto& out = outputs[0]; auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = d.get_command_encoder(s.index); MLXEncoder enc(d, ce);
  if (kind_ == 0) {
    tk::launch_extract_patches_2d(enc, x, out, x.shape(0), x.shape(1), x.shape(2), x.shape(3),
                                  a_, b_, c_, d_, e_, f_, type_to_name(x));
  } else if (kind_ == 1) {
    tk::launch_interpolate_position_2d(enc, x, out, x.shape(0), x.shape(1), a_, b_, x.shape(2),
                                       c_, type_to_name(x));
  } else {
    tk::launch_avg_pool2d_tokens(enc, x, out, x.shape(0), x.shape(1), x.shape(2), x.shape(3),
                                 a_, b_, c_, d_, e_ != 0, type_to_name(x));
  }
}

#define PATCH_NO_AUTODIFF(NAME)                                                  \
std::vector<array> NAME::jvp(const std::vector<array>&, const std::vector<array>&,\
                             const std::vector<int>&) { throw std::runtime_error("PatchOps has no jvp"); }\
std::vector<array> NAME::vjp(const std::vector<array>&, const std::vector<array>&,\
                             const std::vector<int>&, const std::vector<array>&) { throw std::runtime_error("PatchOps has no vjp"); }\
std::pair<std::vector<array>, std::vector<int>> NAME::vmap(                      \
    const std::vector<array>&, const std::vector<int>&) { throw std::runtime_error("PatchOps has no vmap"); }
PATCH_NO_AUTODIFF(PatchOps)
PATCH_NO_AUTODIFF(FactorizedPosition2D)
PATCH_NO_AUTODIFF(PatchOps3D)
PATCH_NO_AUTODIFF(PoolTokensByPosition)

void PatchOps3D::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PatchOps3D has no CPU implementation.");
}
void PatchOps3D::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0]; auto& out = outputs[0]; auto& s = stream();
  auto& dev = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = dev.get_command_encoder(s.index); MLXEncoder enc(dev, ce);
  tk::launch_extract_patches_3d(
      enc, x, out, x.shape(0), x.shape(1), x.shape(2), x.shape(3), x.shape(4),
      kt_, kh_, kw_, st_, sh_, sw_, pt_, ph_, pw_, type_to_name(x));
}

void FactorizedPosition2D::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("FactorizedPosition2D has no CPU implementation.");
}
void FactorizedPosition2D::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& ids = inputs[0]; auto& table = inputs[1]; auto& mask = inputs[2];
  auto& out = outputs[0]; auto& s = stream(); auto& dev = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = dev.get_command_encoder(s.index); MLXEncoder enc(dev, ce);
  tk::launch_factorized_position_2d(
      enc, ids, table, mask, out, ids.shape(0), ids.shape(1), table.shape(1),
      table.shape(2), type_to_name(table));
}

void PoolTokensByPosition::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("PoolTokensByPosition has no CPU implementation.");
}
void PoolTokensByPosition::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0]; auto& ids = inputs[1]; auto& mask = inputs[2];
  auto& out = outputs[0]; auto& out_mask = outputs[1];
  auto& s = stream(); auto& dev = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  out_mask.set_data(allocator::malloc_or_wait(out_mask.nbytes()));
  auto& ce = dev.get_command_encoder(s.index); MLXEncoder enc(dev, ce);
  tk::launch_pool_tokens_by_position(
      enc, x, ids, mask, out, out_mask, x.shape(0), x.shape(1), x.shape(2),
      output_length_, kernel_size_, source_width_, type_to_name(x));
}

}  // namespace mlx::core
