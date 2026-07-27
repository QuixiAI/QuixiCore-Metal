// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "rotary/rotary.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// Operation Implementation
///////////////////////////////////////////////////////////////////////////////

array rotary(
    const array& x,
    const array& cos,
    const array& sin,
    bool interleaved /* = false */,
    StreamOrDevice s /* = {} */
) {
  assert(x.dtype() == bfloat16 && cos.dtype() == bfloat16 && sin.dtype() == bfloat16);
  assert(x.ndim() == 4 && "rotary: x must be (B, H, N, D)");
  const int D = x.shape(-1);
  const int N = x.shape(-2);
  assert((D == 64 || D == 128) && "rotary: head dim must be 64 or 128");
  assert(cos.shape(-1) == D / 2 && sin.shape(-1) == D / 2 &&
         cos.shape(-2) == N && sin.shape(-2) == N &&
         "rotary: cos/sin must be (N, D/2)");

  return array(
      /* const std::vector<int>& shape = */ x.shape(),
      /* Dtype dtype = */ bfloat16,
      /* std::shared_ptr<Primitive> primitive = */
      std::make_shared<Rotary>(to_stream(s), interleaved),
      /* const std::vector<array>& inputs = */ {x, cos, sin});
}

namespace {

int validate_extended_rope(
    const array& x,
    const array& cos,
    const array& sin,
    int rotary_dim,
    const char* op) {
  if (x.dtype() != bfloat16 || x.ndim() != 4) {
    throw std::invalid_argument(std::string(op) + ": x must be (B,H,N,D) bfloat16");
  }
  const int D = x.shape(-1);
  if (!(D == 64 || D == 128 || D == 256 || D == 512)) {
    throw std::invalid_argument(std::string(op) + ": head dim must be 64, 128, 256 or 512");
  }
  const int rd = rotary_dim == 0 ? D : rotary_dim;
  if (rd <= 0 || rd > D || rd % 2 != 0) {
    throw std::invalid_argument(std::string(op) +
                                ": rotary_dim must be positive, even, and <= head dim");
  }
  if (cos.ndim() != 2 || sin.ndim() != 2 || cos.shape() != sin.shape() ||
      cos.shape(1) != rd / 2) {
    throw std::invalid_argument(std::string(op) +
                                ": cos/sin must both be (max_pos, rotary_dim/2)");
  }
  return rd;
}

std::vector<array> extended_rope_inputs(
    const array& x,
    const array& cos,
    const array& sin,
    const array& positions,
    StreamOrDevice s) {
  return {contiguous(x, false, s),
          contiguous(astype(cos, bfloat16, s), false, s),
          contiguous(astype(sin, bfloat16, s), false, s),
          contiguous(astype(positions, int32, s), false, s)};
}

} // namespace

array rotary_positioned(
    const array& x,
    const array& cos,
    const array& sin,
    const array& positions,
    int rotary_dim,
    bool interleaved,
    StreamOrDevice s) {
  const int rd = validate_extended_rope(x, cos, sin, rotary_dim, "rotary_positioned");
  const int B = x.shape(0);
  const int N = x.shape(2);
  const bool batched = positions.ndim() == 2;
  if (!((positions.ndim() == 1 && positions.shape(0) == N) ||
        (batched && positions.shape(0) == B && positions.shape(1) == N))) {
    throw std::invalid_argument("rotary_positioned: positions must be (N,) or (B,N)");
  }
  return array(
      x.shape(), bfloat16,
      std::make_shared<RotaryPositioned>(to_stream(s), rd, interleaved, batched),
      extended_rope_inputs(x, cos, sin, positions, s));
}

array mrope(
    const array& x,
    const array& cos,
    const array& sin,
    const array& positions,
    const std::vector<int>& sections,
    int rotary_dim,
    bool section_interleaved,
    StreamOrDevice s) {
  const int rd = validate_extended_rope(x, cos, sin, rotary_dim, "mrope");
  const int B = x.shape(0);
  const int N = x.shape(2);
  if (sections.size() != 3 || sections[0] < 0 || sections[1] < 0 || sections[2] < 0 ||
      sections[0] + sections[1] + sections[2] != rd / 2) {
    throw std::invalid_argument(
        "mrope: sections must be three nonnegative pair counts summing to rotary_dim/2");
  }
  if (section_interleaved &&
      (sections[0] != (rd / 2 + 2) / 3 || sections[1] != (rd / 2 + 1) / 3 ||
       sections[2] != (rd / 2) / 3)) {
    throw std::invalid_argument(
        "mrope: interleaved sections must describe the THWTHW... axis counts");
  }
  const bool batched = positions.ndim() == 3;
  if (!((positions.ndim() == 2 && positions.shape(0) == 3 && positions.shape(1) == N) ||
        (batched && positions.shape(0) == B && positions.shape(1) == 3 &&
         positions.shape(2) == N))) {
    throw std::invalid_argument("mrope: positions must be (3,N) or (B,3,N)");
  }
  return array(
      x.shape(), bfloat16,
      std::make_shared<MRope>(to_stream(s), sections, rd, section_interleaved, batched),
      extended_rope_inputs(x, cos, sin, positions, s));
}

array vision_rope_2d(
    const array& x, const array& cos, const array& sin,
    const array& positions, bool global_split, StreamOrDevice s) {
  if (x.dtype() != bfloat16 || x.ndim() != 4 ||
      !(x.shape(3) == 64 || x.shape(3) == 128 || x.shape(3) == 256 || x.shape(3) == 512) ||
      cos.ndim() != 2 || sin.shape() != cos.shape() || cos.shape(1) != x.shape(3) / 4 ||
      positions.ndim() != 3 || positions.shape(0) != x.shape(0) ||
      positions.shape(1) != x.shape(2) || positions.shape(2) != 2) {
    throw std::invalid_argument(
        "vision_rope_2d: need x(B,H,N,D) bf16, cos/sin(P,D/4), positions(B,N,2)");
  }
  return array(
      x.shape(), bfloat16,
      std::make_shared<VisionRoPE2D>(to_stream(s), global_split),
      extended_rope_inputs(x, cos, sin, positions, s));
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Common Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void Rotary::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // No CPU fallback yet; use mx.fast.rope for a reference.
  assert(false);
}

void Rotary::eval_cpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Metal Backend Implementation
///////////////////////////////////////////////////////////////////////////////

void Rotary::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 3);
  auto& x = inputs[0];
  auto& cos = inputs[1];
  auto& sin = inputs[2];
  auto& out = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  const int D = x.shape(-1);
  const unsigned N = static_cast<unsigned>(x.shape(-2));
  const uint32_t M = static_cast<uint32_t>(x.size() / D); // B*H*N rows

  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_rotary(enc, x, cos, sin, out, M, N, D, interleaved_);
}

///////////////////////////////////////////////////////////////////////////////
// Primitive Transforms
///////////////////////////////////////////////////////////////////////////////

std::vector<array> Rotary::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {
  throw std::runtime_error("Rotary has no jvp implementation.");
}

std::vector<array> Rotary::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>&) {
  throw std::runtime_error("Rotary has no vjp implementation.");
}

std::pair<std::vector<array>, std::vector<int>> Rotary::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  throw std::runtime_error("Rotary has no vmap implementation.");
}

bool Rotary::is_equivalent(const Primitive& other) const {
  return interleaved_ == static_cast<const Rotary&>(other).interleaved_;
}

void RotaryPositioned::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("RotaryPositioned has no CPU implementation.");
}

void RotaryPositioned::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const uint32_t B = static_cast<uint32_t>(x.shape(0));
  const uint32_t H = static_cast<uint32_t>(x.shape(1));
  const uint32_t N = static_cast<uint32_t>(x.shape(2));
  const int D = x.shape(3);
  const uint32_t M = B * H * N;
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_rotary_positioned(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], out, M, N, H, D,
      static_cast<uint32_t>(rotary_dim_), batched_positions_ ? N : 0,
      interleaved_ ? 1 : 0);
}

std::vector<array> RotaryPositioned::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RotaryPositioned has no jvp implementation.");
}
std::vector<array> RotaryPositioned::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("RotaryPositioned has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> RotaryPositioned::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("RotaryPositioned has no vmap implementation.");
}
bool RotaryPositioned::is_equivalent(const Primitive& other) const {
  auto& o = static_cast<const RotaryPositioned&>(other);
  return rotary_dim_ == o.rotary_dim_ && interleaved_ == o.interleaved_ &&
         batched_positions_ == o.batched_positions_;
}

void MRope::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("MRope has no CPU implementation.");
}

void MRope::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0];
  auto& out = outputs[0];
  auto& s = stream();
  auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const uint32_t B = static_cast<uint32_t>(x.shape(0));
  const uint32_t H = static_cast<uint32_t>(x.shape(1));
  const uint32_t N = static_cast<uint32_t>(x.shape(2));
  const int D = x.shape(3);
  const uint32_t M = B * H * N;
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mrope(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], out, M, N, H, D,
      static_cast<uint32_t>(rotary_dim_), batched_positions_ ? 3 * N : 0,
      static_cast<uint32_t>(sections_[0]), static_cast<uint32_t>(sections_[1]),
      static_cast<uint32_t>(sections_[2]), section_interleaved_ ? 1 : 0);
}

std::vector<array> MRope::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MRope has no jvp implementation.");
}
std::vector<array> MRope::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("MRope has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> MRope::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("MRope has no vmap implementation.");
}
bool MRope::is_equivalent(const Primitive& other) const {
  auto& o = static_cast<const MRope&>(other);
  return sections_ == o.sections_ && rotary_dim_ == o.rotary_dim_ &&
         section_interleaved_ == o.section_interleaved_ &&
         batched_positions_ == o.batched_positions_;
}

void VisionRoPE2D::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("VisionRoPE2D has no CPU implementation.");
}
void VisionRoPE2D::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& x = inputs[0]; auto& out = outputs[0]; auto& s = stream();
  auto& dev = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  auto& ce = dev.get_command_encoder(s.index); MLXEncoder enc(dev, ce);
  tk::launch_vision_rope_2d(
      enc, inputs[0], inputs[1], inputs[2], inputs[3], out,
      x.shape(0), x.shape(1), x.shape(2), x.shape(3), inputs[1].shape(0),
      global_split_ ? 1 : 0);
}
std::vector<array> VisionRoPE2D::jvp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("VisionRoPE2D has no jvp implementation.");
}
std::vector<array> VisionRoPE2D::vjp(
    const std::vector<array>&, const std::vector<array>&, const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("VisionRoPE2D has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> VisionRoPE2D::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("VisionRoPE2D has no vmap implementation.");
}

} // namespace mlx::core
