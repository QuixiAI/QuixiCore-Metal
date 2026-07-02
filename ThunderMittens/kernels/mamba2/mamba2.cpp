// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "mamba2/mamba2.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "tk_mlx_launch.h"
#endif

namespace mlx::core {

array ssd_chunked(const array& Cq, const array& Bm, const array& X, const array& cl,
                  StreamOrDevice s) {
  const int B = Cq.shape(0), H = Cq.shape(1), N = Cq.shape(2), D = Cq.shape(3);
  const int C = N / 64;   // chunk L = 64 (must match SSD_CHUNK_L in the metal)
  auto kv = array({B, H, C, D, D}, float32,
                  std::make_shared<SsdChunkKV>(to_stream(s)), {Bm, X, cl});
  auto sex = array({B, H, C, D, D}, float32,
                   std::make_shared<SsdChunkScan>(to_stream(s)), {kv, cl});
  return array(Cq.shape(), bfloat16,
               std::make_shared<SsdChunkOut>(to_stream(s)), {Cq, Bm, X, cl, sex});
}

array mamba2(const array& C, const array& B, const array& X, const array& cumlog,
             StreamOrDevice s) {
  assert(C.dtype() == bfloat16 && B.dtype() == bfloat16 && X.dtype() == bfloat16);
  assert(cumlog.dtype() == float32);
  assert(C.shape() == B.shape() && B.shape() == X.shape());
  const int N = C.shape(2), D = C.shape(3);
  assert((D == 64 || D == 128) && "mamba2 supports D in {64, 128}");
  assert(N % 8 == 0 && "mamba2: N must be a multiple of 8");
  // The chunked pipeline holds a D x D state quadrant per simdgroup (rt_fl<D,D>), which is only
  // register-feasible at D=64; D=128 uses the quadratic materialized kernel.
  if (D == 64 && N % 64 == 0 && N >= 128) {
    return ssd_chunked(C, B, X, cumlog, s);   // linear-time chunked pipeline
  }
  return array(C.shape(), bfloat16,
               std::make_shared<Mamba2>(to_stream(s)), {C, B, X, cumlog});
}

void SsdChunkKV::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void SsdChunkKV::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& bm = inputs[0]; auto& x = inputs[1]; auto& cl = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = bm.shape(0), H = bm.shape(1), N = bm.shape(2), D = bm.shape(3);
  const int C = out.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_kv(enc, bm, x, cl, out, static_cast<unsigned>(N),
                          static_cast<unsigned>(H), B, C, D);
}
std::vector<array> SsdChunkKV::jvp(const std::vector<array>&, const std::vector<array>&,
                                   const std::vector<int>&) {
  throw std::runtime_error("SsdChunkKV has no jvp implementation.");
}
std::vector<array> SsdChunkKV::vjp(const std::vector<array>&, const std::vector<array>&,
                                   const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("SsdChunkKV has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SsdChunkKV::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SsdChunkKV has no vmap implementation.");
}
bool SsdChunkKV::is_equivalent(const Primitive&) const { return true; }

void SsdChunkScan::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void SsdChunkScan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& sin = inputs[0]; auto& cl = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = sin.shape(0), H = sin.shape(1), C = sin.shape(2), D = sin.shape(3);
  const int N = cl.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_scan(enc, sin, cl, out, static_cast<unsigned>(C),
                            static_cast<unsigned>(N), B * H, D);
}
std::vector<array> SsdChunkScan::jvp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&) {
  throw std::runtime_error("SsdChunkScan has no jvp implementation.");
}
std::vector<array> SsdChunkScan::vjp(const std::vector<array>&, const std::vector<array>&,
                                     const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("SsdChunkScan has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SsdChunkScan::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SsdChunkScan has no vmap implementation.");
}
bool SsdChunkScan::is_equivalent(const Primitive&) const { return true; }

void SsdChunkOut::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); }
void SsdChunkOut::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cq = inputs[0]; auto& bm = inputs[1]; auto& x = inputs[2];
  auto& cl = inputs[3]; auto& sex = inputs[4];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = cq.shape(0), H = cq.shape(1), N = cq.shape(2), D = cq.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_out(enc, cq, bm, x, cl, sex, out, static_cast<unsigned>(N),
                           static_cast<unsigned>(H), B, D);
}
std::vector<array> SsdChunkOut::jvp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&) {
  throw std::runtime_error("SsdChunkOut has no jvp implementation.");
}
std::vector<array> SsdChunkOut::vjp(const std::vector<array>&, const std::vector<array>&,
                                    const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("SsdChunkOut has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> SsdChunkOut::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("SsdChunkOut has no vmap implementation.");
}
bool SsdChunkOut::is_equivalent(const Primitive&) const { return true; }

// ---------- chunked linear-time backward ----------
void SsdBwdQkv::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cq = inputs[0]; auto& dy = inputs[1]; auto& cl = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = cq.shape(0), H = cq.shape(1), N = cq.shape(2), D = cq.shape(3);
  const int C = out.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_bwd_qkv(enc, cq, dy, cl, out, static_cast<unsigned>(N), static_cast<unsigned>(H),
                         B, C, D);
}
void SsdBwdQscan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& qin = inputs[0]; auto& cl = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = qin.shape(0), H = qin.shape(1), C = qin.shape(2), D = qin.shape(3);
  const int N = cl.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_bwd_qscan(enc, qin, cl, out, static_cast<unsigned>(C), static_cast<unsigned>(N),
                           B * H, D);
}
void SsdBwdOutRow::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cq = inputs[0]; auto& bm = inputs[1]; auto& x = inputs[2];
  auto& cl = inputs[3]; auto& dy = inputs[4]; auto& p = inputs[5];
  auto& dC = outputs[0]; auto& r = outputs[1]; auto& ri = outputs[2];
  auto& s = stream(); auto& d = metal::device(s.device);
  dC.set_data(allocator::malloc_or_wait(dC.nbytes()));
  r.set_data(allocator::malloc_or_wait(r.nbytes()));
  ri.set_data(allocator::malloc_or_wait(ri.nbytes()));
  const int B = cq.shape(0), H = cq.shape(1), N = cq.shape(2), D = cq.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_bwd_out_row(enc, cq, bm, x, cl, dy, p, dC, r, ri, static_cast<unsigned>(N),
                             static_cast<unsigned>(H), B, D);
}
void SsdBwdOutCol::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cq = inputs[0]; auto& bm = inputs[1]; auto& x = inputs[2];
  auto& cl = inputs[3]; auto& dy = inputs[4]; auto& qex = inputs[5]; auto& qtex = inputs[6];
  auto& dB = outputs[0]; auto& dX = outputs[1]; auto& cc = outputs[2]; auto& ci = outputs[3];
  auto& s = stream(); auto& d = metal::device(s.device);
  dB.set_data(allocator::malloc_or_wait(dB.nbytes()));
  dX.set_data(allocator::malloc_or_wait(dX.nbytes()));
  cc.set_data(allocator::malloc_or_wait(cc.nbytes()));
  ci.set_data(allocator::malloc_or_wait(ci.nbytes()));
  const int B = cq.shape(0), H = cq.shape(1), N = cq.shape(2), D = cq.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_bwd_out_col(enc, cq, bm, x, cl, dy, qex, qtex, dB, dX, cc, ci,
                             static_cast<unsigned>(N), static_cast<unsigned>(H), B, D);
}
#define TK_SSD_BWD_NO_AD(CLASS)                                                      \
  void CLASS::eval_cpu(const std::vector<array>&, std::vector<array>&) { assert(false); } \
  std::vector<array> CLASS::jvp(const std::vector<array>&, const std::vector<array>&, \
                                const std::vector<int>&) {                           \
    throw std::runtime_error(#CLASS " has no jvp."); }                              \
  std::vector<array> CLASS::vjp(const std::vector<array>&, const std::vector<array>&, \
                                const std::vector<int>&, const std::vector<array>&) { \
    throw std::runtime_error(#CLASS " has no vjp."); }                              \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(                       \
      const std::vector<array>&, const std::vector<int>&) {                          \
    throw std::runtime_error(#CLASS " has no vmap."); }
TK_SSD_BWD_NO_AD(SsdBwdQkv)
TK_SSD_BWD_NO_AD(SsdBwdQscan)
TK_SSD_BWD_NO_AD(SsdBwdOutRow)
TK_SSD_BWD_NO_AD(SsdBwdOutCol)

std::vector<array> ssd_chunked_bwd(const array& C, const array& B, const array& X,
                                   const array& cumlog, const array& dY, StreamOrDevice s) {
  const int Bsz = C.shape(0), H = C.shape(1), N = C.shape(2), D = C.shape(3);
  const int Cn = N / 64;
  auto C_c = contiguous(C, false, s), B_c = contiguous(B, false, s), X_c = contiguous(X, false, s);
  auto cl_c = contiguous(cumlog, false, s), dY_c = contiguous(dY, false, s);
  // Forward state P_c = sum_{j<chunk c} exp(cl[r_{c-1}]-cl_j) X_j B_j^T (X rows, B cols): the
  // forward ssd_chunk_kv builds B_j X_j^T, so feed (X,B) SWAPPED to get X_j B_j^T directly (then
  // the row inter term is mma_AB(dY_i, P), no transpose-on-read).
  auto pkv = array({Bsz, H, Cn, D, D}, float32, std::make_shared<SsdChunkKV>(to_stream(s)),
                   {X_c, B_c, cl_c});
  auto P = array({Bsz, H, Cn, D, D}, float32, std::make_shared<SsdChunkScan>(to_stream(s)),
                 {pkv, cl_c});
  // Reverse state Q_c = sum_{i>chunk c} exp(cl_i-cl[r_c]) C_i dY_i^T (C rows, dY cols) for dX, and
  // its transpose Qt_c = sum ... dY_i C_i^T (dY rows, C cols) for dB — built by SWAPPING (C,dY).
  auto qkv = array({Bsz, H, Cn, D, D}, float32, std::make_shared<SsdBwdQkv>(to_stream(s)),
                   {C_c, dY_c, cl_c});
  auto qex = array({Bsz, H, Cn, D, D}, float32, std::make_shared<SsdBwdQscan>(to_stream(s)),
                   {qkv, cl_c});
  auto qtkv = array({Bsz, H, Cn, D, D}, float32, std::make_shared<SsdBwdQkv>(to_stream(s)),
                    {dY_c, C_c, cl_c});
  auto qtex = array({Bsz, H, Cn, D, D}, float32, std::make_shared<SsdBwdQscan>(to_stream(s)),
                    {qtkv, cl_c});
  // Row-owned: dC, r_intra (rowsum), r_inter. Col-owned: dB, dX, c_intra (colsum), c_inter.
  auto row = array::make_arrays(
      {C.shape(), {Bsz, H, N}, {Bsz, H, N}}, {bfloat16, float32, float32},
      std::make_shared<SsdBwdOutRow>(to_stream(s)), {C_c, B_c, X_c, cl_c, dY_c, P});
  auto col = array::make_arrays(
      {C.shape(), C.shape(), {Bsz, H, N}, {Bsz, H, N}}, {bfloat16, bfloat16, float32, float32},
      std::make_shared<SsdBwdOutCol>(to_stream(s)), {C_c, B_c, X_c, cl_c, dY_c, qex, qtex});
  auto dcl = subtract(add(row[1], row[2], s), add(col[2], col[3], s), s);
  return {row[0], col[0], col[1], dcl};
}

void Mamba2::eval(const std::vector<array>&, std::vector<array>&) { assert(false); }
void Mamba2::eval_cpu(const std::vector<array>& in, std::vector<array>& out) { eval(in, out); }

void Mamba2::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 4);
  auto& C = inputs[0]; auto& B = inputs[1]; auto& X = inputs[2]; auto& cumlog = inputs[3];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int Bsz = C.shape(0), H = C.shape(1), N = C.shape(2), D = C.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mamba2(enc, C, B, X, cumlog, out, static_cast<unsigned>(N),
                    static_cast<unsigned>(H), Bsz, D);
}

std::vector<array> Mamba2::jvp(const std::vector<array>&, const std::vector<array>&,
                               const std::vector<int>&) {
  throw std::runtime_error("Mamba2 has no jvp implementation.");
}
std::vector<array> Mamba2::vjp(const std::vector<array>&, const std::vector<array>&,
                               const std::vector<int>&, const std::vector<array>&) {
  throw std::runtime_error("Mamba2 has no vjp implementation.");
}
std::pair<std::vector<array>, std::vector<int>> Mamba2::vmap(
    const std::vector<array>&, const std::vector<int>&) {
  throw std::runtime_error("Mamba2 has no vmap implementation.");
}
bool Mamba2::is_equivalent(const Primitive&) const { return true; }

// ----------------------------- mamba2_bwd -----------------------------

std::vector<array> mamba2_bwd(const array& C, const array& B, const array& X, const array& cumlog,
                              const array& dY, bool force_quadratic /* = false */,
                              StreamOrDevice s /* = {} */) {
  assert(C.dtype() == bfloat16 && B.dtype() == bfloat16 && X.dtype() == bfloat16 &&
         dY.dtype() == bfloat16);
  assert(cumlog.dtype() == float32);
  assert(C.shape() == B.shape() && B.shape() == X.shape() && X.shape() == dY.shape());
  const int Bsz = C.shape(0), H = C.shape(1), N = C.shape(2), D = C.shape(3);
  assert((D == 64 || D == 128) && "mamba2_bwd supports D in {64, 128}");
  assert(N % 8 == 0 && "mamba2_bwd: N must be a multiple of 8");
  // Chunked linear-time backward, gated exactly like the forward chunked route (the chunked
  // kernels hold a D x D state, register-feasible only at D=64); else the quadratic backward.
  if (!force_quadratic && D == 64 && N % 64 == 0 && N >= 128) {
    return ssd_chunked_bwd(C, B, X, cumlog, dY, s);
  }
  auto C_c = contiguous(C, false, s);
  auto B_c = contiguous(B, false, s);
  auto X_c = contiguous(X, false, s);
  auto cl_c = contiguous(cumlog, false, s);
  auto dY_c = contiguous(dY, false, s);

  auto row = array::make_arrays(
      {C.shape(), {Bsz, H, N}}, {bfloat16, float32},
      std::make_shared<Mamba2BwdRow>(to_stream(s)), {C_c, B_c, X_c, cl_c, dY_c});
  auto col = array::make_arrays(
      {C.shape(), C.shape(), {Bsz, H, N}}, {bfloat16, bfloat16, float32},
      std::make_shared<Mamba2BwdCol>(to_stream(s)), {C_c, B_c, X_c, cl_c, dY_c});
  // dcumlog = rowsum(M) - colsum(M)
  auto dcl = subtract(row[1], col[2], s);
  return {row[0], col[0], col[1], dcl};
}

void Mamba2BwdRow::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("Mamba2BwdRow has no CPU implementation.");
}
void Mamba2BwdRow::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& C = inputs[0]; auto& B = inputs[1]; auto& X = inputs[2];
  auto& cl = inputs[3]; auto& dY = inputs[4];
  auto& dC = outputs[0]; auto& r = outputs[1];
  auto& s = stream(); auto& d = metal::device(s.device);
  dC.set_data(allocator::malloc_or_wait(dC.nbytes()));
  r.set_data(allocator::malloc_or_wait(r.nbytes()));
  const int Bsz = C.shape(0), H = C.shape(1), N = C.shape(2), D = C.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mamba2_bwd_row(enc, C, B, X, cl, dY, dC, r, static_cast<unsigned>(N),
                            static_cast<unsigned>(H), Bsz, D);
}

void Mamba2BwdCol::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("Mamba2BwdCol has no CPU implementation.");
}
void Mamba2BwdCol::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& C = inputs[0]; auto& B = inputs[1]; auto& X = inputs[2];
  auto& cl = inputs[3]; auto& dY = inputs[4];
  auto& dB = outputs[0]; auto& dX = outputs[1]; auto& cc = outputs[2];
  auto& s = stream(); auto& d = metal::device(s.device);
  dB.set_data(allocator::malloc_or_wait(dB.nbytes()));
  dX.set_data(allocator::malloc_or_wait(dX.nbytes()));
  cc.set_data(allocator::malloc_or_wait(cc.nbytes()));
  const int Bsz = C.shape(0), H = C.shape(1), N = C.shape(2), D = C.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_mamba2_bwd_col(enc, C, B, X, cl, dY, dB, dX, cc, static_cast<unsigned>(N),
                            static_cast<unsigned>(H), Bsz, D);
}

#define TK_MAMBA_BWD_NO_AUTODIFF(CLASS, LABEL)                               \
  std::vector<array> CLASS::jvp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&) {                                             \
    throw std::runtime_error(LABEL " has no jvp implementation.");           \
  }                                                                          \
  std::vector<array> CLASS::vjp(                                             \
      const std::vector<array>&, const std::vector<array>&,                  \
      const std::vector<int>&, const std::vector<array>&) {                  \
    throw std::runtime_error(LABEL " has no vjp implementation.");           \
  }                                                                          \
  std::pair<std::vector<array>, std::vector<int>> CLASS::vmap(               \
      const std::vector<array>&, const std::vector<int>&) {                  \
    throw std::runtime_error(LABEL " has no vmap implementation.");          \
  }

TK_MAMBA_BWD_NO_AUTODIFF(Mamba2BwdRow, "Mamba2BwdRow")
TK_MAMBA_BWD_NO_AUTODIFF(Mamba2BwdCol, "Mamba2BwdCol")

} // namespace mlx::core
