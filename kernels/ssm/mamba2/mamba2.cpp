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

array ssd_chunked(const array& Cq_in, const array& Bm_in, const array& X_in, const array& cl_in,
                  StreamOrDevice s) {
  // Kernels read raw buffers, so a transposed/strided view from the Python wrapper would be read
  // as garbage past row 0 ([[mlx-strided-view-contiguity]]). No-op when already row-contiguous.
  auto Cq = contiguous(Cq_in, false, s), Bm = contiguous(Bm_in, false, s);
  auto X = contiguous(X_in, false, s), cl = contiguous(cl_in, false, s);
  const int B = Cq.shape(0), H = Cq.shape(1), N = Cq.shape(2), D = Cq.shape(3);
  const int C = N / 64;   // chunk L = 64 (must match SSD_CHUNK_L in the metal)
  auto kv = array({B, H, C, D, D}, float32,
                  std::make_shared<SsdChunkKV>(to_stream(s)), {Bm, X, cl});
  // The scanned state is stored BF16: the out mma consumes bf16 anyway, so results are
  // identical and the dominant state-read traffic halves.
  auto sex = array({B, H, C, D, D}, bfloat16,
                   std::make_shared<SsdChunkScan>(to_stream(s)), {kv, cl});
  return array(Cq.shape(), bfloat16,
               std::make_shared<SsdChunkOut>(to_stream(s)), {Cq, Bm, X, cl, sex});
}

array mamba2(const array& C_in, const array& B_in, const array& X_in, const array& cumlog_in,
             StreamOrDevice s) {
  // Force row-contiguity for the quadratic-kernel path too (the chunked path re-forces in
  // ssd_chunked; both are no-ops when already contiguous). [[mlx-strided-view-contiguity]]
  auto C = contiguous(C_in, false, s), B = contiguous(B_in, false, s);
  auto X = contiguous(X_in, false, s), cumlog = contiguous(cumlog_in, false, s);
  assert(C.dtype() == bfloat16 && B.dtype() == bfloat16 && X.dtype() == bfloat16);
  assert(cumlog.dtype() == float32);
  assert(C.shape() == B.shape() && B.shape() == X.shape());
  const int N = C.shape(2), D = C.shape(3);
  assert((D == 64 || D == 128) && "mamba2 supports D in {64, 128}");
  assert(N % 8 == 0 && "mamba2: N must be a multiple of 8");
  // The chunked D x D state is 64x64 quadrant-tiled (QB = D/64), so both head dims run chunked.
  // Route thresholds are MEASURED (M-series, cooperative out kernel): the quadratic kernel wins
  // below N=2048 at D=64 and below N=4096 at D=128.
  const int chunk_min = (D == 64) ? 2048 : 4096;
  if (N % 64 == 0 && N >= chunk_min) {
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
void SsdChunkGstate::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cq = inputs[0]; auto& dy = inputs[1]; auto& cl = inputs[2];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = cq.shape(0), H = cq.shape(1), N = cq.shape(2), D = cq.shape(3);
  const int C = out.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_gstate(enc, cq, dy, cl, out, static_cast<unsigned>(N),
                              static_cast<unsigned>(H), B, C, D);
}
void SsdChunkRscan::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& gin = inputs[0]; auto& cl = inputs[1];
  auto& out = outputs[0];
  auto& s = stream(); auto& d = metal::device(s.device);
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  const int B = gin.shape(0), H = gin.shape(1), C = gin.shape(2), D = gin.shape(3);
  const int N = cl.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_rscan(enc, gin, cl, out, static_cast<unsigned>(C),
                             static_cast<unsigned>(N), B * H, D);
}
void SsdChunkBwdRow::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cq = inputs[0]; auto& bm = inputs[1]; auto& x = inputs[2];
  auto& cl = inputs[3]; auto& dy = inputs[4]; auto& sex = inputs[5];
  auto& dC = outputs[0]; auto& r = outputs[1]; auto& ri = outputs[2];
  auto& s = stream(); auto& d = metal::device(s.device);
  dC.set_data(allocator::malloc_or_wait(dC.nbytes()));
  r.set_data(allocator::malloc_or_wait(r.nbytes()));
  ri.set_data(allocator::malloc_or_wait(ri.nbytes()));
  const int B = cq.shape(0), H = cq.shape(1), N = cq.shape(2), D = cq.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_bwd_row(enc, cq, bm, x, cl, dy, sex, dC, r, ri, static_cast<unsigned>(N),
                               static_cast<unsigned>(H), B, D);
}
void SsdChunkBwdCol::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& cq = inputs[0]; auto& bm = inputs[1]; auto& x = inputs[2];
  auto& cl = inputs[3]; auto& dy = inputs[4]; auto& dkv = inputs[5];
  auto& dB = outputs[0]; auto& dX = outputs[1]; auto& cc = outputs[2]; auto& ci = outputs[3];
  auto& s = stream(); auto& d = metal::device(s.device);
  dB.set_data(allocator::malloc_or_wait(dB.nbytes()));
  dX.set_data(allocator::malloc_or_wait(dX.nbytes()));
  cc.set_data(allocator::malloc_or_wait(cc.nbytes()));
  ci.set_data(allocator::malloc_or_wait(ci.nbytes()));
  const int B = cq.shape(0), H = cq.shape(1), N = cq.shape(2), D = cq.shape(3);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_chunk_bwd_col(enc, cq, bm, x, cl, dy, dkv, dB, dX, cc, ci,
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
TK_SSD_BWD_NO_AD(SsdChunkGstate)
TK_SSD_BWD_NO_AD(SsdChunkRscan)
TK_SSD_BWD_NO_AD(SsdChunkBwdRow)
TK_SSD_BWD_NO_AD(SsdChunkBwdCol)

std::vector<array> ssd_chunked_bwd(const array& C, const array& B, const array& X,
                                   const array& cumlog, const array& dY, StreamOrDevice s) {
  const int Bsz = C.shape(0), H = C.shape(1), N = C.shape(2), D = C.shape(3);
  const int Cn = N / 64;
  auto C_c = contiguous(C, false, s), B_c = contiguous(B, false, s), X_c = contiguous(X, false, s);
  auto cl_c = contiguous(cumlog, false, s), dY_c = contiguous(dY, false, s);
  // Forward chunk states Sex (recomputed exactly as in the forward: kv -> exclusive scan, bf16).
  // bwd_row consumes Sex directly — its inter term loads dY row-layout and Sex col-layout, so no
  // transposed copy of the state is ever materialized.
  auto kv = array({Bsz, H, Cn, D, D}, float32, std::make_shared<SsdChunkKV>(to_stream(s)),
                  {B_c, X_c, cl_c});
  auto sex = array({Bsz, H, Cn, D, D}, bfloat16, std::make_shared<SsdChunkScan>(to_stream(s)),
                   {kv, cl_c});
  // ONE reverse chain: gradient states G_c = sum_{t in c} exp(cl_t - cl[r_{c-1}]) C_t^T dY_t,
  // then the reverse decayed suffix dKV_c = G_{c+1} + lam_{c+1} dKV_{c+1} (bf16). bwd_col loads
  // dKV in both row and col register layouts, so the old second (transposed) chain is gone.
  auto g = array({Bsz, H, Cn, D, D}, float32, std::make_shared<SsdChunkGstate>(to_stream(s)),
                 {C_c, dY_c, cl_c});
  auto dkv = array({Bsz, H, Cn, D, D}, bfloat16, std::make_shared<SsdChunkRscan>(to_stream(s)),
                   {g, cl_c});
  // Row-owned: dC, r_intra (rowsum), r_inter = <dC_inter, C_i>. Col-owned: dB, dX, cc_intra
  // (colsum), ci_inter = <dX_inter, X_j>. dcl = (r + ri) - (cc + ci), all in-kernel.
  auto row = array::make_arrays(
      {C.shape(), {Bsz, H, N}, {Bsz, H, N}}, {bfloat16, float32, float32},
      std::make_shared<SsdChunkBwdRow>(to_stream(s)), {C_c, B_c, X_c, cl_c, dY_c, sex});
  auto col = array::make_arrays(
      {C.shape(), C.shape(), {Bsz, H, N}, {Bsz, H, N}}, {bfloat16, bfloat16, float32, float32},
      std::make_shared<SsdChunkBwdCol>(to_stream(s)), {C_c, B_c, X_c, cl_c, dY_c, dkv});
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
  // Chunked linear-time backward above the same measured crossovers as the forward (both head
  // dims, via 64x64 state quadrants); else the quadratic backward (in-kernel fp32 dcumlog).
  const int chunk_min = (D == 64) ? 2048 : 4096;
  if (!force_quadratic && N % 64 == 0 && N >= chunk_min) {
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

// ----------------------------- ssd_decode -----------------------------
// Single-token SSD decode step: S' = alpha*S + x⊗k ; y = S'·q (readout after the write) — the
// O(D^2) generation step for mamba2 / lin_attn_decay. MLX arrays are immutable, so the update is
// FUNCTIONAL: the kernel reads Sin and writes a fresh Sout (row-owned, alias-safe by design).
std::vector<array> ssd_decode(const array& S, const array& alpha, const array& x,
                              const array& k, const array& q, StreamOrDevice s) {
  assert(S.dtype() == float32 && alpha.dtype() == float32 && x.dtype() == float32 &&
         k.dtype() == float32 && q.dtype() == float32);
  const int B = S.shape(0), H = S.shape(1), D = S.shape(2);
  assert(S.shape(3) == D && "ssd_decode: S must be square (B,H,D,D)");
  assert((D == 64 || D == 128) && "ssd_decode supports D in {64, 128}");
  (void)B; (void)H;
  return array::make_arrays(
      {{B, H, D}, S.shape()}, {float32, float32},
      std::make_shared<SsdDecode>(to_stream(s)), {S, alpha, x, k, q});
}

void SsdDecode::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& S = inputs[0]; auto& alpha = inputs[1]; auto& x = inputs[2];
  auto& k = inputs[3]; auto& q = inputs[4];
  auto& y = outputs[0]; auto& Sout = outputs[1];
  auto& s = stream(); auto& d = metal::device(s.device);
  y.set_data(allocator::malloc_or_wait(y.nbytes()));
  Sout.set_data(allocator::malloc_or_wait(Sout.nbytes()));
  const int B = S.shape(0), H = S.shape(1), D = S.shape(2);
  auto& ce = d.get_command_encoder(s.index);
  MLXEncoder enc(d, ce);
  tk::launch_ssd_decode(enc, S, alpha, x, k, q, Sout, y, static_cast<unsigned>(H), B, D);
}
TK_MAMBA_BWD_NO_AUTODIFF(SsdDecode, "SsdDecode")
void SsdDecode::eval_cpu(const std::vector<array>&, std::vector<array>&) {
  throw std::runtime_error("SsdDecode has no CPU implementation.");
}

} // namespace mlx::core
