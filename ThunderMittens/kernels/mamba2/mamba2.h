// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

/** Mamba-2 / SSD forward (materialized chunked form):
 *  Y_t = sum_{j<=t} (C_t . B_j) * exp(cumlog_t - cumlog_j) * X_j.
 *  C,B,X are (B,H,N,D) bf16; cumlog (B,H,N) fp32 = cumsum(log a); D=64, N a multiple of 8. */
array mamba2(const array& C, const array& B, const array& X, const array& cumlog,
             StreamOrDevice s = {});

/** Mamba-2 / SSD backward (quadratic). Given dY, returns [dC, dB, dX, dcumlog], all matching the
 *  forward shapes (dC/dB/dX (B,H,N,D) bf16; dcumlog (B,H,N) f32 = rowsum(M) - colsum(M)).
 *  D in {64,128}. To turn dcumlog into d(log a) / da, reverse-cumsum then divide by a (host). */
std::vector<array> mamba2_bwd(const array& C, const array& B, const array& X, const array& cumlog,
                              const array& dY, StreamOrDevice s = {});

class Mamba2BwdRow : public Primitive {
 public:
  explicit Mamba2BwdRow(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "Mamba2BwdRow"; }
  void print(std::ostream& os) override { os << "Mamba2BwdRow"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

class Mamba2BwdCol : public Primitive {
 public:
  explicit Mamba2BwdCol(Stream stream) : Primitive(stream) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "Mamba2BwdCol"; }
  void print(std::ostream& os) override { os << "Mamba2BwdCol"; }
  bool is_equivalent(const Primitive&) const override { return true; }
};

// Chunked linear-time SSD pipeline primitives (used automatically for N % 64 == 0,
// N >= 128; shared by mamba2 and lin_attn_decay): SsdChunkKV -> SsdChunkScan -> SsdChunkOut.
class SsdChunkKV : public Primitive {
 public:
  explicit SsdChunkKV(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SsdChunkKV"; }
  void print(std::ostream& os) override { os << "SsdChunkKV"; }
  bool is_equivalent(const Primitive& other) const override;
};

class SsdChunkScan : public Primitive {
 public:
  explicit SsdChunkScan(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SsdChunkScan"; }
  void print(std::ostream& os) override { os << "SsdChunkScan"; }
  bool is_equivalent(const Primitive& other) const override;
};

class SsdChunkOut : public Primitive {
 public:
  explicit SsdChunkOut(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "SsdChunkOut"; }
  void print(std::ostream& os) override { os << "SsdChunkOut"; }
  bool is_equivalent(const Primitive& other) const override;
};

/** Chunked SSD composition used by mamba2 and lin_attn_decay for N%64==0, N>=128. */
array ssd_chunked(const array& Cq, const array& Bm, const array& X, const array& cl,
                  StreamOrDevice s);

class Mamba2 : public Primitive {
 public:
  explicit Mamba2(Stream stream) : Primitive(stream) {};
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override;
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override;
  std::vector<array> jvp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&) override;
  std::vector<array> vjp(const std::vector<array>&, const std::vector<array>&,
                         const std::vector<int>&, const std::vector<array>&) override;
  std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>&, const std::vector<int>&) override;
  const char* name() const { return "Mamba2"; }

  void print(std::ostream& os) override { os << "Mamba2"; }
  bool is_equivalent(const Primitive& other) const override;
  void eval(const std::vector<array>&, std::vector<array>&);
};

} // namespace mlx::core
