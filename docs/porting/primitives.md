# ThunderMittens MSL Substrate — Coverage & Gaps

The substrate (`ThunderMittens/include/`) is a ~90% complete Metal port of ThunderKittens'
primitive layer. Umbrella header: `include/tk.metal`. Most kernel ports compose existing
primitives and need **no new substrate code**.

## Present and validated

- **Types:** register tiles/vectors (`rt`, `rv`, `crt`, `crv`), shared tiles/vectors (`st`, `sv`,
  `cst`, `csv`), global layouts (`gl`, `cgl`). `TILE_DIM=8`, `SIMD_THREADS=32`.
- **MMA:** `simdgroup_matrix` wrappers `mma_AB / mma_ABt / mma_AtB / mma_AtBt` (+ `mm_*`), with
  full register-layout (row/col) handling. `include/ops/warp/register/tile/mma.metal`.
- **Memory:** global↔register, global↔shared, shared↔register load/store for tiles and vectors,
  with on-the-fly dtype conversion (bf16↔fp32↔fp16). Warp- and group-level.
- **Compute:** elementwise maps (`exp`, `exp2`, `log`, `abs`, `relu`, `sqrt`, `rsqrt`,
  `add/sub/mul/div/max/min`, `fma_*`) and row/col reductions (`row_max/row_sum/...`, vec
  `sum/max/min`) for register and shared tiles/vectors. `swap_layout` register transpose.
  Shared-memory swizzle.

## Recently implemented

- **`sqrt` / `rsqrt`** — added to `common/base_ops.metal` and the register & shared, vec & tile
  `maps.metal` wrappers (mirroring `exp`/`relu`). Validated on-device by the `rv_rsqrt` unit test
  (`tests/unit/warp/register/vec/maps.{metal,cpp}`, 18 cases across float/half/bf16 × align/ortho/
  naive). Useful for a future tile/vector RMS-style normalization; LayerNorm still uses scalar
  `metal::rsqrt` inline (its rsqrt argument is a reduced scalar, not a vector).
- **`tanh` / `gelu`** — added to `base_ops.metal` + the four map files. Metal has **no `tanh`
  intrinsic**, so `tanh` is computed from `exp`: `1 - 2/(exp(2x)+1)` (stable for both signs). `gelu`
  is the tanh approximation `0.5*x*(1+tanh(0.7978845608*(x + 0.044715*x^3)))`, matching
  `mx.nn.gelu_approx`. `tanh` validated on-device by `rv_tanh` (18 cases); `gelu` is exercised
  end-to-end by `kernels/gelu/`.
- **`make_causal`** (already in `conversions.metal`) is used by `kernels/attn_causal/` to mask the
  diagonal block (note: its `val` arg is `thread const&`, so pass a thread-local, not a `constant`).

## Gaps (and whether they matter)

| Gap | Impact | Plan |
|---|---|---|
| **Async copy / `cp.async` / TMA** | None — Metal has no direct equivalent | Intentionally skipped. Use sync `load`, or stage via shared + `threadgroup_barrier` when a kernel needs overlap. |
| **Complex MMA** | Only complex kernels (fftconv) | `crt`/`crv` types exist; add complex-multiply MMA wrappers when porting fftconv. Deferred until a complex kernel drives it. |
| **Subtile integration / some layout-conversion edges** | Low | Noted as TODO in `st.metal` and register `conversions.metal`; address per-kernel as needed. |
| **Shared allocator / non-default max shared mem** | Low | `utils.metal` TODO; relevant for large shared-tile kernels (GEMM staging). Deferred until a staging kernel drives it. |

> Note: the warp-level `global→shared` tile load/store is **implemented** (the active
> `meta::load`/`meta::store` path in `ops/warp/memory/tile/global_to_shared.metal`; the commented
> blocks there are superseded experiments) — a previous revision of this doc wrongly listed it as a gap.

## Primitive unit tests

A C++/Metal unit-test harness lives in `tests/unit/`, driven by `tests/unit/unit_tests.cpp` and
gated by `tests/unit/testing_commons/testing_flags.hpp` (`ENABLE_TESTS`). It is now enabled for the
focused leaf suites the LayerNorm kernel depends on — warp register-vector reductions, vec maps, and
naive `rv` global↔register memory (flip to `TEST_ALL` for the full sweep).

**Status: working — 126/126 primitive tests pass on-device** (incl. 18 `rv_rsqrt` + 18 `rv_tanh`). Build & run from the repo root:

```
xcodebuild -project ThunderMittens.xcodeproj -scheme ThunderMittens -configuration Debug build CODE_SIGNING_ALLOWED=NO
"$(find ~/Library/Developer/Xcode/DerivedData -path '*Build/Products/Debug/ThunderMittens' -type f | head -1)"
```

**Resolved Xcode blocker (was: `error: Multiple commands produce '…/<kernel>.cpp.o.h'`).** The
project uses an Xcode-16 synchronized root group over `ThunderMittens/`, auto-including every file.
Two distinct problems caused the failure, both now fixed:
1. **CMake build artifacts in the synced tree.** `kernels/build/.../<kernel>.cpp.o.d` dependency
   files have a `.d` extension, which Xcode classifies as DTrace scripts and compiles to
   `<kernel>.cpp.o.h`; CMake writes two copies per kernel (`_ext` + `mlx_ext` targets) → "Multiple
   commands produce". Fixed durably by relocating the build dir **out of the synced tree** via
   `kernels/setup.cfg` (`build_base = ../../build` → repo-root `/build`). Directory-level
   `membershipExceptions` do *not* cascade, so excluding the dir in the project does not work — the
   artifacts must not live under `ThunderMittens/` at all.
2. **Duplicate `main`.** `kernels/attn_fwd/correctness/c_attn.m` is a standalone Obj-C Metal harness
   with its own `main()`, which collided with `unit_tests.cpp`. Fixed by adding `c_attn.m` (and
   `layernorm.cpp`, for consistency with the other kernel `.cpp`) to the target's
   `membershipExceptions` in `ThunderMittens.xcodeproj/project.pbxproj`.

The MLX Python correctness tests in `kernels/*/correctness/` exercise these same `rv`
reduction/map/load/store paths end-to-end as well.
