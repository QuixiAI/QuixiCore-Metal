# Agent Instructions

This is the QuixiCore Metal backend. Kernel work must be correctness-first,
measurement-driven, and recorded in the performance notebook.

## Read First

- User-facing overview: `README.md`.
- Repository layout: `docs/repository-structure.md`.
- Performance operating guide: `perf/perf.md`.
- Optimization notebook: `perf/optimization_status.md`.
- Baseline index: `perf/baseline_status.md`.
- Kernel metadata: `.quixicore/kernels.yaml` and
  `.quixicore/quant-formats.yaml`.

## Performance Optimization Requirement

Before committing any kernel implementation, kernel routing change, benchmark
change, or performance claim, the agent must complete at least one focused
performance optimization run on an affected kernel.

A valid run includes:

- The kernel, integration path, dtype/format, and shape set.
- Correctness for the touched path.
- Baseline/current timing and candidate timing when testing a variant.
- Apple Silicon model, macOS version, Xcode/Metal toolchain, command line,
  warmups, iterations, median, and variance or min/max.
- A keep/reject decision in `perf/optimization_status.md`.

If an Apple Silicon/Metal runtime is unavailable, do not commit a kernel
optimization or speedup claim. Stop and report the blocker, or restrict the
commit to docs/scaffolding with no performance claim.

Pure documentation and metadata-only commits may skip the kernel perf run, but
they must not claim a performance improvement.

## How To Optimize

- Start from `perf/perf.md`; form a bottleneck hypothesis before editing.
- Change one meaningful factor at a time: tile shape, launch geometry, memory
  layout, fusion, barrier placement, dequant strategy, routing threshold, or
  specialization.
- Compare against framework and naive baselines where available.
- Keep only wins that pass correctness, improve realistic priority shapes, and
  do not regress supported edge shapes or tolerances.
- Store raw output under `perf/results/`; copy durable conclusions into
  `perf/optimization_status.md`. Do not commit large profiler traces.

## Useful Commands

```bash
scripts/build
scripts/test correctness -q
scripts/test parity -q
scripts/test mps -q
.venv/bin/python perf/bench_kernels.py --backend mlx --preset quick --kernel <kernel>
.venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel <kernel>
```

Use Xcode/Instruments or Metal command-buffer timing when benchmark results do
not explain a bottleneck.

## Engineering Hygiene

- Check `git status` before editing. Do not revert user changes.
- Keep backend-local optimizations behind the public QuixiCore contract.
- Update metadata, tests, docs, and bindings when changing public behavior.
- Do not import reference implementation code unless licensing and provenance
  have been reviewed.
- Keep commits scoped and descriptive.
