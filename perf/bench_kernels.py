#!/usr/bin/env python3
"""QuixiCore Metal kernel benchmark harness (schema v1).

Covers every active kernel family under kernels/ with, per case:
  - the tk target kernel,
  - a framework baseline (mx.* / mx.fast.* / torch.*) when one exists,
  - a naive decomposed baseline for fused/quant kernels (e.g. dequantize(wq) @ x),
  - a one-shot correctness check (max abs/rel error vs a float64 numpy reference),
  - derived throughput (GB/s, weight-only GB/s for packed weights, GFLOP/s).

Run from the repo root:

    .venv/bin/python perf/bench_kernels.py --backend mlx --preset smoke --kernel all
    .venv/bin/python perf/bench_kernels.py --backend mlx --preset comprehensive --kernel qgemv --formats q4_0,q8_0
    .venv/bin/python perf/bench_kernels.py --backend torch --preset quick --kernel attn,softmax

Each run writes:

    perf/results/YYYY-MM-DD/<run-id>/run.json       (environment + invocation metadata)
    perf/results/YYYY-MM-DD/<run-id>/results.jsonl  (schema v1, one row per case)
    perf/results/YYYY-MM-DD/<run-id>/summary.md     (human-readable table)

Cases self-skip (recorded with a reason, not fatal) when a kernel, format, or framework
is unavailable. perf/results/ is git-ignored; copy summaries into
perf/optimization_status.md.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BINDINGS_DIR = REPO_ROOT / "bindings" / "python"
PYTORCH_MPS_BINDINGS_DIR = REPO_ROOT / "bindings" / "pytorch_mps"
for _path in (PYTHON_BINDINGS_DIR, PYTORCH_MPS_BINDINGS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- backend
class Backend:
    """Thin adapter so cases are written once and run on MLX or PyTorch-MPS."""

    def __init__(self, name):
        self.name = name
        if name == "mlx":
            import mlx.core as mx
            self.mx = mx
            self._dtypes = {"f32": mx.float32, "f16": mx.float16, "bf16": mx.bfloat16}
        elif name == "torch":
            import torch
            self.torch = torch
            if not torch.backends.mps.is_available():
                raise RuntimeError("torch MPS not available")
            self._dtypes = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
        else:
            raise ValueError(name)

    def array(self, np_arr, dtype="f32"):
        if self.name == "mlx":
            return self.mx.array(np_arr).astype(self._dtypes[dtype])
        return self.torch.from_numpy(np.ascontiguousarray(np_arr)).to(self._dtypes[dtype]).to("mps")

    def int_array(self, np_arr):
        if self.name == "mlx":
            return self.mx.array(np_arr)
        return self.torch.from_numpy(np.ascontiguousarray(np_arr)).to("mps")

    def raw_array(self, np_arr):
        """uint8/int8 buffers passed through untouched (packed quant weights)."""
        return self.int_array(np_arr)

    def sync(self, val=None):
        if self.name == "mlx":
            if val is not None:
                self.mx.eval(val)
            else:
                self.mx.synchronize()
        else:
            self.torch.mps.synchronize()

    def to_numpy(self, val):
        if self.name == "mlx":
            return np.array(val.astype(self.mx.float32))
        return val.detach().to("cpu", self.torch.float32).numpy()

    def tk(self):
        import tk
        return tk


# --------------------------------------------------------------------------- timing
def time_thunk(fn, be, warmup, iters, min_sample_ms=2.0):
    """Median/p20/p80 per-call latency in ms.

    Small kernels are batched (several calls per sync) so per-call submit+sync latency
    (~0.2 ms) does not swamp the kernel time; the reported number is throughput-style
    per-call latency. Kernels above min_sample_ms run one call per sample.
    """
    # Warm by TIME, not call count: GPU clocks decay whenever the host does setup work
    # between cases, and a handful of sub-ms calls will not re-ramp them.
    t0 = time.perf_counter()
    calls = 0
    while calls < warmup or time.perf_counter() - t0 < 0.05:
        be.sync(fn())
        calls += 1
    t0 = time.perf_counter()
    be.sync(fn())
    est_ms = 1e3 * (time.perf_counter() - t0)
    batch = max(1, min(64, math.ceil(min_sample_ms / max(est_ms, 1e-3))))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        outs = [fn() for _ in range(batch)]
        be.sync(outs)
        samples.append(1e3 * (time.perf_counter() - t0) / batch)
    samples.sort()
    n = len(samples)
    med = statistics.median(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples)
    return {
        "ms": med,
        "p20_ms": samples[max(0, int(0.20 * n) - 1)] if n > 1 else med,
        "p80_ms": samples[min(n - 1, int(0.80 * n))] if n > 1 else med,
        "cv": (stdev / mean) if mean > 0 else 0.0,
        "batch": batch,
    }


# --------------------------------------------------------------------------- case model
@dataclass
class Case:
    kernel: str                     # family, e.g. "qgemv"
    variant: str                    # e.g. "q4_0" or "N4096_K4096"
    shape: dict                     # named dims
    dtype: str                      # I/O dtype
    fmt: str | None = None          # quant format when applicable
    target: object = None           # () -> device output (thunk)
    baselines: dict = field(default_factory=dict)   # name -> thunk
    ref: object = None              # () -> float64 numpy reference (or np array)
    out_to_numpy: object = None     # optional: convert target output -> numpy
    bytes_moved: float | None = None      # conservative total bytes (read+write)
    weight_bytes: float | None = None     # packed-weight bytes only (quant decode metric)
    flops: float | None = None
    notes: str = ""


def _rel_err(out, ref):
    """max|diff| / max|ref| — the repo's correctness-test convention."""
    return float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9))


def run_case(case, be, warmup, iters, check):
    row = {
        "schema": SCHEMA_VERSION,
        "kernel": case.kernel,
        "variant": case.variant,
        "shape": case.shape,
        "dtype": case.dtype,
        "format": case.fmt,
        "status": "ok",
        "notes": case.notes,
    }
    # one-shot correctness check against the numpy reference
    if check and case.ref is not None:
        out = case.target()
        be.sync(out)
        if case.out_to_numpy is not None:
            out_np = case.out_to_numpy(out)
        else:
            out_np = be.to_numpy(out)
        ref_np = case.ref() if callable(case.ref) else case.ref
        ref_np = np.asarray(ref_np, dtype=np.float64)
        out_np = np.asarray(out_np, dtype=np.float64)
        if out_np.shape != ref_np.shape:
            raise RuntimeError(f"shape mismatch out {out_np.shape} vs ref {ref_np.shape}")
        row["max_abs_err"] = float(np.max(np.abs(out_np - ref_np)))
        row["max_rel_err"] = _rel_err(out_np, ref_np)
    # timing: target then baselines
    t = time_thunk(case.target, be, warmup, iters)
    row["target_ms"] = t["ms"]
    row["target_p20_ms"] = t["p20_ms"]
    row["target_p80_ms"] = t["p80_ms"]
    row["target_cv"] = round(t["cv"], 4)
    row["batch"] = t["batch"]
    row["baselines"] = {}
    for name, thunk in case.baselines.items():
        try:
            b = time_thunk(thunk, be, warmup, iters)
            row["baselines"][name] = {
                "ms": b["ms"],
                "p20_ms": b["p20_ms"],
                "p80_ms": b["p80_ms"],
                "cv": round(b["cv"], 4),
                "batch": b["batch"],
                "speedup": (b["ms"] / t["ms"]) if t["ms"] > 0 else None,
            }
        except Exception as e:  # noqa: BLE001
            row["baselines"][name] = {"error": f"{type(e).__name__}: {e}"}
    # derived throughput
    sec = t["ms"] / 1e3
    if case.bytes_moved:
        row["gbps"] = case.bytes_moved / sec / 1e9
    if case.weight_bytes:
        row["weight_gbps"] = case.weight_bytes / sec / 1e9
    if case.flops:
        row["gflops"] = case.flops / sec / 1e9
    return row


# --------------------------------------------------------------------------- output
def _git_label():
    try:
        c = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _env_meta(backend_name):
    meta = {
        "git": _git_label(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": None,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        meta["device"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                        capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    if backend_name == "mlx":
        import mlx.core as mx
        meta["mlx"] = mx.__version__
    else:
        import torch
        meta["torch"] = torch.__version__
    return meta


def write_outputs(rows, meta, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (out_dir / "results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # summary table
    lines = ["# QuixiCore Metal kernel benchmarks", ""]
    lines.append(f"- `{meta['git']}` · {meta.get('device','?')} · backend `{meta['backend']}` · "
                 f"preset `{meta['preset']}` · warmup/iters {meta['warmup']}/{meta['iters']}")
    lines.append("")
    lines.append("| kernel | variant | shape | tk ms | best baseline | base ms | speedup | GB/s | W-GB/s | GFLOP/s | rel err |")
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['kernel']} | {r['variant']} | {_shape_str(r['shape'])} "
                         f"| _skip_ | {r.get('skip_reason','')} | | | | | | |")
            continue
        bl_name, bl = "", {}
        valid = {k: v for k, v in r.get("baselines", {}).items() if "ms" in v}
        if valid:
            bl_name = min(valid, key=lambda k: valid[k]["ms"])
            bl = valid[bl_name]
        lines.append(
            f"| {r['kernel']} | {r['variant']} | {_shape_str(r['shape'])} "
            f"| {r['target_ms']:.4f} | {bl_name} | {bl.get('ms', float('nan')):.4f} "
            f"| {bl.get('speedup', float('nan')):.2f} "
            f"| {r.get('gbps', float('nan')):.1f} | {r.get('weight_gbps', float('nan')):.1f} "
            f"| {r.get('gflops', float('nan')):.0f} | {r.get('max_rel_err', float('nan')):.2e} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _shape_str(shape):
    return "×".join(str(v) for v in shape.values()) if isinstance(shape, dict) else str(shape)


# --------------------------------------------------------------------------- registry
KERNEL_BUILDERS = {}   # name -> builder(be, preset, formats) -> yields Case


def register(name):
    def deco(fn):
        KERNEL_BUILDERS[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------- shared helpers
# Packed-quant block geometry: fmt -> (block_k, block_bytes). Weight bytes for (N,K) =
# N * (K/block_k) * block_bytes.
BLOCK_INFO = {
    "q8_0": (32, 34), "q4_0": (32, 18), "q4_K": (256, 144), "kU4B8": (128, 66), "kU4": (128, 68),
    "fp8_e4m3": (32, 34), "fp4_e2m1": (32, 18), "mxfp8": (32, 33), "nvfp4": (16, 9),
    "mxfp4": (32, 17), "bitnet": (32, 10), "iq4_nl": (32, 18), "iq4_xs": (256, 136),
    "iq2_xxs": (256, 66), "iq2_xs": (256, 74), "iq3_xxs": (256, 98), "iq1_s": (256, 50),
    "q4_1": (32, 20), "q5_0": (32, 22), "q5_1": (32, 24), "q2_K": (256, 84), "q3_K": (256, 110),
    "q5_K": (256, 176), "q6_K": (256, 210), "e5m2": (32, 34), "fp8_block": (128, 130),
    "mxfp6_e3m2": (32, 25), "mxfp6_e2m3": (32, 25), "hqq": (64, 36),
}

WCACHE = RESULTS_ROOT / ".wcache"   # packed-weight cache (results/ is git-ignored)


def _packed_weight(fmt, N, K, seed=0):
    """quantize() a (N,K) normal(0,0.3) weight, cached on disk (encoders can be slow)."""
    from tk.quant import QUANT_FORMATS
    WCACHE.mkdir(parents=True, exist_ok=True)
    key = WCACHE / f"{fmt}_{N}x{K}_s{seed}"
    wq_p, wdq_p = key.with_suffix(".wq.npy"), key.with_suffix(".wdq.npy")
    quantize, dequantize = QUANT_FORMATS[fmt]
    if wq_p.exists() and wdq_p.exists():
        return np.load(wq_p), np.load(wdq_p)
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
    wq = quantize(W)
    wdq = dequantize(wq).astype(np.float32)
    np.save(wq_p, wq)
    np.save(wdq_p, wdq)
    return wq, wdq


def _mx_nn():
    import mlx.nn as nn
    return nn


def _gelu_tanh_np(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _pick(preset, smoke, quick, comprehensive):
    return {"smoke": smoke, "quick": quick, "comprehensive": comprehensive}[preset]


# --------------------------------------------------------------------------- row/elementwise
@register("layernorm")
def layernorm_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(1)
    shapes = _pick(preset, [(4096, 1024)],
                   [(4096, 1024), (16384, 256)],
                   [(r, d) for r in (4096, 16384, 65536) for d in (256, 512, 768, 1024)])
    for N, D in shapes:
        x = rng.standard_normal((N, D)).astype(np.float32)
        w = rng.standard_normal(D).astype(np.float32)
        b = rng.standard_normal(D).astype(np.float32)
        x_d, w_d, b_d = be.array(x, "bf16"), be.array(w, "bf16"), be.array(b, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            baselines["mx.fast.layer_norm"] = lambda x_d=x_d, w_d=w_d, b_d=b_d: \
                mx.fast.layer_norm(x_d, w_d, b_d, 1e-5)
        else:
            F = be.torch.nn.functional
            baselines["F.layer_norm"] = lambda x_d=x_d, w_d=w_d, b_d=b_d: \
                F.layer_norm(x_d, (x_d.shape[-1],), w_d, b_d, 1e-5)
        xb = be.to_numpy(x_d).astype(np.float64)   # bf16-rounded input
        mu = xb.mean(-1, keepdims=True)
        ref = (xb - mu) / np.sqrt(xb.var(-1, keepdims=True) + 1e-5) \
            * be.to_numpy(w_d).astype(np.float64) + be.to_numpy(b_d).astype(np.float64)
        yield Case("layernorm", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, w_d=w_d, b_d=b_d: tk.layernorm(x_d, w_d, b_d),
                   baselines=baselines, ref=ref,
                   bytes_moved=2 * N * D * 2)


@register("rms_norm")
def rms_norm_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(2)
    shapes = _pick(preset, [(4096, 1024)],
                   [(4096, 1024), (16384, 256)],
                   [(r, d) for r in (4096, 16384, 65536) for d in (256, 512, 768, 1024)])
    for N, D in shapes:
        x = rng.standard_normal((N, D)).astype(np.float32)
        w = rng.standard_normal(D).astype(np.float32)
        x_d, w_d = be.array(x, "bf16"), be.array(w, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            baselines["mx.fast.rms_norm"] = lambda x_d=x_d, w_d=w_d: mx.fast.rms_norm(x_d, w_d, 1e-5)
        else:
            F = be.torch.nn.functional
            baselines["F.rms_norm"] = lambda x_d=x_d, w_d=w_d: \
                F.rms_norm(x_d, (x_d.shape[-1],), w_d, 1e-5)
        xb = be.to_numpy(x_d).astype(np.float64)
        ref = xb / np.sqrt((xb ** 2).mean(-1, keepdims=True) + 1e-5) \
            * be.to_numpy(w_d).astype(np.float64)
        yield Case("rms_norm", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, w_d=w_d: tk.rms_norm(x_d, w_d),
                   baselines=baselines, ref=ref, bytes_moved=2 * N * D * 2)


@register("mean_pool_rms_l2")
def mean_pool_rms_l2_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(7)
    shapes = _pick(preset, [(128, 768)],
                   [(128, 768), (512, 1024)],
                   [(m, d) for m in (128, 512, 2048) for d in (256, 512, 768, 1024)])
    for M, D in shapes:
        x = rng.standard_normal((M, D)).astype(np.float32)
        w = rng.standard_normal(D).astype(np.float32)
        x_d, w_d = be.array(x, "bf16"), be.array(w, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            baselines["mx compose"] = lambda x_d=x_d, w_d=w_d, mx=mx: (
                lambda n: n * mx.rsqrt((n * n).sum() + 1e-12))(
                (lambda p: p * mx.rsqrt((p * p).mean() + 1e-6) * w_d.astype(mx.float32))(
                    x_d.astype(mx.float32).mean(axis=0)))
        else:
            t = be.torch
            baselines["torch compose"] = lambda x_d=x_d, w_d=w_d, t=t: (
                lambda n: n / t.sqrt((n * n).sum() + 1e-12))(
                (lambda p: p * t.rsqrt((p * p).mean() + 1e-6) * w_d.float())(
                    x_d.float().mean(0)))
        wb = be.to_numpy(w_d).astype(np.float64)
        pb = be.to_numpy(x_d).astype(np.float64).mean(0)
        nb = pb / np.sqrt((pb * pb).mean() + 1e-6) * wb
        ref = nb / np.sqrt((nb * nb).sum() + 1e-12)
        yield Case("mean_pool_rms_l2", f"M{M}_D{D}", {"M": M, "D": D}, "bf16",
                   target=lambda x_d=x_d, w_d=w_d: tk.mean_pool_rms_l2(x_d, w_d),
                   baselines=baselines, ref=ref, bytes_moved=(M + 2) * D * 2)


@register("softmax")
def softmax_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(3)
    shapes = _pick(preset, [(4096, 1024)],
                   [(4096, 1024), (16384, 256)],
                   [(r, d) for r in (4096, 16384, 65536) for d in (256, 512, 768, 1024)])
    for N, D in shapes:
        x = rng.standard_normal((N, D)).astype(np.float32)
        x_d = be.array(x, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            baselines["mx.softmax"] = lambda x_d=x_d: mx.softmax(x_d, axis=-1)
        else:
            baselines["torch.softmax"] = lambda x_d=x_d: be.torch.softmax(x_d, dim=-1)
        xb = be.to_numpy(x_d).astype(np.float64)
        e = np.exp(xb - xb.max(-1, keepdims=True))
        ref = e / e.sum(-1, keepdims=True)
        yield Case("softmax", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d: tk.softmax(x_d),
                   baselines=baselines, ref=ref, bytes_moved=2 * N * D * 2)


@register("gelu")
def gelu_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(4)
    shapes = _pick(preset, [(4096, 1024)],
                   [(4096, 1024), (16384, 1024)],
                   [(r, d) for r in (4096, 16384, 65536) for d in (256, 1024)])
    for N, D in shapes:
        x = rng.standard_normal((N, D)).astype(np.float32)
        x_d = be.array(x, "bf16")
        baselines = {}
        if be.name == "mlx":
            nn = _mx_nn()
            baselines["mx.nn.gelu_approx"] = lambda x_d=x_d: nn.gelu_approx(x_d)
        else:
            F = be.torch.nn.functional
            baselines["F.gelu_tanh"] = lambda x_d=x_d: F.gelu(x_d, approximate="tanh")
        ref = _gelu_tanh_np(be.to_numpy(x_d).astype(np.float64))
        yield Case("gelu", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d: tk.gelu(x_d),
                   baselines=baselines, ref=ref, bytes_moved=2 * N * D * 2)


@register("add_rt")
def add_rt_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(5)
    shapes = _pick(preset, [(4096, 1024, "bf16")],
                   [(4096, 1024, "bf16"), (16384, 1024, "f32")],
                   [(4096, 1024, "bf16"), (16384, 1024, "bf16"), (16384, 1024, "f32"),
                    (65536, 1024, "bf16"), (4096, 4096, "f16")])
    for N, D, dt in shapes:
        x = rng.standard_normal((N, D)).astype(np.float32)
        y = rng.standard_normal((N, D)).astype(np.float32)
        x_d, y_d = be.array(x, dt), be.array(y, dt)
        add = (lambda x_d=x_d, y_d=y_d: x_d + y_d)
        ref = be.to_numpy(x_d).astype(np.float64) + be.to_numpy(y_d).astype(np.float64)
        esize = 4 if dt == "f32" else 2
        yield Case("add_rt", f"N{N}_D{D}_{dt}", {"N": N, "D": D}, dt,
                   target=lambda x_d=x_d, y_d=y_d: tk.add_rt(x_d, y_d),
                   baselines={"framework_add": add}, ref=ref,
                   bytes_moved=3 * N * D * esize)


@register("rotary")
def rotary_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(6)
    shapes = _pick(preset, [(1, 32, 1024, 128, False)],
                   [(1, 32, 2048, 128, False), (1, 32, 2048, 64, False),
                    (1, 32, 2048, 128, True)],
                   [(b, h, n, d, il) for (b, h) in ((1, 32), (8, 32)) for n in (512, 2048, 4096)
                    for d in (64, 128) for il in (False, True)])
    for B, H, N, D, interleaved in shapes:
        x = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        inv_freq = 10000.0 ** (-(np.arange(0, D, 2, dtype=np.float32) / D))
        ang = np.arange(N, dtype=np.float32)[:, None] * inv_freq[None, :]
        cos, sin = np.cos(ang), np.sin(ang)
        x_d = be.array(x, "bf16")
        cos_d, sin_d = be.array(cos, "bf16"), be.array(sin, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            freqs = mx.array(1.0 / inv_freq)
            baselines["mx.fast.rope"] = lambda x_d=x_d, freqs=freqs, il=interleaved: \
                mx.fast.rope(x_d, dims=D, traditional=il, base=None, scale=1.0,
                             offset=0, freqs=freqs)
        # ref: split-half / interleaved rotation in float64
        xb = be.to_numpy(x_d).astype(np.float64)
        cb = np.cos(ang).astype(np.float64)[None, None]
        sb = np.sin(ang).astype(np.float64)[None, None]
        ref = np.empty_like(xb)
        if interleaved:
            x1, x2 = xb[..., 0::2], xb[..., 1::2]
            ref[..., 0::2] = x1 * cb - x2 * sb
            ref[..., 1::2] = x1 * sb + x2 * cb
        else:
            h = D // 2
            x1, x2 = xb[..., :h], xb[..., h:]
            ref[..., :h] = x1 * cb - x2 * sb
            ref[..., h:] = x1 * sb + x2 * cb
        yield Case("rotary", f"B{B}H{H}N{N}D{D}{'_il' if interleaved else ''}",
                   {"B": B, "H": H, "N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, c=cos_d, s=sin_d, il=interleaved:
                       tk.rotary(x_d, c, s, interleaved=il),
                   baselines=baselines, ref=ref, bytes_moved=2 * B * H * N * D * 2)


@register("glu")
def glu_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(7)
    modes = _pick(preset, ["swiglu"], ["swiglu", "geglu"],
                  ["swiglu", "geglu", "reglu", "swiglu_oai", "geglu_erf", "geglu_quick"])
    shapes = _pick(preset, [(4096, 4096)], [(16384, 4096)], [(4096, 4096), (16384, 11008)])
    for mode in modes:
        for N, D in shapes:
            x = rng.standard_normal((N, D)).astype(np.float32)
            g = rng.standard_normal((N, D)).astype(np.float32)
            x_d, g_d = be.array(x, "bf16"), be.array(g, "bf16")
            baselines = {}
            if be.name == "mlx" and mode == "swiglu":
                mx = be.mx
                baselines["mx_composed_silu_mul"] = lambda x_d=x_d, g_d=g_d: \
                    (x_d * mx.sigmoid(x_d)) * g_d
            elif be.name == "torch" and mode == "swiglu":
                F = be.torch.nn.functional
                baselines["torch_silu_mul"] = lambda x_d=x_d, g_d=g_d: F.silu(x_d) * g_d
            yield Case("glu", f"{mode}_N{N}_D{D}", {"N": N, "D": D}, "bf16", fmt=mode,
                       target=lambda x_d=x_d, g_d=g_d, m=mode: tk.glu(x_d, g_d, mode=m),
                       baselines=baselines, ref=None, bytes_moved=3 * N * D * 2)


@register("hadamard")
def hadamard_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(8)
    shapes = _pick(preset, [(16384, 128)], [(16384, 128), (16384, 512)],
                   [(r, d) for r in (4096, 65536) for d in (64, 128, 256, 512)])
    for N, D in shapes:
        x = rng.standard_normal((N, D)).astype(np.float32)
        x_d = be.array(x, "f16")
        # Hadamard matrix baseline (matmul)
        H = np.array([[1.0]])
        while H.shape[0] < D:
            H = np.block([[H, H], [H, -H]])
        h_d = be.array(H / math.sqrt(D), "f16")
        if be.name == "mlx":
            mm = (lambda x_d=x_d, h_d=h_d: be.mx.matmul(x_d, h_d))
        else:
            mm = (lambda x_d=x_d, h_d=h_d: x_d @ h_d)
        ref = be.to_numpy(x_d).astype(np.float64) @ (H / math.sqrt(D))
        yield Case("hadamard", f"N{N}_D{D}", {"N": N, "D": D}, "f16",
                   target=lambda x_d=x_d: tk.hadamard(x_d),
                   baselines={"matmul_H": mm}, ref=ref, bytes_moved=2 * N * D * 2)


@register("add_norm")
def add_norm_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(9)
    shapes = _pick(preset, [(4096, 1024)], [(4096, 1024), (16384, 1024)],
                   [(4096, 1024), (16384, 1024), (65536, 1024)])
    for N, D in shapes:
        x = rng.standard_normal((N, D)).astype(np.float32)
        r = rng.standard_normal((N, D)).astype(np.float32)
        w = rng.standard_normal(D).astype(np.float32)
        x_d, r_d, w_d = be.array(x, "bf16"), be.array(r, "bf16"), be.array(w, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def composed(x_d=x_d, r_d=r_d, w_d=w_d):
                s = x_d + r_d
                return mx.fast.rms_norm(s, w_d, 1e-5), s
            baselines["mx_add_then_rms_norm"] = composed
        yield Case("add_norm", f"rms_add_N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, r_d=r_d, w_d=w_d: tk.rms_norm_add(x_d, r_d, w_d),
                   baselines=baselines, ref=None, bytes_moved=4 * N * D * 2)


# --------------------------------------------------------------------------- GEMM / fusion
@register("matmul")
def matmul_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(10)
    shapes = _pick(preset, [(1024, 1024, 1024, "bf16")],
                   [(1024, 1024, 1024, "bf16"), (2048, 2048, 2048, "bf16"),
                    (4096, 4096, 1024, "bf16")],
                   [(s, s, s, dt) for s in (256, 512, 1024, 2048) for dt in ("bf16", "f32")]
                   + [(11008, 4096, 512, "bf16"), (4096, 11008, 512, "bf16"),
                      (4096, 4096, 32, "bf16")])
    for N, K, M, dt in shapes:
        x = (0.1 * rng.standard_normal((N, K))).astype(np.float32)
        y = (0.1 * rng.standard_normal((K, M))).astype(np.float32)
        x_d, y_d = be.array(x, dt), be.array(y, dt)
        if be.name == "mlx":
            mm = (lambda x_d=x_d, y_d=y_d: be.mx.matmul(x_d, y_d))
        else:
            mm = (lambda x_d=x_d, y_d=y_d: x_d @ y_d)
        ref = be.to_numpy(x_d).astype(np.float64) @ be.to_numpy(y_d).astype(np.float64)
        yield Case("matmul", f"custom_{N}x{K}x{M}_{dt}", {"N": N, "K": K, "M": M}, dt,
                   target=lambda x_d=x_d, y_d=y_d: tk.matmul_custom(x_d, y_d),
                   baselines={"framework_matmul": mm}, ref=ref,
                   flops=2.0 * N * K * M)
        yield Case("matmul", f"staged_{N}x{K}x{M}_{dt}", {"N": N, "K": K, "M": M}, dt,
                   target=lambda x_d=x_d, y_d=y_d: tk.gemm_staged(x_d, y_d),
                   baselines={"framework_matmul": mm,
                              "tk.matmul_custom": (lambda x_d=x_d, y_d=y_d:
                                                   tk.matmul_custom(x_d, y_d))},
                   ref=ref, flops=2.0 * N * K * M)


@register("flux")
def flux_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(11)
    shapes = _pick(preset, [(1024, 1024, 1024)],
                   [(1024, 1024, 1024), (2048, 2048, 2048)],
                   [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 1024)])
    for N, K, M in shapes:
        x = (0.1 * rng.standard_normal((N, K))).astype(np.float32)
        w = (0.1 * rng.standard_normal((K, M))).astype(np.float32)
        bias = rng.standard_normal(M).astype(np.float32)
        gate = rng.standard_normal(M).astype(np.float32)
        resid = rng.standard_normal((N, M)).astype(np.float32)
        x_d, w_d = be.array(x, "bf16"), be.array(w, "bf16")
        b_d, g_d, r_d = be.array(bias, "bf16"), be.array(gate, "bf16"), be.array(resid, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx, nn = be.mx, _mx_nn()
            baselines["mx_matmul_then_gelu"] = lambda x_d=x_d, w_d=w_d, b_d=b_d: \
                nn.gelu_approx(mx.matmul(x_d, w_d) + b_d)
        else:
            F = be.torch.nn.functional
            baselines["torch_matmul_then_gelu"] = lambda x_d=x_d, w_d=w_d, b_d=b_d: \
                F.gelu(x_d @ w_d + b_d, approximate="tanh")
        yield Case("flux", f"gelu_{N}x{K}x{M}", {"N": N, "K": K, "M": M}, "bf16",
                   target=lambda x_d=x_d, w_d=w_d, b_d=b_d: tk.flux_gelu(x_d, w_d, b_d),
                   baselines=baselines, ref=None, flops=2.0 * N * K * M)
        baselines2 = {}
        if be.name == "mlx":
            mx = be.mx
            baselines2["mx_matmul_then_gate"] = \
                lambda x_d=x_d, w_d=w_d, b_d=b_d, g_d=g_d, r_d=r_d: \
                (mx.matmul(x_d, w_d) + b_d) * g_d + r_d
        yield Case("flux", f"gate_{N}x{K}x{M}", {"N": N, "K": K, "M": M}, "bf16",
                   target=lambda x_d=x_d, w_d=w_d, b_d=b_d, g_d=g_d, r_d=r_d:
                       tk.flux_gate(x_d, w_d, b_d, g_d, r_d),
                   baselines=baselines2, ref=None, flops=2.0 * N * K * M)


# --------------------------------------------------------------------------- attention
def _sdpa_baseline(be, q_d, k_d, v_d, D, causal):
    if be.name == "mlx":
        mx = be.mx
        scale = 1.0 / math.sqrt(D)
        if causal:
            N = q_d.shape[2]
            rows = mx.arange(N)[:, None]
            cols = mx.arange(N)[None, :]
            mask = mx.where(cols > rows, float("-inf"), 0.0).astype(mx.float32)
            return lambda: mx.fast.scaled_dot_product_attention(q_d, k_d, v_d, scale=scale,
                                                                mask=mask)
        return lambda: mx.fast.scaled_dot_product_attention(q_d, k_d, v_d, scale=scale, mask=None)
    F = be.torch.nn.functional
    return lambda: F.scaled_dot_product_attention(q_d, k_d, v_d, is_causal=causal)


@register("attn")
def attn_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(12)
    shapes = _pick(preset, [(1, 8, 1024, 64)],
                   [(1, 8, 1024, 64), (1, 8, 1024, 128), (1, 8, 2048, 128)],
                   [(2, 16, n, d) for n in (512, 1024, 2048, 4096) for d in (64, 128)])
    for B, H, N, D in shapes:
        q = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        k = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        v = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        q_d, k_d, v_d = be.array(q, "bf16"), be.array(k, "bf16"), be.array(v, "bf16")
        flops = 4.0 * B * H * N * N * D
        for variant, fn, causal, softcap in (
                ("fwd", tk.attn_fwd, False, 0.0),
                ("causal", tk.attn_causal, True, 0.0),
                ("multiwarp", tk.attn_multiwarp, False, 0.0),
                ("fwd_softcap", tk.attn_fwd, False, 30.0),
                ("causal_softcap", tk.attn_causal, True, 30.0)):
            baselines = ({"sdpa": _sdpa_baseline(be, q_d, k_d, v_d, D, causal)}
                         if softcap == 0.0 else {})
            if variant == "multiwarp":
                baselines["tk.attn_fwd"] = lambda q_d=q_d, k_d=k_d, v_d=v_d: \
                    tk.attn_fwd(q_d, k_d, v_d)
            target = ((lambda fn=fn, q_d=q_d, k_d=k_d, v_d=v_d:
                       fn(q_d, k_d, v_d)) if variant == "multiwarp" else
                      (lambda fn=fn, q_d=q_d, k_d=k_d, v_d=v_d, softcap=softcap:
                       fn(q_d, k_d, v_d, softcap=softcap)))
            yield Case("attn", f"{variant}_B{B}H{H}N{N}D{D}",
                       {"B": B, "H": H, "N": N, "D": D}, "bf16",
                       target=target,
                       baselines=baselines, ref=None,
                       flops=flops / (2 if causal else 1))


@register("beam")
def beam_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(151)
    # (B, beam_width, V)
    shapes = _pick(preset, [(4, 4, 32000)],
                   [(4, 4, 32000), (8, 8, 32000)],
                   [(4, 4, 128256), (8, 8, 128256), (16, 4, 128256)])
    for B, BM, V in shapes:
        logits = (rng.standard_normal((B * BM, V))).astype(np.float32)
        cum = rng.standard_normal((B, BM)).astype(np.float32)
        lg = be.array(logits, "bf16")
        cm = be.array(cum, "f32")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def framework_beam(lg=lg, cm=cm, B=B, BM=BM, V=V):
                # fair equivalent: also unravels to (token, parent) and gathers the scores
                logp = lg.astype(mx.float32) - mx.logsumexp(lg.astype(mx.float32), axis=1,
                                                            keepdims=True)
                sc = (logp.reshape(B, BM, V) + cm.reshape(B, BM, 1)).reshape(B, BM * V)
                idx = mx.argpartition(-sc, BM, axis=1)[:, :BM]
                return idx % V, idx // V, mx.take_along_axis(sc, idx, axis=1)
            baselines["framework_logsoftmax_topk"] = framework_beam
        yield Case("beam", f"advance_B{B}_BM{BM}_V{V}", {"B": B, "BM": BM, "V": V}, "bf16",
                   target=lambda lg=lg, cm=cm, BM=BM: tk.beam_advance(lg, cm, BM),
                   baselines=baselines, ref=None, bytes_moved=float(B * BM * V * 2))


@register("cross_entropy")
def cross_entropy_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(141)
    shapes = _pick(preset, [(1024, 32000)],
                   [(1024, 32000), (2048, 128256)],
                   [(1024, 32000), (2048, 128256), (4096, 128256)])
    for T, V in shapes:
        logits = (rng.standard_normal((T, V))).astype(np.float32)
        tgt = rng.integers(0, V, size=(T,)).astype(np.int32)
        lg = be.array(logits, "bf16")
        tg = be.int_array(tgt)
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def framework_ce(lg=lg, tg=tg):
                lse = mx.logsumexp(lg.astype(mx.float32), axis=1)
                xy = mx.take_along_axis(lg.astype(mx.float32), tg[:, None].astype(mx.int32), axis=1)[:, 0]
                return (lse - xy).mean()
            baselines["framework_lse_gather"] = framework_ce
        yield Case("cross_entropy", f"fwd_T{T}_V{V}", {"T": T, "V": V}, "bf16",
                   target=lambda lg=lg, tg=tg: tk.cross_entropy(lg, tg, reduction="mean"),
                   baselines=baselines, ref=None, bytes_moved=float(T * V * 2))


@register("lm_head")
def lm_head_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(131)
    # (T, V, K). Decode regime: small T, large vocab.
    shapes = _pick(preset, [(1, 32000, 2048)],
                   [(1, 32000, 4096), (8, 32000, 4096)],
                   [(1, 32000, 4096), (8, 32000, 4096), (1, 128256, 4096)])
    for T, V, K in shapes:
        h = (0.5 * rng.standard_normal((T, K))).astype(np.float32)
        W = (0.5 * rng.standard_normal((V, K))).astype(np.float32)
        h_d, W_d = be.array(h, "bf16"), be.array(W, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def matmul_argmax(h_d=h_d, W_d=W_d):
                return tk.argmax_sample(mx.matmul(h_d, W_d.T))
            baselines["matmul+argmax"] = matmul_argmax
        # target = the no-materialization fused kernel; baseline = matmul+sampler (= the DEFAULT
        # tk.lm_head_sample path). Fused ties/wins only at very large V; matmul wins the common case.
        yield Case("lm_head", f"argmax_T{T}_V{V}_K{K}", {"T": T, "V": V, "K": K}, "bf16",
                   target=lambda h_d=h_d, W_d=W_d:
                       tk.lm_head_sample(h_d, W_d, mode="argmax", fused=True),
                   baselines=baselines, ref=None, bytes_moved=float(V * K * 2))


@register("attn_varlen")
def attn_varlen_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(122)
    bs = 16
    # (cu_seqlens, ctxs, H, H_KV, D) — ragged packed queries reading K/V from the paged cache.
    cfgs = _pick(
        preset,
        [([0, 128, 384], [128, 256], 8, 8, 64)],
        [([0, 128, 384], [128, 256], 8, 8, 128),
         ([0, 512, 1024, 1536], [512, 512, 512], 16, 4, 128)],
        [([0, 512, 1024, 1536], [512, 512, 512], 16, 4, 128),
         ([0, 1024, 3072], [1024, 2048], 32, 8, 128)])
    for cu, ctxs, H, H_KV, D in cfgs:
        total_q = cu[-1]
        scale = 1.0 / np.sqrt(D)
        q = (0.3 * rng.standard_normal((total_q, H, D))).astype(np.float32)
        nb = sum((c + bs - 1) // bs for c in ctxs) + 2
        mbk = max((c + bs - 1) // bs for c in ctxs)
        kc = (0.3 * rng.standard_normal((nb, bs, H_KV, D))).astype(np.float32)
        vc = (0.3 * rng.standard_normal((nb, bs, H_KV, D))).astype(np.float32)
        bt = np.full((len(ctxs), mbk), -1, np.int32)
        blk = 1
        for b in range(len(ctxs)):
            for c in range((ctxs[b] + bs - 1) // bs):
                bt[b, c] = blk
                blk += 1
        q_d, kc_d, vc_d = be.array(q, "bf16"), be.array(kc, "bf16"), be.array(vc, "bf16")
        bt_d, cl_d = be.int_array(bt), be.int_array(np.array(ctxs, np.int32))
        # flops ~ 2 * sum_b (qkv passes) = 4 * H * D * sum_b qlen_b * ctx_b (causal ~ /2 of dense ctx)
        flops = 0.0
        for b in range(len(ctxs)):
            qlen = cu[b + 1] - cu[b]
            flops += 4.0 * H * D * qlen * ctxs[b]
        yield Case("attn_varlen",
                   f"prefill_B{len(ctxs)}_tq{total_q}_H{H}kv{H_KV}_D{D}",
                   {"B": len(ctxs), "total_q": total_q, "H": H, "H_KV": H_KV, "D": D}, "bf16",
                   target=lambda q_d=q_d, kc_d=kc_d, vc_d=vc_d, bt_d=bt_d, cl_d=cl_d,
                   cu=cu, scale=scale:
                       tk.attn_varlen_prefill(q_d, kc_d, vc_d, bt_d, cl_d, cu, scale=float(scale)),
                   baselines={}, ref=None, flops=flops)


@register("attn_bwd")
def attn_bwd_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(13)
    shapes = _pick(preset, [(1, 8, 1024, 64, False)],
                   [(1, 8, 1024, 64, False), (1, 8, 1024, 128, True)],
                   [(1, 8, n, d, c) for n in (512, 1024, 2048) for d in (64, 128)
                    for c in (False, True)])
    for B, H, N, D, causal in shapes:
        q = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        k = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        v = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        do = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        q_d, k_d, v_d = be.array(q, "bf16"), be.array(k, "bf16"), be.array(v, "bf16")
        do_d = be.array(do, "bf16")
        o_d, L_d = tk.attn_fwd_l(q_d, k_d, v_d, causal=causal)
        be.sync((o_d, L_d))
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            scale = 1.0 / math.sqrt(D)
            neg = mx.where(mx.arange(N)[None, :] > mx.arange(N)[:, None],
                           float("-inf"), 0.0).astype(mx.float32) if causal else None

            def attn_ref(qq, kk, vv, neg=neg, scale=scale):
                s = (qq.astype(mx.float32) @ kk.swapaxes(-1, -2).astype(mx.float32)) * scale
                if neg is not None:
                    s = s + neg
                p = mx.softmax(s, axis=-1)
                return (p @ vv.astype(mx.float32)).astype(mx.bfloat16)
            baselines["mx_vjp_naive"] = \
                lambda fn=attn_ref, q_d=q_d, k_d=k_d, v_d=v_d, do_d=do_d: \
                mx.vjp(fn, [q_d, k_d, v_d], [do_d])[1]
        yield Case("attn_bwd", f"{'causal' if causal else 'fwd'}_B{B}H{H}N{N}D{D}",
                   {"B": B, "H": H, "N": N, "D": D}, "bf16",
                   target=lambda q_d=q_d, k_d=k_d, v_d=v_d, o_d=o_d, do_d=do_d, L_d=L_d,
                                 c=causal: tk.attn_bwd(q_d, k_d, v_d, o_d, do_d, L_d, causal=c),
                   baselines=baselines, ref=None,
                   flops=10.0 * B * H * N * N * D / (2 if causal else 1))


@register("attn_q")
def attn_q_cases(be, preset, formats):
    tk = be.tk()
    from tk.quant import quantize_kv, dequantize_kv
    rng = np.random.default_rng(14)
    fmts = formats or _pick(preset, ["q8_0"], ["q8_0", "q4_0", "fp8_e4m3", "mxfp8"],
                            ["q8_0", "q4_0", "fp8_e4m3", "mxfp8"])
    shapes = _pick(preset, [(1, 8, 1024, 128)],
                   [(1, 8, 1024, 128)],
                   [(1, 8, 1024, 64), (1, 8, 2048, 128), (2, 8, 2048, 128)])
    for B, H, N, D in shapes:
        q = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        k = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        v = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        q_d = be.array(q, "bf16")
        for fmt in fmts:
            kq = quantize_kv(k, fmt)
            vq = quantize_kv(v, fmt)
            dk, dv = dequantize_kv(kq, fmt), dequantize_kv(vq, fmt)
            kq_d, vq_d = be.raw_array(kq), be.raw_array(vq)
            dk_d, dv_d = be.array(dk, "bf16"), be.array(dv, "bf16")
            baselines = {"sdpa_on_dequant": _sdpa_baseline(be, q_d, dk_d, dv_d, D, False),
                         "tk.attn_fwd_on_dequant": (lambda q_d=q_d, dk_d=dk_d, dv_d=dv_d:
                                                    tk.attn_fwd(q_d, dk_d, dv_d))}
            bk, bb = BLOCK_INFO[fmt]
            kv_bytes = 2 * B * H * N * (D // bk) * bb
            yield Case("attn_q", f"{fmt}_B{B}H{H}N{N}D{D}",
                       {"B": B, "H": H, "N": N, "D": D}, "bf16", fmt=fmt,
                       target=lambda q_d=q_d, kq_d=kq_d, vq_d=vq_d, f=fmt:
                           tk.attn_q(q_d, kq_d, vq_d, format=f),
                       baselines=baselines, ref=None,
                       flops=4.0 * B * H * N * N * D, weight_bytes=kv_bytes)
            if fmt in ("q8_0", "fp8_e4m3", "mxfp8") and N % 32 == 0:
                yield Case("attn_q", f"{fmt}_mw_B{B}H{H}N{N}D{D}",
                           {"B": B, "H": H, "N": N, "D": D}, "bf16", fmt=fmt,
                           target=lambda q_d=q_d, kq_d=kq_d, vq_d=vq_d, f=fmt:
                               tk.attn_q(q_d, kq_d, vq_d, format=f, multiwarp=True),
                           baselines={"tk.attn_q_singlewarp":
                                      (lambda q_d=q_d, kq_d=kq_d, vq_d=vq_d, f=fmt:
                                       tk.attn_q(q_d, kq_d, vq_d, format=f))},
                           ref=None, flops=4.0 * B * H * N * N * D, weight_bytes=kv_bytes)


# --------------------------------------------------------------------------- linear attention
@register("linear_attn")
def linear_attn_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(15)
    shapes = _pick(preset, [(1, 8, 1024, 64)],
                   [(1, 8, 2048, 64), (2, 8, 4096, 64)],
                   [(1, 8, 512, 64), (1, 8, 2048, 64), (2, 8, 4096, 64), (2, 16, 8192, 64)])
    for B, H, N, D in shapes:
        q = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        k = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        v = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        q_d, k_d, v_d = be.array(q, "bf16"), be.array(k, "bf16"), be.array(v, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            baselines["mx_composed"] = lambda q_d=q_d, k_d=k_d, v_d=v_d: \
                mx.matmul(q_d, mx.matmul(k_d.swapaxes(-1, -2), v_d))
        yield Case("linear_attn", f"B{B}H{H}N{N}D{D}", {"B": B, "H": H, "N": N, "D": D}, "bf16",
                   target=lambda q_d=q_d, k_d=k_d, v_d=v_d: tk.linear_attn(q_d, k_d, v_d),
                   baselines=baselines, ref=None,
                   flops=4.0 * B * H * N * D * D)
        # causal variant (chunked scan) — naive baseline is O(N^2), only for small N*B*H
        baselines_c = {}
        if be.name == "mlx" and B * H * N * N <= 2 ** 28:
            mx = be.mx
            maskc = (np.arange(N)[None, :] <= np.arange(N)[:, None])
            mask_d = mx.array(maskc.astype(np.float32))
            baselines_c["mx_masked_naive"] = lambda q_d=q_d, k_d=k_d, v_d=v_d, m=mask_d: \
                mx.matmul(mx.matmul(q_d.astype(mx.float32),
                                    k_d.swapaxes(-1, -2).astype(mx.float32)) * m,
                          v_d.astype(mx.float32))
        yield Case("linear_attn", f"causal_B{B}H{H}N{N}D{D}",
                   {"B": B, "H": H, "N": N, "D": D}, "bf16",
                   target=lambda q_d=q_d, k_d=k_d, v_d=v_d: tk.lin_attn_causal(q_d, k_d, v_d),
                   baselines=baselines_c, ref=None, flops=4.0 * B * H * N * D * D)


@register("lin_attn_decay")
def lin_attn_decay_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(16)
    shapes = _pick(preset, [(1, 8, 1024, 64)], [(1, 8, 2048, 64)],
                   [(1, 8, 1024, 64), (1, 8, 4096, 64), (2, 16, 4096, 64)])
    for B, H, N, D in shapes:
        q = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        k = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        v = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        slopes = np.linspace(0.05, 0.5, H).astype(np.float32)
        q_d, k_d, v_d = be.array(q, "bf16"), be.array(k, "bf16"), be.array(v, "bf16")
        yield Case("lin_attn_decay", f"B{B}H{H}N{N}D{D}",
                   {"B": B, "H": H, "N": N, "D": D}, "bf16",
                   target=lambda q_d=q_d, k_d=k_d, v_d=v_d, s=slopes:
                       tk.lin_attn_decay(q_d, k_d, v_d, s),
                   baselines={}, ref=None, flops=4.0 * B * H * N * D * D,
                   notes="public API rebuilds the decay ramp in numpy per call")


@register("hedgehog")
def hedgehog_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(17)
    shapes = _pick(preset, [(1, 8, 1024, 64)], [(1, 8, 2048, 64)],
                   [(1, 8, 2048, 64), (2, 16, 4096, 64)])
    for B, H, N, D in shapes:
        q = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        k = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        v = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        q_d, k_d, v_d = be.array(q, "bf16"), be.array(k, "bf16"), be.array(v, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def composed(q_d=q_d, k_d=k_d, v_d=v_d):
                fq = mx.exp(q_d.astype(mx.float32) - q_d.astype(mx.float32).max(-1, keepdims=True))
                fk = mx.exp(k_d.astype(mx.float32) - k_d.astype(mx.float32).max(-1, keepdims=True))
                return mx.matmul(fq, mx.matmul(fk.swapaxes(-1, -2), v_d.astype(mx.float32)))
            baselines["mx_composed"] = composed
        yield Case("hedgehog", f"B{B}H{H}N{N}D{D}", {"B": B, "H": H, "N": N, "D": D}, "bf16",
                   target=lambda q_d=q_d, k_d=k_d, v_d=v_d: tk.hedgehog(q_d, k_d, v_d),
                   baselines=baselines, ref=None, flops=4.0 * B * H * N * D * D)


@register("based")
def based_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(18)
    shapes = _pick(preset, [(1, 8, 1024)], [(1, 8, 2048)], [(1, 8, 2048), (2, 16, 4096)])
    for B, H, N in shapes:
        q = (0.5 * rng.standard_normal((B, H, N, 16))).astype(np.float32)
        k = (0.5 * rng.standard_normal((B, H, N, 16))).astype(np.float32)
        v = (0.5 * rng.standard_normal((B, H, N, 64))).astype(np.float32)
        q_d, k_d, v_d = be.array(q, "bf16"), be.array(k, "bf16"), be.array(v, "bf16")
        yield Case("based", f"B{B}H{H}N{N}", {"B": B, "H": H, "N": N, "DQK": 16, "DVO": 64},
                   "bf16",
                   target=lambda q_d=q_d, k_d=k_d, v_d=v_d: tk.based(q_d, k_d, v_d),
                   baselines={}, ref=None, flops=4.0 * B * H * N * N * 40)


@register("mamba2")
def mamba2_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(19)
    shapes = _pick(preset, [(1, 8, 1024, 64)], [(1, 8, 2048, 64)],
                   [(1, 8, 2048, 64), (2, 16, 4096, 64)])
    for B, H, N, D in shapes:
        C = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        Bm = (0.5 * rng.standard_normal((B, H, N, D))).astype(np.float32)
        X = rng.standard_normal((B, H, N, D)).astype(np.float32)
        a = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, N)))) * 0.5 + 0.5
        cumlog = np.cumsum(np.log(a), axis=-1).astype(np.float32)
        C_d, B_d, X_d = be.array(C, "bf16"), be.array(Bm, "bf16"), be.array(X, "bf16")
        cl_d = be.array(cumlog, "f32")
        yield Case("mamba2", f"B{B}H{H}N{N}D{D}", {"B": B, "H": H, "N": N, "D": D}, "bf16",
                   target=lambda C_d=C_d, B_d=B_d, X_d=X_d, cl_d=cl_d:
                       tk.mamba2(C_d, B_d, X_d, cl_d),
                   baselines={}, ref=None, flops=4.0 * B * H * N * D * D)
        dY_d = be.array(rng.standard_normal((B, H, N, D)).astype(np.float32), "bf16")
        yield Case("mamba2", f"bwd_B{B}H{H}N{N}D{D}", {"B": B, "H": H, "N": N, "D": D}, "bf16",
                   target=lambda C_d=C_d, B_d=B_d, X_d=X_d, cl_d=cl_d, dY_d=dY_d:
                       tk.mamba2_bwd(C_d, B_d, X_d, cl_d, dY_d),
                   baselines={}, ref=None, flops=12.0 * B * H * N * N * D)


# --------------------------------------------------------------------------- quantized
QGEMV_FMTS = {
    "smoke": ["q8_0", "q4_0"],
    "quick": ["q8_0", "q4_0", "q4_K", "q6_K", "iq4_nl", "fp8_e4m3", "mxfp4", "nvfp4",
              "bitnet", "hqq"],
    "comprehensive": list(BLOCK_INFO),
}


@register("qgemv")
def qgemv_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(20)
    fmts = formats or QGEMV_FMTS[preset]
    # NOTE: the giant lm-head shape (32000, 4096) is deliberately NOT in comprehensive — over the
    # ~10 comprehensive formats its 524 MB fp32 oracle + 262 MB fp16 copy per format OOMs the sweep.
    # Bench it on its own with `--kernel qgemv --formats q4_0,q8_0` if you want lm-head-vocab numbers.
    shapes = _pick(preset, [(4096, 4096)],
                   [(4096, 4096), (11008, 4096)],
                   [(4096, 4096), (11008, 4096), (4096, 11008),
                    (3840, 2560), (13824, 2560), (2560, 6912)])
    for N, K in shapes:
        x = rng.standard_normal((K, 1)).astype(np.float32)
        x_d = be.array(x, "f16")
        for fmt in fmts:
            bk, bb = BLOCK_INFO[fmt]
            if K % bk:
                continue
            wq, wdq = _packed_weight(fmt, N, K)
            wq_d = be.raw_array(wq)
            w_half = be.array(wdq, "f16")
            if be.name == "mlx":
                mm = (lambda w=w_half, x_d=x_d: be.mx.matmul(w, x_d))
            else:
                mm = (lambda w=w_half, x_d=x_d: w @ x_d)
            baselines = {"fp16_matmul": mm}
            if be.name == "mlx" and fmt in ("q4_0", "q8_0"):
                mx = be.mx
                bits, gs = (4, 32) if fmt == "q4_0" else (8, 64)
                mw, msc, mb = mx.quantize(mx.array(wdq).astype(mx.float16),
                                          group_size=gs, bits=bits)
                x_row = mx.array(x.T).astype(mx.float16)
                mx.eval(mw, msc, mb, x_row)
                baselines[f"mlx_q{bits}_gs{gs}"] = \
                    lambda x_row=x_row, mw=mw, msc=msc, mb=mb, gs=gs, bits=bits: \
                    mx.quantized_matmul(x_row, mw, msc, mb, transpose=True,
                                        group_size=gs, bits=bits)
            ref = (wdq @ be.to_numpy(x_d)).astype(np.float16)  # oracle fp16 (halves host residency)
            del wdq                         # free the 524 MB fp32 dequant before the next format
            yield Case("qgemv", f"{fmt}_N{N}_K{K}", {"N": N, "K": K, "M": 1}, "f16", fmt=fmt,
                       target=lambda wq_d=wq_d, x_d=x_d, f=fmt: tk.qgemv(wq_d, x_d, format=f),
                       baselines=baselines, ref=ref,
                       weight_bytes=N * (K // bk) * bb, flops=2.0 * N * K)


@register("qgemv_int")
def qgemv_int_cases(be, preset, formats):
    tk = be.tk()
    from tk.quant import quantize_w8a8, quantize_act_int8, quantize_bitnet
    rng = np.random.default_rng(21)
    shapes = _pick(preset, [(4096, 4096)], [(4096, 4096), (11008, 4096)],
                   [(4096, 4096), (11008, 4096), (32000, 4096), (3840, 2560), (13824, 2560)])
    for N, K in shapes:
        W = (rng.standard_normal((N, K)) * 0.3).astype(np.float32)
        X = rng.standard_normal((K, 1)).astype(np.float32)
        Wq, w_scale = quantize_w8a8(W)
        _, Xq, xs = quantize_act_int8(X)
        a_scale = float(xs[0, 0])
        wq_d, xq_d = be.raw_array(Wq), be.raw_array(Xq)
        ws_d = be.array(w_scale, "f16")
        as_d = be.array(np.array([a_scale], np.float32), "f16")
        x_d = be.array(X, "f16")
        from tk.quant import QUANT_FORMATS
        q8_quant, _ = QUANT_FORMATS["q8_0"]
        wq8 = q8_quant(W)
        wq8_d = be.raw_array(wq8)
        yield Case("qgemv_int", f"w8a8_N{N}_K{K}", {"N": N, "K": K, "M": 1}, "int8",
                   fmt="w8a8",
                   target=lambda wq_d=wq_d, xq_d=xq_d, ws_d=ws_d, as_d=as_d:
                       tk.qgemv_w8a8(wq_d, xq_d, ws_d, as_d),
                   baselines={"tk.qgemv_q8_0": (lambda wq8_d=wq8_d, x_d=x_d:
                                                tk.qgemv(wq8_d, x_d, format="q8_0"))},
                   ref=None, weight_bytes=float(N * K), flops=2.0 * N * K)
        Wq2 = quantize_bitnet(W)
        wq2_d = be.raw_array(Wq2)
        yield Case("qgemv_int", f"w2a8_N{N}_K{K}", {"N": N, "K": K, "M": 1}, "int8",
                   fmt="w2a8",
                   target=lambda wq2_d=wq2_d, xq_d=xq_d, as_d=as_d:
                       tk.qgemv_w2a8(wq2_d, xq_d, as_d),
                   baselines={"tk.qgemv_bitnet": (lambda wq2_d=wq2_d, x_d=x_d:
                                                  tk.qgemv(wq2_d, x_d, format="bitnet"))},
                   ref=None, weight_bytes=N * (K // 32) * 10.0, flops=2.0 * N * K)


@register("qgemm")
def qgemm_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(22)
    fmts = formats or _pick(preset, ["q4_0"], ["q4_0", "q8_0", "fp8_e4m3"],
                            ["q4_0", "q8_0", "q4_K", "fp8_e4m3", "bitnet"])
    m_sweep = _pick(preset, [128], [32, 128, 512], [32, 64, 128, 256, 512])
    NK = _pick(preset, [(4096, 4096)], [(4096, 4096)], [(4096, 4096), (11008, 4096)])
    for N, K in NK:
        for fmt in fmts:
            bk, bb = BLOCK_INFO[fmt]
            if K % bk:
                continue
            wq, wdq = _packed_weight(fmt, N, K)
            wq_d = be.raw_array(wq)
            w_half = be.array(wdq, "f16")
            for M in m_sweep:
                x = rng.standard_normal((K, M)).astype(np.float32)
                x_d = be.array(x, "f16")
                if be.name == "mlx":
                    mm = (lambda w=w_half, x_d=x_d: be.mx.matmul(w, x_d))
                else:
                    mm = (lambda w=w_half, x_d=x_d: w @ x_d)
                baselines = {"fp16_matmul": mm}
                if be.name == "mlx":
                    baselines["tk.qgemm_direct"] = \
                        lambda wq_d=wq_d, x_d=x_d, f=fmt: tk.qgemm_direct(wq_d, x_d, format=f)
                yield Case("qgemm", f"{fmt}_N{N}_K{K}_M{M}", {"N": N, "K": K, "M": M}, "f16",
                           fmt=fmt,
                           target=lambda wq_d=wq_d, x_d=x_d, f=fmt: tk.qgemm(wq_d, x_d, format=f),
                           baselines=baselines, ref=None,
                           weight_bytes=N * (K // bk) * bb, flops=2.0 * N * K * M)


@register("qgemm_fp8_scaled")
def qgemm_fp8_scaled_cases(be, preset, formats):
    """Rank-1-scaled W8A8 FP8 GEMM, including the serving-critical M sweep.

    The framework baseline materializes the same decoded-and-scaled operands in fp16. It is not a
    storage-equivalent baseline, but isolates the decode/scale overhead from dense matmul throughput.
    """
    tk = be.tk()
    from tk.quant import _e4m3_decode_arr
    rng = np.random.default_rng(221)
    shapes = _pick(preset,
                   [(256, 256, 32)],
                   [(4096, 4096, 32), (4096, 4096, 128), (4096, 4096, 512)],
                   [(4096, 4096, 32), (4096, 4096, 64), (4096, 4096, 128),
                    (4096, 4096, 256), (4096, 4096, 512),
                    (11008, 4096, 32), (11008, 4096, 128)])
    for N, K, M in shapes:
        # Restrict magnitudes to finite E4M3 codes representative of quantized normal operands.
        wq = rng.integers(0, 0x77, (N, K), dtype=np.uint8)
        wq |= (rng.integers(0, 2, (N, K), dtype=np.uint8) << 7)
        xq = rng.integers(0, 0x77, (K, M), dtype=np.uint8)
        xq |= (rng.integers(0, 2, (K, M), dtype=np.uint8) << 7)
        ws = rng.uniform(1.0e-3, 3.0e-3, N).astype(np.float16)
        xs = rng.uniform(1.0e-3, 3.0e-3, M).astype(np.float16)
        wq_d, xq_d = be.raw_array(wq), be.raw_array(xq)
        ws_d, xs_d = be.array(ws, "f16"), be.array(xs, "f16")
        w_d = be.array(_e4m3_decode_arr(wq).astype(np.float32) * ws[:, None], "f16")
        x_d = be.array(_e4m3_decode_arr(xq).astype(np.float32) * xs[None, :], "f16")
        if be.name == "mlx":
            dense = lambda w_d=w_d, x_d=x_d: be.mx.matmul(w_d, x_d)
        else:
            dense = lambda w_d=w_d, x_d=x_d: w_d @ x_d
        yield Case("qgemm_fp8_scaled", f"N{N}_K{K}_M{M}",
                   {"N": N, "K": K, "M": M}, "fp8", fmt="fp8_e4m3",
                   target=lambda wq_d=wq_d, xq_d=xq_d, ws_d=ws_d, xs_d=xs_d:
                       tk.qgemm_fp8_scaled(wq_d, xq_d, ws_d, xs_d),
                   baselines={"fp16_matmul": dense}, ref=None,
                   weight_bytes=float(N * K), flops=2.0 * N * K * M,
                   notes="codes-only E4M3 operands with rank-1 fp16 scales")


@register("qgemm_fp8_block2d")
def qgemm_fp8_block2d_cases(be, preset, formats):
    """Codes-only E4M3 weight GEMM with one fp16 scale per 128x128 weight tile."""
    tk = be.tk()
    from tk.quant import _e4m3_decode_arr
    rng = np.random.default_rng(222)
    shapes = _pick(preset,
                   [(256, 256, 64)],
                   [(4096, 4096, 32), (4096, 4096, 128), (4096, 4096, 512)],
                   [(4096, 4096, 32), (4096, 4096, 64), (4096, 4096, 128),
                    (4096, 4096, 256), (4096, 4096, 512),
                    (11008, 4096, 32), (11008, 4096, 128)])
    for N, K, M in shapes:
        wq = rng.integers(0, 0x77, (N, K // 128, 128), dtype=np.uint8)
        wq |= (rng.integers(0, 2, wq.shape, dtype=np.uint8) << 7)
        scale = rng.uniform(1.0e-3, 3.0e-3, (N // 128, K // 128)).astype(np.float16)
        x = (0.2 * rng.standard_normal((K, M))).astype(np.float32)
        wq_d, scale_d, x_d = be.raw_array(wq), be.array(scale, "f16"), be.array(x, "f16")
        scale_rows = np.repeat(scale.astype(np.float32), 128, axis=0)
        w = (_e4m3_decode_arr(wq).astype(np.float32) * scale_rows[:, :, None]).reshape(N, K)
        w_d = be.array(w, "f16")
        if be.name == "mlx":
            dense = lambda w_d=w_d, x_d=x_d: be.mx.matmul(w_d, x_d)
        else:
            dense = lambda w_d=w_d, x_d=x_d: w_d @ x_d
        yield Case("qgemm_fp8_block2d", f"N{N}_K{K}_M{M}",
                   {"N": N, "K": K, "M": M}, "fp8", fmt="fp8_block2d",
                   target=lambda wq_d=wq_d, x_d=x_d, scale_d=scale_d:
                       tk.qgemm_fp8_block2d(wq_d, x_d, scale_d),
                   baselines={"fp16_matmul": dense}, ref=None,
                   weight_bytes=float(N * K + (N // 128) * (K // 128) * 2),
                   flops=2.0 * N * K * M,
                   notes="codes-only E4M3 weights with 128x128 fp16 block scales")


@register("qflux")
def qflux_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(23)
    fmts = formats or _pick(preset, ["q4_0"], ["q4_0", "q8_0"], ["q4_0", "q8_0", "fp8_e4m3"])
    shapes = _pick(preset, [(4096, 4096, 128)], [(4096, 4096, 128)],
                   [(4096, 4096, 32), (4096, 4096, 128), (11008, 4096, 128)])
    for N, K, M in shapes:
        x = rng.standard_normal((K, M)).astype(np.float32)
        bias = rng.standard_normal(M).astype(np.float32)
        x_d, b_d = be.array(x, "f16"), be.array(bias, "f16")
        for fmt in fmts:
            bk, bb = BLOCK_INFO[fmt]
            wq, wdq = _packed_weight(fmt, N, K)
            wq_d = be.raw_array(wq)
            w_half = be.array(wdq, "f16")
            baselines = {}
            if be.name == "mlx":
                mx, nn = be.mx, _mx_nn()
                baselines["mx_matmul_then_gelu"] = lambda w=w_half, x_d=x_d, b_d=b_d: \
                    nn.gelu_approx(mx.matmul(w, x_d) + b_d)
                baselines["tk.qgemm_then_mx_gelu"] = lambda wq_d=wq_d, x_d=x_d, b_d=b_d, f=fmt: \
                    _mx_nn().gelu_approx(tk.qgemm(wq_d, x_d, format=f) + b_d)
            yield Case("qflux", f"{fmt}_N{N}_K{K}_M{M}", {"N": N, "K": K, "M": M}, "f16",
                       fmt=fmt,
                       target=lambda wq_d=wq_d, x_d=x_d, b_d=b_d, f=fmt:
                           tk.qflux_gelu(wq_d, x_d, b_d, format=f),
                       baselines=baselines, ref=None,
                       weight_bytes=N * (K // bk) * bb, flops=2.0 * N * K * M)


# --------------------------------------------------------------------------- complex
@register("cmplx_matmul")
def cmplx_matmul_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(24)
    shapes = _pick(preset, [(512, 512, 512)], [(512, 512, 512), (1024, 1024, 1024)],
                   [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)])
    for N, K, M in shapes:
        A = (0.3 * rng.standard_normal((2, N, K))).astype(np.float32)
        B_ = (0.3 * rng.standard_normal((2, K, M))).astype(np.float32)
        a_d, b_d = be.array(A, "bf16"), be.array(B_, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def composed(a_d=a_d, b_d=b_d):
                ar, ai = a_d[0], a_d[1]
                br, bi = b_d[0], b_d[1]
                return mx.stack([ar @ br - ai @ bi, ar @ bi + ai @ br])
            baselines["mx_4matmul"] = composed
        yield Case("cmplx_matmul", f"{N}x{K}x{M}", {"N": N, "K": K, "M": M}, "bf16",
                   target=lambda a_d=a_d, b_d=b_d: tk.cmplx_matmul(a_d, b_d),
                   baselines=baselines, ref=None, flops=8.0 * N * K * M)


@register("fftconv")
def fftconv_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(25)
    shapes = _pick(preset, [(2, 8, 32)], [(2, 8, 32), (4, 16, 32)],
                   [(2, 8, 16), (2, 8, 32), (4, 16, 32), (8, 32, 32)])
    for B, H, S in shapes:
        N = S * S
        u = rng.standard_normal((B, H, N)).astype(np.float32)
        kf_t = rng.standard_normal((H, N)).astype(np.float32)

        def _fft_matrix(n):
            a = np.arange(n)
            return np.exp(-2j * np.pi * a * a.reshape(-1, 1) / n)

        def _ifft_matrix(n):
            a = np.arange(n)
            return np.exp(2j * np.pi * a * a.reshape(-1, 1) / n)

        def _twiddle(n, m, sign):
            na = np.arange(n).reshape(-1, 1)
            ma = np.arange(m)
            return np.exp(sign * 2j * np.pi * na * ma / (n * m))

        def _stack(m):
            return be.array(np.stack([m.real, m.imag]).astype(np.float32), "f32")
        F, Finv = _fft_matrix(S), _ifft_matrix(S)
        TW, TWI = _twiddle(S, S, -1), _twiddle(S, S, +1) / N
        kf = np.fft.fft(kf_t, n=N).reshape(H, S, S).transpose(0, 2, 1)
        xr = u.reshape(B, H, S, S)
        X = be.array(np.stack([xr, np.zeros_like(xr)]), "f32")
        KF = be.array(np.stack([kf.real, kf.imag]).astype(np.float32), "f32")
        fm, tw, fi, ti = _stack(F), _stack(TW), _stack(Finv), _stack(TWI)
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            u_d = mx.array(u)
            k_d = mx.array(kf_t)
            baselines["mx.fft_conv"] = lambda u_d=u_d, k_d=k_d: \
                mx.fft.irfft(mx.fft.rfft(u_d) * mx.fft.rfft(k_d)[None], n=N)
        ref = np.fft.ifft(np.fft.fft(u, n=N) * np.fft.fft(kf_t, n=N)[None],
                          n=N).real.reshape(B, H, S, S)
        yield Case("fftconv", f"B{B}H{H}S{S}", {"B": B, "H": H, "S": S}, "f32",
                   target=lambda X=X, fm=fm, tw=tw, fi=fi, ti=ti, KF=KF:
                       tk.fftconv(X, fm, tw, fi, ti, KF),
                   baselines=baselines, ref=ref,
                   flops=32.0 * B * H * S ** 3)   # ~4 complex (S,S)@(S,S) GEMMs per (b,h)


# --------------------------------------------------------------------------- serving kernels
@register("indexer_quant")
def indexer_quant_cases(be, preset, formats):
    """DeepSeek-V3.2 indexer K quant-and-cache (bandwidth-bound: reads bf16 K, writes u8 codes)."""
    tk = be.tk()
    rng = np.random.default_rng(242)
    for T, hd in _pick(preset, [(4096, 128)], [(16384, 128), (16384, 256)],
                       [(65536, 128), (65536, 256)]):
        k = (0.3 * rng.standard_normal((T, hd))).astype(np.float32)
        nq = (hd + 127) // 128
        code0 = np.zeros((T, hd), np.uint8)
        scale0 = np.zeros((T, nq), np.float32)
        k_d = be.array(k, "bf16")
        c_d, s_d = be.raw_array(code0), be.array(scale0, "f32")
        sm = be.int_array(np.arange(T, dtype=np.int32))
        for ue8m0 in (False, True):
            suffix = "_ue8m0" if ue8m0 else ""
            yield Case("indexer_quant", f"T{T}_hd{hd}{suffix}", {"T": T, "hd": hd},
                       "bf16", fmt="fp8_e4m3_ue8m0" if ue8m0 else "fp8_e4m3",
                       target=lambda k_d=k_d, sm=sm, c_d=c_d, s_d=s_d, ue8m0=ue8m0:
                           tk.indexer_k_quant_and_cache(
                               k_d, sm, c_d, s_d, ue8m0=ue8m0)[0],
                       baselines={}, ref=None, bytes_moved=float(T * hd * 2 + T * hd))


@register("kv_gather_fp8")
def kv_gather_fp8_cases(be, preset, formats):
    """fp8 KV gather + upconvert (reads 1-byte codes) vs bf16 gather (reads 2-byte). The read
    path for a paged fp8 prefix cache; the win is halving the cache read bandwidth."""
    tk = be.tk()
    rng = np.random.default_rng(241)
    for nb, bs, H, D in _pick(preset, [(64, 16, 8, 64)],
                              [(256, 16, 8, 64), (256, 16, 8, 128)],
                              [(512, 16, 8, 64), (512, 16, 8, 128),
                               (1024, 16, 8, 64), (1024, 16, 8, 128)]):
        total = nb * bs
        K = (0.3 * rng.standard_normal((total, H, D))).astype(np.float32)
        V = (0.3 * rng.standard_normal((total, H, D))).astype(np.float32)
        bt = be.int_array(np.arange(nb, dtype=np.int32).reshape(1, nb))
        cu = be.int_array(np.array([0, total], np.int32))
        for fmt, fmt_code, qmax in (("e4m3", 0, 448.0), ("e5m2", 1, 57344.0)):
            ks = np.maximum(np.abs(K).max(axis=(0, 2)) / qmax, 1e-8).astype(np.float32)
            vs = np.maximum(np.abs(V).max(axis=(0, 2)) / qmax, 1e-8).astype(np.float32)
            ksa, vsa = be.array(ks, "f32"), be.array(vs, "f32")
            kc, vc = tk.kv_cache_scatter_fp8(
                be.array(K, "bf16"), be.array(V, "bf16"),
                be.int_array(np.arange(total, dtype=np.int64)), nb, bs, ksa, vsa, fmt=fmt)
            yield Case("kv_gather_fp8", f"{fmt}_nb{nb}_H{H}_D{D}",
                       {"nb": nb, "H": H, "D": D}, "fp8", fmt=f"fp8_{fmt}",
                       target=lambda kc=kc, vc=vc, bt=bt, cu=cu, ksa=ksa, vsa=vsa,
                                     t=total, fmt_code=fmt_code:
                           tk.kv_cache_gather_fp8(
                               kc, vc, bt, cu, ksa, vsa, t, fmt=fmt_code)[0],
                       baselines={}, ref=None, bytes_moved=float(2 * total * H * D))


@register("kv_scatter_fp8")
def kv_scatter_fp8_cases(be, preset, formats):
    """FP8 KV-cache producer, including both attention head sizes."""
    tk = be.tk()
    rng = np.random.default_rng(240)
    shapes = _pick(preset, [(512, 8, 64)], [(512, 8, 64), (512, 8, 128)],
                   [(1, 8, 64), (1, 8, 128), (512, 8, 64), (512, 8, 128),
                    (4096, 8, 64), (4096, 8, 128)])
    for T, H, D in shapes:
        bs = 16
        nb = (T + bs - 1) // bs
        key = (0.3 * rng.standard_normal((T, H, D))).astype(np.float32)
        value = (0.3 * rng.standard_normal((T, H, D))).astype(np.float32)
        key_d, value_d = be.array(key, "bf16"), be.array(value, "bf16")
        slots_d = be.int_array(np.arange(T, dtype=np.int64))
        for fmt, qmax in (("e4m3", 448.0), ("e5m2", 57344.0)):
            ks = np.maximum(np.abs(key).max(axis=(0, 2)) / qmax, 1e-8).astype(np.float32)
            vs = np.maximum(np.abs(value).max(axis=(0, 2)) / qmax, 1e-8).astype(np.float32)
            ks_d, vs_d = be.array(ks, "f32"), be.array(vs, "f32")
            yield Case("kv_scatter_fp8", f"{fmt}_T{T}_H{H}_D{D}",
                       {"T": T, "H": H, "D": D}, "bf16", fmt=f"fp8_{fmt}",
                       target=lambda key_d=key_d, value_d=value_d, slots_d=slots_d,
                                     nb=nb, bs=bs, ks_d=ks_d, vs_d=vs_d, fmt=fmt:
                           tk.kv_cache_scatter_fp8(key_d, value_d, slots_d, nb, bs,
                                                   ks_d, vs_d, fmt=fmt)[0],
                       baselines={}, ref=None,
                       bytes_moved=float(2 * T * H * D * 2 + 2 * T * H * D))


@register("paged_attn")
def paged_attn_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(26)
    shapes = _pick(preset, [(4, 16, 4, 128, 512)],
                   [(8, 32, 8, 64, 2048), (8, 32, 8, 128, 2048)],
                   [(8, 32, 8, 64, 2048), (8, 32, 8, 128, 2048),
                    (16, 32, 8, 128, 4096), (8, 32, 8, 128, 8192)])
    for B, H, H_KV, D, ctx in shapes:
        block_size = 16
        max_blocks = (ctx + block_size - 1) // block_size
        num_blocks = B * max_blocks
        q = (0.1 * rng.standard_normal((B, H, D))).astype(np.float32)
        kc = (0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32)
        vc = (0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32)
        bt = np.arange(B * max_blocks, dtype=np.int32).reshape(B, max_blocks)
        cl = np.full((B,), ctx, dtype=np.int32)
        q_d, kc_d, vc_d = be.array(q, "bf16"), be.array(kc, "bf16"), be.array(vc, "bf16")
        bt_d, cl_d = be.int_array(bt), be.int_array(cl)
        kv_bytes = 2.0 * B * ctx * H_KV * D * 2
        args = (q_d, kc_d, vc_d, bt_d, cl_d)
        v1 = (lambda a=args: tk.paged_attention(*a))
        yield Case("paged_attn", f"v1_B{B}H{H}ctx{ctx}", {"B": B, "H": H, "ctx": ctx, "D": D},
                   "bf16", target=v1, baselines={}, ref=None, bytes_moved=kv_bytes)
        yield Case("paged_attn", f"staged_B{B}H{H}ctx{ctx}", {"B": B, "H": H, "ctx": ctx, "D": D},
                   "bf16",
                   target=lambda a=args: tk.paged_attention_staged(*a),
                   baselines={"tk.paged_attention_v1": v1}, ref=None, bytes_moved=kv_bytes)
        parts = _pick(preset, [256], [256], [128, 256, 512, 1024])
        for ps in parts:
            yield Case("paged_attn", f"v2_p{ps}_B{B}H{H}ctx{ctx}",
                       {"B": B, "H": H, "ctx": ctx, "D": D}, "bf16",
                       target=lambda a=args, ps=ps: tk.paged_attention_v2(*a, partition_size=ps),
                       baselines={"tk.paged_attention_v1": v1}, ref=None, bytes_moved=kv_bytes)
        # fp8 KV cache read path (dequant-on-read cost)
        codes_k = rng.integers(0, 127, kc.shape, dtype=np.uint8)
        codes_v = rng.integers(0, 127, vc.shape, dtype=np.uint8)
        kc8_d, vc8_d = be.raw_array(codes_k), be.raw_array(codes_v)
        yield Case("paged_attn", f"v1_fp8_e4m3_B{B}H{H}ctx{ctx}",
                   {"B": B, "H": H, "ctx": ctx, "D": D}, "fp8",
                   target=lambda q_d=q_d, kc8_d=kc8_d, vc8_d=vc8_d, bt_d=bt_d, cl_d=cl_d:
                       tk.paged_attention_fp8(q_d, kc8_d, vc8_d, bt_d, cl_d, 0.01, 0.01),
                   baselines={"tk.paged_attention_bf16": v1}, ref=None, bytes_moved=kv_bytes / 2)
        yield Case("paged_attn", f"v2_fp8_e4m3_B{B}H{H}ctx{ctx}",
                   {"B": B, "H": H, "ctx": ctx, "D": D}, "fp8",
                   target=lambda q_d=q_d, kc8_d=kc8_d, vc8_d=vc8_d, bt_d=bt_d, cl_d=cl_d:
                       tk.paged_attention_v2_fp8(q_d, kc8_d, vc8_d, bt_d, cl_d,
                                                 0.01, 0.01, partition_size=256),
                   baselines={"tk.paged_attention_v2_bf16":
                              (lambda a=args: tk.paged_attention_v2(*a, partition_size=256))},
                   ref=None, bytes_moved=kv_bytes / 2)
        yield Case("paged_attn", f"v1_fp8_e5m2_B{B}H{H}ctx{ctx}",
                   {"B": B, "H": H, "ctx": ctx, "D": D}, "fp8",
                   target=lambda q_d=q_d, kc8_d=kc8_d, vc8_d=vc8_d, bt_d=bt_d, cl_d=cl_d:
                       tk.paged_attention_fp8(q_d, kc8_d, vc8_d, bt_d, cl_d, 0.01, 0.01,
                                              fmt="e5m2"),
                   baselines={"tk.paged_attention_bf16": v1}, ref=None, bytes_moved=kv_bytes / 2)
        yield Case("paged_attn", f"v2_fp8_e5m2_B{B}H{H}ctx{ctx}",
                   {"B": B, "H": H, "ctx": ctx, "D": D}, "fp8",
                   target=lambda q_d=q_d, kc8_d=kc8_d, vc8_d=vc8_d, bt_d=bt_d, cl_d=cl_d:
                       tk.paged_attention_v2_fp8(q_d, kc8_d, vc8_d, bt_d, cl_d,
                                                 0.01, 0.01, partition_size=256, fmt="e5m2"),
                   baselines={"tk.paged_attention_v2_bf16":
                              (lambda a=args: tk.paged_attention_v2(*a, partition_size=256))},
                   ref=None, bytes_moved=kv_bytes / 2)


@register("mla")
def mla_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(27)
    shapes = _pick(preset, [(4, 16, 512)], [(8, 32, 2048)],
                   [(8, 32, 2048), (16, 32, 4096), (8, 16, 8192)])
    for B, NH, ctx in shapes:
        block_size = 16
        num_blocks = (ctx + block_size - 1) // block_size
        q = (0.1 * rng.standard_normal((B, NH, 576))).astype(np.float32)
        cache = (0.1 * rng.standard_normal((num_blocks, block_size, 576))).astype(np.float32)
        bt = (np.arange(B * num_blocks, dtype=np.int32).reshape(B, num_blocks)) % num_blocks
        cl = np.full((B,), ctx, dtype=np.int32)
        q_d, c_d = be.array(q, "bf16"), be.array(cache, "bf16")
        bt_d, cl_d = be.int_array(bt), be.int_array(cl)
        yield Case("mla", f"decode_B{B}H{NH}ctx{ctx}", {"B": B, "H": NH, "ctx": ctx}, "bf16",
                   target=lambda q_d=q_d, c_d=c_d, bt_d=bt_d, cl_d=cl_d:
                       tk.mla_decode(q_d, c_d, bt_d, cl_d),
                   baselines={}, ref=None, bytes_moved=float(B * ctx * 576 * 2))

        # V4 cache: 448 E4M3 values with seven UE8M0 scales plus 64 bf16 RoPE values.
        total = num_blocks * block_size
        kv = (0.3 * rng.standard_normal((total, 512))).astype(np.float32)
        cos = np.ones((total, 32), dtype=np.float32)
        sin = np.zeros((total, 32), dtype=np.float32)
        data0 = np.zeros((num_blocks, block_size, 576), dtype=np.uint8)
        scale0 = np.zeros((num_blocks, block_size, 8), dtype=np.uint8)
        data_d, scale_d = tk.mla_kv_insert_fp8(
            be.array(kv, "bf16"), be.array(cos, "bf16"), be.array(sin, "bf16"),
            be.int_array(np.arange(total, dtype=np.int32)),
            be.int_array(np.arange(total, dtype=np.int64)),
            be.raw_array(data0), be.raw_array(scale0))
        be.sync([data_d, scale_d])
        q8_d = be.array(q[:, :, :512], "bf16")
        yield Case("mla", f"decode_fp8_B{B}H{NH}ctx{ctx}",
                   {"B": B, "H": NH, "ctx": ctx}, "fp8", fmt="fp8_e4m3_ue8m0",
                   target=lambda q8_d=q8_d, data_d=data_d, scale_d=scale_d, bt_d=bt_d, cl_d=cl_d:
                       tk.mla_decode_fp8(q8_d, data_d, scale_d, bt_d, cl_d),
                   baselines={}, ref=None,
                   bytes_moved=float(B * ctx * (448 + 64 * 2 + 8)))
        kv_d = be.array(kv, "bf16")
        cos_d, sin_d = be.array(cos, "bf16"), be.array(sin, "bf16")
        pos_d = be.int_array(np.arange(total, dtype=np.int32))
        slot_d = be.int_array(np.arange(total, dtype=np.int64))
        data0_d, scale0_d = be.raw_array(data0), be.raw_array(scale0)
        yield Case("mla", f"insert_fp8_T{total}", {"T": total, "D": 512}, "bf16",
                   fmt="fp8_e4m3_ue8m0",
                   target=lambda kv_d=kv_d, cos_d=cos_d, sin_d=sin_d, pos_d=pos_d,
                                 slot_d=slot_d, data0_d=data0_d, scale0_d=scale0_d:
                       tk.mla_kv_insert_fp8(
                           kv_d, cos_d, sin_d, pos_d, slot_d, data0_d, scale0_d)[0],
                   baselines={}, ref=None,
                   bytes_moved=float(total * (512 * 2 + 576 + 8)))


@register("logit_transforms")
def logit_transform_cases(be, preset, formats):
    """Sampler-zoo transforms: bandwidth-bound (T,V) passes; ms + GB/s (2*T*V*4)."""
    tk = be.tk()
    rng = np.random.default_rng(35)
    shapes = _pick(preset, [(256, 32000)], [(256, 32000), (1024, 32000)],
                   [(256, 128256), (1024, 128256), (2048, 128256)])
    for T, V in shapes:
        x = rng.standard_normal((T, V)).astype(np.float32)
        x_d = be.array(x, "f32")
        prev = rng.integers(0, 512, (T, 256)).astype(np.int32)
        lens = np.full(T, 256, np.int32)
        brk = np.array([-1], np.int32)
        prev_d, lens_d, brk_d = be.int_array(prev), be.int_array(lens), be.int_array(brk)
        for name, thunk in [
            ("quadratic", lambda x_d=x_d: tk.quadratic_transform(x_d, 0.3, 1.5)),
            ("nsigma", lambda x_d=x_d: tk.top_nsigma_mask(x_d, 1.5)),
            ("top_a", lambda x_d=x_d: tk.top_a_mask(x_d, 0.2)),
            ("eta", lambda x_d=x_d: tk.eta_cutoff_mask(x_d, 2e-3)),
            ("xtc", lambda x_d=x_d: tk.xtc_mask(x_d, 0.1, 1.0, seed=7)),
            ("dry_L256", lambda x_d=x_d, p=prev_d, l=lens_d, b=brk_d:
                 tk.dry_penalty(x_d, p, l, b, 1.2)),
        ]:
            yield Case("logit_transforms", f"{name}_T{T}_V{V}", {"T": T, "V": V}, "f32",
                       target=thunk, baselines={}, ref=None, bytes_moved=2.0 * T * V * 4)


def _fake_quant_deq_np(x):
    amax = np.abs(x).max(axis=-1)
    scale = (amax / 127.0).astype(np.float32)
    inv = np.where(scale > 0, 1.0 / scale, 0.0).astype(np.float32)
    codes = np.clip(np.rint(x * inv[..., None]), -127, 127).astype(np.int8)
    return codes.astype(np.float32) * scale.astype(np.float16).astype(np.float32)[..., None]


def _swiglu_np(x, gate):
    return x / (1.0 + np.exp(-x)) * gate


def _fake_quant_deq_baseline(be, x):
    tk = be.tk()
    codes, scale = tk.quantize_per_token_int8(x)
    if be.name == "mlx":
        return codes.astype(x.dtype) * scale[..., None].astype(x.dtype)
    return codes.to(x.dtype) * scale[:, None].to(x.dtype)


def _ternary_deq_baseline(be, w, group_k):
    if be.name == "mlx":
        mx = be.mx
        *prefix, K = w.shape
        wg = mx.reshape(w, (*prefix, K // group_k, group_k))
        s = mx.maximum(mx.mean(mx.abs(wg), axis=-1, keepdims=True), 1.0e-5)
        q = mx.clip(mx.round(wg / s), -1, 1)
        return mx.reshape(q * s.astype(mx.float16), w.shape)
    torch = be.torch
    *prefix, K = w.shape
    wg = w.reshape(*prefix, K // group_k, group_k)
    s = torch.clamp(torch.mean(torch.abs(wg), dim=-1, keepdim=True), min=1.0e-5)
    q = torch.clamp(torch.round(wg / s), -1, 1)
    return (q * s.to(torch.float16)).reshape(w.shape)


def _ternary_pt_deq_baseline(be, w):
    if be.name == "mlx":
        mx = be.mx
        if len(w.shape) == 2:
            s = mx.maximum(mx.mean(mx.abs(w)), 1.0e-5)
        else:
            s = mx.maximum(mx.mean(mx.abs(w), axis=(-2, -1), keepdims=True), 1.0e-5)
        return mx.clip(mx.round(w / s), -1, 1) * s.astype(mx.float16)
    torch = be.torch
    if len(w.shape) == 2:
        s = torch.clamp(torch.mean(torch.abs(w)), min=1.0e-5)
    else:
        s = torch.clamp(torch.mean(torch.abs(w), dim=(-2, -1), keepdim=True), min=1.0e-5)
    return torch.clamp(torch.round(w / s), -1, 1) * s.to(torch.float16)


def _logsumexp_np(z):
    m = z.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True))).squeeze(-1)


def _kd_topk_grad_np(logits, idx, prob, grad_out, invtemp, tail_mode):
    z = logits * invtemp
    lse = _logsumexp_np(z)
    q = np.exp(z - lse[:, None])
    grad = np.zeros_like(logits, np.float32)
    for t in range(idx.shape[0]):
        valid = idx[t] >= 0
        ii = idx[t, valid]
        pp = prob[t, valid]
        P = pp.sum()
        S = q[t, ii].sum()
        if tail_mode == 0:
            pt = pp / max(P, 1e-30)
            grad[t] = q[t]
            grad[t, ii] -= pt
        else:
            tail = max(1.0 - P, 0.0)
            tail_c = tail / max(1.0 - S, 1e-30) if tail > 0 else 0.0
            grad[t] = (P - tail_c * S) * q[t]
            grad[t, ii] += -pp + tail_c * q[t, ii]
        grad[t] *= grad_out[t] * invtemp
    return grad


def _adamw_masked_param_np(p, g, m, v, lr, b1, b2, eps, wd, step, active, mask_mode):
    m_new = b1 * m + (1.0 - b1) * g
    v_new = b2 * v + (1.0 - b2) * g * g
    mhat = m_new / (1.0 - b1 ** step)
    vhat = v_new / (1.0 - b2 ** step)
    decay = wd * active.astype(np.float32)
    p_new = p - lr * (mhat / (np.sqrt(vhat) + eps) + decay * p)
    if mask_mode == 0:
        p_new = np.where(active, p_new, p)
    return p_new.astype(np.float32)


@register("fake_quant")
def fake_quant_cases(be, preset, formats):
    """BitNet one-pass fake quant: current port vs decomposed quantize_per_token_int8 + dequant."""
    tk = be.tk()
    rng = np.random.default_rng(36)
    shapes = _pick(preset, [(512, 2048)], [(512, 2880), (4096, 2880)],
                   [(512, 2880), (4096, 2880), (4096, 11008)])
    for T, D in shapes:
        x = (0.5 * rng.standard_normal((T, D))).astype(np.float32)
        g = (0.5 * rng.standard_normal((T, D))).astype(np.float32)
        x_d, g_d = be.array(x, "bf16"), be.array(g, "bf16")
        x_ref = be.to_numpy(x_d)
        g_ref = be.to_numpy(g_d)
        yield Case("fake_quant", f"plain_T{T}_D{D}", {"T": T, "D": D}, "bf16",
                   target=lambda x_d=x_d: tk.fake_quant_int8(x_d),
                   out_to_numpy=lambda o: be.to_numpy(o[0]),
                   baselines={"quantize_then_dequant":
                              lambda x_d=x_d: _fake_quant_deq_baseline(be, x_d)},
                   ref=_fake_quant_deq_np(x_ref),
                   bytes_moved=float(2 * T * D * 2 + T * D + T * 4))
        yield Case("fake_quant", f"swiglu_T{T}_D{D}", {"T": T, "D": D}, "bf16",
                   target=lambda x_d=x_d, g_d=g_d: tk.silu_mul_fake_quant_int8(x_d, g_d),
                   out_to_numpy=lambda o: be.to_numpy(o[0]),
                   baselines={"swiglu_then_quant_dequant":
                              lambda x_d=x_d, g_d=g_d:
                                  _fake_quant_deq_baseline(be, tk.swiglu(x_d, g_d))},
                   ref=_fake_quant_deq_np(_swiglu_np(x_ref, g_ref)),
                   bytes_moved=float(3 * T * D * 2 + T * D + T * 4))


@register("fake_quant_fp8")
def fake_quant_fp8_cases(be, preset, formats):
    """Per-tensor E4M3 fake quantization with exact RNE midpoint handling."""
    tk = be.tk()
    rng = np.random.default_rng(361)
    shapes = _pick(preset, [(512, 2048)], [(512, 2880), (4096, 2880)],
                   [(512, 2880), (4096, 2880), (4096, 11008)])
    for T, D in shapes:
        x_d = be.array((0.5 * rng.standard_normal((T, D))).astype(np.float32), "bf16")
        yield Case("fake_quant_fp8", f"T{T}_D{D}", {"T": T, "D": D}, "bf16",
                   fmt="fp8_e4m3",
                   target=lambda x_d=x_d: tk.fake_quant_fp8(x_d)[0],
                   baselines={}, ref=None, bytes_moved=float(3 * T * D * 2))


@register("weight_quant_ternary")
def weight_quant_ternary_cases(be, preset, formats):
    """BitNet weight quantization port vs framework decomposed dequant-only baseline."""
    from tk.quant import dequantize_bitnet, quantize_bitnet

    tk = be.tk()
    rng = np.random.default_rng(37)
    shapes = _pick(preset, [(512, 2048)], [(512, 2880), (4096, 2880)],
                   [(512, 2880), (4096, 2880), (4096, 11008)])
    for N, K in shapes:
        w = (0.05 * rng.standard_normal((N, K))).astype(np.float32)
        w_d = be.array(w, "bf16")
        w_ref = be.to_numpy(w_d)
        q_ref = quantize_bitnet(w_ref)
        yield Case("weight_quant_ternary", f"group32_N{N}_K{K}",
                   {"N": N, "K": K, "group_k": 32}, "bf16",
                   target=lambda w_d=w_d: tk.weight_quant_ternary(w_d, group_k=32),
                   out_to_numpy=lambda o: be.to_numpy(o[1]),
                   baselines={"framework_deq_only":
                              lambda w_d=w_d: _ternary_deq_baseline(be, w_d, 32)},
                   ref=dequantize_bitnet(q_ref), bytes_moved=float(3 * N * K * 2))
    for E, N, K in _pick(preset, [(2, 256, 2048)], [(2, 512, 2880)],
                         [(2, 512, 2880), (4, 512, 2880)]):
        w = (0.05 * rng.standard_normal((E, N, K))).astype(np.float32)
        w_d = be.array(w, "bf16")
        w_ref = be.to_numpy(w_d)
        ref = np.zeros_like(w_ref, dtype=np.float32)
        for e in range(E):
            s = max(float(np.abs(w_ref[e]).mean()), 1e-5)
            ref[e] = np.clip(np.rint(w_ref[e] / s), -1, 1) * np.float16(s).astype(np.float32)
        yield Case("weight_quant_ternary", f"pt_E{E}_N{N}_K{K}",
                   {"E": E, "N": N, "K": K}, "bf16",
                   target=lambda w_d=w_d: tk.weight_quant_ternary_pt(w_d),
                   out_to_numpy=lambda o: be.to_numpy(o[1]),
                   baselines={"framework_deq_only":
                              lambda w_d=w_d: _ternary_pt_deq_baseline(be, w_d)},
                   ref=ref, bytes_moved=float(3 * E * N * K * 2))


@register("kd_kl_topk")
def kd_kl_topk_cases(be, preset, formats):
    """Sparse teacher KD-KL fwd+bwd over top-k teacher caches."""
    tk = be.tk()
    rng = np.random.default_rng(38)
    shapes = _pick(preset, [(256, 32000, 32)], [(256, 32000, 32), (1024, 32000, 32)],
                   [(256, 32000, 32), (1024, 32000, 32), (1024, 128256, 32)])
    for T, V, K in shapes:
        logits = (rng.standard_normal((T, V)) * 1.3).astype(np.float32)
        raw = rng.random((T, K)).astype(np.float32)
        prob = raw / raw.sum(axis=-1, keepdims=True)
        idx = np.empty((T, K), np.int32)
        for t in range(T):
            idx[t] = rng.choice(V, size=K, replace=False)
        grad_out = rng.standard_normal(T).astype(np.float32)
        logits_d = be.array(logits, "f32")
        idx_d, prob_d, grad_d = be.int_array(idx), be.array(prob, "f32"), be.array(grad_out, "f32")
        invtemp = 0.5
        for tail_mode in [0, 1]:
            yield Case("kd_kl_topk", f"tail{tail_mode}_T{T}_V{V}_K{K}",
                       {"T": T, "V": V, "K": K}, "f32",
                       target=lambda logits_d=logits_d, idx_d=idx_d, prob_d=prob_d, grad_d=grad_d,
                                     invtemp=invtemp, tail_mode=tail_mode:
                           (lambda loss_lse:
                            (loss_lse[0],
                             tk.kd_kl_topk_bwd(logits_d, idx_d, prob_d, loss_lse[1],
                                               grad_d, invtemp=invtemp, tail_mode=tail_mode)))(
                               tk.kd_kl_topk_fwd(logits_d, idx_d, prob_d,
                                                 invtemp=invtemp, tail_mode=tail_mode)),
                       out_to_numpy=lambda o: be.to_numpy(o[1]),
                       baselines={}, ref=_kd_topk_grad_np(logits, idx, prob, grad_out,
                                                          invtemp, tail_mode),
                       bytes_moved=float((2 * T * V + 2 * T * K) * 4))


@register("act_quant")
def act_quant_cases(be, preset, formats):
    """Fused gated-activation -> quant vs the unfused glu -> quantize_per_token chain
    (the fusion win = eliminating the (rows, D) bf16 intermediate round-trip)."""
    tk = be.tk()
    rng = np.random.default_rng(34)
    shapes = _pick(preset, [(512, 2048)], [(512, 2880), (4096, 2880)],
                   [(512, 2880), (4096, 2880), (4096, 11008)])
    for T, D in shapes:
        x = (0.5 * rng.standard_normal((T, D))).astype(np.float32)
        g = (0.5 * rng.standard_normal((T, D))).astype(np.float32)
        x_d, g_d = be.array(x, "bf16"), be.array(g, "bf16")
        baselines = {"glu_then_quant":
                     lambda x_d=x_d, g_d=g_d:
                         tk.quantize_per_token_int8(tk.swiglu(x_d, g_d))[0]}
        yield Case("act_quant", f"int8_T{T}_D{D}", {"T": T, "D": D}, "bf16",
                   target=lambda x_d=x_d, g_d=g_d: tk.silu_mul_quant_int8(x_d, g_d)[0],
                   baselines=baselines, ref=None,
                   bytes_moved=(2.0 * T * D * 2 + T * D))
        fp8_baselines = {"glu_then_quant":
                         lambda x_d=x_d, g_d=g_d:
                             tk.quantize_per_token_fp8(tk.swiglu(x_d, g_d))[0]}
        yield Case("act_quant", f"fp8_T{T}_D{D}", {"T": T, "D": D}, "bf16",
                   fmt="fp8_e4m3",
                   target=lambda x_d=x_d, g_d=g_d: tk.silu_mul_quant_fp8(x_d, g_d)[0],
                   baselines=fp8_baselines, ref=None,
                   bytes_moved=(2.0 * T * D * 2 + T * D))
        yield Case("act_quant", f"fp8_oai_T{T}_D{D}", {"T": T, "D": D}, "bf16",
                   fmt="fp8_e4m3",
                   target=lambda x_d=x_d, g_d=g_d:
                       tk.silu_mul_quant_fp8(x_d, g_d, act="swiglu_oai")[0],
                   baselines={}, ref=None,
                   bytes_moved=(2.0 * T * D * 2 + T * D))
        yield Case("act_quant", f"int8_oai_T{T}_D{D}", {"T": T, "D": D}, "bf16",
                   target=lambda x_d=x_d, g_d=g_d:
                       tk.silu_mul_quant_int8(x_d, g_d, act="swiglu_oai")[0],
                   baselines={}, ref=None,
                   bytes_moved=(2.0 * T * D * 2 + T * D))
        group_size = 128 if D % 128 == 0 else 64
        yield Case("act_quant", f"fp8_group_ue8m0_T{T}_D{D}_G{group_size}",
                   {"T": T, "D": D, "G": group_size}, "bf16", fmt="fp8_e4m3_ue8m0",
                   target=lambda x_d=x_d, g_d=g_d, group_size=group_size:
                       tk.silu_mul_quant_fp8_group(
                           x_d, g_d, group_size=group_size, ue8m0=True)[0],
                   baselines={}, ref=None,
                   bytes_moved=(2.0 * T * D * 2 + T * D))
        yield Case("act_quant", f"fp8_group_oai_ue8m0_T{T}_D{D}_G{group_size}",
                   {"T": T, "D": D, "G": group_size}, "bf16", fmt="fp8_e4m3_ue8m0",
                   target=lambda x_d=x_d, g_d=g_d, group_size=group_size:
                       tk.silu_mul_quant_fp8_group(
                           x_d, g_d, group_size=group_size, ue8m0=True,
                           act="swiglu_oai")[0],
                   baselines={}, ref=None,
                   bytes_moved=(2.0 * T * D * 2 + T * D))


@register("gdn")
def gdn_cases(be, preset, formats):
    """GatedDeltaNet recurrence (Qwen3-Next mixer). Sequential over time per (req, hv, dv)
    simdgroup; decode (seq=1 per request) and prefill shapes. No framework baseline."""
    tk = be.tk()
    rng = np.random.default_rng(33)
    shapes = _pick(preset, [([1] * 16, 2, 8, 64, 64)],
                   [([1] * 64, 2, 8, 128, 128), ([2048] * 2, 2, 8, 128, 128)],
                   [([1] * 64, 2, 8, 128, 128), ([2048] * 2, 2, 8, 128, 128),
                    ([512] * 8, 4, 16, 128, 128)])
    for lens, Hk, Hv, Dk, Dv in shapes:
        T, R = sum(lens), len(lens)
        cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        q = (0.3 * rng.standard_normal((T, Hk, Dk))).astype(np.float32)
        k = (0.3 * rng.standard_normal((T, Hk, Dk))).astype(np.float32)
        v = (0.3 * rng.standard_normal((T, Hv, Dv))).astype(np.float32)
        g = rng.uniform(0.9, 1.0, (T, Hv)).astype(np.float32)
        beta = rng.uniform(0.2, 0.8, (T, Hv)).astype(np.float32)
        pool = np.zeros((R, Hv, Dv, Dk), np.float32)
        slots = np.arange(R, dtype=np.int32)
        args = [be.array(q, "bf16"), be.array(k, "bf16"), be.array(v, "bf16"),
                be.array(g, "bf16"), be.array(beta, "bf16"), be.array(pool, "f32"),
                be.int_array(cu), be.int_array(slots)]
        label = f"R{R}_L{lens[0]}_Hv{Hv}_Dk{Dk}"
        yield Case("gdn", label, {"R": R, "L": lens[0], "Hv": Hv, "Dk": Dk}, "bf16",
                   target=lambda a=args: tk.gdn_recur(*a)[0],
                   baselines={}, ref=None,
                   flops=2.0 * T * Hv * (2 * Dv * Dk + Dv * Dk))


@register("selective_scan")
def selective_scan_cases(be, preset, formats):
    """Mamba-1 S6 scan: sequential-in-time, parallel-over-state. No framework baseline
    (a lazily-traced per-step composition is pathological); ms + GB/s over the io tensors."""
    tk = be.tk()
    rng = np.random.default_rng(32)
    shapes = _pick(preset, [(2, 1024, 128, 16)],
                   [(2, 2048, 512, 16), (2, 2048, 512, 128)],
                   [(2, 2048, 512, 16), (2, 2048, 2048, 16), (2, 4096, 512, 128)])
    for b, d, L, N in shapes:
        u = (0.5 * rng.standard_normal((b, d, L))).astype(np.float32)
        delta = (0.3 * rng.standard_normal((b, d, L))).astype(np.float32)
        A = (-np.exp(0.5 * rng.standard_normal((d, N)))).astype(np.float32)
        B = (0.5 * rng.standard_normal((b, 1, N, L))).astype(np.float32)
        C = (0.5 * rng.standard_normal((b, 1, N, L))).astype(np.float32)
        h0 = np.zeros((b, d, N), np.float32)
        u_d, dl_d = be.array(u, "bf16"), be.array(delta, "bf16")
        A_d = be.array(A, "f32")
        B_d, C_d = be.array(B, "bf16"), be.array(C, "bf16")
        h_d = be.array(h0, "f32")
        yield Case("selective_scan", f"B{b}_d{d}_L{L}_N{N}", {"B": b, "d": d, "L": L, "N": N},
                   "bf16",
                   target=lambda u_d=u_d, dl_d=dl_d, A_d=A_d, B_d=B_d, C_d=C_d, h_d=h_d:
                       tk.selective_scan(u_d, dl_d, A_d, B_d, C_d, h_d)[0],
                   baselines={}, ref=None,
                   bytes_moved=2.0 * (3 * b * d * L * 2 + 2 * b * N * L * 2))


@register("qk_norm_rope")
def qk_norm_rope_cases(be, preset, formats):
    """Fused per-head QK-RMSNorm + RoPE on packed QKV vs an mx-ops composition (per-head
    fast.rms_norm + fast.rope over reshaped heads + V concat). Qwen3-8B / gpt-oss shapes."""
    tk = be.tk()
    rng = np.random.default_rng(31)
    shapes = _pick(preset, [(512, 8, 2, 2, 128)],
                   [(512, 32, 8, 8, 128), (4096, 32, 8, 8, 128)],
                   [(512, 32, 8, 8, 128), (4096, 32, 8, 8, 128), (4096, 64, 8, 8, 64)])
    for T, hq, hk, hv, D in shapes:
        HT = hq + hk + hv
        qkv = (0.3 * rng.standard_normal((T, HT * D))).astype(np.float32)
        qw = (1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)
        kw = (1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32)
        half = D // 2
        inv = 1.0 / (10000.0 ** (np.arange(half) / half))
        ang = np.outer(np.arange(8192), inv)
        cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        pos = rng.integers(0, 8192, T).astype(np.int32)
        qkv_d = be.array(qkv, "bf16")
        qw_d, kw_d = be.array(qw, "bf16"), be.array(kw, "bf16")
        c_d, s_d = be.array(cos, "bf16"), be.array(sin, "bf16")
        p_d = be.int_array(pos)
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def composed(qkv_d=qkv_d, qw_d=qw_d, kw_d=kw_d, p_d=p_d, T=T, hq=hq, hk=hk,
                         hv=hv, D=D):
                x = qkv_d.reshape(T, hq + hk + hv, D)
                q = mx.fast.rms_norm(x[:, :hq], qw_d, 1e-6)
                k = mx.fast.rms_norm(x[:, hq:hq + hk], kw_d, 1e-6)
                # positions vary per token; fast.rope wants an offset — approximate the cost
                q = mx.fast.rope(q.transpose(1, 0, 2), D, traditional=False, base=10000.0,
                                 scale=1.0, offset=0).transpose(1, 0, 2)
                k = mx.fast.rope(k.transpose(1, 0, 2), D, traditional=False, base=10000.0,
                                 scale=1.0, offset=0).transpose(1, 0, 2)
                return mx.concatenate([q, k, x[:, hq + hk:]], axis=1).reshape(T, -1)
            baselines["mx_composed"] = composed
        yield Case("qk_norm_rope", f"T{T}_hq{hq}hk{hk}hv{hv}_D{D}",
                   {"T": T, "hq": hq, "D": D}, "bf16",
                   target=lambda qkv_d=qkv_d, qw_d=qw_d, kw_d=kw_d, c_d=c_d, s_d=s_d,
                          p_d=p_d, hq=hq, hk=hk, hv=hv:
                       tk.qk_norm_rope(qkv_d, qw_d, kw_d, c_d, s_d, p_d, hq, hk, hv),
                   baselines=baselines, ref=None,
                   bytes_moved=2.0 * T * (hq + hk + hv) * D * 2)


@register("moe_route")
def moe_route_cases(be, preset, formats):
    """DeepSeek grouped routing vs the plain top-k router (same T,E,K): the grouped kernel
    adds scoring + bias + a two-level masked_topk; success = within noise of moe_route_topk."""
    tk = be.tk()
    rng = np.random.default_rng(30)
    shapes = _pick(preset, [(512, 64, 4, 2, 4)],
                   [(512, 256, 8, 4, 8), (4096, 256, 8, 4, 8)],
                   [(512, 256, 8, 4, 8), (4096, 256, 8, 4, 8), (4096, 384, 1, 1, 8)])
    for T, E, n_group, topk_group, K in shapes:
        logits = rng.standard_normal((T, E)).astype(np.float32)
        bias = (0.1 * rng.standard_normal(E)).astype(np.float32)
        lg_d, b_d = be.array(logits, "f32"), be.array(bias, "f32")
        yield Case("moe_route", f"grouped_T{T}_E{E}_g{n_group}_k{K}",
                   {"T": T, "E": E, "g": n_group, "K": K}, "f32",
                   target=lambda lg_d=lg_d, b_d=b_d, K=K, g=n_group, tg=topk_group:
                       tk.moe_route_grouped(lg_d, K, g, tg, bias=b_d, scoring="sigmoid",
                                            routed_scaling_factor=2.5),
                   baselines={"moe_route_topk":
                              lambda lg_d=lg_d, K=K: tk.moe_route_topk(lg_d, K)},
                   ref=None, bytes_moved=2.0 * T * E * 4)


@register("moe_q")
def moe_q_cases(be, preset, formats):
    """Quantized grouped expert GEMMs vs the dense bf16 grouped GEMM. Decode-shape MoE is
    expert-weight-bandwidth-bound, so the packed/dense byte ratio bounds the speedup;
    weight-GB/s (packed bytes decoded per second) is the headline metric. E=4 keeps host
    packing tractable — per-tile kernel work is independent of E."""
    tk = be.tk()
    from tk.quant import quantize_expert_stack
    rng = np.random.default_rng(29)
    fmts = formats or _pick(preset, ["mxfp4"], ["mxfp4", "mxfp8", "q8_0"],
                            ["mxfp4", "mxfp8", "kU4", "fp8_e4m3", "q8_0", "nvfp4", "q4_K"])
    # (E, K_dim, N_out, rows): gpt-oss-ish (2880x2880) decode/prefill + a Qwen3-MoE-ish rect.
    shapes = _pick(preset, [(4, 1024, 1024, 32)],
                   [(4, 2880, 2880, 32), (4, 2880, 2880, 512)],
                   [(4, 2880, 2880, 32), (4, 2880, 2880, 512), (4, 2880, 2880, 4096),
                    (4, 2048, 768, 512)])
    for E, K_dim, N_out, rows in shapes:
        tiles = rows // 32
        eot = (np.arange(tiles, dtype=np.int32) * E // max(tiles, 1)).astype(np.int32)
        A = (0.1 * rng.standard_normal((rows, K_dim))).astype(np.float32)
        Wd = (0.1 * rng.standard_normal((E, K_dim, N_out))).astype(np.float32)
        A_d, eot_d = be.array(A, "bf16"), be.int_array(eot)
        W_dense = be.array(Wd, "bf16")
        dense = (lambda A_d=A_d, W=W_dense, eot_d=eot_d:
                 tk.moe_grouped_gemm_rect(A_d, W, eot_d))
        for fmt in fmts:
            bk, bb = BLOCK_INFO[fmt]
            if K_dim % bk or K_dim % 32:
                continue
            Wq = quantize_expert_stack(Wd, fmt)
            Wq_d = be.raw_array(Wq)
            yield Case("moe_q", f"rect_{fmt}_K{K_dim}_N{N_out}_rows{rows}",
                       {"E": E, "K": K_dim, "N": N_out, "rows": rows}, "bf16", fmt=fmt,
                       target=lambda A_d=A_d, Wq_d=Wq_d, eot_d=eot_d, f=fmt:
                           tk.moe_grouped_gemm_rect_q(A_d, Wq_d, eot_d, format=f),
                       baselines={"dense_bf16_rect": dense}, ref=None,
                       weight_bytes=tiles * N_out * (K_dim // bk) * bb,
                       flops=2.0 * rows * K_dim * N_out)
        del Wd, W_dense

    # Fused quantized SwiGLU GEMM1, gpt-oss config (swiglu_oai + expert bias).
    sw_fmts = ["mxfp4"] if formats is None else [f for f in formats if f in fmts]
    sw_shapes = _pick(preset, [(4, 1024, 512, 32)], [(4, 2880, 2880, 32), (4, 2880, 2880, 512)],
                      [(4, 2880, 2880, 32), (4, 2880, 2880, 512), (4, 2880, 2880, 4096)])
    for E, Hd, inter, rows in sw_shapes:
        tiles = rows // 32
        eot = (np.arange(tiles, dtype=np.int32) * E // max(tiles, 1)).astype(np.int32)
        A = (0.1 * rng.standard_normal((rows, Hd))).astype(np.float32)
        W1 = (0.1 * rng.standard_normal((E, Hd, 2 * inter))).astype(np.float32)
        bias = (0.1 * rng.standard_normal((E, 2 * inter))).astype(np.float32)
        A_d, eot_d = be.array(A, "bf16"), be.int_array(eot)
        W1_dense, b_d = be.array(W1, "bf16"), be.array(bias, "bf16")
        for fmt in sw_fmts:
            bk, bb = BLOCK_INFO[fmt]
            if Hd % bk:
                continue
            W1q_d = be.raw_array(quantize_expert_stack(W1, fmt))
            yield Case("moe_q", f"swiglu_oai_{fmt}_H{Hd}_I{inter}_rows{rows}",
                       {"E": E, "H": Hd, "inter": inter, "rows": rows}, "bf16", fmt=fmt,
                       target=lambda A_d=A_d, W1q_d=W1q_d, eot_d=eot_d, b_d=b_d, f=fmt:
                           tk.moe_grouped_gemm_swiglu_q(A_d, W1q_d, eot_d, format=f,
                                                        bias=b_d, act="swiglu_oai",
                                                        alpha=1.702, limit=7.0),
                       baselines={"dense_bf16_swiglu":
                                  lambda A_d=A_d, W=W1_dense, eot_d=eot_d:
                                      tk.moe_grouped_gemm_swiglu(A_d, W, eot_d)},
                       ref=None,
                       weight_bytes=tiles * 2 * inter * (Hd // bk) * bb,
                       flops=2.0 * rows * Hd * 2 * inter)
        del W1, W1_dense


@register("moe")
def moe_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(28)
    shapes = _pick(preset, [(8, 1024, 512)], [(8, 2048, 2048)],
                   [(8, 2048, 2048), (16, 4096, 4096)])
    for E, Hd, rows in shapes:
        tiles = rows // 32
        x = (0.1 * rng.standard_normal((rows, Hd))).astype(np.float32)
        W = (0.1 * rng.standard_normal((E, Hd, Hd))).astype(np.float32)
        eot = (np.arange(tiles, dtype=np.int32) * E // max(tiles, 1)).astype(np.int32)
        x_d, W_d, eot_d = be.array(x, "bf16"), be.array(W, "bf16"), be.int_array(eot)
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def per_expert(x_d=x_d, W_d=W_d, E=E, rows=rows):
                outs = []
                seg = rows // E
                for e in range(E):
                    outs.append(mx.matmul(x_d[e * seg:(e + 1) * seg], W_d[e]))
                return mx.concatenate(outs)
            baselines["mx_per_expert_loop"] = per_expert
        yield Case("moe", f"grouped_E{E}_H{Hd}_rows{rows}", {"E": E, "H": Hd, "rows": rows},
                   "bf16",
                   target=lambda x_d=x_d, W_d=W_d, eot_d=eot_d:
                       tk.moe_grouped_gemm(x_d, W_d, eot_d),
                   baselines=baselines, ref=None, flops=2.0 * rows * Hd * Hd)

    # End-to-end MoE MLP: all-GPU schedule (moe_pad_schedule + moe_gather) vs the old
    # host-side numpy schedule (the removed host round-trip is the win being measured).
    mlp_shapes = _pick(preset, [(256, 512, 512, 8, 2)], [(1024, 2048, 1024, 8, 2)],
                       [(1024, 2048, 1024, 8, 2), (4096, 2048, 1024, 16, 2)])
    for T, Hd, I, E, K in mlp_shapes:
        x = (0.1 * rng.standard_normal((T, Hd))).astype(np.float32)
        rl = rng.standard_normal((T, E)).astype(np.float32)
        W1 = (0.1 * rng.standard_normal((E, Hd, 2 * I))).astype(np.float32)
        W2 = (0.1 * rng.standard_normal((E, I, Hd))).astype(np.float32)
        x_d, rl_d = be.array(x, "bf16"), be.array(rl, "f32")
        W1_d, W2_d = be.array(W1, "bf16"), be.array(W2, "bf16")
        baselines = {}
        if be.name == "mlx":
            mx = be.mx

            def host_glued(x_d=x_d, rl_d=rl_d, W1_d=W1_d, W2_d=W2_d, E=E, K=K):
                ids, w = tk.moe_route_topk(rl_d, K)
                sidx, offsets, _ = tk.moe_permute(ids, E)
                mx.eval(sidx, offsets)                      # the host sync being removed
                so, off = np.array(sidx), np.array(offsets)
                counts = np.diff(off)
                off_pad = np.concatenate(
                    [[0], np.cumsum(((counts + 31) // 32) * 32)]).astype(np.int64)
                total_pad = int(off_pad[-1])
                eot = np.zeros(total_pad // 32, np.int32)
                tb = off_pad // 32
                for e in range(E):
                    eot[tb[e]:tb[e + 1]] = e
                padpos = np.zeros(len(so), np.int64)
                for e in range(E):
                    s, en = int(off[e]), int(off[e + 1])
                    padpos[s:en] = off_pad[e] + np.arange(en - s)
                gidx = np.zeros(total_pad, np.int64)
                gidx[padpos] = so // K
                inv = np.argsort(so)
                px = x_d[mx.array(gidx)]
                eot_d = mx.array(eot)
                h = tk.moe_grouped_gemm_swiglu(px, W1_d, eot_d)
                op = tk.moe_grouped_gemm_rect(h, W2_d, eot_d)
                return tk.moe_finalize(op, mx.array(padpos[inv].astype(np.int32)), w, K)
            baselines["host_glued_schedule"] = host_glued
        yield Case("moe", f"mlp_T{T}_H{Hd}_I{I}_E{E}_K{K}",
                   {"T": T, "H": Hd, "I": I, "E": E, "K": K}, "bf16",
                   target=lambda x_d=x_d, rl_d=rl_d, W1_d=W1_d, W2_d=W2_d, K=K:
                       tk.moe_mlp(x_d, rl_d, W1_d, W2_d, K),
                   baselines=baselines, ref=None, flops=2.0 * T * K * 3.0 * Hd * I)


@register("quant_rt")
def quant_rt_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(29)
    shapes = _pick(preset, [(4096, 1024)], [(4096, 1024), (16384, 1024)],
                   [(4096, 1024), (16384, 1024), (65536, 1024)])
    for N, Dm in shapes:
        x = (rng.standard_normal((N, Dm)) * 2.0).astype(np.float32)
        x_d = be.array(x, "f16")
        yield Case("quant_rt", f"per_tensor_fp8_N{N}_D{Dm}", {"N": N, "D": Dm}, "f16",
                   target=lambda x_d=x_d: tk.quantize_per_tensor_fp8(x_d),
                   baselines={}, ref=None, bytes_moved=3.0 * N * Dm)
        yield Case("quant_rt", f"per_token_fp8_N{N}_D{Dm}", {"N": N, "D": Dm}, "f16",
                   target=lambda x_d=x_d: tk.quantize_per_token_fp8(x_d),
                   baselines={}, ref=None, bytes_moved=3.0 * N * Dm)
        yield Case("quant_rt", f"per_group_fp8_ue8m0_N{N}_D{Dm}",
                   {"N": N, "D": Dm, "G": 128}, "f16", fmt="fp8_e4m3_ue8m0",
                   target=lambda x_d=x_d:
                       tk.quantize_per_group_fp8(x_d, group_size=128, ue8m0=True),
                   baselines={}, ref=None, bytes_moved=3.0 * N * Dm)


# --------------------------------------------------------------------------- Wave-5 kernels
# Training-side backward (numpy oracles from the standard formulas, on bf16-rounded inputs).
def _rms_bwd_ref(xb, wb, dyb, eps=1e-5):
    D = xb.shape[-1]
    rstd = 1.0 / np.sqrt((xb ** 2).mean(-1, keepdims=True) + eps)
    return rstd * (wb * dyb) - (rstd ** 3 / D) * xb * (wb * dyb * xb).sum(-1, keepdims=True)


def _ln_bwd_ref(xb, wb, dyb, eps=1e-5):
    mu = xb.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(xb.var(-1, keepdims=True) + eps)
    xhat = (xb - mu) * rstd
    dxhat = dyb * wb
    return rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))


def _gelu_bwd_ref(xb, dyb):
    k = 0.7978845608028654
    z = k * (xb + 0.044715 * xb ** 3)
    t = np.tanh(z)
    gp = 0.5 * (1.0 + t) + 0.5 * xb * (1.0 - t ** 2) * k * (1.0 + 3.0 * 0.044715 * xb ** 2)
    return dyb * gp


_NORM_BWD_SHAPES = ([(4096, 1024)], [(4096, 1024), (16384, 256)],
                    [(r, d) for r in (4096, 16384, 65536) for d in (256, 512, 1024)])


@register("rms_norm_bwd")
def rms_norm_bwd_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(210)
    for N, D in _pick(preset, *_NORM_BWD_SHAPES):
        x, w, dy = (rng.standard_normal((N, D)).astype(np.float32),
                    rng.standard_normal(D).astype(np.float32),
                    rng.standard_normal((N, D)).astype(np.float32))
        x_d, w_d, dy_d = be.array(x, "bf16"), be.array(w, "bf16"), be.array(dy, "bf16")
        xb, wb, dyb = (be.to_numpy(x_d).astype(np.float64), be.to_numpy(w_d).astype(np.float64),
                       be.to_numpy(dy_d).astype(np.float64))
        ref = _rms_bwd_ref(xb, wb, dyb)
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            baselines["mx.fast.rms_norm.vjp"] = lambda x_d=x_d, w_d=w_d, dy_d=dy_d: \
                mx.vjp(lambda t: mx.fast.rms_norm(t, w_d, 1e-5), [x_d], [dy_d])[1][0]
        yield Case("rms_norm_bwd", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, w_d=w_d, dy_d=dy_d: tk.rms_norm_backward(x_d, w_d, dy_d),
                   out_to_numpy=lambda o: be.to_numpy(o[0]),
                   baselines=baselines, ref=ref, bytes_moved=3 * N * D * 2)


@register("layernorm_bwd")
def layernorm_bwd_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(211)
    for N, D in _pick(preset, *_NORM_BWD_SHAPES):
        x, w, dy = (rng.standard_normal((N, D)).astype(np.float32),
                    rng.standard_normal(D).astype(np.float32),
                    rng.standard_normal((N, D)).astype(np.float32))
        x_d, w_d, dy_d = be.array(x, "bf16"), be.array(w, "bf16"), be.array(dy, "bf16")
        xb, wb, dyb = (be.to_numpy(x_d).astype(np.float64), be.to_numpy(w_d).astype(np.float64),
                       be.to_numpy(dy_d).astype(np.float64))
        ref = _ln_bwd_ref(xb, wb, dyb)
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            b_d = be.array(rng.standard_normal(D).astype(np.float32), "bf16")
            baselines["mx.fast.layer_norm.vjp"] = lambda x_d=x_d, w_d=w_d, b_d=b_d, dy_d=dy_d: \
                mx.vjp(lambda t: mx.fast.layer_norm(t, w_d, b_d, 1e-5), [x_d], [dy_d])[1][0]
        yield Case("layernorm_bwd", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, w_d=w_d, dy_d=dy_d: tk.layernorm_backward(x_d, w_d, dy_d),
                   out_to_numpy=lambda o: be.to_numpy(o[0]),
                   baselines=baselines, ref=ref, bytes_moved=3 * N * D * 2)


@register("gelu_bwd")
def gelu_bwd_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(212)
    shapes = _pick(preset, [(4096, 1024)], [(4096, 1024), (16384, 1024)],
                   [(r, d) for r in (4096, 16384, 65536) for d in (256, 1024)])
    for N, D in shapes:
        x, dy = rng.standard_normal((N, D)).astype(np.float32), rng.standard_normal((N, D)).astype(np.float32)
        x_d, dy_d = be.array(x, "bf16"), be.array(dy, "bf16")
        xb, dyb = be.to_numpy(x_d).astype(np.float64), be.to_numpy(dy_d).astype(np.float64)
        ref = _gelu_bwd_ref(xb, dyb)
        yield Case("gelu_bwd", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, dy_d=dy_d: tk.gelu_backward(x_d, dy_d),
                   baselines={}, ref=ref, bytes_moved=3 * N * D * 2)


# Sampling / grammar masking.
@register("min_p_sample")
def min_p_sample_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(220)
    shapes = _pick(preset, [(256, 32000)], [(256, 32000), (1024, 32000)],
                   [(256, 128256), (1024, 128256), (2048, 128256)])
    for T, V in shapes:
        lg = be.array(rng.standard_normal((T, V)).astype(np.float32), "f32")
        yield Case("min_p_sample", f"T{T}_V{V}", {"T": T, "V": V}, "f32",
                   target=lambda lg=lg: tk.min_p_sample(lg, 0.05, temperature=1.0, seed=0),
                   baselines={}, ref=None, bytes_moved=float(T * V * 4))


@register("apply_token_bitmask")
def apply_token_bitmask_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(221)
    NEG = -3.4028234663852886e38   # SMP_NEG_INF; f32 keeps it finite (bf16 would round to -inf)
    shapes = _pick(preset, [(256, 32000)], [(256, 32000), (1024, 32000)],
                   [(512, 128256), (1024, 128256)])
    for T, V in shapes:
        logits = rng.standard_normal((T, V)).astype(np.float32)
        nwords = (V + 31) // 32
        allow = rng.integers(0, 2, size=(T, V)).astype(np.uint32)
        allow[:, 0] = 1                                  # keep >=1 token per row
        ap = np.zeros((T, nwords * 32), np.uint32)
        ap[:, :V] = allow
        ap = ap.reshape(T, nwords, 32)
        packed = np.zeros((T, nwords), np.uint32)
        for j in range(32):
            packed |= ap[:, :, j] << np.uint32(j)
        lg = be.array(logits, "f32")
        bm_d = be.int_array(packed.view(np.int32))
        ref = np.where(allow.astype(bool), logits.astype(np.float64), NEG)
        yield Case("apply_token_bitmask", f"T{T}_V{V}", {"T": T, "V": V}, "f32",
                   target=lambda lg=lg, bm_d=bm_d: tk.apply_token_bitmask(lg, bm_d),
                   baselines={}, ref=ref, bytes_moved=float(T * V * 4))


@register("lm_head_q")
def lm_head_q_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(224)
    fmts = formats or ["q4_0", "q8_0"]
    shapes = _pick(preset, [(1, 32000, 4096)],
                   [(1, 32000, 4096), (8, 32000, 4096)],
                   [(1, 128256, 4096), (8, 128256, 4096)])
    for T, V, K in shapes:
        h_d = be.array((0.5 * rng.standard_normal((T, K))).astype(np.float32), "bf16")
        for fmt in fmts:
            bk, bb = BLOCK_INFO[fmt]
            if K % bk:
                continue
            wq, wdq = _packed_weight(fmt, V, K)
            wq_d = be.raw_array(wq)
            w_half = be.array(wdq, "f16")
            del wdq
            baselines = {"dense_matmul_topk": (lambda h_d=h_d, w_half=w_half:
                                               tk.lm_head_sample(h_d, w_half, mode="topk", k=8))}
            yield Case("lm_head_q", f"topk_{fmt}_T{T}_V{V}_K{K}", {"T": T, "V": V, "K": K},
                       "bf16", fmt=fmt,
                       target=lambda h_d=h_d, wq_d=wq_d, f=fmt:
                           tk.lm_head_sample(h_d, wq_d, mode="topk", k=8, format=f),
                       baselines=baselines, ref=None, weight_bytes=float(V * (K // bk) * bb))


@register("lm_head_beam")
def lm_head_beam_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(225)
    fmts = formats or ["q4_0", "q8_0"]
    # Beam decode is dominated by the projection. The baseline keeps an fp16
    # dequantized head resident, materializes logits, then uses the fused dense
    # beam primitive; the target reads packed weights and never creates logits.
    shapes = _pick(preset, [(1, 4, 8192, 1024)],
                   [(1, 4, 32000, 4096), (4, 4, 32000, 4096)],
                   [(1, 4, 32000, 4096), (4, 4, 32000, 4096),
                    (1, 8, 128256, 4096)])
    for B, BM, V, K in shapes:
        h = (0.25 * rng.standard_normal((B * BM, K))).astype(np.float32)
        cum = (0.1 * rng.standard_normal((B, BM))).astype(np.float32)
        h_d = be.array(h, "f16")
        cum_d = be.array(cum, "f32")
        for fmt in fmts:
            if fmt not in ("q4_0", "q8_0", "mxfp8", "nvfp4", "mxfp4"):
                continue
            bk, bb = BLOCK_INFO[fmt]
            wq, wdq = _packed_weight(fmt, V, K, seed=225)
            wq_d = be.raw_array(wq)
            w_half = be.array(wdq, "f16")
            del wdq
            if be.name == "mlx":
                mx = be.mx
                dense = lambda h_d=h_d, w_half=w_half: mx.matmul(h_d, w_half.T)
                padded_rows = ((B * BM + 31) // 32) * 32
                h_cols = mx.swapaxes(h_d, 0, 1)
                if padded_rows != B * BM:
                    h_cols = mx.concatenate(
                        (h_cols, mx.zeros((K, padded_rows - B * BM), dtype=h_d.dtype)),
                        axis=1)
                mx.eval(h_cols)

                def packed_matmul_beam(wq_d=wq_d, h_cols=h_cols, cum_d=cum_d,
                                       BM=BM, rows=B * BM, fmt=fmt):
                    logits = mx.swapaxes(tk.qgemm(wq_d, h_cols, format=fmt), 0, 1)[:rows]
                    return tk.beam_advance(logits, cum_d, BM)
            else:
                dense = lambda h_d=h_d, w_half=w_half: h_d @ w_half.T

            def dense_beam(dense=dense, cum_d=cum_d, BM=BM):
                return tk.beam_advance(dense(), cum_d, BM)

            yield Case(
                "lm_head_beam", f"{fmt}_B{B}_BM{BM}_V{V}_K{K}",
                {"B": B, "BM": BM, "V": V, "K": K}, "f16", fmt=fmt,
                target=lambda h_d=h_d, wq_d=wq_d, cum_d=cum_d, BM=BM, fmt=fmt:
                    tk.lm_head_beam_advance(h_d, wq_d, cum_d, BM, format=fmt),
                baselines={
                    "resident_fp16_matmul+beam": dense_beam,
                    **({"packed_qgemm32+beam": packed_matmul_beam}
                       if be.name == "mlx" else {}),
                }, ref=None,
                weight_bytes=float(V * (K // bk) * bb),
                flops=2.0 * B * BM * V * K)


# Embedding / multimodal.
@register("embedding_lookup")
def embedding_lookup_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(222)
    shapes = _pick(preset, [(1024, 4096, 32000)],
                   [(1024, 4096, 32000), (4096, 4096, 32000)],
                   [(4096, 4096, 128256), (8192, 4096, 128256)])
    for T, D, VOCAB in shapes:
        # all ids valid: table[id] + pos is unambiguous (padding-id semantics are in the test suite)
        ids = rng.integers(0, VOCAB, size=(T,)).astype(np.int32)
        tab_d = be.array((0.02 * rng.standard_normal((VOCAB, D))).astype(np.float32), "bf16")
        pos_d = be.array((0.02 * rng.standard_normal((T, D))).astype(np.float32), "bf16")
        ids_d = be.int_array(ids)
        tb, pb = be.to_numpy(tab_d).astype(np.float64), be.to_numpy(pos_d).astype(np.float64)
        ref = tb[ids] + pb
        baselines = {}
        if be.name == "mlx":
            mx = be.mx
            baselines["mx.take+add"] = lambda ids_d=ids_d, tab_d=tab_d, pos_d=pos_d: \
                tab_d[mx.maximum(ids_d, 0)] + pos_d
        else:
            baselines["torch.index+add"] = lambda ids_d=ids_d, tab_d=tab_d, pos_d=pos_d: \
                tab_d[ids_d.clamp(min=0)] + pos_d
        yield Case("embedding_lookup", f"T{T}_D{D}_V{VOCAB}", {"T": T, "D": D, "V": VOCAB}, "bf16",
                   target=lambda ids_d=ids_d, tab_d=tab_d, pos_d=pos_d:
                       tk.embedding_lookup(ids_d, tab_d, pos_table=pos_d, scale=1.0),
                   baselines=baselines, ref=ref, bytes_moved=float(2 * T * D * 2))


@register("merge_multimodal_spans")
def merge_multimodal_spans_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(223)
    shapes = _pick(preset, [(4096, 4096, 512)],
                   [(4096, 4096, 512), (8192, 4096, 1024)],
                   [(8192, 8192, 2048), (16384, 8192, 4096)])
    for T, D, M in shapes:
        text_d = be.array((0.02 * rng.standard_normal((T, D))).astype(np.float32), "bf16")
        modal_d = be.array((0.02 * rng.standard_normal((M, D))).astype(np.float32), "bf16")
        src = np.full((T,), -1, np.int32)
        pos = rng.choice(T, size=min(M, T), replace=False)
        src[pos] = rng.integers(0, M, size=len(pos)).astype(np.int32)
        src_d = be.int_array(src)
        txb, mdb = be.to_numpy(text_d).astype(np.float64), be.to_numpy(modal_d).astype(np.float64)
        ref = np.where(src[:, None] >= 0, mdb[np.clip(src, 0, M - 1)], txb)
        yield Case("merge_multimodal_spans", f"T{T}_D{D}_M{M}", {"T": T, "D": D, "M": M}, "bf16",
                   target=lambda text_d=text_d, modal_d=modal_d, src_d=src_d:
                       tk.merge_multimodal_spans(text_d, modal_d, src_d),
                   baselines={}, ref=ref, bytes_moved=float(2 * T * D * 2))


# Serving: cascade / speculative / beam KV / varlen scheduler (measured; correctness in the suites).
@register("cascade_attention")
def cascade_attention_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(225)
    shapes = _pick(preset, [(4, 16, 4, 128, 512, 512)],
                   [(8, 32, 8, 128, 1024, 1024)],
                   [(8, 32, 8, 128, 2048, 2048), (16, 32, 8, 128, 4096, 2048)])
    for B, H, H_KV, D, pfx, ctx in shapes:
        block_size = 16
        max_blocks = (ctx + block_size - 1) // block_size
        num_blocks = B * max_blocks
        q_d = be.array((0.1 * rng.standard_normal((B, H, D))).astype(np.float32), "bf16")
        pk_d = be.array((0.1 * rng.standard_normal((pfx, H_KV, D))).astype(np.float32), "bf16")
        pv_d = be.array((0.1 * rng.standard_normal((pfx, H_KV, D))).astype(np.float32), "bf16")
        kc_d = be.array((0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32), "bf16")
        vc_d = be.array((0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32), "bf16")
        bt_d = be.int_array(np.arange(B * max_blocks, dtype=np.int32).reshape(B, max_blocks))
        cl_d = be.int_array(np.full((B,), ctx, dtype=np.int32))
        scale = 1.0 / np.sqrt(D)
        # Baseline: the SAME attention with NO prefix sharing — build a full per-request paged cache
        # holding the shared prefix prepended to each request's suffix, then time paged_attention_v2.
        # cascade should win by amortizing the shared prefix's KV reads across the batch.
        full_ctx = pfx + ctx
        fmax_blocks = (full_ctx + block_size - 1) // block_size
        fnum_blocks = B * fmax_blocks
        pk_np = be.to_numpy(pk_d).astype(np.float32)   # (pfx, H_KV, D)
        pv_np = be.to_numpy(pv_d).astype(np.float32)
        fk = np.zeros((fnum_blocks, block_size, H_KV, D), np.float32)
        fv = np.zeros((fnum_blocks, block_size, H_KV, D), np.float32)
        for b in range(B):                             # prefix ++ this request's suffix, per request
            seq_k = np.concatenate([pk_np, be.to_numpy(kc_d)[b * max_blocks:(b + 1) * max_blocks]
                                    .astype(np.float32).reshape(-1, H_KV, D)[:ctx]], axis=0)
            seq_v = np.concatenate([pv_np, be.to_numpy(vc_d)[b * max_blocks:(b + 1) * max_blocks]
                                    .astype(np.float32).reshape(-1, H_KV, D)[:ctx]], axis=0)
            flat_k = fk[b * fmax_blocks:(b + 1) * fmax_blocks].reshape(-1, H_KV, D)
            flat_v = fv[b * fmax_blocks:(b + 1) * fmax_blocks].reshape(-1, H_KV, D)
            flat_k[:full_ctx] = seq_k
            flat_v[:full_ctx] = seq_v
        fk_d = be.array(fk, "bf16")
        fv_d = be.array(fv, "bf16")
        fbt_d = be.int_array(np.arange(fnum_blocks, dtype=np.int32).reshape(B, fmax_blocks))
        fcl_d = be.int_array(np.full((B,), full_ctx, dtype=np.int32))
        yield Case("cascade_attention", f"B{B}H{H}pfx{pfx}ctx{ctx}",
                   {"B": B, "H": H, "pfx": pfx, "ctx": ctx, "D": D}, "bf16",
                   target=lambda q_d=q_d, pk_d=pk_d, pv_d=pv_d, kc_d=kc_d, vc_d=vc_d, bt_d=bt_d,
                   cl_d=cl_d, scale=scale:
                       tk.cascade_attention(q_d, pk_d, pv_d, kc_d, vc_d, bt_d, cl_d,
                                            scale=float(scale), partition_size=256),
                   baselines={"tk.paged_full_prefix++suffix":
                              (lambda q_d=q_d, fk_d=fk_d, fv_d=fv_d, fbt_d=fbt_d, fcl_d=fcl_d,
                               scale=scale:
                               tk.paged_attention_v2(q_d, fk_d, fv_d, fbt_d, fcl_d,
                                                     scale=float(scale), partition_size=256))},
                   ref=None, bytes_moved=2.0 * B * (pfx + ctx) * H_KV * D * 2)


@register("spec_verify_linear")
def spec_verify_linear_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(226)
    shapes = _pick(preset, [(8, 4, 32000)], [(16, 4, 32000), (32, 8, 32000)],
                   [(32, 8, 128256), (64, 8, 128256)])
    for B, S, V in shapes:
        dt_d = be.int_array(rng.integers(0, V, size=(B, S)).astype(np.int32))
        dp = np.abs(rng.standard_normal((B, S, V))).astype(np.float32)
        dp /= dp.sum(-1, keepdims=True)
        tp = np.abs(rng.standard_normal((B, S + 1, V))).astype(np.float32)
        tp /= tp.sum(-1, keepdims=True)
        dp_d, tp_d = be.array(dp, "f32"), be.array(tp, "f32")
        bonus_d = be.int_array(rng.integers(0, V, size=(B,)).astype(np.int32))
        u_d = be.array(rng.random((B, S)).astype(np.float32), "f32")
        yield Case("spec_verify_linear", f"B{B}_S{S}_V{V}", {"B": B, "S": S, "V": V}, "f32",
                   target=lambda dt_d=dt_d, dp_d=dp_d, tp_d=tp_d, bonus_d=bonus_d, u_d=u_d:
                       tk.spec_verify_linear(dt_d, dp_d, tp_d, bonus_d, u_d, 0),
                   baselines={}, ref=None, bytes_moved=float(B * (2 * S + 1) * V * 4))


@register("beam_reorder_kv")
def beam_reorder_kv_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(227)
    shapes = _pick(preset, [(4, 4, 8, 128, 512)], [(8, 8, 8, 128, 1024)],
                   [(8, 8, 8, 128, 2048), (16, 4, 8, 128, 4096)])
    for B, BM, H_KV, D, ctx in shapes:
        block_size = 16
        max_blocks = (ctx + block_size - 1) // block_size
        nbeams = B * BM
        num_blocks = nbeams * max_blocks
        kc_d = be.array((0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32), "bf16")
        vc_d = be.array((0.1 * rng.standard_normal((num_blocks, block_size, H_KV, D))).astype(np.float32), "bf16")
        bt_d = be.int_array(np.arange(nbeams * max_blocks, dtype=np.int32).reshape(nbeams, max_blocks))
        parent_d = be.int_array(rng.integers(0, BM, size=(B, BM)).astype(np.int32))
        seq_d = be.int_array(np.full((nbeams,), ctx, np.int32))
        yield Case("beam_reorder_kv", f"B{B}BM{BM}ctx{ctx}", {"B": B, "BM": BM, "ctx": ctx, "D": D},
                   "bf16",
                   target=lambda kc_d=kc_d, vc_d=vc_d, bt_d=bt_d, parent_d=parent_d, seq_d=seq_d:
                       tk.beam_reorder_kv(kc_d, vc_d, bt_d, parent_d, seq_d),
                   baselines={}, ref=None, bytes_moved=2.0 * nbeams * ctx * H_KV * D * 2)


@register("beam_build_copy_pairs")
def beam_build_copy_pairs_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(228)
    shapes = _pick(preset, [(4, 4, 512)], [(8, 8, 1024)], [(8, 8, 2048), (16, 4, 4096)])
    for B, BM, ctx in shapes:
        block_size = 16
        max_blocks = (ctx + block_size - 1) // block_size
        nbeams = B * BM
        bt = np.arange(nbeams * max_blocks, dtype=np.int32).reshape(nbeams, max_blocks)
        parent = rng.integers(0, BM, size=(B, BM)).astype(np.int32)
        seq = np.full((nbeams,), ctx, np.int32)
        bt_d = be.int_array(bt)
        parent_d = be.int_array(parent)
        seq_d = be.int_array(seq)
        # deterministic numpy oracle: mirror the on-device (src,dst) pair layout (fixed seeds).
        pref = np.full((nbeams * max_blocks, 2), -1, np.int64)
        for gid in range(nbeams * max_blocks):
            gb, c = gid // max_blocks, gid % max_blocks
            b, k = gb // BM, gb % BM
            p = int(parent[b, k])
            if p != k:
                nblk = (int(seq[gb]) + block_size - 1) // block_size
                if c < nblk:
                    s, d = int(bt[b * BM + p, c]), int(bt[gb, c])
                    if s >= 0 and d >= 0:
                        pref[gid] = (s, d)
        yield Case("beam_build_copy_pairs", f"B{B}BM{BM}ctx{ctx}", {"B": B, "BM": BM, "ctx": ctx},
                   "int",
                   target=lambda parent_d=parent_d, bt_d=bt_d, seq_d=seq_d:
                       tk.beam_build_copy_pairs(parent_d, bt_d, seq_d, block_size),
                   out_to_numpy=lambda o: be.to_numpy(o).reshape(-1, 2),
                   baselines={}, ref=pref, bytes_moved=float(nbeams * max_blocks * 2 * 8))


@register("varlen_build_worklist")
def varlen_build_worklist_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(229)
    for B in _pick(preset, [16], [16, 64], [64, 128, 256]):
        qlens = rng.integers(1, 512, size=(B,)).astype(np.int32)
        cu = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
        max_tiles = int(sum((int(q) + 7) // 8 for q in qlens)) + 4
        cu_d = be.int_array(cu)
        # deterministic numpy oracle for the tile->seq map (output 2): each length-q sequence emits
        # ceil(q/8) tiles all tagged with its batch index, then -1-padded to max_tiles.
        ts = []
        for b in range(B):
            ts += [b] * ((int(qlens[b]) + 7) // 8)
        tsref = np.full(max_tiles, -1, np.int32)
        tsref[:len(ts)] = np.array(ts, np.int32)
        yield Case("varlen_build_worklist", f"B{B}", {"B": B, "max_tiles": max_tiles}, "int",
                   target=lambda cu_d=cu_d, mt=max_tiles: tk.varlen_build_worklist(cu_d, mt),
                   out_to_numpy=lambda o: be.to_numpy(o[2]).astype(np.int32),
                   baselines={}, ref=tsref, bytes_moved=float(max_tiles * 4))


@register("typical_p_sample")
def typical_p_sample_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(230)
    for T, V in _pick(preset, [(256, 32000)], [(256, 32000), (1024, 32000)],
                      [(256, 128256), (1024, 128256), (2048, 128256)]):
        lg = be.array(rng.standard_normal((T, V)).astype(np.float32), "f32")
        yield Case("typical_p_sample", f"T{T}_V{V}", {"T": T, "V": V}, "f32",
                   target=lambda lg=lg: tk.typical_p_sample(lg, 0.9, temperature=1.0, seed=0),
                   baselines={}, ref=None, bytes_moved=float(T * V * 4))


@register("apply_bad_words")
def apply_bad_words_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(231)
    for T, V in _pick(preset, [(256, 32000)], [(256, 32000), (1024, 32000)],
                      [(512, 128256), (1024, 128256)]):
        maxbad = 16
        logits = rng.standard_normal((T, V)).astype(np.float32)
        bad = rng.integers(0, V, size=(T, maxbad)).astype(np.int32)
        blen = rng.integers(0, maxbad + 1, size=(T,)).astype(np.int32)
        lg = be.array(logits, "f32")
        bad_d, blen_d = be.int_array(bad), be.int_array(blen)
        yield Case("apply_bad_words", f"T{T}_V{V}", {"T": T, "V": V}, "f32",
                   target=lambda lg=lg, bad_d=bad_d, blen_d=blen_d:
                       tk.apply_bad_words(lg, bad_d, blen_d),
                   baselines={}, ref=None, bytes_moved=float(T * V * 4))


@register("dropout")
def dropout_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(232)
    for N, D in _pick(preset, [(4096, 4096)], [(4096, 4096), (16384, 4096)],
                      [(16384, 4096), (65536, 4096), (16384, 11008)]):
        x_d = be.array(rng.standard_normal((N, D)).astype(np.float32), "bf16")
        yield Case("dropout", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d: tk.dropout(x_d, 0.1, 0),
                   baselines={}, ref=None, bytes_moved=float(2 * N * D * 2))


@register("adamw")
def adamw_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(233)
    for numel in _pick(preset, [4 << 20], [4 << 20, 16 << 20], [16 << 20, 64 << 20]):
        p_d = be.array(rng.standard_normal((numel,)).astype(np.float32), "f32")
        g_d = be.array(rng.standard_normal((numel,)).astype(np.float32), "f32")
        m_d = be.array(np.zeros((numel,), np.float32), "f32")
        v_d = be.array(np.zeros((numel,), np.float32), "f32")
        yield Case("adamw", f"numel{numel}", {"numel": numel}, "f32",
                   target=lambda p_d=p_d, g_d=g_d, m_d=m_d, v_d=v_d:
                       tk.adamw(p_d, g_d, m_d, v_d, step=1),
                   baselines={}, ref=None, bytes_moved=float(5 * numel * 4))


@register("adamw_masked")
def adamw_masked_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(239)
    lr, b1, b2, eps, wd, step = 1.0e-3, 0.9, 0.999, 1.0e-8, 0.05, 3
    seg_size = 256
    for numel in _pick(preset, [4 << 20], [4 << 20, 16 << 20], [16 << 20, 64 << 20]):
        p = rng.standard_normal((numel,)).astype(np.float32) * 0.1
        g = rng.standard_normal((numel,)).astype(np.float32)
        m = np.zeros((numel,), np.float32)
        v = np.zeros((numel,), np.float32)
        mask = (rng.random((numel + seg_size - 1) // seg_size) > 0.35).astype(np.uint8)
        active = np.repeat(mask.astype(bool), seg_size)[:numel]
        p_d, g_d = be.array(p, "f32"), be.array(g, "f32")
        m_d, v_d, mask_d = be.array(m, "f32"), be.array(v, "f32"), be.raw_array(mask)
        for mode in [0, 1]:
            yield Case("adamw_masked", f"mode{mode}_numel{numel}",
                       {"numel": numel, "seg_size": seg_size, "active": float(active.mean())},
                       "f32",
                       target=lambda p_d=p_d, g_d=g_d, m_d=m_d, v_d=v_d, mask_d=mask_d, mode=mode:
                           tk.adamw_masked(p_d, g_d, m_d, v_d, lr=lr, beta1=b1, beta2=b2,
                                           eps=eps, weight_decay=wd, step=step, mask=mask_d,
                                           seg_size=seg_size, mask_mode=mode),
                       out_to_numpy=lambda o: be.to_numpy(o[0]),
                       baselines={"unmasked_adamw":
                                  lambda p_d=p_d, g_d=g_d, m_d=m_d, v_d=v_d:
                                      tk.adamw(p_d, g_d, m_d, v_d, lr=lr, beta1=b1,
                                               beta2=b2, eps=eps, weight_decay=wd,
                                               step=step)},
                       ref=_adamw_masked_param_np(p, g, m, v, lr, b1, b2, eps, wd,
                                                  step, active, mode),
                       bytes_moved=float(5 * numel * 4 + mask.size))


@register("rms_norm_add")
def rms_norm_add_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(234)
    # the fused add+norm forward is register-tile (D in {256,512,768,1024})
    for N, D in _pick(preset, [(4096, 1024)], [(16384, 1024), (65536, 768)],
                      [(65536, 1024), (131072, 1024), (65536, 512)]):
        x_d = be.array(rng.standard_normal((N, D)).astype(np.float32), "bf16")
        r_d = be.array(rng.standard_normal((N, D)).astype(np.float32), "bf16")
        w_d = be.array(rng.standard_normal((D,)).astype(np.float32), "bf16")
        yield Case("rms_norm_add", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, r_d=r_d, w_d=w_d: tk.rms_norm_add(x_d, r_d, w_d),
                   baselines={}, ref=None, bytes_moved=float(3 * N * D * 2))


@register("norm_quant_block")
def norm_quant_block_cases(be, preset, formats):
    """Fused rms_norm(x+residual) -> per-128-block int8 vs the unfused chain
    (rms_norm_add -> quantize_per_group). The win is eliminating the (N,D) bf16 round-trip."""
    tk = be.tk()
    rng = np.random.default_rng(240)
    for N, D in _pick(preset, [(4096, 1024)], [(16384, 1024), (65536, 768)],
                      [(65536, 1024), (131072, 1024)]):
        x_d = be.array(rng.standard_normal((N, D)).astype(np.float32), "bf16")
        r_d = be.array(rng.standard_normal((N, D)).astype(np.float32), "bf16")
        w_d = be.array(rng.standard_normal((D,)).astype(np.float32), "bf16")
        base = {"norm_then_group_quant":
                (lambda x_d=x_d, r_d=r_d, w_d=w_d:
                 tk.quantize_per_group_int8(tk.rms_norm_add(x_d, r_d, w_d)[0], group_size=128)[0])}
        yield Case("norm_quant_block", f"N{N}_D{D}", {"N": N, "D": D}, "bf16",
                   target=lambda x_d=x_d, r_d=r_d, w_d=w_d:
                       tk.rms_norm_add_per_block(x_d, r_d, w_d, int8=True)[0],
                   baselines=base, ref=None, bytes_moved=float(3 * N * D * 2))
        yield Case("norm_quant_block", f"fp8_ue8m0_N{N}_D{D}",
                   {"N": N, "D": D, "G": 128}, "bf16", fmt="fp8_e4m3_ue8m0",
                   target=lambda x_d=x_d, r_d=r_d, w_d=w_d:
                       tk.rms_norm_add_per_block(
                           x_d, r_d, w_d, int8=False, ue8m0=True)[0],
                   baselines={}, ref=None, bytes_moved=float(3 * N * D * 2))


@register("embedding_backward")
def embedding_backward_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(235)
    # heavy id duplication (small vocab, many tokens) is the sorted path's target regime
    for T, D, V in _pick(preset, [(8192, 2048, 256)], [(8192, 2048, 256), (32768, 4096, 512)],
                         [(32768, 4096, 512), (131072, 4096, 1024)]):
        ids = rng.integers(0, V, size=(T,)).astype(np.int32)
        ids_d = be.int_array(ids)
        dY_d = be.array((0.1 * rng.standard_normal((T, D))).astype(np.float32), "bf16")
        baselines = {"tk.embedding_backward.sorted":
                     (lambda ids_d=ids_d, dY_d=dY_d, V=V:
                      tk.embedding_backward(ids_d, dY_d, V, method="sorted"))}
        yield Case("embedding_backward", f"T{T}_D{D}_V{V}", {"T": T, "D": D, "V": V}, "bf16",
                   target=lambda ids_d=ids_d, dY_d=dY_d, V=V:
                       tk.embedding_backward(ids_d, dY_d, V, method="atomic"),
                   baselines=baselines, ref=None, bytes_moved=float(T * D * 2))


@register("spec_verify_tree")
def spec_verify_tree_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(236)
    from tk import spec_build_tree_pointers
    for B, N, V in _pick(preset, [(8, 7, 32000)], [(16, 7, 32000), (32, 15, 32000)],
                         [(32, 15, 128256), (64, 31, 128256)]):
        parents = [-1] + [max(0, (c - 1) // 2) for c in range(1, N)]
        nt, ns = spec_build_tree_pointers(parents, N)
        nt_b = np.broadcast_to(nt, (B, N)).copy(); ns_b = np.broadcast_to(ns, (B, N)).copy()
        draft = rng.integers(0, V, size=(B, N - 1)).astype(np.int32)
        tp = np.abs(rng.standard_normal((B, N, V))).astype(np.float32); tp /= tp.sum(-1, keepdims=True)
        d_d, tp_d = be.int_array(draft), be.array(tp, "f32")
        nt_d, ns_d = be.int_array(nt_b), be.int_array(ns_b)
        yield Case("spec_verify_tree", f"B{B}_N{N}_V{V}", {"B": B, "N": N, "V": V}, "f32",
                   target=lambda d_d=d_d, tp_d=tp_d, nt_d=nt_d, ns_d=ns_d:
                       tk.spec_verify_tree(d_d, tp_d, nt_d, ns_d, 0),
                   baselines={}, ref=None, bytes_moved=float(B * N * V * 4))


@register("spec_compact")
def spec_compact_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(237)
    for B, S in _pick(preset, [(256, 8)], [(256, 8), (1024, 8)], [(1024, 8), (4096, 8)]):
        Sp1 = S + 1
        acc = rng.integers(0, S + 1, size=B).astype(np.int32)
        sl = rng.integers(1, 100, size=B).astype(np.int32)
        ot = np.full((B, Sp1), -1, np.int32)
        for b in range(B):
            for j in range(int(acc[b]) + 1):
                ot[b, j] = rng.integers(0, 32000)
        ot_d, acc_d, sl_d = be.int_array(ot), be.int_array(acc), be.int_array(sl)
        yield Case("spec_compact", f"B{B}_S{S}", {"B": B, "S": S}, "int",
                   target=lambda ot_d=ot_d, acc_d=acc_d, sl_d=sl_d:
                       tk.spec_compact(ot_d, acc_d, sl_d),
                   baselines={}, ref=None, bytes_moved=float(B * Sp1 * 4))


@register("build_dynamic_tree")
def build_dynamic_tree_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(238)
    for B, N in _pick(preset, [(64, 33)], [(64, 33), (128, 65)], [(128, 129), (256, 129)]):
        parents = np.full((B, N), -1, np.int32)
        for b in range(B):
            for c in range(1, N):
                parents[b, c] = rng.integers(0, c)
        p_d = be.int_array(parents)
        yield Case("build_dynamic_tree", f"B{B}_N{N}", {"B": B, "N": N}, "int",
                   target=lambda p_d=p_d: tk.build_dynamic_tree(p_d),
                   baselines={}, ref=None, bytes_moved=float(B * N * 4))


# --------------------------------------------------------------------------- focused kernel extensions
def _gelu_erf_np(x):
    """The A&S erf-GELU approximation used by the fused Metal kernels."""
    z = x * np.float32(0.7071067811865475)
    az = np.abs(z)
    t = 1.0 / (1.0 + np.float32(0.3275911) * az)
    p = (((((np.float32(1.061405429) * t - np.float32(1.453152027)) * t
            + np.float32(1.421413741)) * t - np.float32(0.284496736)) * t
          + np.float32(0.254829592)) * t)
    erf = np.copysign(1.0 - p * np.exp(-az * az), z)
    return 0.5 * x * (1.0 + erf)


def _gelu_erf_device(be, x):
    if be.name == "mlx":
        return 0.5 * x * (1.0 + be.mx.erf(x * (1.0 / math.sqrt(2.0))))
    return be.torch.nn.functional.gelu(x, approximate="none")


@register("attn_decode_bh")
def attn_decode_bh_cases(be, preset, formats):
    """Batched head-major GQA decode, including long-context priority shapes."""
    tk = be.tk()
    rng = np.random.default_rng(260)
    shapes = _pick(
        preset,
        [(1, 4, 2, 17, 32)],
        [(4, 8, 2, 256, 32), (4, 16, 4, 512, 128),
         (4, 16, 4, 1024, 128), (4, 16, 4, 2048, 128)],
        [(16, 32, 8, 4096, 128)])
    for B, Hq, Hkv, T, D in shapes:
        q = (0.2 * rng.standard_normal((B, Hq, D))).astype(np.float32)
        k = (0.2 * rng.standard_normal((B, Hkv, T, D))).astype(np.float32)
        v = (0.2 * rng.standard_normal((B, Hkv, T, D))).astype(np.float32)
        q_d, k_d, v_d = be.array(q), be.array(k), be.array(v)
        group = Hq // Hkv

        def attn_base(q_d=q_d, k_d=k_d, v_d=v_d, T=T):
            return tk.attn_decode_bh(q_d, k_d, v_d, T, use_kernel=False)

        score = np.einsum(
            "bhd,bhtd->bht", q, np.repeat(k, group, axis=1)) / math.sqrt(D)
        prob = np.exp(score - score.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        ref = np.einsum("bht,bhtd->bhd", prob, np.repeat(v, group, axis=1))
        yield Case(
            "attn_decode_bh", f"B{B}_H{Hq}_{Hkv}_T{T}_D{D}",
            {"B": B, "Hq": Hq, "Hkv": Hkv, "T": T, "D": D}, "f32",
            target=lambda q_d=q_d, k_d=k_d, v_d=v_d, T=T:
                tk.attn_decode_bh(q_d, k_d, v_d, T, use_kernel=True),
            baselines={"framework_sdpa": attn_base}, ref=ref,
            bytes_moved=float((q.size + k.size + v.size + q.size) * 4))


@register("edge_mlp")
def edge_mlp_cases(be, preset, formats):
    """Fixed factorized 512-to-256-to-7 pairwise edge MLP."""
    tk = be.tk()
    rng = np.random.default_rng(263)
    B, L = _pick(preset, (1, 8), (1, 64), (4, 128))
    hidden = (0.05 * rng.standard_normal((B, L, 256))).astype(np.float32)
    first_w = (0.04 * rng.standard_normal((256, 512))).astype(np.float32)
    first_b = (0.03 * rng.standard_normal(256)).astype(np.float32)
    second_w = (0.04 * rng.standard_normal((7, 256))).astype(np.float32)
    second_b = (0.03 * rng.standard_normal(7)).astype(np.float32)
    h_d, fw_d, fb_d = be.array(hidden), be.array(first_w), be.array(first_b)
    sw_d, sb_d = be.array(second_w), be.array(second_b)
    if be.name == "mlx":
        def edge_base():
            left = be.mx.matmul(h_d, be.mx.swapaxes(fw_d[:, :256], 0, 1))
            right = be.mx.matmul(h_d, be.mx.swapaxes(fw_d[:, 256:], 0, 1)) + fb_d
            act = _gelu_erf_device(be, left[:, :, None, :] + right[:, None, :, :])
            return be.mx.einsum("bijh,ch->bcij", act, sw_d) + sb_d[None, :, None, None]
    else:
        def edge_base():
            left = h_d @ fw_d[:, :256].T
            right = h_d @ fw_d[:, 256:].T + fb_d
            act = _gelu_erf_device(be, left[:, :, None, :] + right[:, None, :, :])
            return be.torch.einsum("bijh,ch->bcij", act, sw_d) + sb_d[None, :, None, None]
    left = hidden @ first_w[:, :256].T
    right = hidden @ first_w[:, 256:].T + first_b
    ref = np.einsum(
        "bijh,ch->bcij", _gelu_erf_np(left[:, :, None] + right[:, None]), second_w)
    ref += second_b[None, :, None, None]
    yield Case(
        "edge_mlp", f"B{B}_L{L}",
        {"B": B, "L": L, "hidden": 256, "classes": 7}, "f32",
        target=lambda: tk.edge_mlp_256x7(
            h_d, fw_d, fb_d, sw_d, sb_d, use_kernel=True),
        baselines={"materialized_pairwise_mlp": edge_base}, ref=ref,
        flops=float(B * L * L * 2 * 256 * 7))


@register("kernel_extensions")
def kernel_extension_cases(be, preset, formats):
    """Focused A/B suite for fused and specialized kernel paths.

    Hypothesis: fusion and packed-format reads should win on real decode/vision
    shapes by removing intermediate tensors or expanded weights. Small shapes
    are included to expose launch-bound regressions and guide public routing.
    """
    tk = be.tk()
    rng = np.random.default_rng(260)

    # Dynamic LayerNorm: the existing fixed-width route is the current baseline;
    # this case covers a newly supported dynamic hidden width.
    R, D = _pick(preset, (1, 320), (8, 1536), (64, 1536))
    x_d = be.array(rng.standard_normal((R, D)).astype(np.float32), "bf16")
    w_d = be.array((0.8 + 0.1 * rng.standard_normal(D)).astype(np.float32), "bf16")
    b_d = be.array((0.05 * rng.standard_normal(D)).astype(np.float32), "bf16")
    if be.name == "mlx":
        norm_base = lambda: be.mx.fast.layer_norm(x_d, w_d, b_d, 1e-5)
    else:
        norm_base = lambda: be.torch.nn.functional.layer_norm(x_d, (D,), w_d, b_d, 1e-5)
    xb = be.to_numpy(x_d).astype(np.float64)
    mean = xb.mean(-1, keepdims=True)
    ref = ((xb - mean) / np.sqrt(((xb - mean) ** 2).mean(-1, keepdims=True) + 1e-5) *
           be.to_numpy(w_d).astype(np.float64) + be.to_numpy(b_d).astype(np.float64))
    yield Case("layernorm", f"dynamic_R{R}_D{D}", {"R": R, "D": D}, "bf16",
               target=lambda: tk.layernorm(x_d, w_d, b_d, use_kernel=True),
               baselines={"framework_layer_norm": norm_base}, ref=ref,
               bytes_moved=float(2 * R * D * 2))

    # Materialized-order decode add + LayerNorm.
    R, D = _pick(preset, (1, 37), (8, 256), (64, 256))
    x_d = be.array((0.3 * rng.standard_normal((R, D))).astype(np.float32), "bf16")
    r_d = be.array((0.3 * rng.standard_normal((R, D))).astype(np.float32), "bf16")
    w_d = be.array((0.8 + 0.1 * rng.standard_normal(D)).astype(np.float32), "bf16")
    b_d = be.array((0.05 * rng.standard_normal(D)).astype(np.float32), "bf16")
    if be.name == "mlx":
        def decode_norm_base():
            summed = (x_d.astype(be.mx.float32) + r_d.astype(be.mx.float32)).astype(be.mx.bfloat16)
            return be.mx.fast.layer_norm(summed, w_d, b_d, 1e-5), summed
    else:
        def decode_norm_base():
            summed = (x_d.float() + r_d.float()).to(be.torch.bfloat16)
            return be.torch.nn.functional.layer_norm(summed, (D,), w_d, b_d, 1e-5), summed
    summed = (be.to_numpy(x_d) + be.to_numpy(r_d)).astype(np.float32)
    if be.name == "mlx":
        summed = be.to_numpy(be.array(summed, "bf16"))
    else:
        summed = be.to_numpy(be.array(summed, "bf16"))
    mu = summed.mean(-1, keepdims=True)
    ref = ((summed - mu) / np.sqrt((summed * summed).mean(-1, keepdims=True) - mu * mu + 1e-5) *
           be.to_numpy(w_d) + be.to_numpy(b_d))
    yield Case("decode_layernorm_add", f"R{R}_D{D}", {"R": R, "D": D}, "bf16",
               target=lambda: tk.decode_layernorm_add(x_d, r_d, w_d, b_d),
               out_to_numpy=lambda out: be.to_numpy(out[0]),
               baselines={"add_then_framework_layer_norm": decode_norm_base}, ref=ref,
               bytes_moved=float(4 * R * D * 2))

    # Swin's priority window is 12x12 (N=144) with D=32.
    BW, N, H = _pick(preset, (2, 16, 2), (8, 144, 4), (64, 144, 4))
    qkv = (0.12 * rng.standard_normal((BW, N, 3, H, 32))).astype(np.float32)
    relative = (0.04 * rng.standard_normal((H, N, N))).astype(np.float32)
    windows = min(BW, 4)
    mask = np.zeros((windows, N, N), np.float32)
    if windows > 1:
        mask[1::2, :, N // 2:] = -10.0
    qkv_d, relative_d, mask_d = be.array(qkv), be.array(relative), be.array(mask)
    mask_full = mask[np.arange(BW) % windows]
    def swin_base():
        return tk.swin_attn_d32(
            qkv_d, relative_d, mask_d, windows, use_kernel=False)
    q0 = qkv[:, :, 0].transpose(0, 2, 1, 3)
    k0 = qkv[:, :, 1].transpose(0, 2, 1, 3)
    v0 = qkv[:, :, 2].transpose(0, 2, 1, 3)
    score = q0 @ k0.swapaxes(-1, -2) / math.sqrt(32) + relative[None] + mask_full[:, None]
    prob = np.exp(score - score.max(-1, keepdims=True)); prob /= prob.sum(-1, keepdims=True)
    ref = (prob @ v0).transpose(0, 2, 1, 3)
    yield Case("swin_attn", f"BW{BW}_N{N}_H{H}_D32", {"BW": BW, "N": N, "H": H, "D": 32},
               "f32", target=lambda: tk.swin_attn_d32(
                   qkv_d, relative_d, mask_d, windows, use_kernel=True),
               baselines={"framework_window_sdpa": swin_base}, ref=ref,
               flops=float(4 * BW * H * N * N * 32))

    # Patch merge: real first-stage Swin resolution for quick/comprehensive.
    B, height, width, C = _pick(preset, (1, 8, 8, 32),
                                (1, 96, 96, 128), (4, 96, 96, 128))
    x_d = be.array((0.2 * rng.standard_normal((B, height * width, C))).astype(np.float32), "bf16")
    w_d = be.array((0.8 + 0.1 * rng.standard_normal(4 * C)).astype(np.float32), "bf16")
    b_d = be.array((0.05 * rng.standard_normal(4 * C)).astype(np.float32), "bf16")
    if be.name == "mlx":
        def patch_base():
            image = be.mx.reshape(x_d, (B, height, width, C))
            merged = be.mx.concatenate((image[:, 0::2, 0::2], image[:, 1::2, 0::2],
                                        image[:, 0::2, 1::2], image[:, 1::2, 1::2]), -1)
            return be.mx.fast.layer_norm(be.mx.reshape(merged, (B, -1, 4 * C)), w_d, b_d, 1e-5)
    else:
        def patch_base():
            image = x_d.reshape(B, height, width, C)
            merged = be.torch.cat((image[:, 0::2, 0::2], image[:, 1::2, 0::2],
                                   image[:, 0::2, 1::2], image[:, 1::2, 1::2]), -1)
            return be.torch.nn.functional.layer_norm(merged.reshape(B, -1, 4 * C), (4 * C,), w_d, b_d, 1e-5)
    xb = be.to_numpy(x_d).reshape(B, height, width, C)
    merged = np.concatenate((xb[:, 0::2, 0::2], xb[:, 1::2, 0::2],
                             xb[:, 0::2, 1::2], xb[:, 1::2, 1::2]), -1).reshape(B, -1, 4 * C)
    mu = merged.mean(-1, keepdims=True)
    ref = ((merged - mu) / np.sqrt(((merged - mu) ** 2).mean(-1, keepdims=True) + 1e-5) *
           be.to_numpy(w_d) + be.to_numpy(b_d))
    yield Case("patch_merge", f"B{B}_{height}x{width}_C{C}",
               {"B": B, "H": height, "W": width, "C": C}, "bf16",
               target=lambda: tk.patch_merge_layernorm(x_d, w_d, b_d, height, width),
               baselines={"gather_concat_layer_norm": patch_base}, ref=ref,
               bytes_moved=float(2 * B * height * width * C * 2))

    # Decode linear (dense + q8_0) and tiled Flux erf-GELU.
    B, K, N = _pick(preset, (1, 65, 37), (1, 256, 256), (16, 256, 256))
    x = (0.1 * rng.standard_normal((B, K))).astype(np.float32)
    weight = (0.08 * rng.standard_normal((N, K))).astype(np.float32)
    bias = (0.03 * rng.standard_normal(N)).astype(np.float32)
    x_d, weight_d, bias_d = be.array(x), be.array(weight), be.array(bias)
    dense_base = lambda: _gelu_erf_device(be, x_d @ (weight_d.T if be.name == "torch" else be.mx.swapaxes(weight_d, 0, 1)) + bias_d)
    ref = _gelu_erf_np(x @ weight.T + bias)
    yield Case("decode_linear", f"gelu_B{B}_K{K}_N{N}", {"B": B, "K": K, "N": N}, "f32",
               target=lambda: tk.decode_linear(x_d, weight_d, bias_d, gelu=True),
               baselines={"matmul_bias_erf_gelu": dense_base}, ref=ref,
               flops=float(2 * B * K * N))

    Kq = ((K + 31) // 32) * 32
    xq = (0.1 * rng.standard_normal((B, Kq))).astype(np.float32)
    wq, wdq = _packed_weight("q8_0", N, Kq, seed=261)
    bias = (0.03 * rng.standard_normal(N)).astype(np.float32)
    residual = (0.03 * rng.standard_normal((B, N))).astype(np.float32)
    xq_d, wq_d, wdq_d = be.array(xq), be.raw_array(wq), be.array(wdq)
    bias_d, residual_d = be.array(bias), be.array(residual)
    q8_base = lambda: _gelu_erf_device(be, xq_d @ (wdq_d.T if be.name == "torch" else be.mx.swapaxes(wdq_d, 0, 1)) + bias_d) + residual_d
    ref = _gelu_erf_np(xq @ wdq.T + bias) + residual
    yield Case("decode_linear_q8", f"B{B}_K{Kq}_N{N}", {"B": B, "K": Kq, "N": N}, "f32", fmt="q8_0",
               target=lambda: tk.decode_linear_q8(xq_d, wq_d, bias_d, residual_d, gelu=True),
               baselines={"dequantized_matmul_epilogue": q8_base}, ref=ref,
               weight_bytes=float(wq.nbytes), flops=float(2 * B * Kq * N))

    FN, FK, FM = _pick(preset, (32, 16, 32), (128, 64, 128), (256, 256, 256))
    fa = (0.08 * rng.standard_normal((FN, FK))).astype(np.float32)
    fw = (0.08 * rng.standard_normal((FK, FM))).astype(np.float32)
    fb = (0.03 * rng.standard_normal(FM)).astype(np.float32)
    fa_d, fw_d, fb_d = be.array(fa), be.array(fw), be.array(fb)
    yield Case("flux_gelu_erf", f"N{FN}_K{FK}_M{FM}", {"N": FN, "K": FK, "M": FM}, "f32",
               target=lambda: tk.flux_gelu_erf(fa_d, fw_d, fb_d),
               baselines={"matmul_bias_erf_gelu": lambda: _gelu_erf_device(be, fa_d @ fw_d + fb_d)},
               ref=_gelu_erf_np(fa @ fw + fb), flops=float(2 * FN * FK * FM))

    # Packed row gather for all three audited GGUF formats.
    gather_formats = [f for f in ("q4_0", "q8_0", "q6_K") if formats is None or f in formats]
    for fmt in gather_formats:
        rows, columns, tokens = _pick(preset, (64, 256, 8),
                                     (4096, 1536 if fmt == "q6_K" else 256, 256),
                                     (32768, 1536 if fmt == "q6_K" else 256, 2048))
        packed, dequantized = _packed_weight(fmt, rows, columns, seed=262)
        ids = rng.integers(0, rows, size=tokens, dtype=np.int32)
        packed_d, ids_d = be.raw_array(packed), be.int_array(ids)
        # Match the kernel's one-final-rounding contract: the expanded
        # framework baseline keeps the table in fp32 and casts only after the
        # gathered row has been scaled.
        table_d = be.array(dequantized)
        scale = float(np.sqrt(1536.0))
        if be.name == "mlx":
            gather_base = lambda packed_d=packed_d, ids_d=ids_d, table_d=table_d: (
                be.mx.take(table_d, ids_d, axis=0) * scale).astype(be.mx.float16)
        else:
            gather_base = lambda packed_d=packed_d, ids_d=ids_d, table_d=table_d: (
                table_d.index_select(0, ids_d.long()) * scale).to(be.torch.float16)
        ref = (dequantized[ids] * np.float32(scale)).astype(np.float16)
        yield Case("dequant_gather", f"{fmt}_R{rows}_D{columns}_T{tokens}",
                   {"rows": rows, "D": columns, "tokens": tokens}, "f16", fmt=fmt,
                   target=lambda packed_d=packed_d, ids_d=ids_d, fmt=fmt:
                       tk.dequant_gather(packed_d, ids_d, fmt, scale),
                   baselines={"predequantized_framework_gather": gather_base}, ref=ref,
                   weight_bytes=float(packed.nbytes))

    # Packed fp32 q4_0/q6_K GEMV specializations.
    qfmts = [f for f in ("q4_0", "q6_K") if formats is None or f in formats]
    for fmt in qfmts:
        N, K = _pick(preset, (67, 512),
                     (6144 if fmt == "q4_0" else 1536, 1536),
                     (12288 if fmt == "q4_0" else 6144, 1536))
        packed, dequantized = _packed_weight(fmt, N, K, seed=263)
        x = (0.2 * rng.standard_normal((K, 1))).astype(np.float32)
        packed_d, x_d, dequant_d = be.raw_array(packed), be.array(x), be.array(dequantized)
        gemv_base = lambda x_d=x_d, dequant_d=dequant_d: dequant_d @ x_d
        yield Case("qgemv", f"f32_{fmt}_N{N}_K{K}", {"N": N, "K": K}, "f32", fmt=fmt,
                   target=lambda packed_d=packed_d, x_d=x_d, fmt=fmt: tk.qgemv(packed_d, x_d, fmt),
                   baselines={"predequantized_f32_matmul": gemv_base},
                   ref=dequantized.astype(np.float64) @ x.astype(np.float64),
                   weight_bytes=float(packed.nbytes), flops=float(2 * N * K))

    # Q6_K two-stage vocabulary projection/argmax. Quick uses a substantial
    # vocabulary while keeping the focused run practical on developer laptops.
    V, K = _pick(preset, (1025, 512), (32768, 1536), (131072, 1536))
    packed, dequantized = _packed_weight("q6_K", V, K, seed=264)
    h = (0.15 * rng.standard_normal((1, K))).astype(np.float32)
    packed_d, h_d, dequant_d = be.raw_array(packed), be.array(h), be.array(dequantized)
    if be.name == "mlx":
        lm_base = lambda: be.mx.argmax(be.mx.matmul(h_d, be.mx.swapaxes(dequant_d, 0, 1)), axis=-1)
        lm_packed_base = lambda: be.mx.reshape(
            be.mx.argmax(tk.qgemv(packed_d, be.mx.swapaxes(h_d, 0, 1), "q6_K")[:, 0], axis=0), (1,))
    else:
        lm_base = lambda: be.torch.argmax(h_d @ dequant_d.T, dim=-1).to(be.torch.int32)
        lm_packed_base = lambda: be.torch.argmax(
            tk.qgemv(packed_d, h_d.T, "q6_K")[:, 0], dim=0, keepdim=True).to(be.torch.int32)
    yield Case("lm_head", f"q6_K_argmax_V{V}_K{K}", {"T": 1, "V": V, "K": K}, "f32", fmt="q6_K",
               target=lambda: tk.lm_head_sample(
                   h_d, packed_d, mode="argmax", format="q6_K", fused=True),
               baselines={"predequantized_matmul_argmax": lm_base,
                          "packed_qgemv_argmax": lm_packed_base},
               ref=(h.astype(np.float64) @ dequantized.astype(np.float64).T).argmax(-1),
               weight_bytes=float(packed.nbytes), flops=float(2 * V * K))

    # Grammar-constrained dense LM head with a representative V=357, K=256 shape.
    T, V, K = _pick(preset, (1, 67, 31), (8, 357, 256), (64, 357, 256))
    h = (0.12 * rng.standard_normal((T, K))).astype(np.float32)
    weight = (0.12 * rng.standard_normal((V, K))).astype(np.float32)
    bias = (0.03 * rng.standard_normal(V)).astype(np.float32)
    forbidden = (rng.random((V, V)) < 0.15).astype(np.uint8)
    previous = rng.integers(0, V, size=T, dtype=np.int32)
    forbidden[previous, 0] = 0
    h_d, weight_d, bias_d = be.array(h), be.array(weight), be.array(bias)
    forbidden_d, previous_d = be.raw_array(forbidden), be.int_array(previous)
    if be.name == "mlx":
        def constrained_base():
            logits = be.mx.matmul(h_d, be.mx.swapaxes(weight_d, 0, 1)) + bias_d
            allowed = be.mx.take(forbidden_d, previous_d, axis=0) == 0
            token = be.mx.argmax(be.mx.where(allowed, logits, -float("inf")), axis=-1)
            logprob = be.mx.take_along_axis(logits - be.mx.logsumexp(logits, axis=-1, keepdims=True),
                                            token[:, None], axis=-1)[:, 0]
            return token, logprob
    else:
        def constrained_base():
            logits = h_d @ weight_d.T + bias_d
            allowed = forbidden_d.index_select(0, previous_d.long()) == 0
            token = be.torch.argmax(be.torch.where(allowed, logits, -float("inf")), dim=-1)
            logprob = (logits - be.torch.logsumexp(logits, dim=-1, keepdim=True)).gather(1, token[:, None])[:, 0]
            return token.to(be.torch.int32), logprob
    logits = h @ weight.T + bias
    ref = np.where(forbidden[previous] == 0, logits, -np.inf).argmax(-1)
    yield Case("lm_head_constrained", f"T{T}_V{V}_K{K}", {"T": T, "V": V, "K": K}, "f32",
               target=lambda: tk.lm_head_constrained(h_d, weight_d, forbidden_d, previous_d, bias=bias_d),
               out_to_numpy=lambda out: be.to_numpy(out[0]),
               baselines={"matmul_logsoftmax_mask_argmax": constrained_base}, ref=ref,
               flops=float(2 * T * V * K))


# --------------------------------------------------------------------------- composable decode kernels
def _device_transpose(be, value):
    return value.T if be.name == "torch" else be.mx.swapaxes(value, -1, -2)


def _device_silu(be, value):
    if be.name == "torch":
        return be.torch.nn.functional.silu(value)
    return value * be.mx.sigmoid(value)


@register("quantized_embedding")
def quantized_embedding_cases(be, preset, formats):
    """Packed lookup candidate versus a materialized fp32 table gather."""
    tk = be.tk()
    rng = np.random.default_rng(301)
    shapes = _pick(
        preset, [(1024, 256, 32)],
        [(8192, 1024, 1), (8192, 1024, 256)],
        [(32768, 1536, 1), (32768, 1536, 256), (32768, 4096, 2048)])
    fmt = "q4_0" if formats is None else next((f for f in formats if f in BLOCK_INFO), "q4_0")
    for rows, dimension, tokens in shapes:
        block_k = BLOCK_INFO[fmt][0]
        dimension = ((dimension + block_k - 1) // block_k) * block_k
        packed, dequantized = _packed_weight(fmt, rows, dimension, seed=301)
        ids = rng.integers(0, rows, size=tokens, dtype=np.int32)
        add = (0.02 * rng.standard_normal((tokens, dimension))).astype(np.float32)
        packed_d, ids_d = be.raw_array(packed), be.int_array(ids)
        table_d, add_d = be.array(dequantized), be.array(add)
        if be.name == "mlx":
            baseline = lambda: be.mx.take(table_d, ids_d, axis=0) + add_d
        else:
            baseline = lambda: table_d.index_select(0, ids_d.long()) + add_d
        ref = dequantized[ids] + add
        yield Case(
            "quantized_embedding", f"{fmt}_R{rows}_D{dimension}_T{tokens}",
            {"rows": rows, "D": dimension, "tokens": tokens}, "f32", fmt=fmt,
            target=lambda packed_d=packed_d, ids_d=ids_d, add_d=add_d, fmt=fmt:
                tk.quantized_embedding(
                    packed_d, ids_d, fmt, add=add_d, output_dtype="float32"),
            baselines={"predequantized_gather_add": baseline}, ref=ref,
            bytes_moved=float(packed.nbytes + ids.nbytes + add.nbytes + ref.nbytes),
            weight_bytes=float(packed.nbytes))


@register("quantized_embedding_bag")
def quantized_embedding_bag_cases(be, preset, formats):
    """Packed CSR reduction candidate versus gather/materialize/reduce."""
    tk = be.tk()
    rng = np.random.default_rng(307)
    shapes = _pick(
        preset, [(1024, 256, 16, 4)],
        [(8192, 1024, 128, 8), (8192, 1024, 32, 32)],
        [(32768, 1536, 256, 8), (32768, 4096, 512, 16)])
    fmt = "q4_0" if formats is None else next((f for f in formats if f in BLOCK_INFO), "q4_0")
    for rows, dimension, bags, bag_size in shapes:
        block_k = BLOCK_INFO[fmt][0]
        dimension = ((dimension + block_k - 1) // block_k) * block_k
        packed, dequantized = _packed_weight(fmt, rows, dimension, seed=307)
        ids = rng.integers(0, rows, size=bags * bag_size, dtype=np.int32)
        offsets = np.arange(0, ids.size + 1, bag_size, dtype=np.int32)
        sample_weights = (0.5 + rng.random(ids.size)).astype(np.float32)
        packed_d, ids_d = be.raw_array(packed), be.int_array(ids)
        offsets_d, weights_d = be.int_array(offsets), be.array(sample_weights)
        table_d = be.array(dequantized)
        if be.name == "mlx":
            def baseline():
                gathered = be.mx.take(table_d, ids_d, axis=0)
                weighted = gathered * weights_d[:, None]
                return be.mx.mean(be.mx.reshape(weighted, (bags, bag_size, dimension)), axis=1)
        else:
            def baseline():
                gathered = table_d.index_select(0, ids_d.long())
                weighted = gathered * weights_d[:, None]
                return weighted.reshape(bags, bag_size, dimension).mean(1)
        ref = (dequantized[ids] * sample_weights[:, None]).reshape(
            bags, bag_size, dimension).mean(1)
        yield Case(
            "quantized_embedding_bag",
            f"{fmt}_R{rows}_D{dimension}_B{bags}_L{bag_size}",
            {"rows": rows, "D": dimension, "bags": bags, "bag_size": bag_size},
            "f32", fmt=fmt,
            target=lambda: tk.quantized_embedding_bag(
                packed_d, ids_d, offsets_d, fmt, sample_weights=weights_d,
                mode="mean", output_dtype="float32"),
            baselines={"predequantized_gather_weight_mean": baseline}, ref=ref,
            weight_bytes=float(packed.nbytes))


@register("decode_linear_epilogue")
def decode_linear_epilogue_cases(be, preset, formats):
    """One-dispatch projection/SiLU/residual versus decomposed framework ops."""
    tk = be.tk()
    rng = np.random.default_rng(311)
    shapes = _pick(preset, [(1, 256, 256)], [(1, 1536, 4096)],
                   [(1, 4096, 4096), (4, 4096, 11008)])
    for batch, hidden, output in shapes:
        x = (0.08 * rng.standard_normal((batch, hidden))).astype(np.float32)
        bias = (0.02 * rng.standard_normal(output)).astype(np.float32)
        residual = (0.03 * rng.standard_normal((batch, output))).astype(np.float32)
        x_d, bias_d, residual_d = be.array(x), be.array(bias), be.array(residual)
        packed_formats = [f for f in ("q4_0", "q8_0", "q6_K", "mxfp8", "nvfp4", "mxfp4")
                          if formats is None or f in formats]
        for fmt in [None, *packed_formats]:
            if fmt is None:
                weight = (0.06 * rng.standard_normal((output, hidden))).astype(np.float32)
                weight_d, weight_ref = be.array(weight), weight
                weight_bytes = float(weight.nbytes)
                variant = "dense"
            else:
                packed, weight_ref = _packed_weight(fmt, output, hidden, seed=313)
                weight_d = be.raw_array(packed)
                weight_bytes = float(packed.nbytes)
                variant = fmt
            weight_ref_d = be.array(weight_ref)
            baseline = lambda weight_ref_d=weight_ref_d: (
                _device_silu(be, x_d @ _device_transpose(be, weight_ref_d) + bias_d) + residual_d)
            linear = x @ weight_ref.T + bias
            ref = linear / (1.0 + np.exp(-linear)) + residual
            yield Case(
                "decode_linear_epilogue",
                f"{variant}_B{batch}_K{hidden}_N{output}",
                {"B": batch, "K": hidden, "N": output}, "f32", fmt=fmt,
                target=lambda weight_d=weight_d, fmt=fmt: tk.decode_linear_epilogue(
                    x_d, weight_d, bias_d, residual_d, activation="silu", format=fmt,
                    use_kernel=True),
                baselines={"matmul_bias_silu_residual": baseline}, ref=ref,
                weight_bytes=weight_bytes, flops=float(2 * batch * hidden * output))


@register("decode_swiglu")
def decode_swiglu_cases(be, preset, formats):
    """Fused dual projection versus two materialized matmuls."""
    tk = be.tk()
    rng = np.random.default_rng(317)
    shapes = _pick(preset, [(1, 256, 256)], [(1, 1536, 4096)],
                   [(1, 4096, 11008)])
    for batch, hidden, output in shapes:
        x = (0.07 * rng.standard_normal((batch, hidden))).astype(np.float32)
        x_d = be.array(x)
        packed_formats = [f for f in ("q4_0", "q8_0", "q6_K", "mxfp8", "nvfp4", "mxfp4")
                          if formats is None or f in formats]
        for fmt in [None, *packed_formats]:
            if fmt is None:
                gate = (0.05 * rng.standard_normal((output, hidden))).astype(np.float32)
                up = (0.05 * rng.standard_normal((output, hidden))).astype(np.float32)
                gate_d, up_d = be.array(gate), be.array(up)
                weight_bytes = float(gate.nbytes + up.nbytes)
                variant = "dense"
            else:
                gate_q, gate = _packed_weight(fmt, output, hidden, seed=319)
                up_q, up = _packed_weight(fmt, output, hidden, seed=321)
                gate_d, up_d = be.raw_array(gate_q), be.raw_array(up_q)
                weight_bytes = float(gate_q.nbytes + up_q.nbytes)
                variant = fmt
            gate_ref_d, up_ref_d = be.array(gate), be.array(up)
            baseline = lambda gate_ref_d=gate_ref_d, up_ref_d=up_ref_d: (
                _device_silu(be, x_d @ _device_transpose(be, gate_ref_d)) *
                (x_d @ _device_transpose(be, up_ref_d)))
            gate_value = x @ gate.T
            ref = gate_value / (1.0 + np.exp(-gate_value)) * (x @ up.T)
            yield Case(
                "decode_swiglu", f"{variant}_B{batch}_K{hidden}_N{output}",
                {"B": batch, "K": hidden, "N": output}, "f32", fmt=fmt,
                target=lambda gate_d=gate_d, up_d=up_d, fmt=fmt: tk.decode_swiglu(
                    x_d, gate_d, up_d, format=fmt, use_kernel=True),
                baselines={"two_matmuls_silu_mul": baseline}, ref=ref,
                weight_bytes=weight_bytes, flops=float(4 * batch * hidden * output))


def _allow_mask(rows, vocab):
    packed = np.zeros((len(rows), (vocab + 31) // 32), dtype=np.uint32)
    boolean = np.zeros((len(rows), vocab), dtype=bool)
    for token, values in enumerate(rows):
        boolean[token, values] = True
        for value in values:
            packed[token, value // 32] |= np.uint32(1 << (value % 32))
    return packed.view(np.int32), boolean


@register("lm_head_masked")
def lm_head_masked_cases(be, preset, formats):
    """Packed mask projection versus full logits plus mask/top-k/logsumexp."""
    tk = be.tk()
    rng = np.random.default_rng(331)
    shapes = _pick(preset, [(1, 1025, 256, 32)],
                   [(1, 8192, 1024, 256), (8, 8192, 1024, 64)],
                   [(1, 32768, 1536, 512), (16, 32768, 1536, 128)])
    topk = 4
    fmt = "q4_0" if formats is None else next(
        (f for f in formats if f in ("q4_0", "q8_0", "q6_K", "mxfp8", "nvfp4", "mxfp4")), "q4_0")
    for tokens, vocab, hidden, legal in shapes:
        packed, weight = _packed_weight(fmt, vocab, hidden, seed=331)
        h = (0.08 * rng.standard_normal((tokens, hidden))).astype(np.float32)
        bias = (0.02 * rng.standard_normal(vocab)).astype(np.float32)
        rows = [np.sort(rng.choice(vocab, size=legal, replace=False)).astype(np.int32)
                for _ in range(tokens)]
        allow_words, allow_bool = _allow_mask(rows, vocab)
        h_d, packed_d, weight_d = be.array(h), be.raw_array(packed), be.array(weight)
        bias_d = be.array(bias); allow_d = be.int_array(allow_words)
        allow_bool_d = be.int_array(allow_bool)
        if be.name == "mlx":
            def baseline():
                logits = h_d @ be.mx.swapaxes(weight_d, 0, 1) + bias_d
                masked = be.mx.where(allow_bool_d, logits, -float("inf"))
                ids = be.mx.argpartition(-masked, topk - 1, axis=-1)[:, :topk]
                selected = be.mx.take_along_axis(masked, ids, axis=-1)
                return ids, selected - be.mx.logsumexp(masked, axis=-1, keepdims=True)
        else:
            def baseline():
                logits = h_d @ weight_d.T + bias_d
                masked = be.torch.where(allow_bool_d.bool(), logits, -float("inf"))
                selected, ids = be.torch.topk(masked, topk, dim=-1)
                return ids.to(be.torch.int32), selected - be.torch.logsumexp(
                    masked, dim=-1, keepdim=True)
        logits = h @ weight.T + bias
        ref = np.stack([values[np.lexsort((values, -logits[token, values]))[:topk]]
                        for token, values in enumerate(rows)])
        yield Case(
            "lm_head_masked", f"{fmt}_T{tokens}_V{vocab}_K{hidden}_L{legal}",
            {"T": tokens, "V": vocab, "K": hidden, "legal": legal}, "f32", fmt=fmt,
            target=lambda: tk.lm_head_masked(
                h_d, packed_d, allow_d, bias=bias_d, format=fmt, topk=topk),
            out_to_numpy=lambda out: be.to_numpy(out[0]),
            baselines={"full_matmul_mask_topk_logsoftmax": baseline}, ref=ref,
            weight_bytes=float(packed.nbytes), flops=float(2 * tokens * vocab * hidden))


@register("lm_head_candidates")
def lm_head_candidates_cases(be, preset, formats):
    """CSR candidate projection versus gathered dense candidate weights."""
    tk = be.tk()
    rng = np.random.default_rng(337)
    shapes = _pick(preset, [(1, 1025, 256, 32)],
                   [(1, 8192, 1024, 256), (8, 8192, 1024, 64)],
                   [(16, 32768, 1536, 512)])
    topk = 4
    fmt = "q4_0" if formats is None else next(
        (f for f in formats if f in ("q4_0", "q8_0", "q6_K", "mxfp8", "nvfp4", "mxfp4")), "q4_0")
    for tokens, vocab, hidden, candidates in shapes:
        packed, weight = _packed_weight(fmt, vocab, hidden, seed=337)
        h = (0.08 * rng.standard_normal((tokens, hidden))).astype(np.float32)
        bias = (0.02 * rng.standard_normal(vocab)).astype(np.float32)
        candidate_matrix = np.stack([
            rng.choice(vocab, size=candidates, replace=False).astype(np.int32)
            for _ in range(tokens)])
        candidate_ids = candidate_matrix.reshape(-1)
        offsets = np.arange(0, candidate_ids.size + 1, candidates, dtype=np.int32)
        h_d, packed_d, weight_d = be.array(h), be.raw_array(packed), be.array(weight)
        ids_d, offsets_d, bias_d = be.int_array(candidate_ids), be.int_array(offsets), be.array(bias)
        matrix_d = be.int_array(candidate_matrix)
        if be.name == "mlx":
            def baseline():
                rows = be.mx.take(weight_d, matrix_d, axis=0)
                logits = be.mx.einsum("tk,tck->tc", h_d, rows) + be.mx.take(bias_d, matrix_d)
                ids = be.mx.argpartition(-logits, topk - 1, axis=-1)[:, :topk]
                selected = be.mx.take_along_axis(logits, ids, axis=-1)
                return be.mx.take_along_axis(matrix_d, ids, axis=-1), (
                    selected - be.mx.logsumexp(logits, axis=-1, keepdims=True))
        else:
            def baseline():
                rows = weight_d.index_select(0, matrix_d.reshape(-1).long()).reshape(
                    tokens, candidates, hidden)
                logits = be.torch.einsum("tk,tck->tc", h_d, rows) + bias_d[matrix_d.long()]
                selected, local = be.torch.topk(logits, topk, dim=-1)
                return matrix_d.gather(1, local).to(be.torch.int32), (
                    selected - be.torch.logsumexp(logits, dim=-1, keepdim=True))
        logits = h @ weight.T + bias
        ref = np.stack([
            row[np.lexsort((row, -logits[token, row]))[:topk]]
            for token, row in enumerate(candidate_matrix)])
        yield Case(
            "lm_head_candidates", f"{fmt}_T{tokens}_V{vocab}_K{hidden}_C{candidates}",
            {"T": tokens, "V": vocab, "K": hidden, "candidates": candidates},
            "f32", fmt=fmt,
            target=lambda: tk.lm_head_candidates(
                h_d, packed_d, ids_d, offsets_d, bias=bias_d, format=fmt, topk=topk),
            out_to_numpy=lambda out: be.to_numpy(out[0]),
            baselines={"gather_dense_rows_einsum_topk": baseline}, ref=ref,
            weight_bytes=float(packed.nbytes),
            flops=float(2 * tokens * candidates * hidden))


@register("space_to_depth_norm_linear")
def space_to_depth_norm_linear_cases(be, preset, formats):
    """Fused gather/norm/projection versus a materialized framework chain."""
    tk = be.tk()
    rng = np.random.default_rng(347)
    shapes = _pick(preset, [(1, 8, 8, 32, 64, 2)],
                   [(1, 48, 48, 128, 512, 2), (1, 32, 32, 64, 256, 4)],
                   [(4, 96, 96, 128, 512, 2), (2, 64, 64, 128, 1024, 4)])
    for batch, height, width, channels, output, block in shapes:
        dimension = block * block * channels
        x = (0.1 * rng.standard_normal((batch, height * width, channels))).astype(np.float32)
        norm_weight = (0.8 + 0.1 * rng.standard_normal(dimension)).astype(np.float32)
        norm_bias = (0.02 * rng.standard_normal(dimension)).astype(np.float32)
        projection = (0.04 * rng.standard_normal((output, dimension))).astype(np.float32)
        projection_bias = (0.02 * rng.standard_normal(output)).astype(np.float32)
        x_d, nw_d, nb_d, pw_d, pb_d = [be.array(value) for value in
                                       (x, norm_weight, norm_bias, projection, projection_bias)]
        if be.name == "mlx":
            def baseline():
                image = be.mx.reshape(x_d, (batch, height, width, channels))
                merged = be.mx.concatenate(
                    [image[:, dy::block, dx::block] for dy in range(block) for dx in range(block)],
                    axis=-1)
                merged = be.mx.reshape(merged, (batch, -1, dimension))
                normalized = be.mx.fast.layer_norm(merged, nw_d, nb_d, 1e-5)
                return normalized @ be.mx.swapaxes(pw_d, 0, 1) + pb_d
        else:
            def baseline():
                image = x_d.reshape(batch, height, width, channels)
                merged = be.torch.cat(
                    [image[:, dy::block, dx::block] for dy in range(block) for dx in range(block)],
                    dim=-1).reshape(batch, -1, dimension)
                normalized = be.torch.nn.functional.layer_norm(
                    merged, (dimension,), nw_d, nb_d, 1e-5)
                return normalized @ pw_d.T + pb_d
        image = x.reshape(batch, height, width, channels)
        merged = np.concatenate(
            [image[:, dy::block, dx::block] for dy in range(block) for dx in range(block)],
            axis=-1).reshape(batch, -1, dimension)
        mean = merged.mean(-1, keepdims=True)
        variance = (merged * merged).mean(-1, keepdims=True) - mean * mean
        normalized = (merged - mean) / np.sqrt(np.maximum(variance, 0) + 1e-5)
        ref = (normalized * norm_weight + norm_bias) @ projection.T + projection_bias
        yield Case(
            "space_to_depth_norm_linear",
            f"B{batch}_{height}x{width}_C{channels}_O{output}_S{block}",
            {"B": batch, "H": height, "W": width, "C": channels,
             "O": output, "block": block}, "f32",
            target=lambda: tk.space_to_depth_norm_linear(
                x_d, nw_d, pw_d, height, width, norm_bias=nb_d,
                projection_bias=pb_d, block_size=block, use_kernel=True),
            baselines={"space_to_depth_layernorm_matmul": baseline}, ref=ref,
            flops=float(2 * batch * (height // block) * (width // block) * dimension * output))


@register("decode_cache_attention")
def decode_cache_attention_cases(be, preset, formats):
    """Fused functional cache step versus pre-updated framework SDPA."""
    tk = be.tk()
    rng = np.random.default_rng(353)
    shapes = _pick(preset, [(1, 4, 2, 64, 64)],
                   [(4, 16, 4, 512, 128), (4, 16, 4, 1024, 128),
                    (4, 16, 4, 2048, 128)],
                   [(16, 32, 8, 4096, 128)])
    for batch, heads_q, heads_kv, context, dimension in shapes:
        cache_length = context + 16
        q = (0.1 * rng.standard_normal((batch, heads_q, dimension))).astype(np.float32)
        new_k = (0.1 * rng.standard_normal((batch, heads_kv, dimension))).astype(np.float32)
        new_v = (0.1 * rng.standard_normal((batch, heads_kv, dimension))).astype(np.float32)
        key_cache = (0.1 * rng.standard_normal(
            (batch, heads_kv, cache_length, dimension))).astype(np.float32)
        value_cache = (0.1 * rng.standard_normal(
            (batch, heads_kv, cache_length, dimension))).astype(np.float32)
        positions = np.full(batch, context, dtype=np.int32)
        contexts = np.full(batch, context, dtype=np.int32)
        angles = (np.arange(context + 1, dtype=np.float32)[:, None] + 1) * (
            np.arange(dimension // 2, dtype=np.float32)[None, :] + 1) * 0.0001
        cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        half = dimension // 2; c, s = cos[context], sin[context]
        q_rotated = np.concatenate((q[..., :half] * c - q[..., half:] * s,
                                    q[..., half:] * c + q[..., :half] * s), axis=-1)
        k_rotated = np.concatenate((new_k[..., :half] * c - new_k[..., half:] * s,
                                    new_k[..., half:] * c + new_k[..., :half] * s), axis=-1)
        key_updated, value_updated = key_cache.copy(), value_cache.copy()
        key_updated[:, :, context] = k_rotated; value_updated[:, :, context] = new_v
        arrays = [be.array(value) for value in
                  (q, new_k, new_v, cos, sin, key_cache, value_cache,
                   q_rotated, key_updated, value_updated)]
        q_d, nk_d, nv_d, cos_d, sin_d, kc_d, vc_d, qr_d, ku_d, vu_d = arrays
        positions_d, contexts_d = be.int_array(positions), be.int_array(contexts)
        baseline = lambda: tk.attn_decode_bh(
            qr_d, ku_d, vu_d, context + 1, use_kernel=False)
        equivalent = lambda: tk.decode_cache_attention(
            q_d, nk_d, nv_d, cos_d, sin_d, positions_d, contexts_d, kc_d, vc_d,
            use_kernel=False)
        repetitions = heads_q // heads_kv
        repeated_k = np.repeat(key_updated[:, :, :context + 1], repetitions, axis=1)
        repeated_v = np.repeat(value_updated[:, :, :context + 1], repetitions, axis=1)
        scores = np.einsum("bhd,bhtd->bht", q_rotated, repeated_k) / math.sqrt(dimension)
        probabilities = np.exp(scores - scores.max(-1, keepdims=True))
        probabilities /= probabilities.sum(-1, keepdims=True)
        ref = np.einsum("bht,bhtd->bhd", probabilities, repeated_v)
        yield Case(
            "decode_cache_attention",
            f"B{batch}_H{heads_q}_{heads_kv}_T{context}_D{dimension}",
            {"B": batch, "Hq": heads_q, "Hkv": heads_kv, "T": context,
             "D": dimension}, "f32",
            target=lambda: tk.decode_cache_attention(
                q_d, nk_d, nv_d, cos_d, sin_d, positions_d, contexts_d, kc_d, vc_d,
                use_kernel=True),
            out_to_numpy=lambda out: be.to_numpy(out[0]),
            baselines={"framework_equivalent_functional_step": equivalent,
                       "preupdated_framework_sdpa_output_only": baseline}, ref=ref,
            bytes_moved=float(key_cache.nbytes + value_cache.nbytes +
                              key_updated.nbytes + value_updated.nbytes))


# --------------------------------------------------------------------------- runner
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    ap.add_argument("--preset", choices=["smoke", "quick", "comprehensive"], default="quick")
    ap.add_argument("--kernel", default="all", help="comma list of kernel families, or 'all'")
    ap.add_argument("--formats", default=None, help="comma list of quant formats to restrict to")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-check", action="store_true", help="skip the correctness pass")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    be = Backend(args.backend)
    formats = args.formats.split(",") if args.formats else None
    names = list(KERNEL_BUILDERS) if args.kernel == "all" else args.kernel.split(",")
    unknown = [n for n in names if n not in KERNEL_BUILDERS]
    if unknown:
        print(f"unknown kernels: {unknown}; available: {sorted(KERNEL_BUILDERS)}")
        return 1

    meta = _env_meta(args.backend)
    meta.update(backend=args.backend, preset=args.preset, warmup=args.warmup,
                iters=args.iters, kernels=names, formats=formats)

    # Spin the GPU clocks up before any measurement (the first-timed case otherwise
    # reads 1.5-5x slow while the GPU ramps from idle frequency).
    _warm = be.array(np.random.default_rng(0).standard_normal((2048, 2048)), "f16")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        if be.name == "mlx":
            be.sync(be.mx.matmul(_warm, _warm))
        else:
            be.sync(_warm @ _warm)
    del _warm

    rows = []
    t_start = time.perf_counter()
    for name in names:
        print(f"== {name} ==", flush=True)
        # consume the builder LAZILY: comprehensive quant sweeps hold ~1 GB of host+device
        # weights per case, and materializing the whole family's case list OOMs the process
        gen = iter(KERNEL_BUILDERS[name](be, args.preset, formats))
        while True:
            try:
                case = next(gen)
            except StopIteration:
                break
            except Exception as e:  # noqa: BLE001
                rows.append({"schema": SCHEMA_VERSION, "kernel": name, "variant": "-", "shape": {},
                             "dtype": "-", "format": None, "status": "skip",
                             "skip_reason": f"builder: {type(e).__name__}: {e}"})
                print(f"  SKIP family ({type(e).__name__}: {e})", flush=True)
                break
            try:
                row = run_case(case, be, args.warmup, args.iters, check=not args.no_check)
                rows.append(row)
                bl = {k: v for k, v in row["baselines"].items() if "ms" in v}
                best = min(bl.values(), key=lambda v: v["ms"])["ms"] if bl else float("nan")
                print(f"  {case.variant:28s} {_shape_str(case.shape):>22s} "
                      f"tk {row['target_ms']:8.4f} ms   base {best:8.4f} ms   "
                      f"err {row.get('max_rel_err', float('nan')):.1e}", flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append({"schema": SCHEMA_VERSION, "kernel": case.kernel,
                             "variant": case.variant, "shape": case.shape, "dtype": case.dtype,
                             "format": case.fmt, "status": "skip",
                             "skip_reason": f"{type(e).__name__}: {e}"})
                print(f"  {case.variant:28s} SKIP ({type(e).__name__}: {e})", flush=True)
            del case
            if be.name == "mlx":
                be.mx.metal.clear_cache()   # else the buffer cache keeps every case's device
                                            # weights and comprehensive quant sweeps OOM
    meta["wall_s"] = round(time.perf_counter() - t_start, 1)

    day = _dt.date.today().isoformat()
    run_id = _dt.datetime.now().strftime("%H%M%S") + f"-{args.backend}-{args.preset}"
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / day / run_id
    write_outputs(rows, meta, out_dir)
    print(f"\nwrote {out_dir}/ (run.json, results.jsonl, summary.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
