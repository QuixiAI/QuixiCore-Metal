#!/usr/bin/env python3
"""ThunderMittens kernel benchmark harness (schema v1).

Covers every active kernel family under ThunderMittens/kernels/ with, per case:
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
KERNELS_DIR = REPO_ROOT / "ThunderMittens" / "kernels"
if str(KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(KERNELS_DIR))

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
    for _ in range(warmup):
        be.sync(fn())
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
    lines = ["# ThunderMittens kernel benchmarks", ""]
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
        for variant, fn, causal in (
                ("fwd", tk.attn_fwd, False),
                ("causal", tk.attn_causal, True),
                ("multiwarp", tk.attn_multiwarp, False)):
            baselines = {"sdpa": _sdpa_baseline(be, q_d, k_d, v_d, D, causal)}
            if variant == "multiwarp":
                baselines["tk.attn_fwd"] = lambda q_d=q_d, k_d=k_d, v_d=v_d: \
                    tk.attn_fwd(q_d, k_d, v_d)
            yield Case("attn", f"{variant}_B{B}H{H}N{N}D{D}",
                       {"B": B, "H": H, "N": N, "D": D}, "bf16",
                       target=lambda fn=fn, q_d=q_d, k_d=k_d, v_d=v_d: fn(q_d, k_d, v_d),
                       baselines=baselines, ref=None,
                       flops=flops / (2 if causal else 1))


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
    fmts = formats or _pick(preset, ["q8_0"], ["q8_0", "q4_0", "fp8_e4m3"],
                            ["q8_0", "q4_0", "fp8_e4m3"])
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
            if fmt in ("q8_0", "fp8_e4m3") and N % 32 == 0:
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
    shapes = _pick(preset, [(4096, 4096)],
                   [(4096, 4096), (11008, 4096)],
                   [(4096, 4096), (11008, 4096), (4096, 11008), (32000, 4096),
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
            ref = wdq @ be.to_numpy(x_d)   # f32 (f64 matmul warns spuriously under Accelerate)
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
@register("paged_attn")
def paged_attn_cases(be, preset, formats):
    tk = be.tk()
    rng = np.random.default_rng(26)
    shapes = _pick(preset, [(4, 16, 4, 128, 512)],
                   [(8, 32, 8, 128, 2048)],
                   [(8, 32, 8, 128, 2048), (16, 32, 8, 128, 4096), (8, 32, 8, 128, 8192)])
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
        yield Case("paged_attn", f"v2_fp8_B{B}H{H}ctx{ctx}",
                   {"B": B, "H": H, "ctx": ctx, "D": D}, "fp8",
                   target=lambda q_d=q_d, kc8_d=kc8_d, vc8_d=vc8_d, bt_d=bt_d, cl_d=cl_d:
                       tk.paged_attention_v2_fp8(q_d, kc8_d, vc8_d, bt_d, cl_d,
                                                 0.01, 0.01, partition_size=256),
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

    rows = []
    t_start = time.perf_counter()
    for name in names:
        print(f"== {name} ==")
        try:
            cases = list(KERNEL_BUILDERS[name](be, args.preset, formats))
        except Exception as e:  # noqa: BLE001
            rows.append({"schema": SCHEMA_VERSION, "kernel": name, "variant": "-", "shape": {},
                         "dtype": "-", "format": None, "status": "skip",
                         "skip_reason": f"builder: {type(e).__name__}: {e}"})
            print(f"  SKIP family ({type(e).__name__}: {e})")
            continue
        for case in cases:
            try:
                row = run_case(case, be, args.warmup, args.iters, check=not args.no_check)
                rows.append(row)
                bl = {k: v for k, v in row["baselines"].items() if "ms" in v}
                best = min(bl.values(), key=lambda v: v["ms"])["ms"] if bl else float("nan")
                print(f"  {case.variant:28s} {_shape_str(case.shape):>22s} "
                      f"tk {row['target_ms']:8.4f} ms   base {best:8.4f} ms   "
                      f"err {row.get('max_rel_err', float('nan')):.1e}")
            except Exception as e:  # noqa: BLE001
                rows.append({"schema": SCHEMA_VERSION, "kernel": case.kernel,
                             "variant": case.variant, "shape": case.shape, "dtype": case.dtype,
                             "format": case.fmt, "status": "skip",
                             "skip_reason": f"{type(e).__name__}: {e}"})
                print(f"  {case.variant:28s} SKIP ({type(e).__name__}: {e})")
    meta["wall_s"] = round(time.perf_counter() - t_start, 1)

    day = _dt.date.today().isoformat()
    run_id = _dt.datetime.now().strftime("%H%M%S") + f"-{args.backend}-{args.preset}"
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_ROOT / day / run_id
    write_outputs(rows, meta, out_dir)
    print(f"\nwrote {out_dir}/ (run.json, results.jsonl, summary.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
