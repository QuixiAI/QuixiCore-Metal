"""Encoder quality: round-trip ||dequantize(quantize(W)) - W|| (relative RMS) within ggml-grade
bounds. This locks in the production-grade encoders (ggml-faithful make_qkx2/make_qx/make_q3 ports,
the iq4 scale sweep, and the best-of-floor/ceil e8m0 MX scale) and catches any regression to the
old naive encoders. Random-Gaussian input is near-worst-case for low-bit formats, so the bounds
reflect that (real weight distributions compress better). The kernel decoders are unchanged, so the
existing kernel-vs-oracle suites independently guarantee decode correctness.

    python -m pytest tk/tests/test_encoders.py -v
"""

import numpy as np
import pytest

from tk.quant import QUANT_FORMATS

# per-format relrms ceiling on N(0,1)-scaled weights (current encoders sit comfortably below these;
# the old naive encoders blew past several of them — e.g. mxfp8 0.19, q2_K 0.33).
BOUNDS = {
    "q8_0": 0.012, "q4_0": 0.09, "q4_1": 0.085, "q5_0": 0.05, "q5_1": 0.045,
    "q2_K": 0.32, "q3_K": 0.17, "q4_K": 0.085, "q5_K": 0.045, "q6_K": 0.022,
    "kU4B8": 0.11, "kU4": 0.10, "hqq": 0.09,
    "iq4_nl": 0.085, "iq4_xs": 0.085,
    "iq2_xxs": 0.40, "iq2_xs": 0.34, "iq3_xxs": 0.24, "iq1_s": 0.75,
    "fp8_e4m3": 0.03, "e5m2": 0.06, "fp8_block": 0.035,
    "mxfp8": 0.04, "mxfp4": 0.14, "mxfp6_e3m2": 0.065, "mxfp6_e2m3": 0.035, "nvfp4": 0.11,
    "bitnet": 0.55,
}


@pytest.mark.parametrize("fmt", sorted(BOUNDS))
def test_encoder_roundtrip(fmt):
    quantize, dequantize = QUANT_FORMATS[fmt]
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((256, 512)) * 0.3).astype(np.float32)
    dW = dequantize(quantize(W))
    relrms = float(np.sqrt(((dW - W) ** 2).mean()) / (np.sqrt((W ** 2).mean()) + 1e-12))
    assert relrms < BOUNDS[fmt], f"{fmt} encoder round-trip relrms {relrms:.4f} >= {BOUNDS[fmt]}"


def test_optimizers_match_scalar_ggml():
    """The batched make_qkx2 matches a scalar transcription of ggml's algorithm (float drift only)."""
    from tk.quant import _make_qkx2_quants

    def ref(x, nmax, w, rmin, rdelta, nstep, use_mad):
        mn = min(x.min(), 0.0); mx = x.max()
        if mx == mn:
            return 0.0
        iscale = nmax / (mx - mn); scale = 1 / iscale
        L = np.clip(np.rint(iscale * (x - mn)), 0, nmax)
        def err(sc, mi, Lc):
            d = sc * Lc + mi - x; d = np.abs(d) if use_mad else d * d; return (w * d).sum()
        best = err(scale, mn, L); minv = mn; sw = w.sum(); sx = (w * x).sum()
        for is_ in range(nstep + 1):
            isc = (rmin + rdelta * is_ + nmax) / (mx - mn)
            La = np.clip(np.rint(isc * (x - mn)), 0, nmax)
            sl = (w * La).sum(); sl2 = (w * La * La).sum(); sxl = (w * La * x).sum()
            D = sw * sl2 - sl * sl
            if D > 0:
                ts = (sw * sxl - sx * sl) / D; tm = (sl2 * sx - sl * sxl) / D
                if tm > 0:
                    tm = 0; ts = sxl / sl2
                if err(ts, tm, La) < best:
                    best = err(ts, tm, La); scale = ts; minv = tm
        return scale

    rng = np.random.default_rng(1)
    xb = (rng.standard_normal((64, 32)) * 0.3).astype(np.float32)
    sc_v, _, _ = _make_qkx2_quants(xb, np.abs(xb), 15, -1.0, 0.1, 20, False)
    for i in range(64):
        s = ref(xb[i].astype(np.float64), 15, np.abs(xb[i]).astype(np.float64), -1.0, 0.1, 20, False)
        assert abs(s - sc_v[i]) < 1e-2, f"row {i}: batched {sc_v[i]} vs scalar {s}"


if __name__ == "__main__":
    for f in sorted(BOUNDS):
        test_encoder_roundtrip(f)
    test_optimizers_match_scalar_ggml()
    print("ok")
