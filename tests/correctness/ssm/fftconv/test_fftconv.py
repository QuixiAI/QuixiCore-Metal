"""Correctness test for the Monarch FFT convolution kernel.

Builds the FFT/twiddle matrices (the Cooley-Tukey N=S*S Monarch factorization) and
compares the kernel output to a direct FFT circular convolution (numpy). The kernel
exercises the complex-multiply MMA + transposes + pointwise complex multiplies.

Run from kernels/:  python -m pytest fftconv/correctness/test_fftconv.py -v
"""

import mlx.core as mx
import numpy as np
import pytest

from tk import fftconv


def _fft_matrix(N):
    n = np.arange(N); k = n.reshape(-1, 1)
    return np.exp(-2j * np.pi * n * k / N)


def _ifft_matrix(N):
    n = np.arange(N); k = n.reshape(-1, 1)
    return np.exp(2j * np.pi * n * k / N)


def _twiddle(n, m, sign):
    na = np.arange(n).reshape(-1, 1); ma = np.arange(m)
    return np.exp(sign * 2j * np.pi * na * ma / (n * m))


def _stack(m):
    return mx.array(np.stack([m.real, m.imag]).astype(np.float32))


@pytest.mark.parametrize("shape", [(1, 1, 16), (2, 3, 16), (1, 1, 32), (2, 2, 32)])
def test_fftconv(shape):
    B, H, S = shape
    N = S * S
    rng = np.random.default_rng(0)
    u = rng.standard_normal((B, H, N)).astype(np.float32)
    k = rng.standard_normal((H, N)).astype(np.float32)

    F, Finv = _fft_matrix(S), _ifft_matrix(S)
    TW, TWI = _twiddle(S, S, -1), _twiddle(S, S, +1) / N
    kf = np.fft.fft(k, n=N).reshape(H, S, S).transpose(0, 2, 1)  # reshape + transpose (k_fT)

    xr = u.reshape(B, H, S, S).astype(np.float32)
    X = mx.array(np.stack([xr, np.zeros_like(xr)]))
    KF = mx.array(np.stack([kf.real, kf.imag]).astype(np.float32))

    got = fftconv(X, _stack(F), _stack(TW), _stack(Finv), _stack(TWI), KF)
    mx.eval(got)
    g = np.array(got)

    ref = np.fft.ifft(np.fft.fft(u, n=N) * np.fft.fft(k, n=N)[None], n=N).real.reshape(B, H, S, S)
    assert got.shape == (B, H, S, S)
    rel = np.abs(g - ref).max() / (np.abs(ref).max() + 1e-9)
    assert rel < 2e-2, f"relative diff {rel}"


if __name__ == "__main__":
    for shp in [(1, 1, 16), (1, 1, 32)]:
        test_fftconv(shp)
        print("ok", shp)
