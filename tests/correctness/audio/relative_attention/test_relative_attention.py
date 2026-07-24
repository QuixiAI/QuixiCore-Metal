"""Correctness for blocked Conformer relative-position attention."""

import mlx.core as mx
import numpy as np

import tk


def _reference(q, k, v, relative_k, per_dim, lengths, chunk, left, right, softcap):
    B, T, H, D = q.shape
    out = np.zeros_like(q)
    q_scale = 1.0 / (np.sqrt(D) * np.log(2.0)); k_scale = 1.0 / np.log(2.0)
    learned = np.logaddexp(per_dim, 0.0)
    for b in range(B):
        for t in range(lengths[b]):
            qi = t % chunk; context_start = (t // chunk) * chunk - (left - 1)
            context_length = chunk + left - 1 + right
            for h in range(H):
                qs = q[b, t, h] * q_scale * learned
                scores, values = [], []
                for ci in range(context_length):
                    kt = context_start + ci
                    if kt < 0 or kt >= lengths[b]:
                        continue
                    score = np.dot(qs, k[b, kt, h] * k_scale)
                    ri = ci - qi
                    if 0 <= ri < relative_k.shape[0]:
                        score += np.dot(qs, relative_k[ri, h])
                    if softcap:
                        score = softcap * np.tanh(score / softcap)
                    scores.append(score); values.append(v[b, kt, h])
                scores = np.asarray(scores, np.float32)
                probs = np.exp(scores - scores.max()); probs /= probs.sum()
                out[b, t, h] = probs @ np.asarray(values)
    return out


def test_audio_relative_attention_shift_context_and_lengths():
    rng = np.random.default_rng(81)
    B, T, H, D = 2, 17, 2, 64; chunk, left, right, P = 4, 3, 1, 5
    q = (0.08 * rng.standard_normal((B, T, H, D))).astype(np.float32)
    k = (0.08 * rng.standard_normal((B, T, H, D))).astype(np.float32)
    v = (0.2 * rng.standard_normal((B, T, H, D))).astype(np.float32)
    rel = (0.08 * rng.standard_normal((P, H, D))).astype(np.float32)
    per_dim = (0.2 * rng.standard_normal(D)).astype(np.float32)
    lengths = np.array([17, 11], np.int32)
    got = tk.audio_relative_attention(
        mx.array(q), mx.array(k), mx.array(v), mx.array(rel), mx.array(per_dim),
        mx.array(lengths), chunk, left, right, softcap=5.0)
    mx.eval(got)
    ref = _reference(q, k, v, rel, per_dim, lengths, chunk, left, right, 5.0)
    np.testing.assert_allclose(np.array(got), ref, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(np.array(got)[1, lengths[1]:], 0.0)
