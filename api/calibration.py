"""Probability calibration: temperature scaling and the measures that test it.

Softmax probabilities are not automatically calibrated. A model that says "90%" is
well calibrated only if it is right about 90% of the time at that confidence. This
module fits ONE parameter, a temperature T > 0, on VALIDATION logits
(``softmax(logits / T)``) and measures calibration with:

  * ECE  - expected calibration error, ``n_bins`` equal-width confidence bins,
  * NLL  - mean negative log-likelihood (the quantity T is fitted to minimise),
  * Brier - mean squared error of the full probability vector.

T never changes which class has the highest score, so accuracy and Grad-CAM (which
uses the raw class logit) are unaffected. Nothing here claims calibration was
achieved: ``calibration_report`` returns measured numbers, before and after, on
validation (the fit data) and on the held-out test set (reported, never used to fit).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

T_MIN, T_MAX = 0.05, 20.0
DEFAULT_BINS = 15


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray, iters: int = 80) -> float:
    """Temperature minimising validation NLL: golden-section search on log T (deterministic).

    NLL is convex in 1/T, so a one-dimensional search over [T_MIN, T_MAX] finds the optimum.
    """
    val_logits, val_y = np.asarray(val_logits), np.asarray(val_y, dtype=np.int64)
    if len(val_y) == 0 or len(val_logits) != len(val_y):
        raise ValueError("need matching, non-empty validation logits and labels")
    lo, hi = math.log(T_MIN), math.log(T_MAX)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = nll(val_logits, val_y, math.exp(a)), nll(val_logits, val_y, math.exp(b))
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = nll(val_logits, val_y, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = nll(val_logits, val_y, math.exp(b))
    return float(math.exp((lo + hi) / 2.0))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = DEFAULT_BINS) -> float:
    """Expected calibration error of the top-class confidence (equal-width bins)."""
    probs, y = np.asarray(probs), np.asarray(y, dtype=np.int64)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    probs, y = np.asarray(probs), np.asarray(y, dtype=np.int64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = DEFAULT_BINS) -> list[dict]:
    """Per-bin count, mean confidence and accuracy (for a reliability diagram)."""
    probs, y = np.asarray(probs), np.asarray(y, dtype=np.int64)
    conf, correct = probs.max(axis=1), (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({
            "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
            "mean_confidence": float(conf[m].mean()) if m.any() else None,
            "accuracy": float(correct[m].mean()) if m.any() else None,
        })
    return out


def _measures(logits: np.ndarray, y: np.ndarray, t: float, n_bins: int) -> Dict[str, float]:
    p = softmax(logits, t)
    return {"ece": ece(p, y, n_bins), "nll": nll(logits, y, t), "brier": brier(p, y)}


def calibration_report(
    val_logits: np.ndarray, val_y: np.ndarray, test_logits: np.ndarray, test_y: np.ndarray,
    n_bins: int = DEFAULT_BINS,
) -> dict:
    """Fit T on validation; report ECE / NLL / Brier at T=1 and at the fitted T on both splits."""
    t = fit_temperature(val_logits, val_y)
    return {
        "method": "temperature_scaling",
        "temperature": t,
        "fit_split": "validation",
        "fit_n": int(len(val_y)),
        "n_bins": n_bins,
        "validation": {"before": _measures(val_logits, val_y, 1.0, n_bins),
                       "after": _measures(val_logits, val_y, t, n_bins)},
        "test": {"before": _measures(test_logits, test_y, 1.0, n_bins),
                 "after": _measures(test_logits, test_y, t, n_bins),
                 "reliability_after": reliability_bins(softmax(test_logits, t), test_y, n_bins)},
        "note": ("T is fitted on validation logits only. Test measures are reported, not used to fit "
                 "or select anything. Calibration holds only to the extent the 'after' test numbers show."),
    }
