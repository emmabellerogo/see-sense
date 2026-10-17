"""Temperature scaling and the calibration measures (pure numpy)."""
from __future__ import annotations

import numpy as np
import pytest

from api import calibration as cal


def _overconfident(n=4000, k=5, seed=0, scale=4.0, acc=0.7):
    """Logits that are `scale` times too sharp for a classifier that is right ~`acc` of the time."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    correct = rng.random(n) < acc
    pred = np.where(correct, y, (y + rng.integers(1, k, n)) % k)
    logits = rng.normal(0, 0.5, (n, k))
    logits[np.arange(n), pred] += 1.0
    return logits * scale, y


def test_softmax_rows_sum_to_one_and_temperature_flattens():
    z = np.array([[3.0, 1.0, 0.0]])
    assert np.isclose(cal.softmax(z).sum(), 1.0)
    assert cal.softmax(z, 5.0).max() < cal.softmax(z, 1.0).max()


def test_fitted_temperature_cools_an_overconfident_model_and_lowers_nll():
    logits, y = _overconfident()
    t = cal.fit_temperature(logits, y)
    assert t > 1.5  # the logits were 4x too sharp
    assert cal.nll(logits, y, t) < cal.nll(logits, y, 1.0)
    assert cal.nll(logits, y, t) <= min(cal.nll(logits, y, t * f) for f in (0.7, 0.9, 1.1, 1.4)) + 1e-9


def test_a_calibrated_model_keeps_a_temperature_near_one():
    logits, y = _overconfident(scale=1.0, acc=0.6, seed=2)
    rng = np.random.default_rng(1)
    # Labels drawn from the model's own probabilities are calibrated by construction.
    p = cal.softmax(logits)
    y = np.array([rng.choice(len(r), p=r) for r in p])
    assert 0.85 < cal.fit_temperature(logits, y) < 1.15


def test_temperature_never_changes_the_predicted_class():
    logits, _ = _overconfident()
    assert np.array_equal(cal.softmax(logits, 1.0).argmax(1), cal.softmax(logits, 3.7).argmax(1))


def test_ece_is_zero_for_perfectly_confident_correct_and_large_for_confident_wrong():
    y = np.array([0, 1, 2, 0])
    sure = np.eye(3)[[0, 1, 2, 0]] * 0.999 + 0.0005
    assert cal.ece(sure, y) < 0.01
    assert cal.ece(sure, np.array([1, 2, 0, 1])) > 0.9


def test_report_improves_on_held_out_data_and_records_what_it_measured():
    vl, vy = _overconfident(seed=0)
    tl, ty = _overconfident(seed=1)
    rep = cal.calibration_report(vl, vy, tl, ty)
    assert rep["method"] == "temperature_scaling" and rep["fit_split"] == "validation" and rep["fit_n"] == len(vy)
    assert rep["test"]["after"]["ece"] < rep["test"]["before"]["ece"]
    assert rep["test"]["after"]["nll"] < rep["test"]["before"]["nll"]
    assert {"ece", "nll", "brier"} == set(rep["validation"]["before"])
    assert sum(b["n"] for b in rep["test"]["reliability_after"]) == len(ty)


def test_the_temperature_is_fitted_on_validation_logits_only():
    vl, vy = _overconfident(seed=0)
    t_a = cal.calibration_report(vl, vy, *_overconfident(seed=1))["temperature"]
    t_b = cal.calibration_report(vl, vy, *_overconfident(seed=2, scale=0.5))["temperature"]
    assert t_a == t_b  # changing the test data cannot change the fit


def test_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError):
        cal.fit_temperature(np.zeros((3, 4)), np.zeros(2, dtype=int))
