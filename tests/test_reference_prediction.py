"""Fixed-reference regression: a known checkpoint and image must keep producing the same prediction.

The checkpoint is a seeded, untrained CNN stored as plain numbers (tests/fixtures), the image is a
formula. If preprocessing, normalization, temperature scaling, class order or the network change,
these numbers change and this test says so. Regenerate deliberately with
``python -m tests.fixtures.make_reference``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from api.training import predict_with_cam
from shared import fashion
from tests.fixtures.make_reference import TEMPERATURE, build_artifact_from_weights, reference_image

FIX = Path(__file__).parent / "fixtures"
EXPECTED = json.loads((FIX / "reference_expected.json").read_text())
B64 = build_artifact_from_weights(json.loads((FIX / "reference_weights.json").read_text()))


def test_fixed_reference_image_gives_the_recorded_prediction():
    out = predict_with_cam(B64, reference_image())
    assert out["label"] == EXPECTED["label"] and out["class_name"] == EXPECTED["class_name"]
    assert np.allclose(out["probs"], EXPECTED["probs"], atol=1e-5)
    assert np.isclose(out["confidence"], EXPECTED["confidence"], atol=1e-5)
    assert [t["label"] for t in out["top5"]] == [t["label"] for t in EXPECTED["top5"]]
    assert np.allclose([t["prob"] for t in out["top5"]], [t["prob"] for t in EXPECTED["top5"]], atol=1e-5)
    assert out["temperature"] == TEMPERATURE


def test_the_reference_probabilities_are_a_valid_distribution_in_class_order():
    out = predict_with_cam(B64, reference_image())
    assert len(out["probs"]) == len(fashion.CLASS_NAMES) == 10
    assert abs(sum(out["probs"]) - 1.0) < 1e-6
    assert out["class_names"] == fashion.CLASS_NAMES


def test_prediction_is_deterministic_across_calls():
    a, b = predict_with_cam(B64, reference_image()), predict_with_cam(B64, reference_image())
    assert a["probs"] == b["probs"] and a["cam_png"] == b["cam_png"]
