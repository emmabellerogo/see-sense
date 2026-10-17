"""Regenerate the fixed-reference prediction fixture. Run only when the model/serving math is
changed on purpose:

    python -m tests.fixtures.make_reference

Writes tests/fixtures/reference_weights.json (a seeded, UNTRAINED SmallCNN's weights) and
reference_expected.json (what predict_with_cam returned for the reference image). The regression
test fails if a later change silently alters preprocessing, temperature scaling, ordering, or the CNN.
"""
from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from api.training import ARTIFACT_VERSION, SmallCNN, default_preprocessing, predict_with_cam
from shared import fashion

HERE = Path(__file__).parent
TEMPERATURE = 1.7


def reference_image() -> np.ndarray:
    """A fixed 28x28 uint8 picture: a bright trapezoid 'garment' on a dark background with a gradient."""
    y, x = np.mgrid[0:28, 0:28]
    img = np.zeros((28, 28), dtype=np.float64)
    inside = (y >= 4) & (y <= 24) & (np.abs(x - 13.5) <= 4 + 0.35 * (y - 4))
    img[inside] = 90 + 6 * (y[inside] - 4)
    img[8:12, 4:8] = 200
    return np.clip(img, 0, 255).astype(np.uint8)


def build_artifact_from_weights(weights: dict) -> str:
    model = SmallCNN(28, 10)
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()})
    art = {
        "artifact_version": ARTIFACT_VERSION, "state_dict": model.state_dict(), "img_size": 28,
        "n_classes": 10, "channels": 1, "arch": "baseline", "use_batchnorm": False, "dropout": 0.0,
        "class_names": list(fashion.CLASS_NAMES), "preprocessing": default_preprocessing(),
        "dataset": {"name": "reference-fixture"},
        "calibration": {"method": "temperature_scaling", "temperature": TEMPERATURE,
                        "fit_split": "validation", "fit_n": 0},
        "selected_epoch": 0,
    }
    return base64.b64encode(pickle.dumps(art)).decode("ascii")


if __name__ == "__main__":
    torch.manual_seed(1234)
    weights = {k: v.tolist() for k, v in SmallCNN(28, 10).state_dict().items()}
    (HERE / "reference_weights.json").write_text(json.dumps(weights))
    out = predict_with_cam(build_artifact_from_weights(weights), reference_image())
    expected = {k: out[k] for k in ("label", "class_name", "confidence", "probs", "top5", "temperature")}
    (HERE / "reference_expected.json").write_text(json.dumps(expected, indent=1))
    print("label", out["label"], out["class_name"], "confidence", round(out["confidence"], 4))
