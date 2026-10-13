"""Both CNN variants predict with Grad-CAM, and old baseline artifacts still load."""
from __future__ import annotations

import base64
import io
import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from api.training import (
    ImprovedCNN,
    SmallCNN,
    _to_tensor,
    build_model,
    predict_with_cam,
    run_experiment,
)
from shared.data import IMG_SIZE, N_CLASSES, generate_images

CFG = dict(
    n_rows=400, img_size=IMG_SIZE, noise=0.1, data_seed=7, split_seed=1,
    train_seed=0, lr=0.01, batch_size=32, epochs=2,
)


def _png_size(b64: str):
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


@pytest.mark.parametrize("arch", ["baseline", "improved"])
def test_prediction_and_gradcam_work_for_each_variant(arch):
    r = run_experiment(**CFG, arch=arch)
    img = generate_images(1, seed=3)[0][0]
    out = predict_with_cam(r["model_b64"], img)
    assert len(out["probs"]) == N_CLASSES and abs(sum(out["probs"]) - 1) < 1e-5
    assert out["label"] == int(np.argmax(out["probs"]))
    assert _png_size(out["input_png"]) == _png_size(out["cam_png"]) == (IMG_SIZE, IMG_SIZE)


@pytest.mark.parametrize("model", [SmallCNN(IMG_SIZE, N_CLASSES), ImprovedCNN(IMG_SIZE, N_CLASSES)])
def test_both_models_accept_images_and_output_four_scores(model):
    x = _to_tensor(generate_images(5, seed=1)[0])
    assert model.eval()(x).shape == (5, N_CLASSES)


def test_old_baseline_artifact_without_new_keys_still_loads():
    torch.manual_seed(0)
    old = SmallCNN(IMG_SIZE, N_CLASSES).eval()
    legacy = base64.b64encode(
        pickle.dumps({"state_dict": old.state_dict(), "img_size": IMG_SIZE, "n_classes": N_CLASSES})
    ).decode()
    img = generate_images(1, seed=3)[0][0]
    out = predict_with_cam(legacy, img)
    with torch.no_grad():
        expected = torch.softmax(old(_to_tensor(img[None])), 1)[0].tolist()
    assert np.allclose(out["probs"], expected, atol=1e-6)
    assert _png_size(out["cam_png"]) == (IMG_SIZE, IMG_SIZE)


def test_baseline_records_what_it_actually_uses():
    r = run_experiment(**CFG, arch="baseline", use_batchnorm=True, dropout=0.5)
    assert r["config"]["use_batchnorm"] is False and r["config"]["dropout"] == 0.0
    art = pickle.loads(base64.b64decode(r["model_b64"]))
    assert art["arch"] == "baseline" and art["use_batchnorm"] is False and art["dropout"] == 0.0
    # The ignored options really are ignored: same run either way.
    other = run_experiment(**CFG, arch="baseline", use_batchnorm=False, dropout=0.0)
    assert r["train_loss"] == other["train_loss"] and r["metrics"] == other["metrics"]


def test_improved_records_its_real_options_independently():
    r = run_experiment(**CFG, arch="improved", use_batchnorm=False, dropout=0.4)
    assert r["config"]["use_batchnorm"] is False and r["config"]["dropout"] == 0.4
    m = build_model(IMG_SIZE, N_CLASSES, "improved", use_batchnorm=False, dropout=0.4)
    assert r["param_count"] == sum(p.numel() for p in m.parameters())
