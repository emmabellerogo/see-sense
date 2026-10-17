"""Checkpoints: format 3 contents, legacy (synthetic-shape) formats still load, both CNNs predict with Grad-CAM."""
from __future__ import annotations

import base64
import io
import pickle

import numpy as np
import pytest
import torch
from PIL import Image

from api.training import (
    LEGACY_SYNTHETIC_CLASS_NAMES,
    ImprovedCNN,
    SmallCNN,
    describe_artifact,
    load_artifact,
    predict_with_cam,
    run_experiment,
    to01,
)
from shared import fashion
from shared.synthetic_data import generate_images
from tests.fashion_fixture import tiny_fashion

CFG = dict(split_seed=1, train_seed=0, lr=0.01, batch_size=32, epochs=2)
DATA = tiny_fashion()


def _png_size(b64: str):
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


@pytest.mark.parametrize("arch", ["baseline", "improved"])
def test_prediction_and_gradcam_work_for_each_variant(arch):
    r = run_experiment(**CFG, arch=arch, data=DATA)
    img = DATA.x_test[0]
    out = predict_with_cam(r["model_b64"], img)
    assert len(out["probs"]) == 10 and abs(sum(out["probs"]) - 1) < 1e-6
    assert out["label"] == int(np.argmax(out["probs"]))
    assert out["class_name"] == fashion.CLASS_NAMES[out["label"]]
    assert _png_size(out["input_png"]) == _png_size(out["cam_png"]) == (28, 28)


@pytest.mark.parametrize("model", [SmallCNN(28, 10), ImprovedCNN(28, 10)])
def test_both_models_accept_images_and_output_ten_scores(model):
    assert model.eval()(to01(DATA.x_test[:5])).shape == (5, 10)


def test_top5_is_sorted_has_five_entries_and_comes_from_the_served_probabilities():
    out = predict_with_cam(run_experiment(**CFG, data=DATA)["model_b64"], DATA.x_test[3])
    top = out["top5"]
    assert len(top) == 5
    probs = [t["prob"] for t in top]
    assert probs == sorted(probs, reverse=True)
    assert top[0]["label"] == out["label"] and np.isclose(top[0]["prob"], out["confidence"])
    assert all(np.isclose(t["prob"], out["probs"][t["label"]]) for t in top)
    assert [t["class_name"] for t in top] == [fashion.CLASS_NAMES[t["label"]] for t in top]


def test_checkpoint_stores_labels_preprocessing_architecture_dataset_and_calibration():
    r = run_experiment(**CFG, arch="improved", use_batchnorm=True, dropout=0.25, data=DATA)
    art = pickle.loads(base64.b64decode(r["model_b64"]))
    assert art["artifact_version"] == 3
    assert art["class_names"] == fashion.CLASS_NAMES and art["n_classes"] == 10
    assert art["preprocessing"]["mean"] == fashion.NORM_MEAN and art["preprocessing"]["std"] == fashion.NORM_STD
    assert art["preprocessing"]["size"] == [28, 28] and art["preprocessing"]["mode"] == "L"
    assert (art["arch"], art["use_batchnorm"], art["dropout"], art["img_size"]) == ("improved", True, 0.25, 28)
    assert art["dataset"]["name"] == "fashion-mnist" and art["dataset"]["split_seed"] == 1
    assert art["dataset"]["file_sha256"] and art["dataset"]["class_names"] == fashion.CLASS_NAMES
    assert art["calibration"]["method"] == "temperature_scaling" and art["calibration"]["fit_split"] == "validation"
    assert art["calibration"]["temperature"] > 0 and art["selected_epoch"] == r["best_epoch"]
    info = describe_artifact(r["model_b64"])
    assert info["dataset"] == "fashion-mnist" and info["calibrated"] and info["n_classes"] == 10


def test_temperature_changes_probabilities_but_not_the_predicted_class_or_the_cam():
    r = run_experiment(**CFG, data=DATA)
    art = pickle.loads(base64.b64decode(r["model_b64"]))
    img = DATA.x_test[7]

    def with_t(t):
        a = {**art, "calibration": {**art["calibration"], "temperature": t}}
        return predict_with_cam(base64.b64encode(pickle.dumps(a)).decode(), img)

    cool, plain = with_t(5.0), with_t(1.0)
    assert cool["label"] == plain["label"] and cool["cam_png"] == plain["cam_png"]  # CAM uses raw logits
    assert cool["confidence"] < plain["confidence"] and cool["temperature"] == 5.0
    assert abs(sum(cool["probs"]) - 1) < 1e-9


# ---- historical four-class synthetic-shape checkpoints stay identifiable and loadable ------------
def _legacy(version: int):
    torch.manual_seed(0)
    model = SmallCNN(28, 4).eval()
    obj = {"state_dict": model.state_dict(), "img_size": 28, "n_classes": 4}
    if version == 2:
        obj.update(arch="baseline", use_batchnorm=False, dropout=0.0, artifact_version=2)
    return model, base64.b64encode(pickle.dumps(obj)).decode()


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_synthetic_checkpoints_load_with_their_own_labels_and_scaling(version):
    model, b64 = _legacy(version)
    img = generate_images(1, seed=3)[0][0]
    out = predict_with_cam(b64, img)
    with torch.no_grad():
        expected = torch.softmax(model(to01(img[None])), 1)[0].tolist()  # [0, 1] inputs, no normalization
    assert np.allclose(out["probs"], expected, atol=1e-6)
    assert out["class_name"] == LEGACY_SYNTHETIC_CLASS_NAMES[out["label"]]
    assert out["calibrated"] is False and out["temperature"] == 1.0 and len(out["top5"]) == 4
    assert _png_size(out["cam_png"]) == (28, 28)
    info = describe_artifact(b64)
    assert info["dataset"] == "synthetic-shapes" and info["dataset_source"] == "synthetic_shapes"
    assert info["artifact_version"] == version and info["n_classes"] == 4


def test_a_checkpoint_whose_labels_do_not_match_its_classes_is_refused():
    r = run_experiment(**CFG, data=DATA)
    art = pickle.loads(base64.b64decode(r["model_b64"]))
    art["class_names"] = art["class_names"][:9]
    with pytest.raises(ValueError, match="class_names"):
        load_artifact(base64.b64encode(pickle.dumps(art)).decode())
