"""POST /predict, GET /predictions, drill-down and logging failures, against the in-memory fake DB.

These use a MOCKED database (tests/conftest.py) and a tiny trained model. They prove the API's own
logic: multipart handling, response shape, Grad-CAM size, what is (and is not) written, and that a
logging failure is reported as a failure. They do not prove anything about Supabase itself; the
live checks are in tests/test_live_supabase.py and are skipped unless explicitly enabled.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from api import db
from shared import fashion


def _png(arr: np.ndarray | None = None, size=(28, 28), mode="L", fmt="PNG") -> bytes:
    if arr is None:
        arr = np.random.default_rng(0).integers(0, 255, size[::-1], dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def run_id(client):
    ds = client.post("/datasets", json={"name": "fm"}).json()
    r = client.post("/train", json={"dataset_id": ds["id"], "lr": 0.01, "batch_size": 32, "epochs": 2,
                                    "model_variant": "improved", "experiment_group": "g", "run_name": "full_improved-seed0"})
    assert r.status_code == 200, r.text
    return r.json()["run_id"]


def _post(client, run_id, raw, name="a.png", ctype="image/png"):
    return client.post("/predict", data={"run_id": run_id}, files={"file": (name, raw, ctype)})


# ---- multipart upload ----------------------------------------------------------------------
def test_multipart_upload_returns_prediction_top5_and_gradcam(client, run_id):
    raw = _png()
    r = _post(client, run_id, raw)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["run_id"] == run_id and b["class_name"] == fashion.CLASS_NAMES[b["label"]]
    assert b["class_names"] == fashion.CLASS_NAMES and len(b["probs"]) == 10
    assert abs(sum(b["probs"]) - 1) < 1e-5 and 0 <= b["confidence"] <= 1
    assert len(b["top5"]) == 5 and b["top5"][0]["label"] == b["label"]
    assert [t["prob"] for t in b["top5"]] == sorted((t["prob"] for t in b["top5"]), reverse=True)
    assert b["calibrated"] is True and b["temperature"] > 0
    assert b["image_sha256"] == hashlib.sha256(raw).hexdigest()
    assert b["logged"] is True and b["prediction_id"] == 1


@pytest.mark.parametrize("fmt,ctype", [("JPEG", "image/jpeg"), ("PNG", "image/png")])
def test_larger_rgb_and_jpeg_images_are_converted_to_28x28_grayscale(client, run_id, fmt, ctype):
    rgb = np.random.default_rng(1).integers(0, 255, (120, 90, 3), dtype=np.uint8)
    r = _post(client, run_id, _png(rgb, mode="RGB", fmt=fmt), "x", ctype)
    assert r.status_code == 200
    assert Image.open(io.BytesIO(base64.b64decode(r.json()["input_png"]))).size == (28, 28)


def test_gradcam_overlay_has_the_input_dimensions_and_is_a_colour_png(client, run_id):
    b = _post(client, run_id, _png()).json()
    cam = Image.open(io.BytesIO(base64.b64decode(b["cam_png"])))
    inp = Image.open(io.BytesIO(base64.b64decode(b["input_png"])))
    assert cam.size == inp.size == (28, 28)
    assert cam.mode == "RGB" and inp.mode == "L"
    assert np.asarray(cam)[..., 0].max() > 0  # a heatmap was actually drawn


# ---- logging: linkage, no pixels, failures ----------------------------------------------------
def test_prediction_is_linked_to_its_run_and_stores_hash_and_numbers_only(client, run_id):
    raw = _png()
    b = _post(client, run_id, raw).json()
    store = client._store
    (pred,) = store["predictions"]
    (meta,) = store["image_metadata"]
    assert pred["served_by_run_id"] == run_id and meta["run_id"] == run_id
    assert pred["image_sha256"] == meta["sha256"] == hashlib.sha256(raw).hexdigest()
    assert pred["predicted_label"] == b["label"] and pred["predicted_class"] == b["class_name"]
    assert pred["probabilities"] == b["probs"] and "created_at" in pred
    assert (meta["width"], meta["height"]) == (28, 28)
    blob = json.dumps({"p": store["predictions"], "m": store["image_metadata"]})
    assert b["input_png"] not in blob and b["cam_png"] not in blob
    assert base64.b64encode(raw).decode() not in blob  # the upload itself is never stored
    assert not any(isinstance(v, (bytes, bytearray)) for r in store["predictions"] + store["image_metadata"] for v in r.values())


def test_logging_failure_is_an_error_not_a_successful_result(client, run_id):
    from postgrest.exceptions import APIError

    client._store["fail_log_with"] = APIError({"code": "23503", "message": "fk", "details": None, "hint": None})
    r = _post(client, run_id, _png())
    assert r.status_code == 500 and "cam_png" not in r.json()
    assert client._store["predictions"] == [] and client._store["image_metadata"] == []


@pytest.mark.parametrize("exc_factory", [
    lambda: db.PredictionLogUnavailable("PGRST202"),   # migration 005 function missing
])
def test_unavailable_logging_is_503_and_nothing_is_returned(client, run_id, exc_factory):
    client._store["fail_log_with"] = exc_factory()
    r = _post(client, run_id, _png())
    assert r.status_code == 503 and "logging" in r.json()["detail"]
    assert "logged" not in r.json() and "cam_png" not in r.json()
    assert client._store["predictions"] == [] and client._store["image_metadata"] == []


def test_the_real_db_function_maps_a_missing_rpc_to_prediction_log_unavailable(monkeypatch):
    from postgrest.exceptions import APIError

    class FakeClient:
        def rpc(self, name, params):
            assert name == "log_prediction" and params["p_run_id"] == 1
            raise APIError({"code": "PGRST202", "message": "Could not find the function", "details": None, "hint": None})

    monkeypatch.setattr(db, "get_client", lambda: FakeClient())
    with pytest.raises(db.PredictionLogUnavailable):
        db.log_prediction(1, "h", 28, 28, 0, "x", [0.5, 0.5], 0.5)


def test_the_real_db_function_sends_one_atomic_rpc_with_hash_label_and_probabilities(monkeypatch):
    calls = []

    class Resp:
        data = 41

    class FakeClient:
        def rpc(self, name, params):
            calls.append((name, params))
            return type("Q", (), {"execute": lambda self: Resp()})()

        def table(self, *a):
            raise AssertionError("logging must be one RPC, not separate table inserts")

    monkeypatch.setattr(db, "get_client", lambda: FakeClient())
    assert db.log_prediction(7, "abc", 28, 28, 3, "Dress", [0.1, 0.9], 0.9) == 41
    (name, params), = calls
    assert name == "log_prediction"
    assert params == {"p_run_id": 7, "p_sha256": "abc", "p_width": 28, "p_height": 28, "p_label": 3,
                      "p_class_name": "Dress", "p_probabilities": [0.1, 0.9], "p_confidence": 0.9}


def test_invalid_uploads_are_422_and_log_nothing(client, run_id):
    for payload in (b"", b"definitely not an image"):
        assert _post(client, run_id, payload).status_code == 422
    assert client._store["predictions"] == [] and client._store["image_metadata"] == []


def test_oversize_upload_is_413(client, run_id):
    assert _post(client, run_id, b"x" * (5 * 1024 * 1024 + 1)).status_code == 413


def test_unknown_run_is_404_and_logs_nothing(client):
    assert _post(client, 9999, _png()).status_code == 404
    assert client._store["predictions"] == []


def test_predict_requires_a_file_and_an_integer_run_id(client):
    assert client.post("/predict", data={"run_id": 1}).status_code == 422
    assert client.post("/predict", data={"run_id": "x"}, files={"file": ("a.png", _png(), "image/png")}).status_code == 422


def test_the_removed_synthetic_sample_endpoint_is_gone(client):
    assert client.post("/predict_sample", json={"run_id": 1, "seed": 0}).status_code == 404


# ---- history and drill-down -------------------------------------------------------------------
def test_get_predictions_joins_the_producing_run_and_hides_hashes(client, run_id):
    for s in range(3):
        _post(client, run_id, _png(np.full((28, 28), 40 * s, dtype=np.uint8)))
    rows = client.get("/predictions").json()
    assert [r["id"] for r in rows] == [3, 2, 1]  # newest first
    first = rows[0]
    assert first["served_by_run_id"] == run_id
    assert first["run"]["run_name"] == "full_improved-seed0" and first["run"]["experiment_group"] == "g"
    assert first["run"]["model_variant"] == "improved" and 0 <= first["run"]["accuracy"] <= 1
    assert len(first["probabilities"]) == 10
    text = json.dumps(rows)
    assert "image_sha256" not in text and "sha256" not in text  # the public record has no hash
    assert len(client.get("/predictions?limit=2").json()) == 2
    assert client.get("/predictions?limit=0").status_code == 422


def test_drill_down_from_a_classification_to_its_run(client, run_id):
    _post(client, run_id, _png())
    pred = client.get("/predictions/1").json()
    run = client.get(f"/runs/{pred['served_by_run_id']}").json()
    assert run["id"] == run_id and run["run_name"] == "full_improved-seed0"
    assert run["config"]["dataset"] == "fashion-mnist" and run["calibration"]["temperature"] > 0
    assert client.get("/predictions/999").status_code == 404


def test_the_checkpoint_is_never_part_of_any_public_response(client, run_id):
    _post(client, run_id, _png())
    blob = client.get("/predictions").text + client.get(f"/runs/{run_id}").text + client.get("/runs").text
    assert "model_b64" not in blob and client._store["artifacts"][run_id][:50] not in blob


# ---- database outage ---------------------------------------------------------------------------
@pytest.mark.parametrize("make_exc", [
    lambda: __import__("httpx").ConnectError("secret-host.supabase.co refused"),
    lambda: __import__("httpx").ReadTimeout("timed out"),
    lambda: __import__("postgrest.exceptions", fromlist=["APIError"]).APIError(
        {"code": "PGRST002", "message": "schema cache", "details": None, "hint": None}),
])
def test_database_outage_is_503_without_leaks(client, run_id, monkeypatch, make_exc):
    def boom(*a, **k):
        raise make_exc()
    monkeypatch.setattr(db, "get_run_artifact", boom)
    r = _post(client, run_id, _png())
    assert r.status_code == 503 and r.json() == {"detail": "database unavailable"}
    assert client._store["predictions"] == []


def test_other_database_errors_are_generic_500(client, run_id, monkeypatch):
    from postgrest.exceptions import APIError

    def boom(*a, **k):
        raise APIError({"code": "23505", "message": "duplicate key secret-detail", "details": None, "hint": None})
    monkeypatch.setattr(db, "get_run_artifact", boom)
    r = _post(client, run_id, _png())
    assert r.status_code == 500 and r.json() == {"detail": "database error"}


def test_classes_endpoint_lists_the_ten_labels_in_order(client):
    body = client.get("/classes").json()
    assert body["classes"] == fashion.CLASS_NAMES and body["n_classes"] == 10
