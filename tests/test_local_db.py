"""The optional offline database (SEE_SENSE_LOCAL_DB) used for local runs."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from api.local_db import DEFAULT_DATASET, LocalStore


def test_new_file_is_seeded_with_the_fashion_mnist_dataset(tmp_path):
    store = LocalStore(tmp_path / "db.json")
    ds = store.get_dataset(1)
    assert ds is not None and ds["source"] == "fashion_mnist" and ds["n_classes"] == 10
    assert {k: ds[k] for k in DEFAULT_DATASET} == DEFAULT_DATASET
    assert store.ping()


def test_duplicate_run_name_is_rejected(tmp_path):
    store = LocalStore(tmp_path / "db.json")
    exp = {"experiment_group": "g", "run_name": "baseline-seed0"}
    store.insert_run(1, 0.01, 32, 1, {"accuracy": 1.0, "macro_f1": 1.0}, "AAA", exp)
    assert store.run_name_exists("g", "baseline-seed0")
    with pytest.raises(ValueError):
        store.insert_run(1, 0.01, 32, 1, {"accuracy": 1.0, "macro_f1": 1.0}, "AAA", exp)


def test_log_prediction_writes_both_rows_together_and_rejects_an_unknown_run(tmp_path):
    store = LocalStore(tmp_path / "db.json")
    run = store.insert_run(1, 0.01, 32, 1, {"accuracy": 0.5, "macro_f1": 0.5}, "AAA", {"run_name": "r"})
    pid = store.log_prediction(run["id"], "h" * 64, 28, 28, 3, "Dress", [0.1, 0.9], 0.9)
    data = store._read()
    assert len(data["image_metadata"]) == len(data["predictions"]) == 1 and data["predictions"][0]["id"] == pid
    with pytest.raises(ValueError):
        store.log_prediction(999, "h", 28, 28, 0, "x", [0.5, 0.5], 0.5)
    data = store._read()
    assert len(data["image_metadata"]) == len(data["predictions"]) == 1  # nothing half-written
    pub = store.list_predictions()[0]
    assert "image_sha256" not in pub and pub["runs"]["run_name"] == "r"


def test_api_trains_predicts_and_logs_offline(tmp_path, monkeypatch, tiny_data):
    monkeypatch.setenv("SEE_SENSE_LOCAL_DB", str(tmp_path / "db.json"))
    from api import db, main

    importlib.reload(db)
    importlib.reload(main)
    try:
        c = TestClient(main.app)
        health = c.get("/healthz").json()
        assert health["status"] == "offline" and health["storage"] == "offline"
        assert health["supabase"] is False  # the local file is never reported as Supabase
        r = c.post("/train", json={
            "dataset_id": 1, "lr": 0.01, "batch_size": 16, "epochs": 1,
            "experiment_group": "test", "run_name": "baseline-seed0",
        })
        assert r.status_code == 200, r.text
        run_id = r.json()["run_id"]
        assert c.get(f"/runs/{run_id}").json()["run_name"] == "baseline-seed0"
        import io

        import numpy as np
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.full((28, 28), 90, dtype=np.uint8)).save(buf, format="PNG")
        p = c.post("/predict", data={"run_id": run_id}, files={"file": ("a.png", buf.getvalue(), "image/png")})
        assert p.status_code == 200 and p.json()["logged"] is True and p.json()["cam_png"]
        assert c.get("/predictions").json()[0]["run"]["run_name"] == "baseline-seed0"
        # A file survives a restart: a new store sees the same run.
        assert LocalStore(tmp_path / "db.json").get_run(run_id) is not None
    finally:
        monkeypatch.delenv("SEE_SENSE_LOCAL_DB")
        importlib.reload(db)
        importlib.reload(main)
