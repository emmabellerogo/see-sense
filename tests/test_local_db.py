"""The optional offline database (SEE_SENSE_LOCAL_DB) used for local runs."""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from api.local_db import DEFAULT_DATASET, LocalStore


def test_new_file_is_seeded_with_dataset_1(tmp_path):
    store = LocalStore(tmp_path / "db.json")
    ds = store.get_dataset(1)
    assert ds is not None
    assert {k: ds[k] for k in DEFAULT_DATASET} == DEFAULT_DATASET
    assert store.ping()


def test_duplicate_run_name_is_rejected(tmp_path):
    store = LocalStore(tmp_path / "db.json")
    exp = {"experiment_group": "g", "run_name": "baseline-seed0"}
    store.insert_run(1, 0.01, 32, 1, {"accuracy": 1.0, "macro_f1": 1.0}, "AAA", exp)
    assert store.run_name_exists("g", "baseline-seed0")
    with pytest.raises(ValueError):
        store.insert_run(1, 0.01, 32, 1, {"accuracy": 1.0, "macro_f1": 1.0}, "AAA", exp)


def test_api_trains_and_reads_back_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("SEE_SENSE_LOCAL_DB", str(tmp_path / "db.json"))
    from api import db, main

    importlib.reload(db)
    importlib.reload(main)
    try:
        c = TestClient(main.app)
        assert c.get("/healthz").json()["supabase"] is True
        ds = c.post("/datasets", json={"name": "tiny", "n_rows": 100, "noise": 0.1, "seed": 1}).json()
        r = c.post("/train", json={
            "dataset_id": ds["id"], "lr": 0.01, "batch_size": 16, "epochs": 1,
            "experiment_group": "test", "run_name": "baseline-seed0",
        })
        assert r.status_code == 200, r.text
        run_id = r.json()["run_id"]
        assert c.get(f"/runs/{run_id}").json()["run_name"] == "baseline-seed0"
        p = c.post("/predict_sample", json={"run_id": run_id, "seed": 3})
        assert p.status_code == 200 and p.json()["cam_png"]
        # A file survives a restart: a new store sees the same run.
        assert LocalStore(tmp_path / "db.json").get_run(run_id) is not None
    finally:
        monkeypatch.delenv("SEE_SENSE_LOCAL_DB")
        importlib.reload(db)
        importlib.reload(main)
