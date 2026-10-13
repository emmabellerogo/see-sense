"""Shared pytest fixtures.

The schema, health, and numerical tests run WITHOUT any cloud access by stubbing
the db module with an in-memory store. The Supabase round-trip test is skipped
unless real credentials are present, so `pytest` stays green in CI/offline.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from shared.data import IMG_SIZE, N_CLASSES


# The runs columns migration 002 adds, read from the SQL file itself so the fake
# database below accepts exactly the column names a real one would.
MIGRATION_002 = Path(__file__).parent.parent / "db" / "migrations" / "002_experiment_tracking.sql"
MIGRATION_COLUMNS = tuple(
    re.findall(r"add column if not exists (\w+)", MIGRATION_002.read_text())
)


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the Supabase layer stubbed by an in-memory fake."""
    from fastapi.testclient import TestClient

    from api import db

    store = {
        "datasets": {},
        "runs": {},
        "artifacts": {},
        "image_metadata": [],
    }
    counters = {"datasets": 0, "runs": 0}

    def insert_dataset(name, n_rows, img_size, n_classes, noise, seed):
        counters["datasets"] += 1
        row = {
            "id": counters["datasets"],
            "name": name,
            "n_rows": n_rows,
            "img_size": img_size,
            "n_classes": n_classes,
            "noise": noise,
            "seed": seed,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        store["datasets"][row["id"]] = row
        return row

    def get_dataset(dataset_id):
        return store["datasets"].get(dataset_id)

    def insert_run(dataset_id, lr, batch_size, epochs, metrics, model_b64, experiment=None):
        experiment = dict(experiment or {})
        unknown = set(experiment) - set(MIGRATION_COLUMNS)
        assert not unknown, f"columns not in migration 002: {sorted(unknown)}"
        json.dumps(experiment)  # jsonb columns must be JSON-serializable
        # Mirror a few of the migration's CHECK constraints.
        assert experiment.get("model_variant") in (None, "baseline", "improved")
        assert experiment.get("scheduler") in (None, "none", "cosine", "step")
        for h in ("train_loss_history", "val_loss_history", "val_accuracy_history", "lr_history"):
            assert experiment.get(h) is None or len(experiment[h]) == epochs
        group, name = experiment.get("experiment_group"), experiment.get("run_name")
        if group and name:
            assert not run_name_exists(group, name), "duplicate (experiment_group, run_name)"
        counters["runs"] += 1
        row = {
            "id": counters["runs"],
            "dataset_id": dataset_id,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "created_at": "2026-01-01T00:00:00+00:00",
            **{c: None for c in MIGRATION_COLUMNS},  # old-style rows leave these NULL
            **experiment,
        }
        store["runs"][row["id"]] = row
        store["artifacts"][row["id"]] = model_b64
        return row

    def run_name_exists(experiment_group, run_name):
        return any(
            r["experiment_group"] == experiment_group and r["run_name"] == run_name
            for r in store["runs"].values()
        )

    def get_run(run_id):
        return store["runs"].get(run_id)

    def get_run_artifact(run_id):
        return store["artifacts"].get(run_id)

    def latest_runs(limit=50):
        return list(store["runs"].values())[-limit:][::-1]

    def insert_image_metadata(run_id, sha256, width, height, label, class_name, confidence):
        row = {
            "id": len(store["image_metadata"]) + 1,
            "run_id": run_id,
            "sha256": sha256,
            "width": width,
            "height": height,
            "label": label,
            "class_name": class_name,
            "confidence": confidence,
        }
        store["image_metadata"].append(row)
        return row

    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_dataset", insert_dataset)
    monkeypatch.setattr(db, "get_dataset", get_dataset)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(db, "run_name_exists", run_name_exists)
    monkeypatch.setattr(db, "get_run", get_run)
    monkeypatch.setattr(db, "get_run_artifact", get_run_artifact)
    monkeypatch.setattr(db, "latest_runs", latest_runs)
    monkeypatch.setattr(db, "insert_image_metadata", insert_image_metadata)

    from api.main import app

    with TestClient(app) as c:
        c._store = store  # expose for assertions
        yield c


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))
