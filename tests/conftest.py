"""Shared pytest fixtures.

The schema, health, and numerical tests run WITHOUT any cloud access by stubbing
the db module with an in-memory store, and WITHOUT the real Fashion-MNIST files by
substituting a tiny Fashion-MNIST-shaped dataset (tests/fashion_fixture.py). The live Supabase
tests are skipped unless explicitly requested, so `pytest` stays green in CI/offline.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from shared.fashion import IMG_SIZE, N_CLASSES  # noqa: F401  (re-exported for tests)
from tests.fashion_fixture import tiny_fashion

MIGRATIONS = Path(__file__).parent.parent / "db" / "migrations"


def columns_added(sql_file: str, table: str) -> tuple[str, ...]:
    """Column names a migration adds to ``public.<table>`` (read from the SQL itself)."""
    text = (MIGRATIONS / sql_file).read_text()
    cols: list[str] = []
    for block in re.split(r"(?=alter table )", text):
        if re.match(rf"alter table (?:public\.)?{table}\s", block):
            statement = re.split(r";[ \t]*\n", block, maxsplit=1)[0]
            cols += re.findall(r"add column if not exists (\w+)", statement)
    return tuple(cols)


# The runs columns each migration adds, so the fake database accepts exactly the column
# names a real one would.
MIGRATION_COLUMNS = columns_added("002_experiment_tracking.sql", "runs")
MIGRATION_004_COLUMNS = columns_added("004_predictions.sql", "runs")
MIGRATION_005_COLUMNS = columns_added("005_fashion_mnist_and_atomic_logging.sql", "runs")
DATASET_005_COLUMNS = columns_added("005_fashion_mnist_and_atomic_logging.sql", "datasets")
ALL_RUN_EXTRA = MIGRATION_004_COLUMNS + MIGRATION_005_COLUMNS


@pytest.fixture
def tiny_data(monkeypatch):
    """Make ``fashion.load`` return the tiny dataset (the real files are never opened)."""
    from shared import fashion

    data = tiny_fashion()
    monkeypatch.setattr(fashion, "load", lambda *a, **k: data)
    return data


@pytest.fixture
def client(monkeypatch, tiny_data):
    """A TestClient with the Supabase layer stubbed by an in-memory fake."""
    from fastapi.testclient import TestClient

    from api import db

    store = {
        "datasets": {},
        "runs": {},
        "artifacts": {},
        "image_metadata": [],
        "predictions": [],
        "fail_log_with": None,   # set to an exception instance to make logging fail
    }
    counters = {"datasets": 0, "runs": 0}

    def insert_dataset(name, n_rows, img_size, n_classes, noise=0.0, seed=0,
                       source=None, class_names=None, identity=None):
        counters["datasets"] += 1
        row = {
            "id": counters["datasets"], "name": name, "n_rows": n_rows, "img_size": img_size,
            "n_classes": n_classes, "noise": noise, "seed": seed,
            "created_at": "2026-01-01T00:00:00+00:00",
            "source": source or "synthetic_shapes", "class_names": class_names, "identity": identity,
        }
        assert set(row) - {"id", "created_at"} <= {"name", "n_rows", "img_size", "n_classes", "noise",
                                                    "seed", *DATASET_005_COLUMNS}
        store["datasets"][row["id"]] = row
        return row

    def get_dataset(dataset_id):
        return store["datasets"].get(dataset_id)

    def insert_run(dataset_id, lr, batch_size, epochs, metrics, model_b64, experiment=None):
        experiment = dict(experiment or {})
        unknown = set(experiment) - set(MIGRATION_COLUMNS) - set(ALL_RUN_EXTRA)
        assert not unknown, f"columns not in migrations 002/004/005: {sorted(unknown)}"
        json.dumps(experiment)  # jsonb columns must be JSON-serializable
        # Mirror a few of the migration's CHECK constraints.
        assert experiment.get("model_variant") in (None, "baseline", "improved")
        assert experiment.get("scheduler") in (None, "none", "cosine", "step")
        for h in ("train_loss_history", "val_loss_history", "val_accuracy_history", "lr_history",
                  "train_accuracy_history"):
            assert experiment.get(h) is None or len(experiment[h]) == epochs
        for obj in ("calibration", "test_per_class", "config"):
            assert experiment.get(obj) is None or isinstance(experiment[obj], dict)
        group, name = experiment.get("experiment_group"), experiment.get("run_name")
        if group and name:
            assert not run_name_exists(group, name), "duplicate (experiment_group, run_name)"
        counters["runs"] += 1
        row = {
            "id": counters["runs"], "dataset_id": dataset_id, "lr": lr, "batch_size": batch_size,
            "epochs": epochs, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "created_at": "2026-01-01T00:00:00+00:00",
            **{c: None for c in MIGRATION_COLUMNS + ALL_RUN_EXTRA},  # old-style rows leave these NULL
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

    def log_prediction(run_id, sha256, width, height, label, class_name, probabilities, confidence):
        """Atomic like the Postgres function: both rows are added, or neither."""
        if store["fail_log_with"] is not None:
            raise store["fail_log_with"]
        assert run_id in store["runs"], "served_by_run_id must reference runs (foreign key)"
        n = len(store["predictions"]) + 1
        store["image_metadata"].append({
            "id": len(store["image_metadata"]) + 1, "run_id": run_id, "sha256": sha256,
            "width": width, "height": height, "label": label, "class_name": class_name,
            "confidence": confidence,
        })
        store["predictions"].append({
            "id": n, "image_sha256": sha256, "predicted_label": label, "predicted_class": class_name,
            "probabilities": list(probabilities), "confidence": confidence,
            "served_by_run_id": run_id, "created_at": "2026-01-01T00:00:00+00:00",
        })
        return n

    def _joined(row):
        run = store["runs"][row["served_by_run_id"]]
        public = {k: v for k, v in row.items() if k != "image_sha256"}  # like the real select list
        return {**public, "runs": {k: run.get(k) for k in (
            "run_name", "experiment_group", "model_variant", "accuracy", "macro_f1")}}

    def list_predictions(limit=50):
        return [_joined(r) for r in reversed(store["predictions"])][:limit]

    def get_prediction(prediction_id):
        row = next((r for r in store["predictions"] if r["id"] == prediction_id), None)
        return _joined(row) if row else None

    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "insert_dataset", insert_dataset)
    monkeypatch.setattr(db, "get_dataset", get_dataset)
    monkeypatch.setattr(db, "insert_run", insert_run)
    monkeypatch.setattr(db, "run_name_exists", run_name_exists)
    monkeypatch.setattr(db, "get_run", get_run)
    monkeypatch.setattr(db, "get_run_artifact", get_run_artifact)
    monkeypatch.setattr(db, "latest_runs", latest_runs)
    monkeypatch.setattr(db, "log_prediction", log_prediction)
    monkeypatch.setattr(db, "list_predictions", list_predictions)
    monkeypatch.setattr(db, "get_prediction", get_prediction)

    from api.main import app

    with TestClient(app) as c:
        c._store = store  # expose for assertions
        yield c


def has_supabase_creds() -> bool:
    """True only when credentials are present AND live tests were asked for explicitly.

    The live tests WRITE to the real project (a dataset, a trained run, predictions).
    api/db.py loads .env on import, so credentials can be present without the person
    running pytest intending to touch the cloud; hence the explicit opt-in:

        SEE_SENSE_RUN_LIVE_TESTS=1 pytest tests/test_live_supabase.py
    """
    if os.environ.get("SEE_SENSE_RUN_LIVE_TESTS") == "1":
        # api.db loads .env only when imported, which is after collection; load it here (never overrides
        # variables that are already set, even blank ones, so `env SUPABASE_URL= ...` still disables it).
        from dotenv import load_dotenv

        load_dotenv()
    return bool(
        os.environ.get("SEE_SENSE_RUN_LIVE_TESTS") == "1"
        and os.environ.get("SUPABASE_URL")
        and os.environ.get("SUPABASE_SERVICE_KEY")
    )
