"""API <-> database wiring for experiment records, against the in-memory fake DB.

The fake (tests/conftest.py) accepts only column names that appear in migration
002 and mirrors some of its CHECK constraints. No Supabase access happens here.
"""
from __future__ import annotations

import base64
import copy
import pickle
import types

import numpy as np
import pytest

from api import db
from api.experiments import same_split, split_identity, validate_result
from api.training import run_experiment
from shared.data import N_CLASSES
from tests.conftest import MIGRATION_COLUMNS

EPOCHS = 3
FAST = {"lr": 0.01, "batch_size": 32, "epochs": EPOCHS}


def _dataset(client, noise=0.1, seed=42, n_rows=300, name="d"):
    return client.post(
        "/datasets", json={"name": name, "n_rows": n_rows, "noise": noise, "seed": seed}
    ).json()


def _checkpoint(client, run_id):
    return pickle.loads(base64.b64decode(client._store["artifacts"][run_id]))


# --------------------------------------------------------------------------
# baseline and improved runs: save, then retrieve
# --------------------------------------------------------------------------
def test_old_style_request_saves_a_baseline_run(client):
    ds = _dataset(client)
    # Exactly what a pre-experiment client sends.
    resp = client.post("/train", json={"dataset_id": ds["id"], **FAST})
    assert resp.status_code == 200
    body = resp.json()

    # Baseline records what it really uses, not the request's unused defaults.
    assert body["model_variant"] == "baseline"
    assert body["use_batchnorm"] is False and body["dropout"] == 0.0
    assert body["augment"] is False and body["scheduler"] == "none"
    assert body["artifact_version"] == 2

    got = client.get(f"/runs/{body['run_id']}").json()
    assert got["config"]["arch"] == "baseline" and got["config"]["use_batchnorm"] is False
    assert got["accuracy"] == body["metrics"]["accuracy"]  # existing columns reused
    assert got["macro_f1"] == body["metrics"]["macro_f1"]
    assert got["test_loss"] == body["metrics"]["test_loss"]
    ck = _checkpoint(client, body["run_id"])
    assert ck["arch"] == "baseline" and ck["use_batchnorm"] is False and ck["dropout"] == 0.0


def test_improved_run_saves_settings_curves_metrics_and_checkpoint(client):
    ds = _dataset(client)
    resp = client.post("/train", json={
        "dataset_id": ds["id"], **FAST,
        "model_variant": "improved", "use_batchnorm": True, "dropout": 0.25,
        "augment": True, "augment_max_shift": 1, "scheduler": "cosine",
        "split_seed": 5, "train_seed": 9,
        "experiment_group": "g1", "run_name": "improved-full",
    })
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    got = client.get(f"/runs/{run_id}").json()

    assert (got["experiment_group"], got["run_name"]) == ("g1", "improved-full")
    assert got["model_variant"] == "improved"
    assert got["use_batchnorm"] is True and got["dropout"] == 0.25
    assert got["augment"] is True and got["augment_max_shift"] == 1
    assert got["scheduler"] == "cosine"
    # Seeds: images come from the registered dataset's seed.
    assert (got["data_seed"], got["split_seed"], got["train_seed"]) == (42, 5, 9)
    # Budget and optimizer.
    assert (got["lr"], got["batch_size"], got["epochs"], got["optimizer"]) == (0.01, 32, EPOCHS, "adam")
    assert got["config"]["n_rows"] == 300 and got["config"]["noise"] == 0.1
    # Curves: one entry per epoch; cosine lowers the rate.
    for k in ("train_loss_history", "val_loss_history", "val_accuracy_history", "lr_history"):
        assert len(got[k]) == EPOCHS
    assert got["lr_history"][0] == 0.01 and got["lr_history"][-1] < 0.01
    assert got["best_epoch"] == int(np.argmin(got["val_loss_history"]))
    # Test results.
    cm = np.array(got["confusion_matrix"])
    assert cm.shape == (N_CLASSES, N_CLASSES)
    assert cm.sum() == got["n_test"] == 60 and (got["n_train"], got["n_val"]) == (210, 30)
    assert np.isclose(np.trace(cm) / cm.sum(), got["accuracy"])
    assert got["test_loss"] > 0 and got["param_count"] == 4436 and got["train_seconds"] > 0
    assert len(got["split_sha256"]) == 64

    # The checkpoint went to run_artifacts and is usable for prediction.
    ck = _checkpoint(client, run_id)
    assert (ck["arch"], ck["use_batchnorm"], ck["dropout"]) == ("improved", True, 0.25)
    pred = client.post("/predict_sample", json={"run_id": run_id, "seed": 1, "true_class": 2}).json()
    assert len(pred["probs"]) == N_CLASSES and base64.b64decode(pred["cam_png"])


def test_list_returns_both_variants(client):
    ds = _dataset(client)
    for variant in ("baseline", "improved"):
        r = client.post("/train", json={"dataset_id": ds["id"], **FAST, "model_variant": variant})
        assert r.status_code == 200
    runs = client.get("/runs").json()
    assert {r["model_variant"] for r in runs} == {"baseline", "improved"}


def test_old_rows_without_experiment_data_are_still_readable(client):
    # A row saved the pre-migration way: no `experiment` argument at all.
    row = db.insert_run(1, 0.01, 32, 15, {"accuracy": 0.9, "macro_f1": 0.88}, "abc")
    got = client.get(f"/runs/{row['id']}").json()
    assert got["accuracy"] == 0.9
    assert all(got[c] is None for c in MIGRATION_COLUMNS)
    assert any(r["id"] == row["id"] for r in client.get("/runs").json())


# --------------------------------------------------------------------------
# request validation / duplicates
# --------------------------------------------------------------------------
def test_duplicate_run_name_in_a_group_is_rejected(client):
    ds = _dataset(client)
    body = {"dataset_id": ds["id"], **FAST, "experiment_group": "g", "run_name": "a"}
    assert client.post("/train", json=body).status_code == 200
    assert client.post("/train", json=body).status_code == 409
    assert client.post("/train", json={**body, "experiment_group": "other"}).status_code == 200
    assert len(client._store["runs"]) == 2  # the rejected run saved nothing


@pytest.mark.parametrize("bad", [
    {"model_variant": "huge"}, {"scheduler": "warp"}, {"dropout": 1.0},
    {"test_size": 0.6, "val_size": 0.5}, {"split_seed": -1}, {"run_name": ""},
])
def test_bad_train_requests_are_rejected(client, bad):
    ds = _dataset(client)
    assert client.post("/train", json={"dataset_id": ds["id"], **FAST, **bad}).status_code == 422


def test_unknown_dataset_is_404(client):
    assert client.post("/train", json={"dataset_id": 999, **FAST}).status_code == 404


# --------------------------------------------------------------------------
# split identity needs the dataset and generation settings, not just the hash
# --------------------------------------------------------------------------
def test_baseline_and_improved_on_same_dataset_share_a_split(client):
    ds = _dataset(client)
    ids = [
        client.post("/train", json={"dataset_id": ds["id"], **FAST, "model_variant": v,
                                    "split_seed": 3}).json()["run_id"]
        for v in ("baseline", "improved")
    ]
    a, b = (client.get(f"/runs/{i}").json() for i in ids)
    assert a["split_sha256"] == b["split_sha256"]
    assert same_split(a, b)


def test_same_split_hash_on_different_data_is_not_the_same_split(client):
    clean = _dataset(client, noise=0.1, name="clean")
    noisy = _dataset(client, noise=0.4, name="noisy")
    seeded = _dataset(client, noise=0.1, seed=7, name="reseeded")
    runs = [
        client.get(f"/runs/{client.post('/train', json={'dataset_id': d['id'], **FAST, 'split_seed': 3}).json()['run_id']}").json()
        for d in (clean, noisy, seeded)
    ]
    # The hash only covers row indices, so it is identical across all three...
    assert len({r["split_sha256"] for r in runs}) == 1
    # ...but the images differ, so none of them may be treated as the same split.
    assert not same_split(runs[0], runs[1]) and not same_split(runs[0], runs[2])
    assert same_split(runs[0], copy.deepcopy(runs[0]))


def test_runs_without_metadata_are_never_comparable():
    old = {"dataset_id": 1, "split_sha256": None, "config": None}
    assert split_identity(old) is None and not same_split(old, old)


# --------------------------------------------------------------------------
# validation of a training result
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def good_result():
    return run_experiment(
        n_rows=200, img_size=28, noise=0.1, data_seed=1, split_seed=2, train_seed=3,
        lr=0.01, batch_size=32, epochs=2,
    )


def _broken(result, mutate):
    r = copy.deepcopy(result)
    mutate(r)
    return r


def test_a_genuine_result_validates(good_result):
    validate_result(good_result)


@pytest.mark.parametrize("name,mutate", [
    ("too few rows", lambda r: r["confusion_matrix"].pop()),
    ("ragged row", lambda r: r["confusion_matrix"][0].pop()),
    ("negative count", lambda r: r["confusion_matrix"][0].__setitem__(0, -1)),
    ("float count", lambda r: r["confusion_matrix"][0].__setitem__(0, 1.5)),
    ("bool count", lambda r: r["confusion_matrix"][0].__setitem__(0, True)),
    ("total != test size", lambda r: r["confusion_matrix"][1].__setitem__(1, r["confusion_matrix"][1][1] + 1)),
    ("short history", lambda r: r["val_loss"].pop()),
    ("splits do not cover rows", lambda r: r["split_sizes"].__setitem__("train", 1)),
    ("best_epoch out of range", lambda r: r.__setitem__("best_epoch", 99)),
])
def test_inconsistent_results_are_refused(good_result, name, mutate):
    with pytest.raises(ValueError):
        validate_result(_broken(good_result, mutate))
    with pytest.raises(ValueError):
        db.experiment_columns(_broken(good_result, mutate))


# --------------------------------------------------------------------------
# the column mapping matches migration 002 exactly
# --------------------------------------------------------------------------
def test_mapping_uses_exactly_the_migration_columns(good_result):
    cols = db.experiment_columns(good_result, "g", "n")
    assert set(cols) == set(MIGRATION_COLUMNS) == set(db.EXPERIMENT_COLUMNS)
    assert "accuracy" not in cols and "macro_f1" not in cols  # existing columns
    assert "model_b64" not in cols  # the checkpoint is not experiment metadata
    assert cols["model_variant"] == "baseline" and cols["use_batchnorm"] is False
    assert cols["confusion_matrix"] == good_result["confusion_matrix"]


# --------------------------------------------------------------------------
# the real db.insert_run, with a fake Supabase client
# --------------------------------------------------------------------------
class _FakeSupabase:
    def __init__(self, fail_artifact=False):
        self.calls, self.fail_artifact = [], fail_artifact

    def table(self, name):
        outer = self

        class _T:
            def insert(self, row):
                outer.calls.append(("insert", name, row))
                self._row = row
                return self

            def delete(self):
                outer.calls.append(("delete", name, None))
                return self

            def eq(self, col, val):
                outer.calls.append(("eq", col, val))
                return self

            def execute(self):
                if name == "run_artifacts" and outer.fail_artifact:
                    raise RuntimeError("boom")
                return types.SimpleNamespace(data=[{"id": 7, **getattr(self, "_row", {})}])

        return _T()


def test_insert_run_writes_metadata_to_runs_and_checkpoint_to_run_artifacts(monkeypatch, good_result):
    fake = _FakeSupabase()
    monkeypatch.setattr(db, "get_client", lambda: fake)
    exp = db.experiment_columns(good_result, "g", "n")
    run = db.insert_run(3, 0.01, 32, 2, good_result["metrics"], good_result["model_b64"], exp)

    (_, t1, runs_row), (_, t2, art_row) = fake.calls
    assert (t1, t2) == ("runs", "run_artifacts")
    assert runs_row["accuracy"] == good_result["metrics"]["accuracy"]
    assert runs_row["macro_f1"] == good_result["metrics"]["macro_f1"]
    assert set(runs_row) == {"dataset_id", "lr", "batch_size", "epochs", "accuracy", "macro_f1", *MIGRATION_COLUMNS}
    assert "model_b64" not in runs_row  # never duplicated into runs
    assert art_row == {"run_id": run["id"], "model_b64": good_result["model_b64"]}


def test_failed_checkpoint_save_does_not_leave_a_run_behind(monkeypatch, good_result):
    fake = _FakeSupabase(fail_artifact=True)
    monkeypatch.setattr(db, "get_client", lambda: fake)
    with pytest.raises(RuntimeError):
        db.insert_run(3, 0.01, 32, 2, good_result["metrics"], "ckpt", None)
    assert ("delete", "runs", None) in fake.calls and ("eq", "id", 7) in fake.calls
