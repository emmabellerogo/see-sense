"""Ten-class training end to end on the tiny Fashion-shaped fixture (no real data, no cloud)."""
from __future__ import annotations

import base64
import inspect

import numpy as np
import pytest

from api.training import run_experiment, train_validate
from shared import fashion
from tests.fashion_fixture import tiny_fashion

BASE = dict(split_seed=1, train_seed=0, lr=0.01, batch_size=32, epochs=4)


def test_cnn_learns_all_ten_classes_and_records_everything():
    data = tiny_fashion()
    r = run_experiment(**BASE, arch="improved", augment=True, scheduler="cosine", data=data)
    assert r["metrics"]["accuracy"] > 0.8 and r["metrics"]["macro_f1"] > 0.75
    assert r["train_loss"][-1] < r["train_loss"][0]
    cm = np.array(r["confusion_matrix"])
    assert cm.shape == (10, 10) and cm.sum() == len(data.y_test) == r["split_sizes"]["test"]
    pc = r["per_class"]
    assert all(len(pc[k]) == 10 for k in ("precision", "recall", "f1", "support"))
    assert sum(pc["support"]) == cm.sum()
    assert np.isclose(pc["macro_f1"], r["metrics"]["macro_f1"])
    assert r["split_sizes"]["train"] + r["split_sizes"]["val"] == len(data.y_train)
    assert isinstance(r["model_b64"], str) and base64.b64decode(r["model_b64"])


def test_every_epoch_records_training_and_validation_loss_and_accuracy():
    r = run_experiment(**BASE, arch="baseline", data=tiny_fashion())
    e = BASE["epochs"]
    for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr_history", "epoch_seconds"):
        assert len(r[key]) == e, key
    assert all(0.0 <= a <= 1.0 for a in r["train_accuracy"] + r["val_accuracy"])


def test_the_checkpoint_is_selected_on_validation_loss_not_test():
    r = run_experiment(**BASE, arch="improved", data=tiny_fashion())
    assert r["best_epoch"] == int(np.argmin(r["val_loss"]))


def test_train_validate_has_no_way_to_receive_test_data():
    params = inspect.signature(train_validate).parameters
    assert not any("test" in p for p in params)


def test_validation_images_come_from_the_official_training_images_only():
    data = tiny_fashion()
    r = run_experiment(**BASE, data=data)
    ident = r["dataset"]
    tr, va = fashion.split_train_val(data.y_train, ident["n_val"], BASE["split_seed"])
    assert ident["train_index_sha256"] == fashion.index_sha256(tr)
    assert ident["val_index_sha256"] == fashion.index_sha256(va)
    assert r["split_sizes"]["test"] == len(data.y_test)  # the official test partition, untouched


def test_train_limit_is_a_recorded_subset_and_validation_is_unchanged():
    data = tiny_fashion()
    full = run_experiment(**BASE, data=data)
    sub = run_experiment(**BASE, train_limit=120, data=data)
    assert sub["split_sizes"]["train"] == 120 and sub["config"]["train_limit"] == 120
    assert sub["dataset"]["val_index_sha256"] == full["dataset"]["val_index_sha256"]
    with pytest.raises(ValueError):
        run_experiment(**BASE, train_limit=10_000, data=data)


def test_run_experiment_requires_the_test_partition():
    data = tiny_fashion()
    no_test = fashion.FashionData(data.x_train, data.y_train, None, None, data.file_sha256)
    with pytest.raises(ValueError, match="test partition"):
        run_experiment(**BASE, data=no_test)


@pytest.mark.parametrize("bad", [dict(val_size=0.0), dict(val_size=0.999)])
def test_degenerate_validation_fractions_are_rejected(bad):
    with pytest.raises(ValueError):
        run_experiment(**{**BASE, **bad}, data=tiny_fashion())


def test_pilot_trains_on_train_and_validation_only(monkeypatch):
    from experiments import pilot

    data = tiny_fashion()
    train_only = fashion.FashionData(data.x_train, data.y_train, None, None, data.file_sha256)
    recs = pilot.run_pilot(["baseline", "full_improved"], epochs=2, batch_size=32, lr=0.01, split_seed=0,
                           train_seed=0, val_size=0.1, train_limit=None, log=lambda *_: None, data=train_only)
    assert [r["variant"] for r in recs] == ["baseline", "full_improved"]
    assert all(r["uses_test_data"] is False and "test_accuracy" not in r for r in recs)
    assert all(len(r["epoch_seconds"]) == 2 for r in recs)
    # Handing the pilot a dataset that includes the test partition is refused outright.
    with pytest.raises(AssertionError, match="test partition"):
        pilot.run_pilot(["baseline"], 1, 32, 0.01, 0, 0, 0.1, None, log=lambda *_: None, data=data)
    est = pilot.estimate(recs, epochs=10, seeds=3)
    assert est["runs"] == 2 * 3 and est["sequential_minutes"] > 0


def test_pilot_main_never_opens_test_files(monkeypatch, tmp_path):
    from experiments import pilot

    seen = {}
    data = tiny_fashion()

    def fake_load(path=None, include_test=True, verify=True):
        seen["include_test"] = include_test
        return fashion.FashionData(data.x_train, data.y_train, None, None, data.file_sha256)

    monkeypatch.setattr(fashion, "load", fake_load)
    out = tmp_path / "p.jsonl"
    assert pilot.main(["--epochs", "1", "--batch-size", "32", "--lr", "0.01",
                       "--variants", "baseline", "--out", str(out)]) == 0
    assert seen == {"include_test": False} and out.read_text().count("\n") == 1


def test_train_endpoint_registers_trains_and_reads_back(client):
    ds = client.post("/datasets", json={"name": "fm"}).json()
    assert ds["source"] == "fashion_mnist" and ds["n_classes"] == 10 and len(ds["class_names"]) == 10
    resp = client.post("/train", json={"dataset_id": ds["id"], "lr": 0.01, "batch_size": 32, "epochs": 3,
                                       "model_variant": "improved", "augment": True, "scheduler": "cosine"})
    assert resp.status_code == 200, resp.text
    run = client.get(f"/runs/{resp.json()['run_id']}").json()
    assert run["config"]["dataset"] == "fashion-mnist" and run["config"]["n_classes"] == 10
    assert run["calibration"]["method"] == "temperature_scaling"
    assert len(run["test_per_class"]["precision"]) == 10
    assert len(run["train_accuracy_history"]) == len(run["val_accuracy_history"]) == 3
    assert run["artifact_version"] == 3


def test_a_historical_synthetic_dataset_cannot_be_retrained(client):
    from api import db

    old = db.insert_dataset("old-shapes", 800, 28, 4, 0.1, 42)  # source defaults to synthetic_shapes
    r = client.post("/train", json={"dataset_id": old["id"], "lr": 0.01, "batch_size": 32, "epochs": 1})
    assert r.status_code == 422 and "historical" in r.json()["detail"]


def test_missing_dataset_files_on_the_server_are_a_clear_503(client, monkeypatch):
    def gone(*a, **k):
        raise fashion.DatasetUnavailable("Fashion-MNIST files are missing")
    ds = client.post("/datasets", json={"name": "fm"}).json()
    monkeypatch.setattr(fashion, "load", gone)
    r = client.post("/train", json={"dataset_id": ds["id"], "lr": 0.01, "batch_size": 32, "epochs": 1})
    assert r.status_code == 503 and "missing" in r.json()["detail"]
    assert client.post("/datasets", json={"name": "again"}).status_code == 503
