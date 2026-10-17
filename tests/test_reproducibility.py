"""Reproducibility + training-record checks for run_experiment (tiny fixture, no cloud access)."""
from __future__ import annotations

import base64
import pickle

import numpy as np
import pytest
import torch

from api.training import build_model, make_splits, run_experiment
from shared import fashion
from tests.fashion_fixture import tiny_fashion

BASE = dict(split_seed=11, train_seed=3, lr=0.01, batch_size=32, epochs=4)
DATA = tiny_fashion()


def _run(**kw):
    return run_experiment(**{**BASE, **kw}, data=DATA)


def _state(result):
    return pickle.loads(base64.b64decode(result["model_b64"]))["state_dict"]


def test_same_seeds_give_identical_runs():
    a = _run(arch="improved", augment=True, scheduler="cosine")
    b = _run(arch="improved", augment=True, scheduler="cosine")
    assert a["train_loss"] == b["train_loss"] and a["val_loss"] == b["val_loss"]
    assert a["metrics"] == b["metrics"] and a["confusion_matrix"] == b["confusion_matrix"]
    assert a["split_sha256"] == b["split_sha256"] and a["calibration"] == b["calibration"]
    sa, sb = _state(a), _state(b)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_train_seed_changes_training_but_not_the_split():
    a, b = _run(), _run(train_seed=4)
    assert a["split_sha256"] == b["split_sha256"]
    assert a["train_loss"] != b["train_loss"]


def test_split_seed_changes_the_split():
    assert _run()["split_sha256"] != _run(split_seed=12)["split_sha256"]


def test_baseline_and_improved_share_partitions_and_initial_weights():
    base, imp = _run(arch="baseline"), _run(arch="improved")
    assert base["split_sha256"] == imp["split_sha256"]
    assert base["split_sizes"] == imp["split_sizes"] == {"train": 450, "val": 50, "test": 100}

    torch.manual_seed(BASE["train_seed"])
    m1 = build_model(28, 10, "baseline")
    torch.manual_seed(BASE["train_seed"])
    m2 = build_model(28, 10, "improved", use_batchnorm=True, dropout=0.3)
    assert torch.equal(m1.features[0].weight, m2.features[0].weight)
    assert torch.equal(m1.classifier[-1].weight, m2.classifier[-1].weight)


def test_legacy_single_array_splits_are_disjoint_and_cover_everything():
    tr, va, te = make_splits(400, 0.2, 0.1, split_seed=5)
    assert len(set(tr) | set(va) | set(te)) == 400
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_augmentation_only_touches_training_data():
    a, b = _run(augment=False), _run(augment=True)
    assert a["split_sha256"] == b["split_sha256"]
    assert a["dataset"]["val_index_sha256"] == b["dataset"]["val_index_sha256"]
    assert a["train_loss"] != b["train_loss"]  # training batches changed
    assert a["config"]["augment"] is False and b["config"]["augment"] is True


def test_scheduler_changes_learning_rate_only_when_enabled():
    none = _run(scheduler="none")["lr_history"]
    cos = _run(scheduler="cosine")["lr_history"]
    step = _run(scheduler="step", epochs=6)["lr_history"]
    assert none == [0.01] * 4
    assert cos[0] == 0.01 and all(x > y for x, y in zip(cos, cos[1:]))
    assert step[0] == 0.01 and step[-1] < step[0]


def test_recorded_fields_are_complete_and_consistent():
    r = _run(arch="improved")
    assert r["best_epoch"] == int(np.argmin(r["val_loss"]))  # chosen on validation only
    assert set(r["metrics"]) == {"accuracy", "macro_f1", "test_loss"}
    cm = np.array(r["confusion_matrix"])
    assert cm.shape == (10, 10) and cm.sum() == r["split_sizes"]["test"]
    assert np.isclose(np.trace(cm) / cm.sum(), r["metrics"]["accuracy"])
    # 10-class parameter counts: conv 80 + 1,168, linear 7,850 (+ 48 batch-norm parameters).
    assert r["param_count"] == 9146 and r["train_seconds"] > 0
    assert _run(arch="baseline")["param_count"] == 9098


def test_baseline_records_what_it_actually_uses():
    r = _run(arch="baseline", use_batchnorm=True, dropout=0.5)
    assert r["config"]["use_batchnorm"] is False and r["config"]["dropout"] == 0.0
    art = pickle.loads(base64.b64decode(r["model_b64"]))
    assert art["arch"] == "baseline" and art["use_batchnorm"] is False and art["dropout"] == 0.0
    other = _run(arch="baseline", use_batchnorm=False, dropout=0.0)
    assert r["train_loss"] == other["train_loss"] and r["metrics"] == other["metrics"]


def test_improved_records_its_real_options_independently():
    r = _run(arch="improved", use_batchnorm=False, dropout=0.4)
    assert r["config"]["use_batchnorm"] is False and r["config"]["dropout"] == 0.4
    m = build_model(28, 10, "improved", use_batchnorm=False, dropout=0.4)
    assert r["param_count"] == sum(p.numel() for p in m.parameters())


def test_matched_variants_start_from_identical_first_epoch_when_only_the_scheduler_differs():
    # no_scheduler vs full_improved: the first epoch runs at the same learning rate, so it must match exactly.
    full = _run(arch="improved", augment=True, scheduler="cosine")
    nosch = _run(arch="improved", augment=True, scheduler="none")
    assert full["train_loss"][0] == nosch["train_loss"][0]
    assert full["val_loss"][0] == nosch["val_loss"][0]
    assert full["train_loss"][1] != nosch["train_loss"][1]
