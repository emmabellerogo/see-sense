"""Reproducibility + training-record checks for run_experiment (no cloud access)."""
from __future__ import annotations

import base64
import pickle

import numpy as np
import pytest
import torch

from api.training import (
    _augment,
    build_model,
    make_splits,
    run_experiment,
    train_cnn,
)
from shared.data import IMG_SIZE, N_CLASSES, generate_one

BASE = dict(
    n_rows=400, img_size=IMG_SIZE, noise=0.1,
    data_seed=7, split_seed=11, train_seed=3,
    lr=0.01, batch_size=32, epochs=4,
)


def _state(result):
    return pickle.loads(base64.b64decode(result["model_b64"]))["state_dict"]


def test_same_seeds_give_identical_runs():
    a = run_experiment(**BASE, arch="improved", augment=True, scheduler="cosine")
    b = run_experiment(**BASE, arch="improved", augment=True, scheduler="cosine")
    assert a["train_loss"] == b["train_loss"]
    assert a["val_loss"] == b["val_loss"]
    assert a["metrics"] == b["metrics"]
    assert a["confusion_matrix"] == b["confusion_matrix"]
    assert a["split_sha256"] == b["split_sha256"]
    sa, sb = _state(a), _state(b)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_train_seed_changes_training_but_not_the_split():
    a = run_experiment(**BASE)
    b = run_experiment(**{**BASE, "train_seed": 4})
    assert a["split_sha256"] == b["split_sha256"]
    assert a["train_loss"] != b["train_loss"]


def test_split_seed_changes_the_split():
    a = run_experiment(**BASE)
    b = run_experiment(**{**BASE, "split_seed": 12})
    assert a["split_sha256"] != b["split_sha256"]


def test_baseline_and_improved_share_partitions_and_initial_weights():
    base = run_experiment(**BASE, arch="baseline")
    imp = run_experiment(**BASE, arch="improved")
    assert base["split_sha256"] == imp["split_sha256"]
    assert base["split_sizes"] == imp["split_sizes"] == {"train": 280, "val": 40, "test": 80}

    torch.manual_seed(BASE["train_seed"])
    m1 = build_model(IMG_SIZE, N_CLASSES, "baseline")
    torch.manual_seed(BASE["train_seed"])
    m2 = build_model(IMG_SIZE, N_CLASSES, "improved", use_batchnorm=True, dropout=0.3)
    assert torch.equal(m1.features[0].weight, m2.features[0].weight)
    assert torch.equal(m1.classifier[-1].weight, m2.classifier[-1].weight)


def test_partitions_are_disjoint_and_cover_everything():
    tr, va, te = make_splits(400, 0.2, 0.1, split_seed=5)
    assert len(set(tr) | set(va) | set(te)) == 400
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    tr2, va2, te2 = make_splits(400, 0.2, 0.1, split_seed=5)
    assert (tr == tr2).all() and (va == va2).all() and (te == te2).all()


def test_augmentation_is_label_preserving():
    gen = torch.Generator().manual_seed(0)
    for kind in (0, 1):  # horizontal / vertical bars, noise-free
        imgs = [generate_one(noise=0.0, seed=s, kind=kind)[0] for s in range(40)]
        x = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0).unsqueeze(1)
        out = _augment(x, 2, gen)
        assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
        assert not torch.equal(out, x)
        rows = out[:, 0].sum(dim=2).max(dim=1).values  # strongest row
        cols = out[:, 0].sum(dim=1).max(dim=1).values  # strongest column
        # A horizontal bar must stay row-dominant, a vertical bar column-dominant:
        # a 90-degree rotation would swap these and change the label.
        assert ((rows > 3 * cols) if kind == 0 else (cols > 3 * rows)).all()


def test_augmentation_only_touches_training_data():
    a = run_experiment(**BASE, augment=False)
    b = run_experiment(**BASE, augment=True)
    assert a["split_sha256"] == b["split_sha256"]
    assert a["train_loss"] != b["train_loss"]  # training batches changed
    assert a["config"]["augment"] is False and b["config"]["augment"] is True


def test_scheduler_changes_learning_rate_only_when_enabled():
    none = run_experiment(**BASE, scheduler="none")["lr_history"]
    cos = run_experiment(**BASE, scheduler="cosine")["lr_history"]
    step = run_experiment(**{**BASE, "epochs": 6}, scheduler="step")["lr_history"]
    assert none == [0.01] * 4
    assert cos[0] == 0.01 and all(x > y for x, y in zip(cos, cos[1:]))
    assert step[0] == 0.01 and step[-1] < step[0]


def test_recorded_fields_are_complete_and_consistent():
    r = run_experiment(**BASE, arch="improved")
    e = BASE["epochs"]
    assert len(r["train_loss"]) == len(r["val_loss"]) == len(r["val_accuracy"]) == e
    assert r["best_epoch"] == int(np.argmin(r["val_loss"]))  # chosen on validation only
    assert set(r["metrics"]) == {"accuracy", "macro_f1", "test_loss"}
    cm = np.array(r["confusion_matrix"])
    assert cm.shape == (N_CLASSES, N_CLASSES) and cm.sum() == r["split_sizes"]["test"]
    assert np.isclose(np.trace(cm) / cm.sum(), r["metrics"]["accuracy"])
    assert r["param_count"] == 4436 and r["train_seconds"] > 0
    assert run_experiment(**BASE, arch="baseline")["param_count"] == 4388


def test_train_cnn_wrapper_is_unchanged_for_callers():
    metrics, b64, loss = train_cnn(
        n_rows=400, img_size=IMG_SIZE, noise=0.1, seed=42,
        lr=0.01, batch_size=32, epochs=3,
    )
    assert set(metrics) == {"accuracy", "macro_f1"} and isinstance(b64, str) and len(loss) == 3


@pytest.mark.parametrize("bad", [dict(val_size=0.0), dict(test_size=0.95)])
def test_degenerate_splits_are_rejected(bad):
    with pytest.raises(ValueError):
        run_experiment(**{**BASE, **bad})
