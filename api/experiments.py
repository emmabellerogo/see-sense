"""Integrity checks for experiment records (no database or torch needed).

Two jobs:
  * ``validate_result`` -- refuse to save a training result that does not add up
    (bad confusion matrix, histories of the wrong length, splits that do not
    cover the dataset).
  * ``same_split`` -- decide whether two saved runs really saw the same examples.

Why ``same_split`` looks at more than the split hash: ``split_sha256`` hashes only
the row *indices* of the train/val/test partition, and those depend just on the
row count, the fractions and ``split_seed``. Two runs on different datasets
(different noise or data seed) can therefore share a hash while seeing different
images. Identical examples need the same dataset, the same generation settings
AND the same split.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple


def metrics_from_confusion(cm: List[List[int]]) -> Dict[str, Any]:
    """Per-class precision / recall / F1 / support and macro-F1 from a confusion matrix.

    Rows = true class, columns = predicted class. A class with no predictions (or no
    examples) gets 0.0 for the undefined ratio, the same convention everywhere.
    """
    k = len(cm)
    precision, recall, f1, support = [], [], [], []
    for c in range(k):
        tp = cm[c][c]
        predicted = sum(cm[r][c] for r in range(k))
        actual = sum(cm[c])
        p = tp / predicted if predicted else 0.0
        r = tp / actual if actual else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
        support.append(actual)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support,
            "macro_f1": sum(f1) / k if k else 0.0}


def validate_result(result: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a ``run_experiment`` result is internally inconsistent."""
    cfg, sizes = result["config"], result["split_sizes"]
    epochs = cfg["epochs"]
    n_classes = cfg.get("n_classes", 4)  # records written before the Fashion-MNIST change are 4-class

    if min(sizes.values()) < 1:
        raise ValueError(f"split sizes {sizes} contain an empty partition")
    if "n_rows" in cfg:  # historical synthetic runs: one pool cut into three
        if sum(sizes.values()) != cfg["n_rows"]:
            raise ValueError(f"split sizes {sizes} do not partition {cfg['n_rows']} rows")
    elif cfg.get("train_limit") is None:  # official train images = train + validation; test is separate
        if sizes["train"] + sizes["val"] != result["dataset"]["official_train_images"]:
            raise ValueError(f"train + validation sizes {sizes} do not cover the official training set")
    elif sizes["train"] > cfg["train_limit"]:
        raise ValueError(f"train size {sizes['train']} exceeds train_limit {cfg['train_limit']}")

    cm = result["confusion_matrix"]
    if not isinstance(cm, list) or len(cm) != n_classes:
        raise ValueError(f"confusion matrix must have {n_classes} rows")
    total = 0
    for row in cm:
        if not isinstance(row, list) or len(row) != n_classes:
            raise ValueError(f"confusion matrix must be {n_classes}x{n_classes}")
        for v in row:
            # bool is an int subclass; a count must be a real non-negative integer.
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"confusion matrix counts must be non-negative integers, got {v!r}")
            total += v
    if total != sizes["test"]:
        raise ValueError(f"confusion matrix total {total} != test-set size {sizes['test']}")

    for key in ("train_loss", "val_loss", "val_accuracy", "lr_history"):
        if len(result[key]) != epochs:
            raise ValueError(f"{key} has {len(result[key])} entries, expected {epochs}")
    if "train_accuracy" in result and len(result["train_accuracy"]) != epochs:
        raise ValueError(f"train_accuracy has {len(result['train_accuracy'])} entries, expected {epochs}")
    if not 0 <= result["best_epoch"] < epochs:
        raise ValueError(f"best_epoch {result['best_epoch']} outside 0..{epochs - 1}")
    per_class = result.get("per_class")
    if per_class is not None and any(len(per_class[k]) != n_classes for k in ("precision", "recall", "f1", "support")):
        raise ValueError(f"per-class metrics must have {n_classes} entries")


# Columns that, together, say which images a run trained/validated/tested on.
_GENERATION_KEYS = ("n_rows", "img_size", "noise")
_SPLIT_KEYS = ("test_size", "val_size")


def split_identity(run: Mapping[str, Any]) -> Optional[Tuple]:
    """Everything that fixes a run's examples, or ``None`` if the row lacks it.

    ``run`` is a runs row (or ``Run`` dump). Old rows without experiment metadata
    return ``None`` and so are never reported as comparable. Fashion-MNIST runs
    (``config["dataset"]`` present) are identified by dataset, split seed, validation
    fraction, subset size and the hash of the exact train/validation indices; the
    synthetic-shape runs keep their original identity.
    """
    cfg: Dict[str, Any] = run.get("config") or {}
    if "dataset" in cfg:
        parts: Tuple = (
            run.get("dataset_id"), cfg["dataset"], run.get("split_seed"), cfg.get("val_size"),
            cfg.get("train_limit") or "all", run.get("split_sha256"),
            run.get("n_train"), run.get("n_val"), run.get("n_test"),
        )
    else:
        parts = (
            run.get("dataset_id"),
            *(cfg.get(k) for k in _GENERATION_KEYS),
            run.get("data_seed"),
            run.get("split_seed"),
            *(cfg.get(k) for k in _SPLIT_KEYS),
            run.get("split_sha256"),
            run.get("n_train"),
            run.get("n_val"),
            run.get("n_test"),
        )
    return None if any(p is None for p in parts) else parts


def same_split(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True only if both runs used the same dataset, generation settings and split."""
    ia, ib = split_identity(a), split_identity(b)
    return ia is not None and ia == ib
