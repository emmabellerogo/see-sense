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

from typing import Any, Dict, Mapping, Optional, Tuple

from shared.data import N_CLASSES


def validate_result(result: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a ``run_experiment`` result is internally inconsistent."""
    cfg, sizes = result["config"], result["split_sizes"]
    epochs = cfg["epochs"]

    if sum(sizes.values()) != cfg["n_rows"] or min(sizes.values()) < 1:
        raise ValueError(f"split sizes {sizes} do not partition {cfg['n_rows']} rows")

    cm = result["confusion_matrix"]
    if not isinstance(cm, list) or len(cm) != N_CLASSES:
        raise ValueError(f"confusion matrix must have {N_CLASSES} rows")
    total = 0
    for row in cm:
        if not isinstance(row, list) or len(row) != N_CLASSES:
            raise ValueError(f"confusion matrix must be {N_CLASSES}x{N_CLASSES}")
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
    if not 0 <= result["best_epoch"] < epochs:
        raise ValueError(f"best_epoch {result['best_epoch']} outside 0..{epochs - 1}")


# Columns that, together, say which images a run trained/validated/tested on.
_GENERATION_KEYS = ("n_rows", "img_size", "noise")
_SPLIT_KEYS = ("test_size", "val_size")


def split_identity(run: Mapping[str, Any]) -> Optional[Tuple]:
    """Everything that fixes a run's examples, or ``None`` if the row lacks it.

    ``run`` is a runs row (or ``Run`` dump). Old rows without experiment metadata
    return ``None`` and so are never reported as comparable.
    """
    cfg: Dict[str, Any] = run.get("config") or {}
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
