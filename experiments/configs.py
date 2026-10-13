"""Matched run configurations for the baseline-vs-improved comparison.

Six variants, all trained on the same dataset, split and seeds with the same
epochs, batch size, optimizer (Adam, fixed by the API) and initial learning rate:

    baseline          SmallCNN, no techniques
    full_improved     ImprovedCNN + batch norm + dropout + augmentation + cosine LR
    no_augmentation   full_improved with augmentation off
    no_batchnorm      full_improved with batch norm off
    no_dropout        full_improved with dropout 0
    no_scheduler      full_improved with the LR scheduler off

Each ablation differs from ``full_improved`` in exactly one technique setting;
``check_matched`` proves it for any plan. Nothing here talks to the network.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

VARIANTS = (
    "baseline",
    "full_improved",
    "no_augmentation",
    "no_batchnorm",
    "no_dropout",
    "no_scheduler",
)

# The one request field each ablation is allowed to change relative to full_improved.
ABLATED_FIELD = {
    "no_augmentation": "augment",
    "no_batchnorm": "use_batchnorm",
    "no_dropout": "dropout",
    "no_scheduler": "scheduler",
}

# Fields that must be identical in every run of the same training seed.
MATCHED_FIELDS = (
    "dataset_id", "experiment_group", "lr", "batch_size", "epochs",
    "test_size", "val_size", "split_seed", "train_seed", "augment_max_shift",
)


@dataclass(frozen=True)
class Protocol:
    """Everything that is held fixed across variants. Chosen by the experimenter."""

    dataset_id: int
    epochs: int
    batch_size: int
    lr: float
    split_seed: int
    test_size: float = 0.2
    val_size: float = 0.1
    dropout: float = 0.3          # strength used where dropout is on
    augment_max_shift: int = 2    # pixels, where augmentation is on
    scheduler: str = "cosine"     # used where the scheduler is on

    def __post_init__(self) -> None:
        if not 0.0 < self.dropout < 1.0:
            raise ValueError("dropout must be in (0, 1), otherwise 'no_dropout' changes nothing")
        if self.augment_max_shift < 1:
            raise ValueError("augment_max_shift must be >= 1, otherwise 'no_augmentation' changes nothing")
        if self.scheduler == "none":
            raise ValueError("scheduler must be cosine or step, otherwise 'no_scheduler' changes nothing")


def technique_settings(protocol: Protocol) -> Dict[str, Dict[str, Any]]:
    """Per-variant model/technique switches (the only things that may differ)."""
    full = {
        "model_variant": "improved",
        "use_batchnorm": True,
        "dropout": protocol.dropout,
        "augment": True,
        "scheduler": protocol.scheduler,
    }
    return {
        "baseline": {
            "model_variant": "baseline", "use_batchnorm": False, "dropout": 0.0,
            "augment": False, "scheduler": "none",
        },
        "full_improved": dict(full),
        "no_augmentation": {**full, "augment": False},
        "no_batchnorm": {**full, "use_batchnorm": False},
        "no_dropout": {**full, "dropout": 0.0},
        "no_scheduler": {**full, "scheduler": "none"},
    }


def run_name(variant: str, train_seed: int) -> str:
    """Unique within a group because it includes the training seed."""
    return f"{variant}-seed{train_seed}"


def build_payload(protocol: Protocol, variant: str, train_seed: int, group: str) -> Dict[str, Any]:
    """The exact JSON body for POST /train."""
    return {
        "dataset_id": protocol.dataset_id,
        "lr": protocol.lr,
        "batch_size": protocol.batch_size,
        "epochs": protocol.epochs,
        "test_size": protocol.test_size,
        "val_size": protocol.val_size,
        "augment_max_shift": protocol.augment_max_shift,
        "split_seed": protocol.split_seed,
        "train_seed": train_seed,
        "experiment_group": group,
        "run_name": run_name(variant, train_seed),
        **technique_settings(protocol)[variant],
    }


def build_plan(
    protocol: Protocol, group: str, seeds: Iterable[int], variants: Iterable[str] = VARIANTS
) -> List[Dict[str, Any]]:
    """Seed-major list of request bodies; every seed gets the full variant set."""
    seeds, variants = list(seeds), list(variants)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and distinct")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    plan = [build_payload(protocol, v, s, group) for s in seeds for v in variants]
    check_matched(plan)
    return plan


def check_matched(plan: List[Dict[str, Any]]) -> None:
    """Raise ValueError unless runs are matched except for the declared differences."""
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for p in plan:
        variant = p["run_name"].rsplit("-seed", 1)[0]
        by_seed.setdefault(p["train_seed"], {})[variant] = p
    names = [p["run_name"] for p in plan]
    if len(set(names)) != len(names):
        raise ValueError("run names are not unique")
    for seed, runs in by_seed.items():
        ref = next(iter(runs.values()))
        for v, p in runs.items():
            for f in MATCHED_FIELDS:
                if p[f] != ref[f]:
                    raise ValueError(f"seed {seed}: {v} differs from the others in {f!r}")
        full = runs.get("full_improved")
        for v, field in ABLATED_FIELD.items():
            if v in runs and full is not None:
                diff = {k for k in full if runs[v][k] != full[k] and k != "run_name"}
                if diff != {field}:
                    raise ValueError(f"seed {seed}: {v} differs from full_improved in {sorted(diff)}, expected [{field!r}]")


def fingerprint(plan: List[Dict[str, Any]]) -> str:
    """Stable hash of the plan, to record when the protocol is frozen."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
