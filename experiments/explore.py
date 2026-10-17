"""Protocol exploration on TRAINING / VALIDATION evidence only (Part 2, step 2).

Calls ``api.training.run_experiment`` directly (same code the API runs) for a
grid of candidate settings, and keeps ONLY training/validation quantities. The
test metrics that ``run_experiment`` computes are deleted before anything is
printed or saved, so test data cannot influence the protocol choice.

Exploration uses training seeds 100+ so it never reuses the official seeds.

    python -m experiments.explore difficulty   > explore_difficulty.jsonl
    python -m experiments.explore budget       > explore_budget.jsonl
"""
from __future__ import annotations

import itertools
import json
import sys

from api.training import run_experiment
from experiments.configs import Protocol, technique_settings

EXPLORE_SEEDS = (100, 101)
SPLIT_SEED = 0


def _val_only(result: dict) -> dict:
    """Strip every test-set quantity; keep train/val evidence."""
    vl, va = result["val_loss"], result["val_accuracy"]
    b = result["best_epoch"]
    return {
        "best_epoch": b,
        "best_val_loss": round(vl[b], 4),
        "val_acc_at_best": round(va[b], 4),
        "final_val_loss": round(vl[-1], 4),
        "final_train_loss": round(result["train_loss"][-1], 4),
        "n_val": result["split_sizes"]["val"],
        "seconds": round(result["train_seconds"], 2),
        "val_loss_curve": [round(v, 4) for v in vl],
    }


def run(variant: str, *, n_rows, noise, lr, batch_size, epochs, seed, data_seed=42,
        dropout=0.3, max_shift=2, scheduler="cosine") -> dict:
    proto = Protocol(dataset_id=0, epochs=epochs, batch_size=batch_size, lr=lr,
                     split_seed=SPLIT_SEED, dropout=dropout,
                     augment_max_shift=max_shift, scheduler=scheduler)
    t = technique_settings(proto)[variant]
    res = run_experiment(
        n_rows=n_rows, img_size=28, noise=noise, data_seed=data_seed,
        split_seed=SPLIT_SEED, train_seed=seed, lr=lr, batch_size=batch_size,
        epochs=epochs, arch=t["model_variant"], use_batchnorm=t["use_batchnorm"],
        dropout=t["dropout"], augment=t["augment"],
        augment_max_shift=max_shift, scheduler=t["scheduler"],
    )
    res.pop("metrics"); res.pop("confusion_matrix"); res.pop("model_b64")
    out = {"variant": variant, "n_rows": n_rows, "noise": noise, "lr": lr,
           "batch_size": batch_size, "epochs": epochs, "seed": seed,
           "dropout": dropout, "max_shift": max_shift, "scheduler": scheduler}
    out.update(_val_only(res))
    return out


def emit(row: dict) -> None:
    print(json.dumps(row), flush=True)


def difficulty() -> None:
    """Stage A: is the task hard enough to separate the variants?"""
    for n_rows, noise in itertools.product((800, 2000), (0.1, 0.5, 0.7, 0.9)):
        for variant in ("baseline", "full_improved"):
            for seed in EXPLORE_SEEDS:
                emit(run(variant, n_rows=n_rows, noise=noise, lr=0.01,
                         batch_size=32, epochs=15, seed=seed))


def budget(n_rows: int, noise: float) -> None:
    """Stage B: learning rate x epochs x batch size on the chosen dataset."""
    for lr, bs, epochs in itertools.product((0.003, 0.01, 0.03), (32, 64), (15, 30)):
        for variant in ("baseline", "full_improved", "no_scheduler"):
            for seed in EXPLORE_SEEDS:
                emit(run(variant, n_rows=n_rows, noise=noise, lr=lr,
                         batch_size=bs, epochs=epochs, seed=seed))


def techniques(n_rows: int, noise: float, lr: float, bs: int, epochs: int) -> None:
    """Stage C: untuned technique strengths (dropout, shift) on validation."""
    for dropout, shift in itertools.product((0.2, 0.3, 0.5), (1, 2, 3)):
        for seed in EXPLORE_SEEDS:
            emit(run("full_improved", n_rows=n_rows, noise=noise, lr=lr, batch_size=bs,
                     epochs=epochs, seed=seed, dropout=dropout, max_shift=shift))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "difficulty":
        difficulty()
    elif stage == "budget":
        budget(int(sys.argv[2]), float(sys.argv[3]))
    elif stage == "techniques":
        techniques(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
                   int(sys.argv[5]), int(sys.argv[6]))
    else:
        raise SystemExit(f"unknown stage {stage}")
