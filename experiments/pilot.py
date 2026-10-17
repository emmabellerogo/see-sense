"""Bounded runtime pilot: TRAINING and VALIDATION only. No test data, no database, no API.

    python -m experiments.pilot --epochs 3 --batch-size 128 --lr 0.001 --out pilot-output/pilot.jsonl

Purpose: measure seconds per epoch on this machine for each of the six variants so the
cost of the full comparison can be estimated before it is launched, and look at
training / validation curves to choose a feasible budget. The official test files are
never opened (``fashion.load(include_test=False)``), nothing is written to a database, and
nothing here is a result: pilot numbers are not reported as findings.

Every protocol value is a required argument (no defaults), like the official runner.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch

from api.training import train_validate
from experiments.configs import VARIANTS, Protocol, technique_settings
from shared import fashion


def run_pilot(variants: List[str], epochs: int, batch_size: int, lr: float, split_seed: int,
              train_seed: int, val_size: float, train_limit: Optional[int],
              log: Callable[[str], None] = print, data: Optional[fashion.FashionData] = None) -> List[dict]:
    data = data if data is not None else fashion.load(include_test=False)
    assert data.x_test is None, "the pilot must never load the official test partition"
    n_val = int(round(len(data.y_train) * val_size))
    train_idx, val_idx = fashion.split_train_val(data.y_train, n_val, split_seed)
    if train_limit is not None:
        train_idx = train_idx[:train_limit]
    protocol = Protocol(dataset_id=0, epochs=epochs, batch_size=batch_size, lr=lr, split_seed=split_seed)
    settings = technique_settings(protocol)
    records = []
    for v in variants:
        s = settings[v]
        log(f"== {v}: {len(train_idx)} train / {len(val_idx)} validation images, {epochs} epochs, torch threads={torch.get_num_threads()}")
        out = train_validate(
            data.x_train[train_idx], data.y_train[train_idx], data.x_train[val_idx], data.y_train[val_idx],
            n_classes=fashion.N_CLASSES, img_size=fashion.IMG_SIZE, lr=lr, batch_size=batch_size,
            epochs=epochs, train_seed=train_seed, arch=s["model_variant"], use_batchnorm=s["use_batchnorm"],
            dropout=s["dropout"], augment=s["augment"], augment_max_shift=protocol.augment_max_shift,
            scheduler=s["scheduler"],
            on_epoch=lambda e, r: log(
                f"   epoch {e + 1:>2}: train loss {r['train_loss']:.4f} acc {r['train_accuracy']:.4f} | "
                f"val loss {r['val_loss']:.4f} acc {r['val_accuracy']:.4f} | {r['epoch_seconds']:.1f}s"),
        )
        records.append({
            "variant": v, "kind": "pilot", "uses_test_data": False, "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "train_seed": train_seed, "split_seed": split_seed, "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)), "torch_threads": torch.get_num_threads(),
            "param_count": out["param_count"], "epoch_seconds": out["epoch_seconds"],
            "train_loss": out["train_loss"], "train_accuracy": out["train_accuracy"],
            "val_loss": out["val_loss"], "val_accuracy": out["val_accuracy"], "best_epoch": out["best_epoch"],
        })
    return records


def estimate(records: List[dict], epochs: int, seeds: int, per_epoch: Optional[float] = None) -> dict:
    """Full-comparison wall-clock estimate from measured per-epoch seconds (sequential)."""
    per_variant = {r["variant"]: st.mean(r["epoch_seconds"]) for r in records}
    mean_epoch = per_epoch or st.mean(per_variant.values())
    runs = len(per_variant) * seeds
    return {"mean_epoch_seconds": mean_epoch, "per_variant_epoch_seconds": per_variant,
            "runs": runs, "sequential_minutes": runs * epochs * mean_epoch / 60.0}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--train-limit", type=int, default=None, help="optional subset (pilot sizing only)")
    p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    p.add_argument("--out", help="append one JSON line per variant")
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    variants = [v for v in args.variants.split(",") if v]
    t0 = time.perf_counter()
    records = run_pilot(variants, args.epochs, args.batch_size, args.lr, args.split_seed,
                        args.train_seed, args.val_size, args.train_limit)
    wall = time.perf_counter() - t0
    print(f"\npilot wall-clock {wall:.1f}s for {len(variants)} variants x {args.epochs} epochs")
    est = estimate(records, args.epochs, seeds=1)
    for v, sec in est["per_variant_epoch_seconds"].items():
        print(f"  {v:<16} {sec:6.2f} s/epoch")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
