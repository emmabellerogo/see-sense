"""Read the official run records from Supabase (READ-ONLY) and verify the matched protocol.

    python -m experiments.export_records <group> <out_dir> [--frozen config/fashion_protocol.json]

Credentials come from .env / the environment and are never printed. Only ``select`` calls are made; the
checkpoint blobs (run_artifacts) are not fetched here. Writes <out_dir>/runs.json: the run rows exactly as
PostgREST returned them, the dataset row, and a small provenance header.

IMPORTANT: this is an API read, not a SQL export. Nothing derived from runs.json may be described as
"SQL-sourced from Supabase". Its legitimate uses are (1) this verification and (2) loading a LOCAL Postgres
replica (experiments/sql_check.mjs) to check the queries before the real SQL export is run in Supabase.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

VARIANTS = ["baseline", "full_improved", "no_augmentation", "no_batchnorm", "no_dropout", "no_scheduler"]


def fetch(group: str) -> dict:
    if os.environ.get("SEE_SENSE_LOCAL_DB"):
        raise SystemExit("Refusing: SEE_SENSE_LOCAL_DB is set (offline store); unset it to read Supabase.")
    from api import db  # loads .env

    cl = db.get_client()
    runs = (cl.table("runs").select("*").eq("experiment_group", group).order("id").execute().data)
    ds_ids = sorted({r["dataset_id"] for r in runs})
    datasets = [cl.table("datasets").select("*").eq("id", i).execute().data[0] for i in ds_ids]
    return {"runs": runs, "datasets": datasets}


def verify(runs: list[dict], datasets: list[dict], proto: dict) -> list[str]:
    """Return a list of problems (empty = every check passed)."""
    bad: list[str] = []
    p, seeds = proto["protocol"], proto["seeds"]
    want = {f"{v}-seed{s}" for v in VARIANTS for s in seeds}
    got = [r["run_name"] for r in runs]
    if sorted(got) != sorted(want):
        bad.append(f"run names differ: missing={sorted(want - set(got))} extra={sorted(set(got) - want)}")
    if len(set(got)) != len(got):
        bad.append("duplicate run names")
    if {r["dataset_id"] for r in runs} != {proto["frozen_dataset_id"]}:
        bad.append("runs are not all on the frozen dataset id")
    for d in datasets:
        if d.get("source") != "fashion_mnist" or len(d.get("class_names") or []) != 10:
            bad.append(f"dataset {d.get('id')} is not a 10-class fashion_mnist row")
    for r in runs:
        n, c = r["run_name"], r.get("config") or {}
        checks = {
            "epochs": r["epochs"] == p["epochs"], "batch_size": r["batch_size"] == p["batch_size"],
            "lr": abs(r["lr"] - p["lr"]) < 1e-12, "split_seed": r["split_seed"] == p["split_seed"],
            "optimizer": r["optimizer"] == p["optimizer"],
            "train_seed": r["train_seed"] == int(n.rsplit("-seed", 1)[1]),
            "dataset": c.get("dataset") == "fashion-mnist", "val_size": c.get("val_size") == p["val_size"],
            "train_limit": c.get("train_limit") in (None, p["train_limit"]),
            "sizes": (r["n_train"], r["n_val"], r["n_test"]) == (54000, 6000, 10000),
            "ten_classes": len(c.get("class_names") or []) == 10,
            "history_lengths": all(len(r[k]) == p["epochs"] for k in (
                "train_loss_history", "val_loss_history", "val_accuracy_history",
                "train_accuracy_history", "lr_history")),
            "confusion_10x10": len(r["confusion_matrix"]) == 10 and all(len(x) == 10 for x in r["confusion_matrix"]),
            "calibration": bool(r.get("calibration")) and r["calibration"]["temperature"] > 0,
            "per_class": len((r.get("test_per_class") or {}).get("precision", [])) == 10,
        }
        bad += [f"{n}: {k} failed" for k, ok in checks.items() if not ok]
    for s in seeds:  # matched: same split, same preprocessing within a seed AND across seeds (split_seed fixed)
        sub = [r for r in runs if r["train_seed"] == s]
        for k in ("split_sha256", "n_train", "n_val", "n_test", "dataset_id", "split_seed"):
            if len({json.dumps(r[k]) for r in sub}) != 1:
                bad.append(f"seed {s}: {k} not identical across the six variants")
    if len({r["split_sha256"] for r in runs}) != 1:
        bad.append("split_sha256 differs between seeds although split_seed is fixed")
    if len({json.dumps(r["config"]["preprocessing"], sort_keys=True) for r in runs}) != 1:
        bad.append("preprocessing differs between runs")
    # Declared technique differences only.
    exp = {
        "baseline": dict(use_batchnorm=False, dropout=0, augment=False, scheduler="none"),
        "full_improved": dict(use_batchnorm=True, dropout=p["dropout"], augment=True, scheduler=p["scheduler"]),
        "no_augmentation": dict(use_batchnorm=True, dropout=p["dropout"], augment=False, scheduler=p["scheduler"]),
        "no_batchnorm": dict(use_batchnorm=False, dropout=p["dropout"], augment=True, scheduler=p["scheduler"]),
        "no_dropout": dict(use_batchnorm=True, dropout=0, augment=True, scheduler=p["scheduler"]),
        "no_scheduler": dict(use_batchnorm=True, dropout=p["dropout"], augment=True, scheduler="none"),
    }
    for r in runs:
        v = r["run_name"].rsplit("-seed", 1)[0]
        for k, want_v in exp.get(v, {}).items():
            if r[k] != want_v:
                bad.append(f"{r['run_name']}: {k}={r[k]!r}, expected {want_v!r}")
        if r["augment"] and r["augment_max_shift"] != p["augment_max_shift"]:
            bad.append(f"{r['run_name']}: augment_max_shift {r['augment_max_shift']}")
    return bad


def main(group: str, out_dir: str, frozen: str = "config/fashion_protocol.json") -> int:
    proto = json.loads(Path(frozen).read_text())
    if proto["status"] != "frozen" or proto["experiment_group"] != group:
        raise SystemExit("protocol file is not the frozen protocol of this group")
    data = fetch(group)
    problems = verify(data["runs"], data["datasets"], proto)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    body = {"provenance": {
        "kind": "PostgREST read via service key (NOT a SQL export)", "group": group,
        "read_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_runs": len(data["runs"]), "run_ids": [r["id"] for r in data["runs"]],
        "verification_problems": problems}, **data}
    text = json.dumps(body, indent=1, sort_keys=True)
    (out / "runs.json").write_text(text)
    print(f"runs read: {len(data['runs'])}  ids {min(r['id'] for r in data['runs'])}-{max(r['id'] for r in data['runs'])}")
    print(f"sha256(runs.json) = {hashlib.sha256(text.encode()).hexdigest()}")
    print("matched-protocol verification:", "ALL CHECKS PASSED" if not problems else f"{len(problems)} PROBLEM(S)")
    for x in problems:
        print("  -", x)
    return 1 if problems else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[0], a[1], a[a.index("--frozen") + 1] if "--frozen" in a else "config/fashion_protocol.json"))
