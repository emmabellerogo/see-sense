"""Run the matched baseline / improved / ablation set through the existing API.

Usage (from the repo root; the API must already be running):

    python -m experiments.run_experiments --dry-run  <required options>
    python -m experiments.run_experiments            <required options>

Every run is a normal ``POST /train``, so it is saved to Supabase by the API; this
script holds no credentials and never touches the database itself. It then reads
each run back with ``GET /runs/{id}`` to confirm it was persisted with the
settings that were requested, and checks that all runs of one seed saw the same
dataset and split.

Protocol values (epochs, batch size, learning rate, seeds, ...) have NO defaults
on purpose: choose them, freeze them (see ``--expect-fingerprint``), then run.
Test metrics are hidden unless ``--show-test`` is given, so protocol choices are
made on training / validation evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from api.experiments import same_split
from experiments.configs import VARIANTS, Protocol, build_plan, fingerprint


class ApiError(RuntimeError):
    """The API could not be reached at all."""


class ApiClient:
    """Minimal JSON-over-HTTP client (standard library only)."""

    def __init__(self, base_url: str, timeout: float = 900.0):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def request(self, method: str, path: str, payload: Optional[dict] = None) -> Tuple[int, Any]:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                return exc.code, {"detail": raw.decode(errors="replace")}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiError(f"cannot reach {self.base_url}{path}: {exc}") from exc


# Fields of the POST /train body that must come back unchanged on the saved run.
ECHOED = {
    "model_variant": "model_variant", "use_batchnorm": "use_batchnorm", "dropout": "dropout",
    "augment": "augment", "augment_max_shift": "augment_max_shift", "scheduler": "scheduler",
    "split_seed": "split_seed", "train_seed": "train_seed", "experiment_group": "experiment_group",
    "run_name": "run_name", "lr": "lr", "batch_size": "batch_size", "epochs": "epochs",
    "dataset_id": "dataset_id",
}


def verify_saved(payload: dict, saved: dict) -> List[str]:
    """Problems found comparing what was requested with the run read back."""
    problems = [
        f"{field}: requested {payload[field]!r}, saved {saved.get(col)!r}"
        for field, col in ECHOED.items()
        if payload[field] != saved.get(col)
    ]
    if payload["model_variant"] == "baseline":
        if saved.get("use_batchnorm") is not False or saved.get("dropout") != 0.0:
            problems.append("baseline must be saved with batch norm off and dropout 0")
    for h in ("train_loss_history", "val_loss_history", "val_accuracy_history",
              "train_accuracy_history", "lr_history"):
        if not saved.get(h) or len(saved[h]) != payload["epochs"]:
            problems.append(f"{h} missing or not {payload['epochs']} long")
    if not saved.get("confusion_matrix") or not saved.get("split_sha256"):
        problems.append("confusion matrix or split hash missing")
    cfg = saved.get("config") or {}
    if cfg.get("dataset") != "fashion-mnist" or cfg.get("n_classes") != 10:
        problems.append("run was not trained on the 10-class Fashion-MNIST dataset")
    calib = saved.get("calibration") or {}
    if not calib.get("temperature") or not (saved.get("test_per_class") or {}).get("precision"):
        problems.append("calibration or per-class test metrics missing (migration 005 applied?)")
    if cfg.get("train_limit") != payload.get("train_limit"):
        problems.append(f"train_limit: requested {payload.get('train_limit')!r}, saved {cfg.get('train_limit')!r}")
    return problems


def run_plan(
    api: Any,
    plan: List[dict],
    keep_going: bool = False,
    log: Callable[[str], None] = print,
    show_test: bool = False,
) -> List[Dict[str, Any]]:
    """POST every run, read it back, and return one record per run.

    Statuses: ``saved`` (persisted and verified), ``exists`` (name already used in
    the group; nothing was trained), ``failed``.
    """
    results: List[Dict[str, Any]] = []
    for i, payload in enumerate(plan, 1):
        name = payload["run_name"]
        rec: Dict[str, Any] = {"run_name": name, "train_seed": payload["train_seed"]}
        log(f"[{i}/{len(plan)}] {name} ...")
        try:
            status, body = api.request("POST", "/train", payload)
            rec["http_status"] = status
            if status == 409:
                rec["status"] = "exists"
                log("    already exists in this group; skipped (not re-verified)")
            elif status != 200:
                rec.update(status="failed", error=f"HTTP {status}: {body.get('detail', body)}")
            else:
                rec["run_id"] = body["run_id"]
                g_status, saved = api.request("GET", f"/runs/{body['run_id']}")
                if g_status != 200:
                    rec.update(status="failed", error=f"saved run not readable: HTTP {g_status}")
                else:
                    problems = verify_saved(payload, saved)
                    rec["saved_row"] = saved
                    if problems:
                        rec.update(status="failed", error="; ".join(problems))
                    else:
                        rec["status"] = "saved"
                        v = saved["val_loss_history"]
                        log(
                            f"    saved run_id={rec['run_id']}  params={saved['param_count']}  "
                            f"best_epoch={saved['best_epoch']}  best_val_loss={v[saved['best_epoch']]:.4f}  "
                            f"final_train_loss={saved['train_loss_history'][-1]:.4f}  "
                            f"seconds={saved['train_seconds']:.1f}"
                        )
                        if show_test:
                            log(f"    TEST accuracy={saved['accuracy']:.4f} macro_f1={saved['macro_f1']:.4f}")
        except ApiError as exc:
            rec.update(status="failed", error=str(exc))
        if rec["status"] == "failed":
            log(f"    FAILED: {rec['error']}")
        results.append(rec)
        if rec["status"] == "failed" and not keep_going:
            log("stopping after the first failure (use --keep-going to continue)")
            break
    return results


def check_splits(results: List[Dict[str, Any]]) -> List[str]:
    """Problems if runs of the same seed did not see the same dataset and split."""
    problems: List[str] = []
    first: Dict[int, Dict[str, Any]] = {}
    for rec in results:
        row = rec.get("saved_row")
        if rec["status"] != "saved" or row is None:
            continue
        ref = first.setdefault(rec["train_seed"], rec)
        if ref is not rec and not same_split(ref["saved_row"], row):
            problems.append(f"{rec['run_name']} did not use the same dataset/split as {ref['run_name']}")
    return problems


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-url", default=os.environ.get("SEESENSE_API_URL", "http://127.0.0.1:8000"))
    req = p.add_argument_group("protocol (required, no defaults: choose and freeze these)")
    req.add_argument("--dataset-id", type=int, required=True)
    req.add_argument("--group", required=True, help="experiment_group; run names are unique within it")
    req.add_argument("--seeds", required=True, help="training seeds, comma separated, e.g. 0,1,2")
    req.add_argument("--epochs", type=int, required=True)
    req.add_argument("--batch-size", type=int, required=True)
    req.add_argument("--lr", type=float, required=True, help="initial Adam learning rate")
    req.add_argument("--split-seed", type=int, required=True)
    opt = p.add_argument_group("technique settings and splits")
    opt.add_argument("--dropout", type=float, default=0.3)
    opt.add_argument("--max-shift", type=int, default=2)
    opt.add_argument("--scheduler", choices=["cosine", "step"], default="cosine")
    opt.add_argument("--val-size", type=float, default=0.1, help="fraction of the official training images used for validation")
    opt.add_argument("--train-limit", type=int, default=None,
                     help="optional documented subset of training images (default: all of them)")
    run = p.add_argument_group("running")
    run.add_argument("--variants", default=",".join(VARIANTS), help="subset to run (default: all six)")
    run.add_argument("--dry-run", action="store_true", help="print the plan and fingerprint; no network")
    run.add_argument("--expect-fingerprint", help="abort unless the plan hashes to this (freezes the protocol)")
    run.add_argument("--keep-going", action="store_true")
    run.add_argument("--show-test", action="store_true", help="also print test metrics (official runs only)")
    run.add_argument("--timeout", type=float, default=900.0, help="seconds per request")
    run.add_argument("--out", help="append one JSON line per run (names/ids/status; no metrics)")
    return p


def main(argv: Optional[List[str]] = None, api: Any = None, log: Callable[[str], None] = print) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = Protocol(
            dataset_id=args.dataset_id, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, split_seed=args.split_seed,
            val_size=args.val_size, train_limit=args.train_limit, dropout=args.dropout, augment_max_shift=args.max_shift,
            scheduler=args.scheduler,
        )
        seeds = [int(s) for s in args.seeds.split(",")]
        plan = build_plan(protocol, args.group, seeds, [v for v in args.variants.split(",") if v])
    except ValueError as exc:
        log(f"invalid configuration: {exc}")
        return 2

    fp = fingerprint(plan)
    log(f"plan: {len(plan)} runs, protocol fingerprint {fp}")
    if args.expect_fingerprint and args.expect_fingerprint != fp:
        log("fingerprint does not match --expect-fingerprint; the protocol changed. Aborting.")
        return 2
    if args.dry_run:
        for p in plan:
            log(json.dumps(p, sort_keys=True))
        return 0

    api = api or ApiClient(args.api_url, args.timeout)
    try:
        status, health = api.request("GET", "/healthz")
    except ApiError as exc:
        log(str(exc))
        return 1
    if status != 200 or health.get("status") != "ok":
        log(f"API is not healthy: HTTP {status} {health}")
        return 1

    results = run_plan(api, plan, args.keep_going, log, args.show_test)
    problems = check_splits(results)
    for msg in problems:
        log(f"SPLIT MISMATCH: {msg}")
    counts = {s: sum(r["status"] == s for r in results) for s in ("saved", "exists", "failed")}
    log(f"done: {counts['saved']} saved, {counts['exists']} already existed, "
        f"{counts['failed']} failed, {len(plan) - len(results)} not attempted")
    if args.out:
        with open(args.out, "a") as fh:
            for r in results:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "saved_row"}) + "\n")
    return 0 if not counts["failed"] and len(results) == len(plan) and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
