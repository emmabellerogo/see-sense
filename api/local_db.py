"""Offline stand-in for the Supabase layer, for running the API locally.

Turned on only when ``SEE_SENSE_LOCAL_DB`` is set (in ``.env`` or the shell) to a
JSON file path, e.g. ``SEE_SENSE_LOCAL_DB=.local_db.json``. Then ``api/db.py``
routes every call here instead of Supabase, so the whole API -- ``/train``,
``/runs``, ``/predict`` and the experiment runner -- works with no credentials.

This is for trying things out and checking the plumbing (Part 2 step 1). It is
NOT the project's source of truth: official runs belong in Supabase so the team,
the dashboard and ``docs/comparison_queries.sql`` can see them. Unset the
variable to go back to Supabase.

On first use the file is seeded with dataset id 1 using the same settings as the
shared Supabase project (800 rows, noise 0.1, seed 42), so commands from
``docs/PART1_HANDOFF.md`` work unchanged.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Mapping, Optional

_LOCK = threading.Lock()

DEFAULT_DATASET = {
    "name": "default-shapes",
    "n_rows": 800,
    "img_size": 28,
    "n_classes": 4,
    "noise": 0.1,
    "seed": 42,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalStore:
    """A tiny JSON-file database with the same functions as ``api/db.py``."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        if not self.path.exists():
            self._write({"datasets": [], "runs": [], "run_artifacts": [], "image_metadata": []})
            self.insert_dataset(**DEFAULT_DATASET)

    # -- file helpers ------------------------------------------------------
    def _read(self) -> dict:
        return json.loads(self.path.read_text())

    def _write(self, data: dict) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)

    @staticmethod
    def _next_id(rows: list) -> int:
        return max((r["id"] for r in rows), default=0) + 1

    # -- same interface as api/db.py --------------------------------------
    def ping(self) -> bool:
        try:
            self._read()
            return True
        except Exception:
            return False

    def insert_dataset(self, name, n_rows, img_size, n_classes, noise, seed) -> dict:
        with _LOCK:
            data = self._read()
            row = {
                "id": self._next_id(data["datasets"]), "name": name, "n_rows": n_rows,
                "img_size": img_size, "n_classes": n_classes, "noise": noise,
                "seed": seed, "created_at": _now(),
            }
            data["datasets"].append(row)
            self._write(data)
            return row

    def get_dataset(self, dataset_id: int) -> Optional[dict]:
        return next((r for r in self._read()["datasets"] if r["id"] == dataset_id), None)

    def insert_run(self, dataset_id, lr, batch_size, epochs, metrics, model_b64,
                   experiment: Optional[Mapping[str, Any]] = None) -> dict:
        from api.db import EXPERIMENT_COLUMNS

        with _LOCK:
            data = self._read()
            experiment = dict(experiment or {})
            group, name = experiment.get("experiment_group"), experiment.get("run_name")
            if group and name and any(
                r.get("experiment_group") == group and r.get("run_name") == name
                for r in data["runs"]
            ):
                # Same rule as the unique index in migration 002.
                raise ValueError(f"duplicate (experiment_group, run_name): {group}/{name}")
            row = {
                "id": self._next_id(data["runs"]), "dataset_id": dataset_id, "lr": lr,
                "batch_size": batch_size, "epochs": epochs,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "created_at": _now(),
                **{c: None for c in EXPERIMENT_COLUMNS},
                **experiment,
            }
            data["runs"].append(row)
            data["run_artifacts"].append({"run_id": row["id"], "model_b64": model_b64})
            self._write(data)
            return row

    def run_name_exists(self, experiment_group: str, run_name: str) -> bool:
        return any(
            r.get("experiment_group") == experiment_group and r.get("run_name") == run_name
            for r in self._read()["runs"]
        )

    def get_run(self, run_id: int) -> Optional[dict]:
        return next((r for r in self._read()["runs"] if r["id"] == run_id), None)

    def get_run_artifact(self, run_id: int) -> Optional[str]:
        return next(
            (a["model_b64"] for a in self._read()["run_artifacts"] if a["run_id"] == run_id),
            None,
        )

    def latest_runs(self, limit: int = 50) -> List[dict]:
        return list(reversed(self._read()["runs"]))[:limit]

    def insert_image_metadata(self, run_id, sha256, width, height, label, class_name,
                              confidence) -> dict:
        with _LOCK:
            data = self._read()
            row = {
                "id": self._next_id(data["image_metadata"]), "run_id": run_id,
                "sha256": sha256, "width": width, "height": height, "label": label,
                "class_name": class_name, "confidence": confidence, "created_at": _now(),
            }
            data["image_metadata"].append(row)
            self._write(data)
            return row
