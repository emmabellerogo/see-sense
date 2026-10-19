"""Baseline vs improved Grad-CAM on the SAME official test images (read-only, nothing retrained).

    python -m experiments.gradcam_compare --supabase <baseline_run_name> <improved_run_name> <group> \
        <improved_evidence_dir> <out_dir>

The test images are the 20 indices already chosen for the improved model (``examples.csv`` in
``<improved_evidence_dir>``: 10 correct, 10 incorrect for the improved model). Nothing is re-selected after
looking at either model's maps. For each image both models' own predictions, correctness, confidence, Grad-CAM
share on the garment and blank-map flag are saved, so a shared image can be correct for one model and wrong for
the other. These shared sets do NOT replace each model's own correct/incorrect galleries
(``fashion-official-1-gradcam`` and ``fashion-official-1-gradcam-baseline``).

Before drawing anything it checks that both runs belong to the same group, dataset, split, class labels,
preprocessing and local official files, and that each regenerated test accuracy equals the stored one.
Credentials are read from .env by the existing loader and never printed. No prediction endpoint is called.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from api.training import LoadedModel, cam_for_class, load_artifact  # noqa: E402
from experiments.gradcam_evidence import (  # noqa: E402
    cam_mass_on_garment, classify_all, load_from_supabase,
)
from shared import fashion  # noqa: E402

TOLERANCE = 1e-9


class IdentityError(AssertionError):
    pass


def read_indices(improved_dir: Path) -> Dict[str, List[int]]:
    """The improved model's two galleries, as test indices, in file order."""
    out: Dict[str, List[int]] = {"correct": [], "incorrect": []}
    with open(improved_dir / "examples.csv", newline="") as f:
        for r in csv.DictReader(f):
            out[r["set"]].append(int(r["test_index"]))
    if len(out["correct"]) < 5 or len(out["incorrect"]) < 5:
        raise ValueError("the improved examples.csv must hold at least 5 correct and 5 incorrect examples")
    return out


def check_identity(base_run: dict, imp_run: dict, base_ds: dict, imp_ds: dict,
                   base: LoadedModel, imp: LoadedModel, local_sha: Dict[str, str],
                   group: str, expected_classes: List[str]) -> Dict[str, Any]:
    """Raise IdentityError on any mismatch; otherwise return the checks as plain JSON-safe values."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise IdentityError(msg)

    need(base_run.get("experiment_group") == group and imp_run.get("experiment_group") == group,
         "runs are not both in the official group")
    need(base_run.get("dataset_id") == imp_run.get("dataset_id"), "runs use different datasets")
    need(base_run.get("split_sha256") and base_run.get("split_sha256") == imp_run.get("split_sha256"),
         "runs use different train/validation splits")
    need(base_run.get("train_seed") == imp_run.get("train_seed"), "runs do not share a seed")
    need(base_ds.get("id") == imp_ds.get("id"), "dataset rows differ")
    need(list(base.class_names) == list(imp.class_names) == list(expected_classes), "class labels differ")
    need(base.preprocessing == imp.preprocessing, "preprocessing differs between checkpoints")
    need(base.dataset.get("name") == imp.dataset.get("name") == fashion.DATASET_NAME, "checkpoint dataset name differs")
    need(base.dataset.get("source") == imp.dataset.get("source"), "checkpoint dataset source differs")
    stored = (base_ds.get("identity") or {}).get("file_sha256") or {}
    need(bool(stored) and all(local_sha.get(k) == v for k, v in stored.items() if k in local_sha)
         and all(k in local_sha for k in stored),
         "local official files do not match the dataset identity stored with the runs")
    for run in (base_run, imp_run):
        for col in ("epochs", "batch_size", "lr", "optimizer", "n_train", "n_val", "n_test"):
            need(run.get(col) == base_run.get(col), f"runs differ in {col}")
    return {
        "group": group, "dataset_id": base_run["dataset_id"], "split_sha256": base_run["split_sha256"],
        "train_seed": base_run["train_seed"], "class_names": list(base.class_names),
        "preprocessing": base.preprocessing,
        "local_file_sha256_matches_stored_identity": True,
        "baseline": {"run_id": base_run["id"], "run_name": base_run["run_name"],
                     "artifact_version": base.artifact_version, "temperature": base.temperature},
        "improved": {"run_id": imp_run["id"], "run_name": imp_run["run_name"],
                     "artifact_version": imp.artifact_version, "temperature": imp.temperature},
    }


def is_blank(cam01: np.ndarray) -> bool:
    return float(np.max(cam01)) <= 0.0


def build(base_b64: str, imp_b64: str, base_run: dict, imp_run: dict, base_ds: dict, imp_ds: dict,
          data: fashion.FashionData, indices: Dict[str, List[int]], out: Path, group: str) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    base, imp = load_artifact(base_b64), load_artifact(imp_b64)
    identity = check_identity(base_run, imp_run, base_ds, imp_ds, base, imp, data.file_sha256, group,
                              list(fashion.CLASS_NAMES))
    y = data.y_test.astype(np.int64)
    names = base.class_names
    results = {}
    for tag, b64, run in (("baseline", base_b64, base_run), ("improved", imp_b64, imp_run)):
        res = classify_all(b64, data.x_test)
        acc = float((res["pred"] == y).mean())
        if abs(acc - run["accuracy"]) > TOLERANCE:
            raise AssertionError(f"{tag}: regenerated accuracy {acc} != stored {run['accuracy']}")
        results[tag] = {"res": res, "acc": acc}
    loaded = {"baseline": base, "improved": imp}

    def up(cam):
        return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((28, 28), Image.BILINEAR)) / 255.0

    rows: List[dict] = []
    for set_name, idxs in indices.items():
        fig, axes = plt.subplots(len(idxs), 3, figsize=(7.4, 2.45 * len(idxs) + 0.8), squeeze=False)
        for row_ax, i in zip(axes, idxs):
            gray = data.x_test[i]
            row_ax[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
            row_ax[0].set_title(f"test #{i}  true: {names[y[i]]}", fontsize=7.5)
            rec: Dict[str, Any] = {"set": f"improved_{set_name}_set", "test_index": i, "true": names[y[i]]}
            for ax, tag, short in ((row_ax[1], "baseline", "Base"), (row_ax[2], "improved", "Impr")):
                res = results[tag]["res"]
                p = int(res["pred"][i])
                conf = float(res["probs"][i].max())
                cam = cam_for_class(loaded[tag], gray, p)
                blank = is_blank(cam)
                ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
                ax.imshow(up(cam), cmap="autumn", alpha=np.clip(up(cam), 0, 1) * 0.75)
                ok = p == int(y[i])
                ax.set_title(f"{short}: {names[p]} {100 * conf:.0f}% ({'correct' if ok else 'WRONG'})"
                             + ("  [blank map]" if blank else ""), fontsize=7.2,
                             color="black" if ok else "#a00000")
                rec.update({f"{tag}_pred": names[p], f"{tag}_correct": ok, f"{tag}_conf": f"{conf:.3f}",
                            f"{tag}_cam_mass_on_garment": f"{cam_mass_on_garment(gray, cam):.3f}",
                            f"{tag}_blank_map": blank})
            for ax in row_ax:
                ax.set_xticks([]); ax.set_yticks([])
            rows.append(rec)
        fig.suptitle("Same test images: baseline-seed0 vs full_improved-seed0\n"
                     f"(rows = improved model's {set_name} gallery; each model's own prediction shown)\n"
                     "blank map = no positive evidence for the predicted class", fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(out / f"gradcam_shared_{set_name}_set.png", dpi=110)
        plt.close(fig)

    with open(out / "shared_examples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "selection": "the 20 test indices already chosen for the improved model (examples.csv of fashion-official-1-gradcam); "
                     "not re-selected after viewing any baseline map",
        "n_images": len(rows),
        "baseline_regenerated_accuracy": results["baseline"]["acc"], "baseline_stored_accuracy": base_run["accuracy"],
        "improved_regenerated_accuracy": results["improved"]["acc"], "improved_stored_accuracy": imp_run["accuracy"],
        "identity": identity,
        "counts_on_shared_images": {
            "baseline_correct": sum(r["baseline_correct"] for r in rows),
            "improved_correct": sum(r["improved_correct"] for r in rows),
            "baseline_blank_maps": sum(r["baseline_blank_map"] for r in rows),
            "improved_blank_maps": sum(r["improved_blank_map"] for r in rows),
        },
    }
    (out / "identity_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(base_name: str, imp_name: str, group: str, improved_dir: str, out_dir: str) -> None:
    base_run, base_ds, base_b64 = load_from_supabase(base_name, group)
    imp_run, imp_ds, imp_b64 = load_from_supabase(imp_name, group)
    data = fashion.load()
    s = build(base_b64, imp_b64, base_run, imp_run, base_ds, imp_ds, data,
              read_indices(Path(improved_dir)), Path(out_dir), group)
    print(json.dumps({k: v for k, v in s.items() if k != "identity"}, indent=2))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) != 6 or a[0] != "--supabase":
        raise SystemExit(__doc__)
    main(*a[1:])
