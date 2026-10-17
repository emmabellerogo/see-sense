"""Grad-CAM evidence: genuinely correct and incorrect Fashion-MNIST test predictions.

Loads one saved checkpoint, classifies every official test image with it, checks that the
regenerated test accuracy equals the accuracy stored with the run (otherwise this is not the
same model), and saves Grad-CAM overlays for a deterministic selection of correct and
incorrect predictions.

    python -m experiments.gradcam_evidence <local_db.json> <run_name> <group> <out_dir>
    python -m experiments.gradcam_evidence --supabase <run_name> <group> <out_dir>

Writes <out_dir>/gradcam_correct.png, gradcam_incorrect.png, examples.csv and source.json.
The weights come from run_artifacts; nothing is retrained. --supabase is read-only, takes
credentials from .env / the environment, never prints them, keeps the checkpoint in memory,
and refuses to run if SEE_SENSE_LOCAL_DB is set. The Fashion-MNIST files must be on disk
(``python tools/fetch_fashion_mnist.py``).

SELECTION RULE (fixed here, before any figure is looked at; not chosen by appearance):
  * correct   - for every class, the first correctly classified test image in test order
                (10 examples, one per class);
  * incorrect - wrong predictions sorted by confidence (highest first), taking one example
                per distinct (true -> predicted) pair first, then filling with the next most
                confident wrong predictions, up to 10. "Incorrect" means predicted != true.

``cam_mass_on_garment`` in examples.csv is the share of the Grad-CAM map that falls on pixels
brighter than 20% of the image maximum, a number to support (not replace) reading the figure.
Interpretations of failures are written by a person in docs/ after viewing the figures.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from api import calibration  # noqa: E402
from api.training import cam_for_class, load_artifact, predict_with_cam, to01, normalize  # noqa: E402
from shared import fashion  # noqa: E402

SELECTION_RULE = (
    "correct: first correctly classified test image of each class in test order; "
    "incorrect: wrong predictions by descending confidence, one per distinct (true, predicted) pair "
    "first, then the next most confident, up to 10."
)


def load_from_file(db_file: str, run_name: str, group: str):
    """(run, dataset, model_b64) from an offline SEE_SENSE_LOCAL_DB json file."""
    db = json.loads(Path(db_file).read_text())
    run = next(r for r in db["runs"] if r["experiment_group"] == group and r["run_name"] == run_name)
    ds = next(d for d in db["datasets"] if d["id"] == run["dataset_id"])
    model_b64 = next(a["model_b64"] for a in db["run_artifacts"] if a["run_id"] == run["id"])
    return run, ds, model_b64


def load_from_supabase(run_name: str, group: str, client=None):
    """(run, dataset, model_b64) read (select-only) from Supabase.

    Refuses offline storage: with SEE_SENSE_LOCAL_DB set, api.db would silently
    read a local file instead of the shared project. ``client`` is injectable for tests.
    """
    from dotenv import load_dotenv

    load_dotenv()
    if os.environ.get("SEE_SENSE_LOCAL_DB"):
        raise SystemExit("Refusing --supabase: SEE_SENSE_LOCAL_DB (offline storage) is set. "
                         "Unset it so the shared Supabase project is used.")
    if client is None:
        if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")):
            raise SystemExit("Refusing --supabase: SUPABASE_URL / SUPABASE_SERVICE_KEY are not set.")
        from api.db import get_client

        client = get_client()
    runs = (client.table("runs").select("*").eq("experiment_group", group)
            .eq("run_name", run_name).execute().data)
    if len(runs) != 1:
        raise LookupError(f"expected exactly 1 run {group}/{run_name}, found {len(runs)}")
    run = runs[0]
    ds = client.table("datasets").select("*").eq("id", run["dataset_id"]).limit(1).execute().data
    art = client.table("run_artifacts").select("model_b64").eq("run_id", run["id"]).limit(1).execute().data
    if not ds or not art:
        raise LookupError(f"run {run['id']} is missing its dataset or checkpoint")
    return run, ds[0], art[0]["model_b64"]


def classify_all(model_b64: str, x_test: np.ndarray, chunk: int = 2000) -> Dict[str, np.ndarray]:
    """Raw logits and calibrated probabilities for every test image (batched, eval mode)."""
    import torch

    loaded = load_artifact(model_b64)
    outs = []
    with torch.no_grad():
        for s in range(0, len(x_test), chunk):
            xb = normalize(to01(x_test[s : s + chunk]), loaded.preprocessing)
            outs.append(loaded.model(xb).numpy())
    logits = np.concatenate(outs).astype(np.float64)
    probs = calibration.softmax(logits, loaded.temperature)
    return {"logits": logits, "probs": probs, "pred": probs.argmax(axis=1)}


def select_examples(y_true: np.ndarray, pred: np.ndarray, conf: np.ndarray, n_classes: int,
                    max_incorrect: int = 10) -> Dict[str, List[int]]:
    """Apply SELECTION_RULE. Raises if fewer than 5 correct or 5 incorrect examples exist."""
    correct_idx = np.flatnonzero(pred == y_true)
    picks_c: List[int] = []
    for k in range(n_classes):
        hit = correct_idx[y_true[correct_idx] == k]
        if len(hit):
            picks_c.append(int(hit[0]))
    wrong = np.flatnonzero(pred != y_true)
    wrong = wrong[np.argsort(-conf[wrong], kind="stable")]
    picks_w: List[int] = []
    seen = set()
    for i in wrong:
        key = (int(y_true[i]), int(pred[i]))
        if key not in seen:
            picks_w.append(int(i))
            seen.add(key)
        if len(picks_w) >= max_incorrect:
            break
    for i in wrong:
        if len(picks_w) >= max_incorrect:
            break
        if int(i) not in picks_w:
            picks_w.append(int(i))
    if len(picks_c) < 5 or len(picks_w) < 5:
        raise ValueError(f"need at least 5 correct and 5 incorrect examples, got {len(picks_c)} and {len(picks_w)}")
    return {"correct": picks_c, "incorrect": picks_w}


def cam_mass_on_garment(img: np.ndarray, cam01: np.ndarray) -> float:
    """Share of Grad-CAM mass on pixels brighter than 20% of the image maximum (0 if no CAM)."""
    cam = np.asarray(Image.fromarray((cam01 * 255).astype(np.uint8)).resize(img.shape[::-1]),
                     dtype=np.float64) / 255.0
    total = cam.sum()
    if total <= 0:
        return 0.0
    mask = img.astype(np.float64) > 0.2 * max(float(img.max()), 1.0)
    return float(cam[mask].sum() / total)


def build_evidence(model_b64: str, data: fashion.FashionData, out: Path, run: Optional[Dict[str, Any]] = None,
                   run_name: str = "model") -> Dict[str, Any]:
    """Create the figures, examples.csv and (when ``run`` is given) source.json."""
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_artifact(model_b64)
    names = loaded.class_names
    y = data.y_test.astype(np.int64)
    res = classify_all(model_b64, data.x_test)
    pred, probs = res["pred"], res["probs"]
    conf = probs.max(axis=1)
    acc = float((pred == y).mean())
    if run is not None and abs(acc - run["accuracy"]) > 1e-9:
        raise AssertionError(f"regenerated test accuracy {acc} != stored {run['accuracy']}: not the same model/data")
    picks = select_examples(y, pred, conf, len(names))

    # The serving path (the same function POST /predict uses) must agree with the batch path.
    for i in picks["correct"] + picks["incorrect"]:
        served = predict_with_cam(model_b64, data.x_test[i])
        assert served["label"] == int(pred[i]), f"serving path disagrees on test image {i}"

    def heat(ax, gray, cam):
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax.imshow(cam, cmap="autumn", alpha=np.clip(cam, 0, 1) * 0.75)

    def up(cam):  # 7x7 -> 28x28 for display
        return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((28, 28), Image.BILINEAR)) / 255.0

    def panel(idxs: List[int], title: str, fname: str, show_true_cam: bool) -> None:
        cols = 3 if show_true_cam else 2
        fig, axes = plt.subplots(len(idxs), cols, figsize=(2.4 * cols, 2.5 * len(idxs) + 0.7), squeeze=False)
        for row, i in zip(axes, idxs):
            gray = data.x_test[i]
            row[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
            row[0].set_title(f"test #{i}  true: {names[y[i]]}", fontsize=7.5)
            heat(row[1], gray, up(cam_for_class(loaded, gray, int(pred[i]))))
            row[1].set_title(f"CAM pred: {names[pred[i]]} ({100 * conf[i]:.0f}%)", fontsize=7.5)
            if show_true_cam:
                heat(row[2], gray, up(cam_for_class(loaded, gray, int(y[i]))))
                row[2].set_title(f"CAM if true: {names[y[i]]}", fontsize=7.5)
            for ax in row:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title + "\n(yellow/red = Grad-CAM of the class logit, brighter = more influential)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)

    panel(picks["correct"], f"{run_name}: correct test predictions", "gradcam_correct.png", False)
    panel(picks["incorrect"], f"{run_name}: incorrect test predictions", "gradcam_incorrect.png", True)

    with open(out / "examples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "test_index", "true", "predicted", "confidence", "cam_mass_on_garment",
                    *[f"p_{n}" for n in names]])
        for name, idxs in picks.items():
            for i in idxs:
                mass = cam_mass_on_garment(data.x_test[i], cam_for_class(loaded, data.x_test[i], int(pred[i])))
                w.writerow([name, i, names[y[i]], names[pred[i]], f"{conf[i]:.3f}", f"{mass:.3f}",
                            *[f"{p:.3f}" for p in probs[i]]])
    source = {
        "dataset": fashion.DATASET_NAME, "class_names": names, "selection_rule": SELECTION_RULE,
        "n_test": int(len(y)), "n_correct": int((pred == y).sum()), "n_incorrect": int((pred != y).sum()),
        "regenerated_accuracy": acc, "temperature": loaded.temperature,
        "checkpoint": {"artifact_version": loaded.artifact_version, "dataset": loaded.dataset.get("name")},
    }
    if run is not None:
        source.update({
            "source": "supabase-or-local-store", "experiment_group": run.get("experiment_group"),
            "run_name": run.get("run_name"), "run_id": run["id"], "dataset_id": run.get("dataset_id"),
            "stored_accuracy": run["accuracy"], "split_sha256": run.get("split_sha256"),
            "run_created_at": run.get("created_at"),
        })
    (out / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    return source


def main(db_file: str, run_name: str, group: str, out_dir: str, supabase: bool = False) -> None:
    if supabase:
        run, _ds, model_b64 = load_from_supabase(run_name, group)
    else:
        run, _ds, model_b64 = load_from_file(db_file, run_name, group)
    data = fashion.load()
    src = build_evidence(model_b64, data, Path(out_dir), run, run_name)
    print(f"{run_name}: test accuracy {src['regenerated_accuracy']:.4f} (matches stored), "
          f"{src['n_correct']} correct, {src['n_incorrect']} wrong")
    print((Path(out_dir) / "examples.csv").read_text())


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--supabase":
        main("", *args[1:4], supabase=True)
    else:
        main(*args[:4])
