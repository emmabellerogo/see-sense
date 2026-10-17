"""Grad-CAM evidence: genuinely correct and incorrect test predictions (Part 2, step 6).

Loads one official model, regenerates that run's exact test split (same dataset
seed and split seed), classifies every test image with ``predict_with_cam``
(the same function ``POST /predict`` uses), and saves overlays for a selection
of correct and incorrect predictions.

    python -m experiments.gradcam_evidence <local_db.json> <run_name> <group> <out_dir>
    python -m experiments.gradcam_evidence --supabase <run_name> <group> <out_dir>

Writes <out_dir>/gradcam_correct.png, gradcam_incorrect.png and examples.csv
(--supabase also writes source.json: which run/group/dataset the evidence is from).
The model weights come from run_artifacts; nothing is retrained. --supabase is
read-only, takes credentials from .env / the environment, never prints them, and
keeps the checkpoint in memory only. It refuses to run if SEE_SENSE_LOCAL_DB is set.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from api.training import make_splits, predict_with_cam  # noqa: E402
from shared.data import CLASS_NAMES, generate_images  # noqa: E402


def clean_shapes(n_rows: int, img_size: int, noise: float, seed: int) -> np.ndarray:
    """Noise-free version of every image: replays generate_images' RNG stream.

    generate_images draws the class, then the shape (_draw), then the noise, from
    one Generator; replaying the same calls reproduces the exact shape that was
    drawn before noise was added. Used only to show where the true shape is.
    """
    from shared.data import N_CLASSES, _draw

    rng = np.random.default_rng(seed)
    out = np.empty((n_rows, img_size, img_size), dtype=np.float32)
    for i in range(n_rows):
        k = int(rng.integers(0, N_CLASSES))
        out[i] = _draw(k, img_size, rng)
        rng.normal(0.0, noise, (img_size, img_size))  # consume the noise draw
    return out


def _img(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def _cam_only(model_b64: str, gray: np.ndarray, cls: int) -> np.ndarray:
    """Grad-CAM map (0..1, 28x28) for an arbitrary class, for the side-by-side view."""
    import torch

    from api.training import _grad_cam, _load_model, _to_tensor

    model, _ = _load_model(model_b64)
    cam = _grad_cam(model, _to_tensor(gray[None]), cls)
    return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((28, 28))) / 255.0


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


def main(db_file: str, run_name: str, group: str, out_dir: str, supabase: bool = False) -> None:
    out = Path(out_dir)
    if supabase:
        run, ds, model_b64 = load_from_supabase(run_name, group)
    else:
        run, ds, model_b64 = load_from_file(db_file, run_name, group)
    out.mkdir(parents=True, exist_ok=True)
    cfg = run["config"]

    images, labels = generate_images(ds["n_rows"], img_size=ds["img_size"], noise=ds["noise"], seed=ds["seed"])
    clean = clean_shapes(ds["n_rows"], ds["img_size"], ds["noise"], ds["seed"])
    _, _, test_idx = make_splits(len(images), cfg["test_size"], cfg["val_size"], cfg["split_seed"])

    rows = []
    for i in test_idx:
        res = predict_with_cam(model_b64, images[i])
        rows.append({"index": int(i), "true": labels[i], "pred": res["label"],
                     "conf": res["confidence"], "probs": res["probs"], "res": res})
    acc = np.mean([r["true"] == r["pred"] for r in rows])
    # Must match the stored test accuracy exactly, or this is not the same model/split.
    assert abs(acc - run["accuracy"]) < 1e-9, (acc, run["accuracy"])

    correct = [r for r in rows if r["true"] == r["pred"]]
    wrong = sorted((r for r in rows if r["true"] != r["pred"]), key=lambda r: -r["conf"])
    # Correct: first example of each class in test order, plus the least confident correct one.
    pick_c = [next(r for r in correct if r["true"] == k) for k in range(4)]
    pick_c.append(min((r for r in correct if r not in pick_c), key=lambda r: r["conf"]))
    # Incorrect: cover distinct (true -> pred) confusions first, most confident first; up to 8.
    pick_w, seen = [], set()
    for r in wrong:
        if (r["true"], r["pred"]) not in seen:
            pick_w.append(r); seen.add((r["true"], r["pred"]))
    for r in wrong:
        if len(pick_w) >= 8:
            break
        if r not in pick_w:
            pick_w.append(r)
    pick_w = pick_w[:8]

    def panel(picks, title, fname, show_true_cam):
        cols = 4 if show_true_cam else 3
        fig, axes = plt.subplots(len(picks), cols, figsize=(2.3 * cols, 2.45 * len(picks) + 0.6))

        def heat(ax, gray, cam, shape):
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            ax.imshow(cam, cmap="autumn", alpha=np.clip(cam, 0, 1) * 0.75)
            ax.contour(shape, levels=[0.5], colors="cyan", linewidths=1.0)

        for row_axes, r in zip(axes, picks):
            gray, shape = images[r["index"]], clean[r["index"]]
            row_axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
            row_axes[0].set_title(f"#{r['index']} input (true: {CLASS_NAMES[r['true']]})", fontsize=7.5)
            row_axes[1].imshow(shape, cmap="gray", vmin=0, vmax=1)
            row_axes[1].set_title("true shape before noise", fontsize=7.5)
            heat(row_axes[2], gray, _cam_only(model_b64, gray, r["pred"]), shape)
            row_axes[2].set_title(f"CAM pred: {CLASS_NAMES[r['pred']]} ({100*r['conf']:.0f}%)", fontsize=7.5)
            if show_true_cam:
                heat(row_axes[3], gray, _cam_only(model_b64, gray, r["true"]), shape)
                row_axes[3].set_title(f"CAM if true: {CLASS_NAMES[r['true']]}", fontsize=7.5)
            for ax in row_axes:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title + "\n(cyan outline = true shape; yellow/red = Grad-CAM, brighter = more important)",
                     fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)

    panel(pick_c, f"{run_name}: correct test predictions",
          "gradcam_correct.png", False)
    panel(pick_w, f"{run_name}: incorrect test predictions",
          "gradcam_incorrect.png", True)

    with open(out / "examples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "test_index", "true", "predicted", "confidence",
                    "p_horizontal", "p_vertical", "p_diagonal", "p_block"])
        for name, picks in (("correct", pick_c), ("incorrect", pick_w)):
            for r in picks:
                w.writerow([name, r["index"], CLASS_NAMES[r["true"]], CLASS_NAMES[r["pred"]],
                            f"{r['conf']:.3f}", *[f"{p:.3f}" for p in r["probs"]]])
    if supabase:
        (out / "source.json").write_text(json.dumps({
            "source": "supabase", "experiment_group": group, "run_name": run_name,
            "run_id": run["id"], "dataset_id": run["dataset_id"],
            "dataset": {k: ds[k] for k in ("name", "n_rows", "img_size", "noise", "seed")},
            "split_seed": cfg["split_seed"], "split_sha256": run.get("split_sha256"),
            "stored_accuracy": run["accuracy"], "regenerated_accuracy": float(acc),
            "n_test": len(rows), "n_correct": len(correct), "n_incorrect": len(wrong),
            "run_created_at": run.get("created_at"),
        }, indent=2) + "\n")
    print(f"{run_name}: test accuracy {acc:.4f} (matches stored), {len(correct)} correct, {len(wrong)} wrong")
    print(open(out / "examples.csv").read())


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--supabase":
        main("", *args[1:4], supabase=True)
    else:
        main(*args[:4])
