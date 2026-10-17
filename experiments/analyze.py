"""Turn the SQL query results (docs/comparison_queries.sql) into tables and figures.

Input: a folder with q1.csv ... q7.csv, exactly as the queries return them
(from the Supabase SQL editor, or from experiments/sql_check.mjs locally).

    python -m experiments.analyze <sql_dir> <out_dir> [--classes fashion|legacy-synthetic]

Writes to <out_dir>:
    summary.md                   mean +/- sd per variant, per-technique deltas, audit
    loss_curves.png              train/val loss per variant (mean over seeds, band = min..max)
    accuracy_curves.png          validation accuracy per variant (+ training accuracy when q3 has it)
    per_class_metrics.csv        per-class test precision / recall per variant (from query 7)
    calibration.csv              per-run temperature and ECE / NLL before and after (query 9; optional)
    confusion_matrices.png       summed confusion matrix per variant
    test_metrics_by_seed.png     test accuracy per seed per variant
"""
from __future__ import annotations

import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Two-sided 95% t critical values by degrees of freedom (n_seeds - 1).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

VARIANTS = ["baseline", "full_improved", "no_augmentation", "no_batchnorm", "no_dropout", "no_scheduler"]
# Class order of the two datasets this tool can analyse. Fashion-MNIST is the project's dataset;
# the four synthetic shapes exist only to regenerate the historical official-supabase-1 outputs.
FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
LEGACY_SYNTHETIC_CLASSES = ["horizontal", "vertical", "diagonal", "block"]
CLASS_SETS = {"fashion": FASHION_CLASSES, "legacy-synthetic": LEGACY_SYNTHETIC_CLASSES}
TECHNIQUE = {
    "no_augmentation": "augmentation", "no_batchnorm": "batch norm",
    "no_dropout": "dropout", "no_scheduler": "LR scheduler",
}


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def variant_of(run_name: str) -> str:
    return run_name.rsplit("-seed", 1)[0]


def sd(xs: list[float]) -> float:
    return st.stdev(xs) if len(xs) > 1 else float("nan")


def per_class_metrics(q7: list[dict], classes: list[str]) -> dict:
    """Per-class test precision and recall for every run, from the saved confusion counts.

    ``q7`` rows are (run_name, true_class, predicted_class, n). Precision = TP / predicted-as-class,
    recall = TP / true-in-class. A class that a run never predicts has undefined precision; it is
    scored 0.0 (same convention as the training code's macro-F1) and counted in ``undefined``.
    Returns {variant: {class: {"precision": [per-run...], "recall": [...], "undefined": int}}}.
    """
    k_n = len(classes)
    runs: dict[str, list[list[int]]] = defaultdict(lambda: [[0] * k_n for _ in range(k_n)])
    for r in q7:
        runs[r["run_name"]][classes.index(r["true_class"])][classes.index(r["predicted_class"])] += int(r["n"])
    out: dict = {v: {c: {"precision": [], "recall": [], "undefined": 0} for c in classes} for v in VARIANTS}
    for name in sorted(runs):
        m = runs[name]
        for k, c in enumerate(classes):
            tp = m[k][k]
            predicted = sum(m[i][k] for i in range(k_n))
            actual = sum(m[k])
            cell = out[variant_of(name)][c]
            cell["precision"].append(tp / predicted if predicted else 0.0)
            cell["undefined"] += 0 if predicted else 1
            cell["recall"].append(tp / actual if actual else 0.0)
    return out


def main(sql_dir: Path, out: Path, classes: list[str] | None = None) -> None:
    CLASSES = classes or FASHION_CLASSES
    K = len(CLASSES)
    out.mkdir(parents=True, exist_ok=True)
    q1, q2, q3, q4 = (read(sql_dir / f"q{i}.csv") for i in (1, 2, 3, 4))
    q6, q7 = read(sql_dir / "q6.csv"), read(sql_dir / "q7.csv")
    lines: list[str] = []
    prov = sql_dir / "PROVENANCE.md"
    lines.append(prov.read_text().strip() if prov.exists() else "# SQL provenance\n\n- source: **not recorded** (no PROVENANCE.md next to the CSVs)")
    lines.append("")

    # --- audit (query 1): every count must be 1 per seed --------------------
    audit_ok = all(
        all(v == "1" for k, v in row.items() if k not in ("train_seed", "n_runs")) and row["n_runs"] == "6"
        for row in q1
    )
    lines.append(f"**Matched-configuration audit (query 1):** {'PASS' if audit_ok else 'FAIL'} "
                 f"- {len(q1)} seeds, 6 runs each, one dataset/split/budget per seed.\n")

    # --- per-variant test metrics (query 4) ---------------------------------
    per = defaultdict(lambda: defaultdict(list))
    for r in q4:
        v = variant_of(r["run_name"])
        per[v]["acc"].append(float(r["test_accuracy"]))
        per[v]["f1"].append(float(r["test_macro_f1"]))
        per[v]["loss"].append(float(r["test_loss"]))
        per[v]["seed"].append(int(r["train_seed"]))
        per[v]["params"].append(int(r["param_count"]))
    lines.append("### Test results per variant (mean ± sd across seeds; from query 4)\n")
    lines.append("| Variant | Seeds | Test accuracy | Macro-F1 | Test loss | Params |")
    lines.append("|---|---|---|---|---|---|")
    for v in VARIANTS:
        d = per[v]
        lines.append(
            f"| {v} | {len(d['acc'])} | {100*st.mean(d['acc']):.2f}% ± {100*sd(d['acc']):.2f} "
            f"| {st.mean(d['f1']):.4f} ± {sd(d['f1']):.4f} | {st.mean(d['loss']):.4f} ± {sd(d['loss']):.4f} "
            f"| {d['params'][0]} |")
    lines.append("")

    # --- per-seed accuracy table --------------------------------------------
    seeds = sorted({int(r["train_seed"]) for r in q4})
    lines.append("### Test accuracy by seed\n")
    lines.append("| Variant | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    lines.append("|---|" + "---|" * len(seeds))
    for v in VARIANTS:
        by = dict(zip(per[v]["seed"], per[v]["acc"]))
        lines.append(f"| {v} | " + " | ".join(f"{100*by[s]:.2f}%" for s in seeds) + " |")
    lines.append("")

    # --- ablation deltas (query 6) ------------------------------------------
    deltas = defaultdict(lambda: defaultdict(list))
    for r in q6:
        deltas[r["variant"]]["acc"].append(float(r["delta_accuracy_vs_full"]))
        deltas[r["variant"]]["f1"].append(float(r["delta_macro_f1_vs_full"]))
        deltas[r["variant"]]["loss"].append(float(r["delta_test_loss_vs_full"]))
    lines.append("### Difference from full_improved, same seed (from query 6)\n")
    lines.append("Negative accuracy delta = scored lower than the full model. "
                 "Accuracy deltas in percentage points (pp).\n")
    lines.append("| Variant | Technique removed | Δ accuracy per seed (pp) | Mean Δ acc (pp) | 95% CI of mean Δ (pp, paired t) | Mean Δ macro-F1 | Mean Δ test loss | Seeds lower than full |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v in ["baseline"] + list(TECHNIQUE):
        d = deltas[v]
        per_seed = ", ".join(f"{100*x:+.2f}" for x in d["acc"])
        lower = sum(1 for x in d["acc"] if x < 0)
        n = len(d["acc"])
        half = T95.get(n - 1, 1.96) * 100 * sd(d["acc"]) / math.sqrt(n) if n > 1 else float("nan")
        m = 100 * st.mean(d["acc"])
        ci = f"[{m - half:+.2f}, {m + half:+.2f}]" + (" includes 0" if m - half <= 0 <= m + half else "")
        lines.append(
            f"| {v} | {TECHNIQUE.get(v, 'architecture + all four')} | {per_seed} | {m:+.2f} | {ci} "
            f"| {st.mean(d['f1']):+.4f} | {st.mean(d['loss']):+.4f} | {lower} of {len(d['acc'])} |")
    lines.append("")

    # --- validation evidence (query 2) --------------------------------------
    val = defaultdict(lambda: defaultdict(list))
    for r in q2:
        v = variant_of(r["run_name"])
        val[v]["bvl"].append(float(r["best_val_loss"]))
        val[v]["vacc"].append(float(r["val_accuracy_at_best"]))
        val[v]["ep"].append(int(r["best_epoch"]))
        val[v]["ftl"].append(float(r["final_train_loss"]))
    lines.append("### Validation / training evidence (from query 2)\n")
    lines.append("| Variant | Best val loss | Val acc at best | Best epoch (0-based) per seed | Final train loss |")
    lines.append("|---|---|---|---|---|")
    for v in VARIANTS:
        d = val[v]
        lines.append(f"| {v} | {st.mean(d['bvl']):.4f} | {100*st.mean(d['vacc']):.2f}% "
                     f"| {', '.join(map(str, d['ep']))} | {st.mean(d['ftl']):.4f} |")
    lines.append("")

    # --- final-epoch train vs validation accuracy (query 3): the fit/generalisation picture ----
    last = defaultdict(dict)
    for r in q3:
        if r.get("train_accuracy") not in (None, ""):
            last[(variant_of(r["run_name"]), int(r["train_seed"]))][int(r["epoch"])] = (
                float(r["train_accuracy"]), float(r["val_accuracy"]), float(r["train_loss"]), float(r["val_loss"]))
    if last:
        lines.append("### Fit at the last epoch: training vs validation (from query 3, mean over seeds)\n")
        lines.append("Training accuracy is the whole un-augmented training split scored in eval mode, so it is comparable "
                     "to validation accuracy: a small gap with modest accuracy suggests under-fitting, a large gap "
                     "over-fitting. Training LOSS is different: it is the mean mini-batch loss in training mode on augmented "
                     "data with dropout, so it sits above the validation loss for the regularized variants and is not "
                     "comparable to it.\n")
        lines.append("| Variant | Train acc | Val acc | Gap (pp) | Train loss | Val loss |")
        lines.append("|---|---|---|---|---|---|")
        for v in VARIANTS:
            vals = [last[(v, s_)][max(last[(v, s_)])] for s_ in sorted({k[1] for k in last if k[0] == v})]
            ta, va, tl, vl = (st.mean(x[i] for x in vals) for i in range(4))
            lines.append(f"| {v} | {100*ta:.2f}% | {100*va:.2f}% | {100*(ta-va):+.2f} | {tl:.4f} | {vl:.4f} |")
        lines.append("")

    # --- confusion (query 7) ------------------------------------------------
    cm = {v: [[0] * K for _ in range(K)] for v in VARIANTS}
    for r in q7:
        v = variant_of(r["run_name"])
        cm[v][CLASSES.index(r["true_class"])][CLASSES.index(r["predicted_class"])] += int(r["n"])
    lines.append("### Most common confusions (summed over seeds, from query 7)\n")
    lines.append("| Variant | Errors | Top confusions (true → predicted: count) |")
    lines.append("|---|---|---|")
    for v in VARIANTS:
        offd = [(cm[v][i][j], CLASSES[i], CLASSES[j]) for i in range(K) for j in range(K) if i != j]
        offd.sort(reverse=True)
        top = "; ".join(f"{a} → {b}: {n}" for n, a, b in offd[:5] if n)
        lines.append(f"| {v} | {sum(n for n, _, _ in offd)} | {top} |")
    lines.append("")

    # --- per-class precision / recall (from the same confusion counts) ---------
    pc = per_class_metrics(q7, CLASSES)
    lines.append("### Per-class test precision and recall (mean ± sd over seeds; from query 7)\n")
    lines.append("Computed from each run's saved confusion matrix. Precision = correct / predicted as that "
                 "class; recall = correct / truly that class. A class never predicted by a run has "
                 "precision scored as 0 (shown with *).\n")
    lines.append("| Variant | Class | Precision | Recall |")
    lines.append("|---|---|---|---|")
    csv_rows = [("variant", "class", "mean_precision", "sd_precision", "mean_recall", "sd_recall", "n_runs")]
    for v in VARIANTS:
        for c in CLASSES:
            cell = pc[v][c]
            star = "*" if cell["undefined"] else ""
            lines.append(f"| {v} | {c} | {100*st.mean(cell['precision']):.1f}% ± {100*sd(cell['precision']):.1f}{star} "
                         f"| {100*st.mean(cell['recall']):.1f}% ± {100*sd(cell['recall']):.1f} |")
            csv_rows.append((v, c, f"{st.mean(cell['precision']):.4f}", f"{sd(cell['precision']):.4f}",
                             f"{st.mean(cell['recall']):.4f}", f"{sd(cell['recall']):.4f}", len(cell["recall"])))
    lines.append("")
    with open(out / "per_class_metrics.csv", "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    q10_path = sql_dir / "q10.csv"
    if q10_path.exists():  # stored per-class metrics (query 10) must equal the ones recomputed from query 7
        q10 = read(q10_path)
        worst = 0.0
        by_run: dict = defaultdict(dict)
        for r in q10:
            by_run[r["run_name"]][r["class_name"]] = (float(r["precision"]), float(r["recall"]))
        runs7: dict = defaultdict(lambda: [[0] * K for _ in range(K)])
        for r in q7:
            runs7[r["run_name"]][CLASSES.index(r["true_class"])][CLASSES.index(r["predicted_class"])] += int(r["n"])
        for name, m in runs7.items():
            for k, c in enumerate(CLASSES):
                pred = sum(m[i][k] for i in range(K))
                p_ = m[k][k] / pred if pred else 0.0
                r_ = m[k][k] / sum(m[k])
                worst = max(worst, abs(p_ - by_run[name][c][0]), abs(r_ - by_run[name][c][1]))
        lines.append(f"**Cross-check:** stored per-class precision/recall (query 10, {len(q10)} rows) vs recomputed from "
                     f"confusion matrices (query 7): max absolute difference {worst:.2e}.\n")

    # --- calibration (query 9, optional: runs trained after migration 005) -----------
    q9_path = sql_dir / "q9.csv"
    if q9_path.exists():
        q9 = read(q9_path)
        lines.append("### Calibration: temperature scaling fitted on validation (query 9)\n")
        lines.append("ECE = expected calibration error (15 bins; lower is better); NLL = negative log-likelihood. "
                     "'Before' is plain softmax (T = 1); 'after' divides the logits by the temperature fitted on "
                     "validation data. Test columns are held-out measurements, never used to fit T. "
                     "Mean over seeds.\n")
        lines.append("| Variant | Mean T | Test ECE before | Test ECE after | Test NLL before | Test NLL after |")
        lines.append("|---|---|---|---|---|---|")
        cal = defaultdict(lambda: defaultdict(list))
        for r in q9:
            for key in ("temperature", "test_ece_before", "test_ece_after", "test_nll_before", "test_nll_after"):
                if r.get(key) not in (None, ""):
                    cal[variant_of(r["run_name"])][key].append(float(r[key]))
        for v in VARIANTS:
            d = cal[v]
            if d["temperature"]:
                lines.append(f"| {v} | {st.mean(d['temperature']):.3f} | {st.mean(d['test_ece_before']):.4f} "
                             f"| {st.mean(d['test_ece_after']):.4f} | {st.mean(d['test_nll_before']):.4f} "
                             f"| {st.mean(d['test_nll_after']):.4f} |")
        lines.append("")
        with open(out / "calibration.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(q9[0].keys()) if q9 else ["run_name"])
            w.writeheader()
            w.writerows(q9)

    (out / "summary.md").write_text("\n".join(lines))

    # --- figures ------------------------------------------------------------
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in q3:
        v = variant_of(r["run_name"])
        e, s = int(r["epoch"]), int(r["train_seed"])
        curves[v]["train"][s][e] = float(r["train_loss"])
        curves[v]["val"][s][e] = float(r["val_loss"])
        curves[v]["val_acc"][s][e] = float(r["val_accuracy"])
        if r.get("train_accuracy") not in (None, ""):  # only in exports of runs trained after migration 004
            curves[v]["train_acc"][s][e] = float(r["train_accuracy"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, v in zip(axes.flat, VARIANTS):
        for kind, color in (("train", "#2a6fdb"), ("val", "#d9534f")):
            by_seed = curves[v][kind]
            epochs = sorted(next(iter(by_seed.values())))
            mat = [[by_seed[s][e] for e in epochs] for s in sorted(by_seed)]
            mean = [st.mean(col) for col in zip(*mat)]
            ax.plot([e + 1 for e in epochs], mean, color=color, label=f"{kind} loss")
            ax.fill_between([e + 1 for e in epochs], [min(c) for c in zip(*mat)],
                            [max(c) for c in zip(*mat)], color=color, alpha=0.15)
        ax.set_title(v)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("cross-entropy loss")
    axes[0, 0].legend()
    fig.suptitle("Training vs validation loss (line = mean over seeds, band = min to max)")
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=130)
    plt.close(fig)

    has_train_acc = any(curves[v]["train_acc"] for v in VARIANTS)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, v in zip(axes.flat, VARIANTS):
        for kind, color, label in (("train_acc", "#2a6fdb", "train accuracy"),
                                   ("val_acc", "#d9534f", "validation accuracy")):
            by_seed = curves[v][kind]
            if not by_seed:
                continue
            epochs = sorted(next(iter(by_seed.values())))
            mat = [[100 * by_seed[s][e] for e in epochs] for s in sorted(by_seed)]
            ax.plot([e + 1 for e in epochs], [st.mean(c) for c in zip(*mat)], color=color, label=label)
            ax.fill_between([e + 1 for e in epochs], [min(c) for c in zip(*mat)],
                            [max(c) for c in zip(*mat)], color=color, alpha=0.15)
        ax.set_title(v)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (%)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Validation accuracy (line = mean over seeds, band = min to max)"
                 + ("" if has_train_acc else "\nTraining accuracy was not recorded for these runs"))
    fig.tight_layout()
    fig.savefig(out / "accuracy_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(7.2 * 3, 6.6 * 2))
    for ax, v in zip(axes.flat, VARIANTS):
        m = cm[v]
        ax.imshow(m, cmap="Blues")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, m[i][j], ha="center", va="center", fontsize=8 if K > 5 else 10,
                        color="white" if m[i][j] > max(map(max, m)) / 2 else "black")
        ax.set_xticks(range(K), CLASSES, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(K), CLASSES, fontsize=8)
        ax.set_title(f"{v} ({len(seeds)} seeds summed)")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle("Test confusion matrices (rows = true class, columns = predicted; counts summed over seeds)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.5 * 2, 4.6), sharey=True)
    for ax, key, title in zip(axes, ("precision", "recall"), ("Test precision", "Test recall")):
        grid = [[100 * st.mean(pc[v][c][key]) for c in CLASSES] for v in VARIANTS]
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8,
                        color="black" if val > 60 else "white")
        ax.set_xticks(range(K), CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(VARIANTS)), VARIANTS, fontsize=8)
        ax.set_title(f"{title} (%, mean over seeds)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out / "per_class_precision_recall.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, v in enumerate(VARIANTS):
        accs = [100 * a for a in per[v]["acc"]]
        ax.scatter([k] * len(accs), accs, color="#555", zorder=3)
        ax.hlines(st.mean(accs), k - 0.25, k + 0.25, color="#d9534f", lw=2, zorder=4)
    ax.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=20)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Test accuracy per seed (dots) and mean (red line)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "test_metrics_by_seed.png", dpi=130)
    plt.close(fig)
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--classes")]
    which = "fashion"
    if "--classes" in sys.argv:
        which = sys.argv[sys.argv.index("--classes") + 1]
        args = [a for a in args if a != which]
    main(Path(args[0]), Path(args[1]), CLASS_SETS[which])
