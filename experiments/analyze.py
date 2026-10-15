"""Turn the SQL query results (docs/comparison_queries.sql) into tables and figures.

Input: a folder with q1.csv ... q7.csv, exactly as the queries return them
(from the Supabase SQL editor, or from experiments/sql_check.mjs locally).

    python -m experiments.analyze <sql_dir> <out_dir>

Writes to <out_dir>:
    summary.md                   mean +/- sd per variant, per-technique deltas, audit
    loss_curves.png              train/val loss per variant (mean over seeds, band = min..max)
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
CLASSES = ["horizontal", "vertical", "diagonal", "block"]
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


def main(sql_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    q1, q2, q3, q4 = (read(sql_dir / f"q{i}.csv") for i in (1, 2, 3, 4))
    q6, q7 = read(sql_dir / "q6.csv"), read(sql_dir / "q7.csv")
    lines: list[str] = []

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

    # --- confusion (query 7) ------------------------------------------------
    cm = {v: [[0] * 4 for _ in range(4)] for v in VARIANTS}
    for r in q7:
        v = variant_of(r["run_name"])
        cm[v][CLASSES.index(r["true_class"])][CLASSES.index(r["predicted_class"])] += int(r["n"])
    lines.append("### Most common confusions (summed over seeds, from query 7)\n")
    lines.append("| Variant | Errors | Top confusions (true → predicted: count) |")
    lines.append("|---|---|---|")
    for v in VARIANTS:
        offd = [(cm[v][i][j], CLASSES[i], CLASSES[j]) for i in range(4) for j in range(4) if i != j]
        offd.sort(reverse=True)
        top = "; ".join(f"{a} → {b}: {n}" for n, a, b in offd[:3] if n)
        lines.append(f"| {v} | {sum(n for n, _, _ in offd)} | {top} |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines))

    # --- figures ------------------------------------------------------------
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in q3:
        v = variant_of(r["run_name"])
        e, s = int(r["epoch"]), int(r["train_seed"])
        curves[v]["train"][s][e] = float(r["train_loss"])
        curves[v]["val"][s][e] = float(r["val_loss"])
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

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for ax, v in zip(axes.flat, VARIANTS):
        m = cm[v]
        ax.imshow(m, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, m[i][j], ha="center", va="center",
                        color="white" if m[i][j] > max(map(max, m)) / 2 else "black")
        ax.set_xticks(range(4), CLASSES, rotation=30)
        ax.set_yticks(range(4), CLASSES)
        ax.set_title(f"{v} ({len(seeds)} seeds summed)")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=130)
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
    main(Path(sys.argv[1]), Path(sys.argv[2]))
