"""Loaders for the saved official-supabase-1 experiment evidence.

Pure standard library (no streamlit, pandas or torch) so it can be unit-tested
offline. Every path is resolved from this file's location, never from the
current working directory. Loaders raise EvidenceError with a short readable
message when a file is missing or malformed; they never invent values.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUP = "official-supabase-1"
RESULTS = REPO_ROOT / "experiments" / "results"
SQL_DIR = RESULTS / f"{GROUP}-sql"
ANALYSIS_DIR = RESULTS / f"{GROUP}-analysis"
GRADCAM_DIR = RESULTS / f"{GROUP}-gradcam"
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"
FAILURE_DOC = REPO_ROOT / "docs" / "SUPABASE_GRADCAM_FAILURE_ANALYSIS.md"

EXPECTED_SEEDS = 5
EXPECTED_VARIANTS = (
    "baseline", "full_improved", "no_augmentation",
    "no_batchnorm", "no_dropout", "no_scheduler",
)
# Two-sided 95% t critical value, df = 4 (five paired seeds).
T_CRIT_DF4 = 2.776445105
N_CORRECT_EXPECTED = 5
N_INCORRECT_EXPECTED = 8
UNCERTAIN_BELOW = 0.5  # a correct prediction under this confidence is flagged

FIGURES = {
    "loss_curves": ANALYSIS_DIR / "loss_curves.png",
    "confusion_matrices": ANALYSIS_DIR / "confusion_matrices.png",
    "test_metrics_by_seed": ANALYSIS_DIR / "test_metrics_by_seed.png",
    "gradcam_correct": GRADCAM_DIR / "gradcam_correct.png",
    "gradcam_incorrect": GRADCAM_DIR / "gradcam_incorrect.png",
}

# Ablation variant -> technique removed (the full model keeps the others).
TECHNIQUE_REMOVED = {
    "baseline": "architecture + all four techniques",
    "no_augmentation": "augmentation",
    "no_batchnorm": "batch norm",
    "no_dropout": "dropout",
    "no_scheduler": "LR scheduler",
}


class EvidenceError(Exception):
    """Evidence is missing or incomplete; the message is safe to show a user."""


def _read_csv(path: Path) -> list[dict]:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        raise EvidenceError(f"Missing evidence file: {_rel(path)}") from None
    except csv.Error:
        raise EvidenceError(f"Could not parse {_rel(path)}") from None
    if not rows:
        raise EvidenceError(f"Evidence file is empty: {_rel(path)}")
    return rows


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _num(row: dict, col: str, path: Path) -> float:
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        raise EvidenceError(f"{_rel(path)}: missing or non-numeric column '{col}'") from None


def load_variant_summary(sql_dir: Path = SQL_DIR) -> list[dict]:
    """Query 5: one row per variant, mean/sd over seeds."""
    path = sql_dir / "q5.csv"
    out = []
    for r in _read_csv(path):
        out.append({
            "variant": r.get("variant", ""),
            "n_seeds": int(_num(r, "n_seeds", path)),
            "mean_test_accuracy": _num(r, "mean_test_accuracy", path),
            "sd_test_accuracy": _num(r, "sd_test_accuracy", path),
            "mean_test_macro_f1": _num(r, "mean_test_macro_f1", path),
            "sd_test_macro_f1": _num(r, "sd_test_macro_f1", path),
            "mean_test_loss": _num(r, "mean_test_loss", path),
            "mean_train_seconds": _num(r, "mean_train_seconds", path),
        })
    missing = set(EXPECTED_VARIANTS) - {r["variant"] for r in out}
    if missing:
        raise EvidenceError(f"{_rel(path)}: missing variants {sorted(missing)}")
    return sorted(out, key=lambda r: -r["mean_test_accuracy"])


def load_param_counts(sql_dir: Path = SQL_DIR) -> dict[str, int]:
    """Query 4: parameter count per variant."""
    path = sql_dir / "q4.csv"
    return {r["run_name"].rsplit("-seed", 1)[0]: int(_num(r, "param_count", path))
            for r in _read_csv(path)}


def load_audit(sql_dir: Path = SQL_DIR) -> dict:
    """Query 1: matched-configuration audit. Passes only if every seed has all
    six runs and exactly one value for every configuration column."""
    path = sql_dir / "q1.csv"
    rows = _read_csv(path)
    config_cols = [c for c in rows[0] if c not in ("train_seed", "n_runs")]
    problems = []
    for r in rows:
        if int(_num(r, "n_runs", path)) != len(EXPECTED_VARIANTS):
            problems.append(f"seed {r['train_seed']}: n_runs = {r['n_runs']}")
        for c in config_cols:
            if int(_num(r, c, path)) != 1:
                problems.append(f"seed {r['train_seed']}: {c} has {r[c]} distinct values")
    if len(rows) != EXPECTED_SEEDS:
        problems.append(f"expected {EXPECTED_SEEDS} seeds, found {len(rows)}")
    return {"rows": rows, "config_columns": config_cols,
            "passed": not problems, "problems": problems}


def load_paired_differences(sql_dir: Path = SQL_DIR) -> list[dict]:
    """Query 6 -> mean paired difference vs full_improved with a 95% t interval,
    computed here from the per-seed deltas (percentage points for accuracy)."""
    path = sql_dir / "q6.csv"
    by_variant: dict[str, dict[int, dict]] = {}
    for r in _read_csv(path):
        seed = int(_num(r, "train_seed", path))
        by_variant.setdefault(r["variant"], {})[seed] = {
            "acc_pp": _num(r, "delta_accuracy_vs_full", path) * 100,
            "f1": _num(r, "delta_macro_f1_vs_full", path),
            "loss": _num(r, "delta_test_loss_vs_full", path),
        }
    out = []
    for variant, technique in TECHNIQUE_REMOVED.items():
        seeds = by_variant.get(variant, {})
        if len(seeds) != EXPECTED_SEEDS:
            raise EvidenceError(
                f"{_rel(path)}: {variant} has {len(seeds)} seeds, expected {EXPECTED_SEEDS}"
            )
        accs = [seeds[s]["acc_pp"] for s in sorted(seeds)]
        n = len(accs)
        mean = sum(accs) / n
        sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / (n - 1))
        half = T_CRIT_DF4 * sd / math.sqrt(n)
        lo, hi = mean - half, mean + half
        out.append({
            "variant": variant,
            "technique_removed": technique,
            "per_seed_acc_pp": accs,
            "mean_acc_pp": mean,
            "ci_low_pp": lo,
            "ci_high_pp": hi,
            "ci_includes_zero": lo <= 0 <= hi,
            "mean_f1": sum(seeds[s]["f1"] for s in seeds) / n,
            "mean_loss": sum(seeds[s]["loss"] for s in seeds) / n,
            "seeds_lower": sum(1 for a in accs if a < 0),
            "n_seeds": n,
        })
    return out


def load_gradcam_examples(gradcam_dir: Path = GRADCAM_DIR) -> list[dict]:
    path = gradcam_dir / "examples.csv"
    out = []
    for r in _read_csv(path):
        if r.get("set") not in ("correct", "incorrect"):
            raise EvidenceError(f"{_rel(path)}: unexpected set value {r.get('set')!r}")
        conf = _num(r, "confidence", path)
        out.append({
            "set": r["set"],
            "test_index": int(_num(r, "test_index", path)),
            "true": r["true"],
            "predicted": r["predicted"],
            "confidence": conf,
            "uncertain": r["set"] == "correct" and conf < UNCERTAIN_BELOW,
        })
    n_c = sum(e["set"] == "correct" for e in out)
    n_i = sum(e["set"] == "incorrect" for e in out)
    if (n_c, n_i) != (N_CORRECT_EXPECTED, N_INCORRECT_EXPECTED):
        raise EvidenceError(
            f"{_rel(path)}: expected {N_CORRECT_EXPECTED} correct and "
            f"{N_INCORRECT_EXPECTED} incorrect examples, found {n_c} and {n_i}"
        )
    return out


def load_gradcam_source(gradcam_dir: Path = GRADCAM_DIR) -> dict:
    import json
    path = gradcam_dir / "source.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise EvidenceError(f"Missing or unreadable evidence file: {_rel(path)}") from None


def load_failure_interpretations(doc: Path = FAILURE_DOC) -> dict:
    """Parse the existing write-up. Returns {'examples': {index: {title, text}},
    'limits': str}. Text is taken verbatim; nothing is generated."""
    try:
        text = doc.read_text(encoding="utf-8")
    except OSError:
        raise EvidenceError(f"Missing evidence file: {_rel(doc)}") from None
    sections = re.split(r"^## ", text, flags=re.M)[1:]
    examples, limits = {}, ""
    for sec in sections:
        head, _, body = sec.partition("\n")
        m = re.match(r"Example (\d+): (.*)", head)
        if m:
            examples[int(m.group(1))] = {"title": m.group(2).strip(), "text": body.strip()}
        elif head.strip().lower().startswith("interpretation limits"):
            limits = body.strip()
    if not examples:
        raise EvidenceError(f"No example interpretations found in {_rel(doc)}")
    return {"examples": examples, "limits": limits}


def missing_figures(figures: dict[str, Path] = FIGURES) -> list[str]:
    return [_rel(p) for p in figures.values() if not p.is_file()]


def load_model_card(path: Path = MODEL_CARD) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise EvidenceError(f"Missing file: {_rel(path)}") from None
    if not text.strip():
        raise EvidenceError(f"{_rel(path)} is empty")
    return text
