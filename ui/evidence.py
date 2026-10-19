"""Loaders for saved experiment evidence: the FINAL Fashion-MNIST set and the HISTORICAL synthetic set.

Two evidence sets exist and are never mixed:
  * ``FASHION``    - the final 10-class Fashion-MNIST comparison (files appear only after the
                     official runs have been completed; until then it reports "not available").
  * ``HISTORICAL`` - the earlier 4-class synthetic-shape experiment (group official-supabase-1),
                     kept for transparency and clearly labelled where it is shown.

Pure standard library (no streamlit, pandas or torch) so it can be unit-tested
offline. Every path is resolved from this file's location, never from the
current working directory. Loaders raise EvidenceError with a short readable
message when a file is missing or malformed; they never invent values.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import NamedTuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "experiments" / "results"
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"
PROTOCOL_CONFIG = REPO_ROOT / "config" / "fashion_protocol.json"

FASHION_CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
LEGACY_SYNTHETIC_CLASSES = ("horizontal", "vertical", "diagonal", "block")


def _fashion_group(path: Path = PROTOCOL_CONFIG) -> str:
    try:
        return json.loads(path.read_text(encoding="utf-8"))["experiment_group"]
    except (OSError, ValueError, KeyError):
        return "fashion-official-1"


class EvidenceSet(NamedTuple):
    """Where one experiment's saved evidence lives and what shape it must have."""

    key: str
    title: str
    group: str
    classes: tuple
    failure_doc: Path
    expected_seeds: Optional[int]          # None = whatever the protocol used (>= 1)
    gradcam_counts: Optional[tuple]        # exact (n_correct, n_incorrect), or None = at least 5 each

    @property
    def sql_dir(self) -> Path:
        return RESULTS / f"{self.group}-sql"

    @property
    def analysis_dir(self) -> Path:
        return RESULTS / f"{self.group}-analysis"

    @property
    def gradcam_dir(self) -> Path:
        return RESULTS / f"{self.group}-gradcam"

    @property
    def gradcam_baseline_dir(self) -> Path:
        return RESULTS / f"{self.group}-gradcam-baseline"

    def figure(self, name: str) -> Path:
        folder = self.gradcam_dir if name.startswith("gradcam") else self.analysis_dir
        return folder / f"{name}.png"

    def available(self) -> bool:
        return (self.sql_dir / "q5.csv").is_file()


HISTORICAL = EvidenceSet(
    key="historical-synthetic", title="Historical: 4-class synthetic shapes (NOT Fashion-MNIST)",
    group="official-supabase-1", classes=LEGACY_SYNTHETIC_CLASSES,
    failure_doc=REPO_ROOT / "docs" / "SUPABASE_GRADCAM_FAILURE_ANALYSIS.md",
    expected_seeds=5, gradcam_counts=(5, 8),
)
FASHION = EvidenceSet(
    key="fashion-mnist", title="Final: Fashion-MNIST, 10 classes", group=_fashion_group(),
    classes=FASHION_CLASSES, failure_doc=REPO_ROOT / "docs" / "FASHION_GRADCAM_FAILURE_ANALYSIS.md",
    expected_seeds=None, gradcam_counts=None,
)

# Backwards-compatible names for the historical set.
GROUP = HISTORICAL.group
SQL_DIR, ANALYSIS_DIR, GRADCAM_DIR = HISTORICAL.sql_dir, HISTORICAL.analysis_dir, HISTORICAL.gradcam_dir
FAILURE_DOC = HISTORICAL.failure_doc

# Two-sided 95% t critical values by degrees of freedom (number of paired seeds - 1).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776445105, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

EXPECTED_SEEDS = 5
EXPECTED_VARIANTS = (
    "baseline", "full_improved", "no_augmentation",
    "no_batchnorm", "no_dropout", "no_scheduler",
)
T_CRIT_DF4 = T95[4]
N_CORRECT_EXPECTED = 5
N_INCORRECT_EXPECTED = 8
UNCERTAIN_BELOW = 0.5  # a correct prediction under this confidence is flagged

FIGURE_NAMES = ("loss_curves", "accuracy_curves", "confusion_matrices", "test_metrics_by_seed",
                "per_class_precision_recall", "gradcam_correct", "gradcam_incorrect")
FIGURES = {n: HISTORICAL.figure(n) for n in FIGURE_NAMES}  # historical set (kept for compatibility)

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


def load_audit(sql_dir: Path = SQL_DIR, expected_seeds: Optional[int] = EXPECTED_SEEDS) -> dict:
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
    if expected_seeds is not None and len(rows) != expected_seeds:
        problems.append(f"expected {expected_seeds} seeds, found {len(rows)}")
    return {"rows": rows, "config_columns": config_cols,
            "passed": not problems, "problems": problems}


def load_paired_differences(sql_dir: Path = SQL_DIR, expected_seeds: Optional[int] = EXPECTED_SEEDS) -> list[dict]:
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
        if len(seeds) < 2 or (expected_seeds is not None and len(seeds) != expected_seeds):
            raise EvidenceError(
                f"{_rel(path)}: {variant} has {len(seeds)} seeds, expected "
                f"{expected_seeds if expected_seeds is not None else 'at least 2'}"
            )
        accs = [seeds[s]["acc_pp"] for s in sorted(seeds)]
        n = len(accs)
        mean = sum(accs) / n
        sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / (n - 1))
        half = T95.get(n - 1, 1.96) * sd / math.sqrt(n)
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


CLASS_ORDER = LEGACY_SYNTHETIC_CLASSES


def load_per_class_metrics(analysis_dir: Path = ANALYSIS_DIR, class_order: tuple = CLASS_ORDER) -> list[dict]:
    """Per-class test precision / recall per variant (mean and sd over seeds).

    ``per_class_metrics.csv`` is written by ``experiments.analyze`` from the saved query-7
    confusion counts; nothing is estimated here. Rows follow variant then class order.
    """
    path = analysis_dir / "per_class_metrics.csv"
    out = []
    for r in _read_csv(path):
        out.append({
            "variant": r.get("variant", ""),
            "class": r.get("class", ""),
            "precision": _num(r, "mean_precision", path),
            "precision_sd": _num(r, "sd_precision", path),
            "recall": _num(r, "mean_recall", path),
            "recall_sd": _num(r, "sd_recall", path),
            "n_runs": int(_num(r, "n_runs", path)),
        })
    missing = {(v, c) for v in EXPECTED_VARIANTS for c in class_order} - {(r["variant"], r["class"]) for r in out}
    if missing:
        raise EvidenceError(f"{_rel(path)}: missing variant/class rows {sorted(missing)[:3]}...")
    order = {v: i for i, v in enumerate(EXPECTED_VARIANTS)}
    return sorted(out, key=lambda r: (order[r["variant"]], class_order.index(r["class"])))


def load_gradcam_examples(gradcam_dir: Path = GRADCAM_DIR,
                          counts: Optional[tuple] = (N_CORRECT_EXPECTED, N_INCORRECT_EXPECTED)) -> list[dict]:
    """Rows of examples.csv. ``counts`` is an exact (correct, incorrect) pair, or None to
    require at least five of each."""
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
    if counts is not None and (n_c, n_i) != tuple(counts):
        raise EvidenceError(
            f"{_rel(path)}: expected {counts[0]} correct and {counts[1]} incorrect examples, "
            f"found {n_c} and {n_i}"
        )
    if counts is None and (n_c < 5 or n_i < 5):
        raise EvidenceError(f"{_rel(path)}: need at least 5 correct and 5 incorrect examples, found {n_c} and {n_i}")
    return out


def load_gradcam_source(gradcam_dir: Path = GRADCAM_DIR) -> dict:
    import json
    path = gradcam_dir / "source.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise EvidenceError(f"Missing or unreadable evidence file: {_rel(path)}") from None


BASELINE_RUN_NAME = "baseline-seed0"
BASELINE_COMPARISON_DOC = REPO_ROOT / "docs" / "FASHION_GRADCAM_BASELINE_COMPARISON.md"
SHARED_REQUIRED = ("set", "test_index", "true", "baseline_pred", "baseline_correct", "baseline_conf",
                   "baseline_blank_map", "improved_pred", "improved_correct", "improved_conf", "improved_blank_map")


def _truthy(value: str, path: Path, col: str) -> bool:
    if value not in ("True", "False"):
        raise EvidenceError(f"{_rel(path)}: column {col} must be True or False, found {value!r}")
    return value == "True"


def load_baseline_gradcam(ev: "EvidenceSet") -> dict:
    """The baseline model's own Grad-CAM evidence, the shared-image comparison and blank-map counts.

    Refuses evidence that is not the baseline run of the same group, dataset, split and labels as the
    improved evidence, whose regenerated accuracy differs from the stored one, or that has fewer than
    five correct or five incorrect examples. Returns {'source', 'examples', 'blank', 'shared', 'identity'}.
    """
    import json
    folder = ev.gradcam_baseline_dir
    if not folder.is_dir():
        raise EvidenceError(f"Missing evidence folder: {_rel(folder)}")
    source = load_gradcam_source(folder)
    improved = load_gradcam_source(ev.gradcam_dir)
    if source.get("run_name") != BASELINE_RUN_NAME:
        raise EvidenceError(f"{_rel(folder)}/source.json: expected run {BASELINE_RUN_NAME}, found {source.get('run_name')!r}")
    for key in ("experiment_group", "dataset_id", "split_sha256", "class_names", "n_test"):
        if source.get(key) != improved.get(key):
            raise EvidenceError(f"Baseline and improved evidence differ in {key}: not a fair comparison")
    if source.get("experiment_group") != ev.group:
        raise EvidenceError("Baseline evidence is not from the official group")
    if "stored_accuracy" not in source or abs(source["regenerated_accuracy"] - source["stored_accuracy"]) > 1e-9:
        raise EvidenceError("Baseline regenerated accuracy does not equal the stored accuracy")
    examples = load_gradcam_examples(folder, counts=None)
    blank = {}
    path = folder / "blank_cam_by_class.csv"
    for r in _read_csv(path):
        blank[r["predicted_class"]] = {"n": int(_num(r, "n_predictions", path)),
                                       "blank": int(_num(r, "n_blank_cam", path))}
    if set(blank) != set(ev.classes):
        raise EvidenceError(f"{_rel(path)}: expected one row per class")
    spath = folder / "shared_examples.csv"
    shared = []
    for r in _read_csv(spath):
        missing = [c for c in SHARED_REQUIRED if c not in r]
        if missing:
            raise EvidenceError(f"{_rel(spath)}: missing columns {missing}")
        shared.append({
            "set": r["set"].replace("improved_", "").replace("_set", ""),
            "test_index": int(_num(r, "test_index", spath)), "true": r["true"],
            "baseline_pred": r["baseline_pred"], "baseline_correct": _truthy(r["baseline_correct"], spath, "baseline_correct"),
            "baseline_conf": _num(r, "baseline_conf", spath), "baseline_blank": _truthy(r["baseline_blank_map"], spath, "baseline_blank_map"),
            "improved_pred": r["improved_pred"], "improved_correct": _truthy(r["improved_correct"], spath, "improved_correct"),
            "improved_conf": _num(r, "improved_conf", spath), "improved_blank": _truthy(r["improved_blank_map"], spath, "improved_blank_map"),
        })
    if len(shared) < 10:
        raise EvidenceError(f"{_rel(spath)}: too few shared examples ({len(shared)})")
    imp_idx = {e["test_index"] for e in load_gradcam_examples(ev.gradcam_dir, counts=None)}
    if {s["test_index"] for s in shared} != imp_idx:
        raise EvidenceError(f"{_rel(spath)}: shared images are not the improved model's gallery images")
    try:
        identity = json.loads((folder / "identity_check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise EvidenceError(f"Missing or unreadable evidence file: {_rel(folder / 'identity_check.json')}") from None
    return {"source": source, "examples": examples, "blank": blank, "shared": shared, "identity": identity}


def load_text_doc(doc: Path) -> str:
    try:
        return doc.read_text(encoding="utf-8")
    except OSError:
        raise EvidenceError(f"Missing evidence file: {_rel(doc)}") from None


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


def figures_for(ev: EvidenceSet) -> dict[str, Path]:
    return {n: ev.figure(n) for n in FIGURE_NAMES}


def load_calibration_summary(analysis_dir: Path) -> list[dict]:
    """Per-variant mean temperature and test ECE / NLL before and after, from calibration.csv.

    Raises EvidenceError if calibration was not measured (file missing). Values are averages of
    the saved per-run measurements; nothing is estimated here.
    """
    path = analysis_dir / "calibration.csv"
    by: dict[str, dict[str, list[float]]] = {}
    keys = ("temperature", "test_ece_before", "test_ece_after", "test_nll_before", "test_nll_after")
    for r in _read_csv(path):
        d = by.setdefault(r["run_name"].rsplit("-seed", 1)[0], {k: [] for k in keys})
        for k in keys:
            d[k].append(_num(r, k, path))
    out = []
    for v in EXPECTED_VARIANTS:
        if v in by:
            out.append({"variant": v, "n_runs": len(by[v]["temperature"]),
                        **{k: sum(by[v][k]) / len(by[v][k]) for k in keys}})
    if not out:
        raise EvidenceError(f"{_rel(path)}: no calibration rows")
    return out


def load_provenance(sql_dir: Path) -> dict:
    """What the saved q*.csv files say about where they came from (their PROVENANCE.md).

    Returns {'source': str, 'is_supabase_sql': bool}. The files are only treated as a Supabase SQL
    export when PROVENANCE.md says so (written by experiments/ingest_sql_export.py from a result that
    contains the database's own ``meta`` row); anything else is reported as it describes itself.
    """
    try:
        text = (sql_dir / "PROVENANCE.md").read_text(encoding="utf-8")
    except OSError:
        return {"source": "not recorded", "is_supabase_sql": False}
    m = re.search(r"^- source: \*\*(.+?)\*\*", text, flags=re.M)
    source = m.group(1).strip() if m else "not recorded"
    return {"source": source, "is_supabase_sql": source.lower().startswith("supabase")}


def load_model_card(path: Path = MODEL_CARD) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise EvidenceError(f"Missing file: {_rel(path)}") from None
    if not text.strip():
        raise EvidenceError(f"{_rel(path)} is empty")
    return text
