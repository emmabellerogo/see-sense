"""Data handling for the Recent Classifications tab.

No streamlit import, so it is unit-testable offline. Reads two narrowly scoped views through a
Supabase client built with the anon key; it never writes:
  * `public.recent_predictions` (migration 004): predictions joined to the run that served them,
    with NO image hash; drives the history table and the classification-to-run drill-down;
  * `public.recent_classifications` (migration 003): the older, run-less list, used only as a
    fallback when the predictions view is not available.
The anon key can reach neither the image hashes nor the model checkpoints. Failures become
RecentError with a message that is safe to show: no raw exception text, URLs or keys.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

VIEW = "recent_classifications"                    # migration 003 (legacy, no run join)
PREDICTIONS_VIEW = "recent_predictions"            # migration 004: predictions JOIN runs
PREDICTION_COLUMNS = ("id,created_at,predicted_class,confidence,probabilities,served_by_run_id,"
                      "run_name,experiment_group,model_variant,run_test_accuracy,run_test_macro_f1")
# `runs` is anon-readable (policy in 001) and holds no checkpoint; "*" keeps the drill-down working
# whether or not the newer columns (calibration, dataset config) exist yet.
RUN_COLUMNS = "*"
COLUMNS = "id,run_id,class_name,confidence,created_at"
LIMIT = 200
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]  # matches shared.fashion.CLASS_NAMES


class RecentError(Exception):
    """Safe-to-display failure."""


def fetch_recent(get_client, limit: int = LIMIT) -> list[dict]:
    """Newest-first rows from the view. Raises RecentError on any failure.

    `get_client` is a zero-argument callable returning the anon Supabase client,
    called inside the guarded block so a bad or missing key is handled too.
    """
    try:
        data = (
            get_client().table(VIEW).select(COLUMNS)
            .order("created_at", desc=True).limit(limit)
            .execute().data
        )
    except Exception as exc:  # noqa: BLE001 - not echoed to the user
        # Class name only: messages can contain URLs or keys. Shows in the server log.
        log.warning("recent_classifications query failed: %s.%s",
                    type(exc).__module__, type(exc).__name__)
        raise RecentError(
            "Couldn't load recent classifications right now. The database may be "
            "unreachable or temporarily unavailable. Please try again shortly."
        ) from None
    if not isinstance(data, list):
        raise RecentError("The database returned an unexpected response. Please try again.")
    return data


def fetch_recent_predictions(get_client, limit: int = LIMIT) -> list[dict]:
    """Newest-first rows of the `recent_predictions` view (predictions joined to runs).

    Same safety rules as ``fetch_recent``: anon client, read-only, no raw exception text.
    """
    try:
        data = (
            get_client().table(PREDICTIONS_VIEW).select(PREDICTION_COLUMNS)
            .order("created_at", desc=True).limit(limit)
            .execute().data
        )
    except Exception as exc:  # noqa: BLE001 - not echoed to the user
        log.warning("recent_predictions query failed: %s.%s",
                    type(exc).__module__, type(exc).__name__)
        raise RecentError(
            "Couldn't load recent predictions right now. The database may be unreachable, "
            "or the predictions view has not been created yet."
        ) from None
    if not isinstance(data, list):
        raise RecentError("The database returned an unexpected response. Please try again.")
    return data


def fetch_run(get_client, run_id: int) -> dict | None:
    """The run that produced a prediction, read from `runs` with the anon key (drill-down)."""
    try:
        data = (
            get_client().table("runs").select(RUN_COLUMNS)
            .eq("id", int(run_id)).limit(1).execute().data
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("runs lookup failed: %s.%s", type(exc).__module__, type(exc).__name__)
        raise RecentError("Couldn't load that run right now. Please try again shortly.") from None
    if not isinstance(data, list):
        raise RecentError("The database returned an unexpected response. Please try again.")
    return data[0] if data else None


def clean_prediction_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Like ``clean_rows`` for the joined view; also keeps the probability vector and run info.

    A row needs a class name, a confidence in [0, 1], an integer run id and a probability
    list of numbers. Run fields may be missing (shown as empty); the rest of the row is kept.
    """
    good, skipped = [], 0
    for r in rows:
        try:
            conf = float(r["confidence"])
            run_id = int(r["served_by_run_id"])
            name = r["predicted_class"]
            probs = [float(x) for x in r["probabilities"]]
            if not isinstance(name, str) or not name or not 0.0 <= conf <= 1.0 or not probs:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        acc = r.get("run_test_accuracy")
        good.append({
            "Time": _fmt_time(r.get("created_at")),
            "Class": name,
            "Confidence": conf,
            "Run ID": run_id,
            "Run": r.get("run_name") or "",
            "Variant": r.get("model_variant") or "",
            "Run test accuracy": float(acc) if isinstance(acc, (int, float)) else None,
            "Probabilities": probs,
            "_sort": r.get("created_at") or "",
        })
    good.sort(key=lambda g: g["_sort"], reverse=True)
    for g in good:
        del g["_sort"]
    return good, skipped


def probability_table(probs: list[float], class_names: list[str] | None = None) -> list[dict]:
    """One row per class, in model order, for display. Never truncated to a top-k."""
    names = class_names or CLASSES
    return [{"Class": names[i] if i < len(names) else f"class {i}", "Probability": p}
            for i, p in enumerate(probs)]


def top_k(probs: list[float], k: int = 5, class_names: list[str] | None = None) -> list[dict]:
    """The k most probable classes of a logged probability vector, best first."""
    rows = probability_table(probs, class_names)
    return sorted(rows, key=lambda r: -r["Probability"])[:k]


def run_summary(run: dict) -> list[dict]:
    """Facts about the producing run for the drill-down, from the anon-readable `runs` row.

    Only fields present on the row are shown, so older runs simply have fewer lines. Nothing is
    inferred; the calibration line appears only if a measured temperature was saved.
    """
    cfg = run.get("config") or {}
    dataset = cfg.get("dataset") or "synthetic shapes (historical)"
    rows = [
        ("Run", f"{run.get('run_name') or '-'} (id {run.get('id')})"),
        ("Experiment group", run.get("experiment_group") or "-"),
        ("Dataset", dataset),
        ("Model", f"{run.get('model_variant') or '-'}, {run.get('param_count') or '?'} parameters"),
        ("Techniques", ", ".join(t for t, on in (
            ("batch norm", run.get("use_batchnorm")), ("dropout", (run.get("dropout") or 0) > 0),
            ("augmentation", run.get("augment")), ("LR scheduler", run.get("scheduler") not in (None, "none"))) if on) or "none"),
        ("Training", f"{run.get('epochs')} epochs, batch {run.get('batch_size')}, lr {run.get('lr')}, seed {run.get('train_seed')}"),
        ("Held-out test accuracy / macro-F1",
         f"{run['accuracy']:.4f} / {run['macro_f1']:.4f}" if isinstance(run.get("accuracy"), (int, float))
         and isinstance(run.get("macro_f1"), (int, float)) else "-"),
    ]
    calib = run.get("calibration")
    if isinstance(calib, dict) and calib.get("temperature"):
        rows.append(("Temperature (fitted on validation)", f"{calib['temperature']:.3f}"))
    return [{"Field": k, "Value": str(v)} for k, v in rows]


def _fmt_time(value) -> str:
    if not isinstance(value, str):
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return value[:19]


def clean_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep well-formed rows, newest first. Returns (rows, number_skipped).

    A row needs a string class_name, a numeric confidence in [0, 1] and an
    integer run_id; anything else is skipped rather than shown or guessed at.
    """
    good, skipped = [], 0
    for r in rows:
        try:
            conf = float(r["confidence"])
            run_id = int(r["run_id"])
            name = r["class_name"]
            if not isinstance(name, str) or not name or not 0.0 <= conf <= 1.0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        good.append({
            "Time": _fmt_time(r.get("created_at")),
            "Class": name,
            "Confidence": conf,
            "Run ID": run_id,
            "_sort": r.get("created_at") or "",
        })
    good.sort(key=lambda g: g["_sort"], reverse=True)
    for g in good:
        del g["_sort"]
    return good, skipped


def class_counts(rows: list[dict]) -> dict[str, int]:
    """Counts for every known class (zero when absent), then any unexpected
    class names so nothing is silently dropped."""
    counts = {c: 0 for c in CLASSES}
    for r in rows:
        counts[r["Class"]] = counts.get(r["Class"], 0) + 1
    return counts
