"""Data handling for the Recent Classifications tab.

No streamlit import, so it is unit-testable offline. Reads the narrowly scoped
view `public.recent_classifications` (migration 003) through a Supabase client
built with the anon key; it never writes. Failures become RecentError with a
message that is safe to show: no raw exception text, URLs or keys.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

VIEW = "recent_classifications"
COLUMNS = "id,run_id,class_name,confidence,created_at"
LIMIT = 200
CLASSES = ["horizontal", "vertical", "diagonal", "block"]  # matches shared.data.CLASS_NAMES


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
