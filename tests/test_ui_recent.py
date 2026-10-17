"""Data handling, histogram counts and failure states for Recent Classifications."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ui_recent", Path(__file__).parent.parent / "ui" / "recent.py"
)
r = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(r)


def row(i, name="block", conf=0.9, run=4, ts="2026-10-17T06:38:40.448414+00:00"):
    return {"id": i, "run_id": run, "class_name": name, "confidence": conf, "created_at": ts}


class FakeQuery:
    def __init__(self, data=None, exc=None):
        self.data, self.exc, self.calls = data, exc, []

    def table(self, name):
        self.calls.append(("table", name)); return self

    def select(self, cols):
        self.calls.append(("select", cols)); return self

    def order(self, col, desc=False):
        self.calls.append(("order", col, desc)); return self

    def limit(self, n):
        self.calls.append(("limit", n)); return self

    def execute(self):
        if self.exc:
            raise self.exc
        return type("R", (), {"data": self.data})()


def test_fetch_queries_only_the_view_with_expected_columns():
    q = FakeQuery([row(1)])
    assert r.fetch_recent(lambda: q) == [row(1)]
    assert ("table", "recent_classifications") in q.calls
    assert ("select", "id,run_id,class_name,confidence,created_at") in q.calls
    assert ("order", "created_at", True) in q.calls and ("limit", 200) in q.calls
    assert "sha256" not in str(q.calls)


@pytest.mark.parametrize("exc", [
    RuntimeError("Invalid API key sb_secret_abc123"),
    ConnectionError("https://x.supabase.co timed out"),
])
def test_failures_become_safe_messages(exc):
    with pytest.raises(r.RecentError) as ei:
        r.fetch_recent(lambda: FakeQuery(exc=exc))
    msg = str(ei.value)
    assert "abc123" not in msg and "supabase.co" not in msg and "Invalid API key" not in msg


def test_client_construction_failure_is_also_safe():
    def bad_client():
        raise RuntimeError("Invalid API key")

    with pytest.raises(r.RecentError) as ei:
        r.fetch_recent(bad_client)
    assert "Invalid API key" not in str(ei.value)


def test_non_list_response_is_an_error():
    with pytest.raises(r.RecentError, match="unexpected"):
        r.fetch_recent(lambda: FakeQuery(data={"message": "oops"}))


def test_clean_rows_formats_orders_and_skips_malformed():
    rows, skipped = r.clean_rows([
        row(1, ts="2026-10-17T06:00:00+00:00"),
        row(2, "vertical", 0.5, ts="2026-10-17T07:00:00+00:00"),
        {"id": 3, "run_id": 4, "class_name": "block"},                 # no confidence
        row(4, conf=1.7),                                              # out of range
        row(5, name=None),
        row(6, conf="abc"),
    ])
    assert skipped == 4
    assert [x["Class"] for x in rows] == ["vertical", "block"]         # newest first
    assert rows[0] == {"Time": "2026-10-17 07:00:00 UTC", "Class": "vertical",
                       "Confidence": 0.5, "Run ID": 4}
    assert "id" not in rows[0] and "sha256" not in rows[0]


def test_unparseable_timestamp_does_not_crash():
    rows, skipped = r.clean_rows([row(1, ts="garbage-time")])
    assert skipped == 0 and rows[0]["Time"] == "garbage-time"


def test_counts_include_zero_for_absent_classes():
    rows, _ = r.clean_rows([row(1, "block"), row(2, "block"), row(3, "vertical")])
    assert r.class_counts(rows) == {"horizontal": 0, "vertical": 1, "diagonal": 0, "block": 2}


def test_counts_for_empty_data_are_all_zero():
    assert r.class_counts([]) == {c: 0 for c in r.CLASSES}


def test_unexpected_class_is_counted_not_dropped():
    rows, _ = r.clean_rows([row(1, "triangle")])
    assert r.class_counts(rows)["triangle"] == 1


def test_known_classes_match_the_model():
    from shared.data import CLASS_NAMES
    assert r.CLASSES == list(CLASS_NAMES)


def test_failure_is_logged_by_type_only(caplog):
    with caplog.at_level("WARNING"):
        with pytest.raises(r.RecentError):
            r.fetch_recent(lambda: FakeQuery(exc=ConnectionError("key=sb-secret-xyz https://h.supabase.co")))
    assert "ConnectionError" in caplog.text
    assert "sb-secret-xyz" not in caplog.text and "supabase.co" not in caplog.text
