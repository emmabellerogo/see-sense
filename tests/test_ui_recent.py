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


def row(i, name="Dress", conf=0.9, run=4, ts="2026-10-17T06:38:40.448414+00:00"):
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

    def eq(self, col, val):
        self.calls.append(("eq", col, val)); return self

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
        row(2, "Trouser", 0.5, ts="2026-10-17T07:00:00+00:00"),
        {"id": 3, "run_id": 4, "class_name": "Dress"},                 # no confidence
        row(4, conf=1.7),                                              # out of range
        row(5, name=None),
        row(6, conf="abc"),
    ])
    assert skipped == 4
    assert [x["Class"] for x in rows] == ["Trouser", "Dress"]          # newest first
    assert rows[0] == {"Time": "2026-10-17 07:00:00 UTC", "Class": "Trouser",
                       "Confidence": 0.5, "Run ID": 4}
    assert "id" not in rows[0] and "sha256" not in rows[0]


def test_unparseable_timestamp_does_not_crash():
    rows, skipped = r.clean_rows([row(1, ts="garbage-time")])
    assert skipped == 0 and rows[0]["Time"] == "garbage-time"


def test_counts_include_zero_for_absent_classes():
    rows, _ = r.clean_rows([row(1, "Dress"), row(2, "Dress"), row(3, "Trouser")])
    counts = r.class_counts(rows)
    assert counts["Dress"] == 2 and counts["Trouser"] == 1 and counts["Coat"] == 0
    assert list(counts) == r.CLASSES and len(counts) == 10


def test_counts_for_empty_data_are_all_zero():
    assert r.class_counts([]) == {c: 0 for c in r.CLASSES}


def test_unexpected_class_is_counted_not_dropped():
    rows, _ = r.clean_rows([row(1, "horizontal")])  # e.g. a row left over from the historical shapes model
    assert r.class_counts(rows)["horizontal"] == 1


def test_known_classes_match_the_model():
    from shared.fashion import CLASS_NAMES
    assert r.CLASSES == list(CLASS_NAMES)


def test_failure_is_logged_by_type_only(caplog):
    with caplog.at_level("WARNING"):
        with pytest.raises(r.RecentError):
            r.fetch_recent(lambda: FakeQuery(exc=ConnectionError("key=sb-secret-xyz https://h.supabase.co")))
    assert "ConnectionError" in caplog.text
    assert "sb-secret-xyz" not in caplog.text and "supabase.co" not in caplog.text


# ---- predictions view + drill-down -------------------------------------------------------------
PROBS = [0.01, 0.02, 0.03, 0.55, 0.05, 0.04, 0.2, 0.06, 0.02, 0.02]


def prow(i, name="Dress", conf=0.55, run=4, ts="2026-10-17T06:38:40+00:00", **over):
    return {"id": i, "created_at": ts, "predicted_class": name, "confidence": conf, "probabilities": PROBS,
            "served_by_run_id": run, "run_name": "full_improved-seed0", "experiment_group": "g",
            "model_variant": "improved", "run_test_accuracy": 0.9, "run_test_macro_f1": 0.89, **over}


def test_fetch_recent_predictions_queries_the_hash_free_view():
    q = FakeQuery([prow(1)])
    assert r.fetch_recent_predictions(lambda: q) == [prow(1)]
    assert ("table", "recent_predictions") in q.calls
    selected = next(c[1] for c in q.calls if c[0] == "select")
    assert "sha256" not in selected and "served_by_run_id" in selected and "probabilities" in selected


def test_prediction_fetch_failure_is_safe():
    with pytest.raises(r.RecentError) as ei:
        r.fetch_recent_predictions(lambda: FakeQuery(exc=RuntimeError("key sb_secret_zzz")))
    assert "zzz" not in str(ei.value)


def test_clean_prediction_rows_keep_run_linkage_and_skip_malformed():
    rows, skipped = r.clean_prediction_rows([
        prow(1, ts="2026-10-17T06:00:00+00:00"), prow(2, "Coat", 0.7, run=5, ts="2026-10-17T07:00:00+00:00"),
        prow(3, conf=2.0), {"id": 4}, prow(5, probabilities=[]),
    ])
    assert skipped == 3 and [x["Class"] for x in rows] == ["Coat", "Dress"]
    top = rows[0]
    assert top["Run ID"] == 5 and top["Run"] == "full_improved-seed0" and top["Run test accuracy"] == 0.9
    assert len(top["Probabilities"]) == 10


def test_top_k_is_sorted_and_named_in_class_order():
    t = r.top_k(PROBS, 5)
    assert [x["Class"] for x in t] == ["Dress", "Shirt", "Sneaker", "Coat", "Sandal"]
    assert [x["Probability"] for x in t] == sorted([x["Probability"] for x in t], reverse=True)
    assert len(r.probability_table(PROBS)) == 10


def test_fetch_run_uses_the_anon_runs_table_and_returns_none_when_gone():
    q = FakeQuery([{"id": 4, "run_name": "x"}])
    assert r.fetch_run(lambda: q, 4)["run_name"] == "x"
    assert ("table", "runs") in q.calls
    assert r.fetch_run(lambda: FakeQuery([]), 4) is None
    with pytest.raises(r.RecentError):
        r.fetch_run(lambda: FakeQuery(exc=RuntimeError("boom")), 4)


def test_run_summary_shows_only_what_the_run_row_has():
    full = r.run_summary({"id": 4, "run_name": "full_improved-seed0", "experiment_group": "g", "model_variant": "improved",
                          "use_batchnorm": True, "dropout": 0.3, "augment": True, "scheduler": "cosine", "epochs": 20,
                          "batch_size": 128, "lr": 0.001, "train_seed": 0, "param_count": 9146, "accuracy": 0.9,
                          "macro_f1": 0.89, "config": {"dataset": "fashion-mnist"},
                          "calibration": {"temperature": 1.234}})
    text = {x["Field"]: x["Value"] for x in full}
    assert text["Dataset"] == "fashion-mnist" and "batch norm, dropout, augmentation, LR scheduler" in text["Techniques"]
    assert text["Temperature (fitted on validation)"] == "1.234"
    old = {x["Field"]: x["Value"] for x in r.run_summary({"id": 1, "accuracy": 0.9, "macro_f1": 0.8})}
    assert "synthetic" in old["Dataset"] and "Temperature (fitted on validation)" not in old
