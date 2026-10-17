"""experiments.gradcam_evidence: the Supabase loading path (no network, fake client)."""
from __future__ import annotations

import pytest

from experiments.gradcam_evidence import load_from_supabase


class FakeQuery:
    def __init__(self, tables, name):
        self.rows, self.filters, self.table_name = tables[name], [], name

    def select(self, *_):
        return self

    def eq(self, col, val):
        self.rows = [r for r in self.rows if r.get(col) == val]
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    def execute(self):
        class R: pass
        r = R(); r.data = self.rows
        return r


class FakeClient:
    def __init__(self, tables):
        self.tables, self.written = tables, False

    def table(self, name):
        return FakeQuery(self.tables, name)

    def __getattr__(self, name):  # any write method would be a bug
        raise AssertionError(name)


def tables(n_runs=1):
    return {
        "runs": [{"id": 7, "dataset_id": 3, "experiment_group": "g", "run_name": "r",
                  "accuracy": 0.9, "config": {}}] * n_runs,
        "datasets": [{"id": 3, "n_rows": 10, "img_size": 28, "noise": 0.1, "seed": 1}],
        "run_artifacts": [{"run_id": 7, "model_b64": "CKPT"}],
    }


def test_loads_run_dataset_and_checkpoint(monkeypatch):
    monkeypatch.delenv("SEE_SENSE_LOCAL_DB", raising=False)
    run, ds, ckpt = load_from_supabase("r", "g", client=FakeClient(tables()))
    assert (run["id"], ds["id"], ckpt) == (7, 3, "CKPT")


def test_refuses_offline_storage(monkeypatch):
    monkeypatch.setenv("SEE_SENSE_LOCAL_DB", "x.json")
    with pytest.raises(SystemExit, match="offline storage"):
        load_from_supabase("r", "g", client=FakeClient(tables()))


@pytest.mark.parametrize("n", [0, 2])
def test_requires_exactly_one_run(monkeypatch, n):
    monkeypatch.delenv("SEE_SENSE_LOCAL_DB", raising=False)
    with pytest.raises(LookupError):
        load_from_supabase("r", "g", client=FakeClient(tables(n)))


def test_refuses_without_credentials(monkeypatch):
    monkeypatch.delenv("SEE_SENSE_LOCAL_DB", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="not set"):
        load_from_supabase("r", "g")
