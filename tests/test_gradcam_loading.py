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


# ---- evidence generation on the tiny fixture (no real data, no cloud) --------------------------------------
import csv  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from api.training import run_experiment  # noqa: E402
from experiments import gradcam_evidence as ge  # noqa: E402
from tests.fashion_fixture import tiny_fashion  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    data = tiny_fashion(n_train_per_class=40, n_test_per_class=30)
    r = run_experiment(split_seed=0, train_seed=0, lr=0.003, batch_size=32, epochs=1, arch="baseline", data=data)
    return data, r


def test_selection_rule_picks_one_correct_per_class_and_only_genuine_errors():
    k = 6
    y = np.tile(np.arange(k), 4)                    # 24 images, classes 0..5 repeated
    pred = y.copy()
    wrong = {1: 2, 3: 4, 8: 0, 10: 1, 14: 5, 16: 3, 19: 2, 21: 0}  # index -> wrong prediction
    for i, p in wrong.items():
        pred[i] = (y[i] + 1) % k if p == y[i] else p
    conf = np.linspace(0.5, 0.95, len(y))
    picks = ge.select_examples(y, pred, conf, k, max_incorrect=10)
    assert [int(y[i]) for i in picks["correct"]] == list(range(k))   # first correct example of each class
    assert all(pred[i] == y[i] for i in picks["correct"])
    assert all(pred[i] != y[i] for i in picks["incorrect"])          # "incorrect" is always predicted != true
    assert len(set(picks["incorrect"])) == len(picks["incorrect"]) == len(wrong)
    pairs = [(int(y[i]), int(pred[i])) for i in picks["incorrect"]]
    n_distinct = len(set(pairs))
    assert len(set(pairs[:n_distinct])) == n_distinct                # distinct (true -> predicted) pairs come first
    assert picks["incorrect"][0] == max(wrong, key=lambda i: conf[i])  # most confident error first


def test_too_few_examples_is_an_error_not_a_thin_gallery():
    y = np.array([0, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="at least 5"):
        ge.select_examples(y, y.copy(), np.ones(6), 10)


def test_build_evidence_makes_figures_csv_and_source_and_checks_the_stored_accuracy(tiny_model, tmp_path):
    data, r = tiny_model
    run = {"id": 3, "dataset_id": 1, "experiment_group": "g", "run_name": "full_improved-seed0",
           "accuracy": r["metrics"]["accuracy"], "split_sha256": r["split_sha256"]}
    src = ge.build_evidence(r["model_b64"], data, tmp_path, run, "full_improved-seed0")
    assert (tmp_path / "gradcam_correct.png").is_file() and (tmp_path / "gradcam_incorrect.png").is_file()
    rows = list(csv.DictReader(open(tmp_path / "examples.csv")))
    correct = [x for x in rows if x["set"] == "correct"]
    wrong = [x for x in rows if x["set"] == "incorrect"]
    assert len(wrong) >= 5 and all(x["true"] != x["predicted"] for x in wrong)
    assert all(x["true"] == x["predicted"] for x in correct)
    assert {"p_T-shirt/top", "p_Ankle boot", "cam_mass_on_garment"} <= set(rows[0])
    assert all(0.0 <= float(x["cam_mass_on_garment"]) <= 1.0 for x in rows)
    saved = json.loads((tmp_path / "source.json").read_text())
    assert saved["run_id"] == 3 and saved["regenerated_accuracy"] == pytest.approx(run["accuracy"])
    assert saved["n_correct"] + saved["n_incorrect"] == saved["n_test"] == 300
    assert saved["selection_rule"] and saved["class_names"][0] == "T-shirt/top"
    # The figures are real images with several rows.
    assert Image.open(tmp_path / "gradcam_incorrect.png").size[1] > 500


def test_evidence_refuses_a_checkpoint_that_does_not_reproduce_the_stored_accuracy(tiny_model, tmp_path):
    data, r = tiny_model
    run = {"id": 3, "accuracy": min(1.0, r["metrics"]["accuracy"] + 0.2), "dataset_id": 1}
    with pytest.raises(AssertionError, match="not the same model"):
        ge.build_evidence(r["model_b64"], data, tmp_path, run, "x")


def test_cam_mass_on_garment_is_a_share_between_zero_and_one():
    img = np.zeros((28, 28), dtype=np.uint8)
    img[10:20, 10:20] = 200
    inside = np.zeros((7, 7)); inside[3, 3] = 1.0   # the 4x4-pixel cell around (12..16, 12..16): on the garment
    outside = np.zeros((7, 7)); outside[0, 0] = 1.0  # a corner cell
    assert ge.cam_mass_on_garment(img, inside) == pytest.approx(1.0)
    assert ge.cam_mass_on_garment(img, outside) == pytest.approx(0.0)
    assert ge.cam_mass_on_garment(img, np.zeros((7, 7))) == 0.0
