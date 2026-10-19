"""Baseline-vs-improved Grad-CAM evidence: loading, source identity, example counts and the Compare Models display.

Real-file tests read the committed evidence (``fashion-official-1-gradcam-baseline`` and the improved folder). Rejection
tests copy those folders into a temporary results directory and break one thing at a time. Nothing here touches Supabase;
the evidence itself was generated read-only from the saved checkpoints (see experiments/gradcam_compare.py).
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import gradcam_compare as gc
from ui import evidence as ev

FASHION = ev.FASHION
BASE = FASHION.gradcam_baseline_dir
IMPR = FASHION.gradcam_dir
RECORDS = ev.RESULTS / f"{FASHION.group}-records" / "runs.json"


def _stored_accuracy(run_name: str) -> float:
    runs = json.loads(RECORDS.read_text())["runs"]
    return next(r["accuracy"] for r in runs if r["run_name"] == run_name)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---- the committed evidence -------------------------------------------------------------------------------------
def test_baseline_evidence_loads_and_is_the_official_baseline_run():
    d = ev.load_baseline_gradcam(FASHION)
    s = d["source"]
    assert (s["run_name"], s["run_id"], s["experiment_group"], s["dataset_id"]) == \
        ("baseline-seed0", 53, "fashion-official-1", 24)
    imp = ev.load_gradcam_source(IMPR)
    assert s["split_sha256"] == imp["split_sha256"] and s["class_names"] == imp["class_names"] == list(FASHION.classes)
    assert imp["run_id"] == 54  # the improved evidence is preserved and unchanged in identity


def test_regenerated_baseline_accuracy_equals_the_stored_and_the_saved_record():
    s = ev.load_gradcam_source(BASE)
    assert s["regenerated_accuracy"] == s["stored_accuracy"] == _stored_accuracy("baseline-seed0") == 0.9001
    assert s["n_correct"] + s["n_incorrect"] == s["n_test"] == 10000


def test_baseline_has_at_least_five_genuinely_correct_and_five_incorrect_examples():
    rows = _rows(BASE / "examples.csv")
    correct = [r for r in rows if r["set"] == "correct"]
    wrong = [r for r in rows if r["set"] == "incorrect"]
    assert len(correct) >= 5 and len(wrong) >= 5
    assert all(r["true"] == r["predicted"] for r in correct)
    assert all(r["true"] != r["predicted"] for r in wrong)
    for r in rows:  # the saved probability vector agrees with the stated prediction
        probs = {k[2:]: float(v) for k, v in r.items() if k.startswith("p_")}
        assert max(probs, key=probs.get) == r["predicted"] and len(probs) == 10


def test_blank_map_counts_cover_every_test_image_and_every_class():
    d = ev.load_baseline_gradcam(FASHION)
    assert set(d["blank"]) == set(FASHION.classes) and sum(v["n"] for v in d["blank"].values()) == 10000
    assert all(0 <= v["blank"] <= v["n"] for v in d["blank"].values())


def test_shared_images_are_the_improved_gallery_and_correctness_follows_each_models_own_prediction():
    d = ev.load_baseline_gradcam(FASHION)
    imp_idx = {int(r["test_index"]) for r in _rows(IMPR / "examples.csv")}
    assert {s["test_index"] for s in d["shared"]} == imp_idx and len(d["shared"]) == 20
    for s in d["shared"]:
        assert s["baseline_correct"] == (s["baseline_pred"] == s["true"])
        assert s["improved_correct"] == (s["improved_pred"] == s["true"])
    # one image is right for the baseline and wrong for the improved model: both are labelled separately
    assert any(s["baseline_correct"] and not s["improved_correct"] for s in d["shared"])


def test_identity_file_records_matching_split_labels_preprocessing_and_files():
    i = json.loads((BASE / "identity_check.json").read_text())["identity"]
    assert i["local_file_sha256_matches_stored_identity"] is True
    assert i["split_sha256"] == ev.load_gradcam_source(IMPR)["split_sha256"]
    assert (i["baseline"]["run_id"], i["improved"]["run_id"]) == (53, 54)
    assert i["class_names"] == list(FASHION.classes)


def test_improved_evidence_still_matches_its_stored_accuracy():
    s = ev.load_gradcam_source(IMPR)
    assert s["regenerated_accuracy"] == s["stored_accuracy"] == _stored_accuracy("full_improved-seed0")


def test_written_comparison_is_labelled_an_unreviewed_draft_and_names_the_blank_map_limit():
    text = ev.load_text_doc(ev.BASELINE_COMPARISON_DOC)
    assert "DRAFT" in text and "Neither Emma nor Komal has reviewed it" in text
    assert "does not show that no region supports the class" in text
    assert "improved model won" not in text.lower() and "outperform" not in text.lower()


# ---- rejection paths on temporary copies -------------------------------------------------------------------------
@pytest.fixture()
def results(tmp_path, monkeypatch):
    shutil.copytree(BASE, tmp_path / f"{FASHION.group}-gradcam-baseline")
    shutil.copytree(IMPR, tmp_path / f"{FASHION.group}-gradcam")
    monkeypatch.setattr(ev, "RESULTS", tmp_path)
    return tmp_path


def _edit_json(path: Path, **changes):
    d = json.loads(path.read_text())
    d.update(changes)
    path.write_text(json.dumps(d))


def test_copy_loads_before_it_is_broken(results):
    assert ev.load_baseline_gradcam(FASHION)["source"]["run_id"] == 53


@pytest.mark.parametrize("change,message", [
    ({"run_name": "no_dropout-seed0"}, "expected run baseline-seed0"),
    ({"dataset_id": 99}, "differ in dataset_id"),
    ({"split_sha256": "0" * 64}, "differ in split_sha256"),
    ({"experiment_group": "other"}, "differ in experiment_group"),
    ({"regenerated_accuracy": 0.9}, "does not equal the stored accuracy"),
])
def test_wrong_source_identity_is_refused(results, change, message):
    _edit_json(results / f"{FASHION.group}-gradcam-baseline" / "source.json", **change)
    with pytest.raises(ev.EvidenceError, match=message):
        ev.load_baseline_gradcam(FASHION)


def test_fewer_than_five_examples_of_either_kind_is_refused(results):
    path = results / f"{FASHION.group}-gradcam-baseline" / "examples.csv"
    rows = _rows(path)
    keep = [r for r in rows if r["set"] == "correct"][:4] + [r for r in rows if r["set"] == "incorrect"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(keep)
    with pytest.raises(ev.EvidenceError, match="at least 5 correct"):
        ev.load_baseline_gradcam(FASHION)


def test_shared_images_that_are_not_the_improved_gallery_are_refused(results):
    path = results / f"{FASHION.group}-gradcam-baseline" / "shared_examples.csv"
    rows = _rows(path)
    rows[0]["test_index"] = "9999"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ev.EvidenceError, match="not the improved model's gallery"):
        ev.load_baseline_gradcam(FASHION)


def test_missing_folder_and_missing_blank_table_give_readable_errors(results):
    (results / f"{FASHION.group}-gradcam-baseline" / "blank_cam_by_class.csv").unlink()
    with pytest.raises(ev.EvidenceError):
        ev.load_baseline_gradcam(FASHION)
    shutil.rmtree(results / f"{FASHION.group}-gradcam-baseline")
    with pytest.raises(ev.EvidenceError, match="Missing evidence folder"):
        ev.load_baseline_gradcam(FASHION)


# ---- generation script: identity guard and helpers ---------------------------------------------------------------
def _loaded(**kw):
    base = dict(class_names=list(FASHION.classes), preprocessing={"mean": 0.286, "std": 0.353},
                dataset={"name": "fashion-mnist", "source": "fashion_mnist"}, artifact_version=3, temperature=1.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _args(**over):
    sha = {"a.gz": "x", "b.gz": "y"}
    ds = {"id": 24, "identity": {"file_sha256": dict(sha)}}
    run = dict(id=53, run_name="baseline-seed0", experiment_group="g", dataset_id=24, split_sha256="s", train_seed=0,
               epochs=20, batch_size=128, lr=0.001, optimizer="adam", n_train=54000, n_val=6000, n_test=10000)
    imp = dict(run, id=54, run_name="full_improved-seed0")
    a = dict(base_run=run, imp_run=imp, base_ds=ds, imp_ds=dict(ds), base=_loaded(), imp=_loaded(),
             local_sha=dict(sha), group="g", expected_classes=list(FASHION.classes))
    a.update(over)
    return a


def test_identity_check_accepts_a_matched_pair():
    out = gc.check_identity(**_args())
    assert out["local_file_sha256_matches_stored_identity"] and out["baseline"]["run_id"] == 53


@pytest.mark.parametrize("over,message", [
    (dict(imp_run=dict(_args()["imp_run"], dataset_id=25)), "different datasets"),
    (dict(imp_run=dict(_args()["imp_run"], split_sha256="t")), "different train/validation splits"),
    (dict(imp_run=dict(_args()["imp_run"], train_seed=1)), "share a seed"),
    (dict(imp_run=dict(_args()["imp_run"], epochs=10)), "differ in epochs"),
    (dict(imp=_loaded(class_names=list(reversed(FASHION.classes)))), "class labels differ"),
    (dict(imp=_loaded(preprocessing={"mean": 0.0, "std": 1.0})), "preprocessing differs"),
    (dict(local_sha={"a.gz": "x", "b.gz": "WRONG"}), "do not match the dataset identity"),
    (dict(local_sha={"a.gz": "x"}), "do not match the dataset identity"),
])
def test_identity_check_refuses_any_mismatch(over, message):
    with pytest.raises(gc.IdentityError, match=message):
        gc.check_identity(**_args(**over))


def test_read_indices_returns_the_improved_galleries_and_blank_detection():
    idx = gc.read_indices(IMPR)
    assert len(idx["correct"]) == len(idx["incorrect"]) == 10
    import numpy as np
    assert gc.is_blank(np.zeros((7, 7))) and not gc.is_blank(np.eye(7))


# ---- the Compare Models display ----------------------------------------------------------------------------------
def test_compare_models_shows_the_baseline_gallery_blank_table_shared_images_and_draft_label(monkeypatch):
    from tests.test_ui_app_smoke import PRED_ROW, RUN_ROW, _run

    at = _run(monkeypatch, {"recent_predictions": [PRED_ROW], "runs": [RUN_ROW]})
    assert not at.exception, [e.value for e in at.exception]
    heads = " ".join(h.value for h in [*at.subheader, *at.header])
    assert "Baseline (baseline-seed0, run 53)" in heads and "Improved model (full_improved-seed0, run 54)" in heads
    md = " ".join(m.value for m in at.markdown)
    for needle in ("Baseline: correct predictions (10)", "Baseline: incorrect predictions (10)",
                   "Blank Grad-CAM maps by predicted class", "Same test images, both models",
                   "not yet reviewed by the authors"):
        assert needle in md + " ".join(e.label for e in at.expander), needle
    assert not any("This section is unavailable" in w.value for w in at.warning)
    captions = " ".join(c.value for c in at.caption)
    assert "matching the stored 90.01%" in captions and "nothing was retrained" in captions
    assert "does not mean no region supports the class" in captions
