"""Saved evidence loaders: the historical synthetic set (complete) and the Fashion-MNIST set (parametrized, not yet available)."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ui_evidence", Path(__file__).parent.parent / "ui" / "evidence.py"
)
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


def test_paths_are_anchored_to_repo_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ev.HISTORICAL.sql_dir.is_dir() and ev.HISTORICAL.sql_dir.is_absolute()
    assert ev.REPO_ROOT == Path(__file__).resolve().parent.parent
    assert len(ev.load_variant_summary()) == 6


def test_variant_summary_matches_official_results():
    rows = {r["variant"]: r for r in ev.load_variant_summary()}
    assert set(rows) == set(ev.EXPECTED_VARIANTS)
    assert all(r["n_seeds"] == 5 for r in rows.values())
    assert rows["full_improved"]["mean_test_accuracy"] == pytest.approx(0.9505)
    assert rows["baseline"]["mean_test_accuracy"] == pytest.approx(0.768)
    assert rows["no_batchnorm"]["sd_test_accuracy"] == pytest.approx(0.2293)


def test_audit_passes_on_official_data():
    audit = ev.load_audit()
    assert audit["passed"], audit["problems"]
    assert len(audit["rows"]) == 5


def test_paired_differences_match_summary_md():
    rows = {r["variant"]: r for r in ev.load_paired_differences()}
    assert set(rows) == set(ev.TECHNIQUE_REMOVED)
    # Intervals reported in official-supabase-1-analysis/summary.md
    assert rows["baseline"]["mean_acc_pp"] == pytest.approx(-18.25)
    assert rows["baseline"]["ci_low_pp"] == pytest.approx(-28.89, abs=0.01)
    assert rows["baseline"]["ci_high_pp"] == pytest.approx(-7.61, abs=0.01)
    assert not rows["no_augmentation"]["ci_includes_zero"]
    for v in ("no_dropout", "no_scheduler", "no_batchnorm"):
        assert rows[v]["ci_includes_zero"], v


def test_gradcam_examples_complete_and_uncertain_flag():
    ex = ev.load_gradcam_examples()
    assert sum(e["set"] == "correct" for e in ex) == 5
    assert sum(e["set"] == "incorrect" for e in ex) == 8
    flagged = [e["test_index"] for e in ex if e["uncertain"]]
    assert flagged == [1314]


def test_failure_interpretations_are_the_two_existing_ones():
    interp = ev.load_failure_interpretations()
    assert set(interp["examples"]) == {924, 1783}
    assert "dataset difficulty" in interp["examples"][924]["title"]
    assert "model limitation" in interp["examples"][1783]["title"]
    assert interp["limits"]


def test_all_historical_figures_exist():
    assert ev.missing_figures(ev.figures_for(ev.HISTORICAL) | {}) == [
        str(p.relative_to(ev.REPO_ROOT)) for p in ev.figures_for(ev.HISTORICAL).values() if not p.is_file()]
    assert [f for f in ev.missing_figures() if "per_class" not in f] == []  # historical set predates the per-class figure


def test_missing_file_gives_readable_error(tmp_path):
    with pytest.raises(ev.EvidenceError, match="Missing evidence file"):
        ev.load_variant_summary(tmp_path)


def test_incomplete_gradcam_set_is_rejected(tmp_path):
    shutil.copy(ev.GRADCAM_DIR / "examples.csv", tmp_path / "examples.csv")
    lines = (tmp_path / "examples.csv").read_text().splitlines()
    (tmp_path / "examples.csv").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ev.EvidenceError, match="expected 5 correct and 8 incorrect"):
        ev.load_gradcam_examples(tmp_path)


def test_failed_audit_is_reported(tmp_path):
    src = (ev.SQL_DIR / "q1.csv").read_text().splitlines()
    bad = src[:]
    bad[1] = bad[1].replace(",6,1,", ",5,1,", 1)
    (tmp_path / "q1.csv").write_text("\n".join(bad) + "\n")
    audit = ev.load_audit(tmp_path)
    assert not audit["passed"] and audit["problems"]


def test_non_numeric_value_is_not_silently_accepted(tmp_path):
    (tmp_path / "q5.csv").write_text("variant,n_seeds,mean_test_accuracy\nbaseline,5,oops\n")
    with pytest.raises(ev.EvidenceError):
        ev.load_variant_summary(tmp_path)


# ---- the two evidence sets are separate and honestly labelled ---------------------------------------
def test_historical_and_fashion_sets_never_share_files():
    h, f = ev.HISTORICAL, ev.FASHION
    assert h.group == "official-supabase-1" and f.group != h.group
    assert len({h.sql_dir, f.sql_dir, h.analysis_dir, f.analysis_dir, h.gradcam_dir, f.gradcam_dir}) == 6
    assert h.classes == ("horizontal", "vertical", "diagonal", "block") and len(f.classes) == 10
    assert "NOT Fashion-MNIST" in h.title and "Fashion-MNIST" in f.title


def test_fashion_group_name_comes_from_the_protocol_config():
    import json

    cfg = json.loads((ev.REPO_ROOT / "config" / "fashion_protocol.json").read_text())
    assert ev.FASHION.group == cfg["experiment_group"]


def test_fashion_evidence_is_reported_unavailable_until_it_exists():
    if not ev.FASHION.available():
        with pytest.raises(ev.EvidenceError, match="Missing evidence file"):
            ev.load_variant_summary(ev.FASHION.sql_dir)
        with pytest.raises(ev.EvidenceError):
            ev.load_calibration_summary(ev.FASHION.analysis_dir)


def _write_csv(path, header, rows):
    path.write_text(",".join(header) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")


def test_loaders_accept_a_three_seed_ten_class_set(tmp_path):
    """The Fashion-MNIST protocol may use fewer than five seeds; the loaders must not insist on five."""
    variants = ev.EXPECTED_VARIANTS
    seeds = [0, 1, 2]
    _write_csv(tmp_path / "q1.csv", ["train_seed", "n_runs", "datasets", "split_hashes"],
               [[s, 6, 1, 1] for s in seeds])
    assert ev.load_audit(tmp_path, expected_seeds=None)["passed"]
    assert not ev.load_audit(tmp_path, expected_seeds=5)["passed"]
    rows = [[s, v, 0.0 if v == "full_improved" else -0.01 * (s + 1), 0.0, 0.0] for s in seeds for v in variants
            if v != "full_improved"]
    _write_csv(tmp_path / "q6.csv", ["train_seed", "variant", "delta_accuracy_vs_full",
                                     "delta_macro_f1_vs_full", "delta_test_loss_vs_full"], rows)
    paired = ev.load_paired_differences(tmp_path, expected_seeds=None)
    assert {p["variant"] for p in paired} == set(ev.TECHNIQUE_REMOVED) and paired[0]["n_seeds"] == 3
    with pytest.raises(ev.EvidenceError, match="expected 5"):
        ev.load_paired_differences(tmp_path, expected_seeds=5)

    classes = ev.FASHION.classes
    _write_csv(tmp_path / "per_class_metrics.csv",
               ["variant", "class", "mean_precision", "sd_precision", "mean_recall", "sd_recall", "n_runs"],
               [[v, c, 0.8, 0.01, 0.8, 0.01, 3] for v in variants for c in classes])
    pc = ev.load_per_class_metrics(tmp_path, classes)
    assert len(pc) == 6 * 10 and pc[0]["class"] == "T-shirt/top"
    with pytest.raises(ev.EvidenceError, match="missing variant/class"):
        ev.load_per_class_metrics(tmp_path, ev.HISTORICAL.classes)

    _write_csv(tmp_path / "calibration.csv",
               ["run_name", "temperature", "test_ece_before", "test_ece_after", "test_nll_before", "test_nll_after"],
               [[f"{v}-seed{s}", 1.2, 0.02, 0.01, 0.3, 0.29] for v in variants for s in seeds])
    cal = ev.load_calibration_summary(tmp_path)
    assert len(cal) == 6 and cal[0]["n_runs"] == 3 and cal[0]["test_ece_after"] == pytest.approx(0.01)


def test_gradcam_set_for_fashion_needs_at_least_five_of_each(tmp_path):
    def write(n_c, n_i):
        rows = [["correct", i, "Dress", "Dress", 0.9] for i in range(n_c)] + \
               [["incorrect", 100 + i, "Shirt", "Coat", 0.8] for i in range(n_i)]
        _write_csv(tmp_path / "examples.csv", ["set", "test_index", "true", "predicted", "confidence"], rows)
    write(10, 10)
    assert len(ev.load_gradcam_examples(tmp_path, counts=None)) == 20
    write(10, 4)
    with pytest.raises(ev.EvidenceError, match="at least 5"):
        ev.load_gradcam_examples(tmp_path, counts=None)


def test_fashion_failure_analysis_is_a_labelled_draft_grounded_in_the_saved_examples():
    import csv

    doc = ev.FASHION.failure_doc
    text = doc.read_text()
    assert "DRAFT" in text and "human review" in text and "PENDING" not in text
    interp = ev.load_failure_interpretations(doc)
    assert len(interp["examples"]) >= 2 and interp["limits"]
    rows = list(csv.DictReader(open(ev.FASHION.gradcam_dir / "examples.csv")))
    wrong = {int(r["test_index"]) for r in rows if r["set"] == "incorrect"}
    assert set(interp["examples"]) <= wrong  # every interpreted example is a real, genuinely incorrect prediction
    blank = {r["predicted_class"]: r for r in csv.DictReader(open(ev.FASHION.gradcam_dir / "blank_cam_by_class.csv"))}
    for cls in ("Dress", "Coat"):  # the numbers quoted in the text come from the saved diagnostic
        r = blank[cls]
        assert f"{int(r['n_blank_cam']):,} of {int(r['n_predictions']):,}" in text.replace("1,032 of 1,055", "1,032 of 1,055")
