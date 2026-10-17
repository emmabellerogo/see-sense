"""Checks that the saved official-supabase-1 evidence loads and is complete."""
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
    assert ev.SQL_DIR.is_dir() and ev.SQL_DIR.is_absolute()
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


def test_all_figures_exist():
    assert ev.missing_figures() == []


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
