"""Concepts arithmetic, Model Card loading and the final tab list."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

UI = Path(__file__).parent.parent / "ui"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ui_{name}", UI / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c = _load("concepts")
ev = _load("evidence")


@pytest.mark.parametrize("n,k,s,p,expected", [
    (28, 3, 1, 0, 26), (28, 3, 1, 1, 28), (28, 3, 2, 1, 14),
    (28, 2, 2, 0, 14), (14, 2, 2, 0, 7), (7, 3, 2, 0, 3), (5, 5, 1, 0, 1),
])
def test_output_size_formula(n, k, s, p, expected):
    assert c.out_size(n, k, s, p) == expected


@pytest.mark.parametrize("args", [(2, 3, 1, 0), (0, 3, 1, 0), (28, 3, 0, 0), (28, 3, 1, -1)])
def test_invalid_sizes_raise(args):
    with pytest.raises(ValueError):
        c.out_size(*args)


def test_three_stacked_3x3_receptive_field_is_3_5_7():
    rows = c.trace(c.THREE_3X3, 28)
    assert [r["r"] for r in rows] == [1, 3, 5, 7]
    assert [r["size"] for r in rows] == [28, 26, 24, 22]
    assert all(r["j"] == 1 for r in rows)


def test_project_cnn_trace_matches_the_real_network():
    rows = c.trace(c.OUR_CNN, 28)
    assert [r["size"] for r in rows] == [28, 28, 14, 14, 7]
    assert [r["r"] for r in rows] == [1, 3, 4, 8, 10]       # 10x10 patch per Grad-CAM cell
    assert [r["j"] for r in rows] == [1, 1, 2, 2, 4]


def test_parameter_count_matches_supabase_baseline():
    assert c.our_cnn_param_count() == 4388


def test_architecture_matches_code():
    """The Concepts table must describe the network that is actually trained."""
    torch = pytest.importorskip("torch")
    from api.training import ImprovedCNN, SmallCNN
    model = SmallCNN(28, 4)
    assert sum(p.numel() for p in model.parameters()) == c.our_cnn_param_count()
    assert model.features(torch.zeros(1, 1, 28, 28)).shape[1:] == (16, 7, 7)
    assert sum(p.numel() for p in ImprovedCNN(28, 4).parameters()) == 4436


def test_model_card_loads_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = ev.load_model_card()
    for must in ("official-supabase-1", "95.05%", "76.80%", "full_improved-seed0",
                 "recent_classifications", "not a general photo classifier"):
        assert must in text


def test_model_card_missing_or_empty_is_a_friendly_error(tmp_path):
    with pytest.raises(ev.EvidenceError, match="Missing file"):
        ev.load_model_card(tmp_path / "nope.md")
    (tmp_path / "empty.md").write_text("  \n")
    with pytest.raises(ev.EvidenceError, match="empty"):
        ev.load_model_card(tmp_path / "empty.md")


def test_model_card_numbers_match_saved_results():
    text = ev.load_model_card()
    for r in ev.load_variant_summary():
        assert f"{r['mean_test_accuracy'] * 100:.2f}%" in text, r["variant"]


def test_final_tabs_exactly_in_order():
    tree = ast.parse((UI / "app.py").read_text())
    lists = [
        [e.value for e in n.args[0].elts]
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "tabs"
    ]
    assert lists == [["Concepts", "Try It", "Compare Models", "Recent Classifications", "Model Card"]]
