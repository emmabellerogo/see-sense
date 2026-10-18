"""Render the whole Streamlit app (AppTest) with no network: every tab must draw without an exception.

The Supabase client is replaced by a fake. The committed configuration serves run 54 and the committed official
evidence (group fashion-official-1) is displayed, including its provenance label; a configuration with no served
run must still produce the honest "unavailable" state instead of a guess.
"""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).parent.parent / "ui" / "app.py"


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def select(self, *_):
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_):
        return self

    def eq(self, *_):
        return self

    def execute(self):
        return type("R", (), {"data": self.rows})()


class FakeSupabase:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return FakeTable(self.tables.get(name, []))


PROBS = [0.02, 0.01, 0.03, 0.6, 0.04, 0.05, 0.1, 0.05, 0.05, 0.05]
PRED_ROW = {"id": 1, "created_at": "2026-10-17T06:38:40+00:00", "predicted_class": "Dress", "confidence": 0.6,
            "probabilities": PROBS, "served_by_run_id": 7, "run_name": "full_improved-seed0",
            "experiment_group": "g", "model_variant": "improved", "run_test_accuracy": 0.9, "run_test_macro_f1": 0.89}
RUN_ROW = {"id": 7, "run_name": "full_improved-seed0", "experiment_group": "g", "model_variant": "improved",
           "accuracy": 0.9, "macro_f1": 0.89, "config": {"dataset": "fashion-mnist"}, "epochs": 20}


def _run(monkeypatch, tables):
    import streamlit as st
    import supabase

    st.cache_resource.clear()  # the app caches its anon client; start each test from a clean cache

    monkeypatch.setattr(supabase, "create_client", lambda *a, **k: FakeSupabase(tables))
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    at.secrets["API_URL"] = "http://127.0.0.1:9"
    at.secrets["SUPABASE_URL"] = "https://example.invalid"
    at.secrets["SUPABASE_ANON_KEY"] = "anon"
    return at.run()


def test_app_renders_five_tabs_shows_the_final_evidence_and_labels_its_provenance(monkeypatch):
    at = _run(monkeypatch, {"recent_predictions": [PRED_ROW], "runs": [RUN_ROW]})
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Concepts", "Try It", "Compare Models", "Recent Classifications", "Model Card"]
    text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning) + " ".join(i.value for i in at.info)
    assert "Fashion-MNIST" in at.title[0].value
    assert "ten Fashion-MNIST clothing categories" in text
    assert "Final Fashion-MNIST results are not available yet" not in text
    assert "SQL provenance" in text + " ".join(c.value for c in at.caption)
    assert "Six-variant results" in " ".join(h.value for h in [*at.subheader, *at.header, *at.markdown])
    assert not any("not Fashion-MNIST" in m.value for m in at.warning)  # hidden until the toggle is switched on
    toggle = next(x for x in at.toggle if "historical" in x.label.lower() or "earlier experiment" in x.label.lower())
    assert "NOT Fashion-MNIST" in toggle.label and toggle.value is False
    at = toggle.set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("This is not Fashion-MNIST" in w.value for w in at.warning)  # the historical section is labelled
    assert not any("could not be read" in w.value for w in at.warning)      # every historical section rendered
    assert not any("Try It is unavailable" in w.value for w in at.warning)  # run 54 is configured and in the saved records


def test_without_a_served_run_id_try_it_is_unavailable_instead_of_guessing(monkeypatch, tmp_path):
    at, posts = _try_it(monkeypatch, tmp_path, RUN_OK, Resp(200, _predict_body()), served_run_id=None)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Try It is unavailable" in w.value for w in at.warning) and posts == []


def test_recent_tab_shows_the_prediction_and_drills_down_to_its_run(monkeypatch):
    at = _run(monkeypatch, {"recent_predictions": [PRED_ROW], "runs": [RUN_ROW]})
    assert not at.exception, [e.value for e in at.exception]
    tables = [d.value for d in at.dataframe]
    flat = " ".join(str(t) for t in tables)
    assert "Dress" in flat and "full_improved-seed0" in flat
    assert "sha256" not in flat.lower()
    assert any("Producing run 7" in m.value for m in at.markdown)


def test_recent_tab_falls_back_when_the_predictions_view_is_missing(monkeypatch):
    class Broken(FakeSupabase):
        def table(self, name):
            if name == "recent_predictions":
                raise RuntimeError("relation does not exist")
            return super().table(name)

    import streamlit as st
    import supabase

    st.cache_resource.clear()
    legacy = {"recent_classifications": [{"id": 1, "run_id": 4, "class_name": "block", "confidence": 0.9,
                                          "created_at": "2026-10-17T06:38:40+00:00"}]}
    monkeypatch.setattr(supabase, "create_client", lambda *a, **k: Broken(legacy))
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    at.secrets["API_URL"] = "http://127.0.0.1:9"
    at.secrets["SUPABASE_URL"] = "https://example.invalid"
    at.secrets["SUPABASE_ANON_KEY"] = "anon"
    at.run()
    assert not at.exception
    assert any("predictions view isn't available" in w.value for w in at.warning)


# ---- Try It flow with a mocked prediction service ---------------------------------------------------------
import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

import requests  # noqa: E402


def _tiny_png_b64() -> str:
    from PIL import Image  # test-side only; the UI itself never imports PIL

    buf = io.BytesIO()
    Image.new("RGB", (28, 28), (200, 40, 40)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


TOP5 = [{"label": 3, "class_name": "Dress", "prob": 0.6}, {"label": 6, "class_name": "Shirt", "prob": 0.2},
        {"label": 0, "class_name": "T-shirt/top", "prob": 0.1}, {"label": 4, "class_name": "Coat", "prob": 0.05},
        {"label": 2, "class_name": "Pullover", "prob": 0.03}]


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError
        return self._body


def _predict_body(**over):
    png = _tiny_png_b64()
    return {"run_id": 41, "label": 3, "class_name": "Dress", "confidence": 0.6, "probs": [0.1] * 10, "top5": TOP5,
            "class_names": [], "temperature": 1.3, "calibrated": True, "image_sha256": "ab" * 32,
            "input_png": png, "cam_png": png, "true_class": None, "logged": True, "prediction_id": 5, **over}


def _try_it(monkeypatch, tmp_path, run_body, post_resp, served_run_id=41):
    """Build the app against a configured served model and a fake API; returns (AppTest, post_calls)."""
    import streamlit as st
    import supabase

    ui_dir = str(APP.parent)
    if ui_dir not in sys.path:
        sys.path.insert(0, ui_dir)
    import tryit  # the very module the app imports

    cfg = tmp_path / "served_model.json"
    cfg.write_text(json.dumps({"dataset": "fashion-mnist", "n_classes": 10, "experiment_group": "fashion-test-1",
                               "run_name": "full_improved-seed0", "served_run_id": served_run_id}))
    recs = tmp_path / "fashion-test-1-runs.jsonl"
    recs.write_text(json.dumps({"run_name": "full_improved-seed0", "run_id": 41, "status": "saved"}) + "\n")
    monkeypatch.setattr(tryit, "SERVED_MODEL_CONFIG", cfg)
    monkeypatch.setattr(tryit, "runs_jsonl_for", lambda g: recs)
    posts = []

    def fake_get(url, timeout=None, **k):
        if url.endswith("/healthz"):
            return Resp(200, {"status": "ok"})
        if "/runs/" in url:
            return run_body
        raise AssertionError(url)

    def fake_post(url, **k):
        posts.append((url, k))
        return post_resp

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    st.cache_resource.clear()
    monkeypatch.setattr(supabase, "create_client", lambda *a, **k: FakeSupabase({}))
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60)
    at.secrets["API_URL"] = "http://api.invalid"
    at.secrets["SUPABASE_URL"] = "https://example.invalid"
    at.secrets["SUPABASE_ANON_KEY"] = "anon"
    return at.run(), posts


RUN_OK = Resp(200, {"run_name": "full_improved-seed0", "experiment_group": "fashion-test-1",
                    "config": {"dataset": "fashion-mnist", "n_classes": 10}})


def test_try_it_classifies_a_held_out_example_with_exactly_one_post_and_shows_top5(monkeypatch, tmp_path):
    at, posts = _try_it(monkeypatch, tmp_path, RUN_OK, Resp(200, _predict_body()))
    assert not at.exception, [e.value for e in at.exception]
    btn = next(b for b in at.button if b.label == "Classify this example")
    at = btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(posts) == 1 and posts[0][0] == "http://api.invalid/predict" and posts[0][1]["data"] == {"run_id": 41}
    assert any("Predicted: **Dress**" in s.value for s in at.success)
    flat = " ".join(str(d.value) for d in at.dataframe)
    assert "Shirt" in flat and "T-shirt/top" in flat and "Pullover" in flat       # the five classes are shown
    assert any("Logged to Supabase as prediction #5" in c.value for c in at.caption)
    assert any("Ground truth for this held-out example" in c.value for c in at.caption)


def test_try_it_never_retries_a_failed_prediction(monkeypatch, tmp_path):
    at, posts = _try_it(monkeypatch, tmp_path, RUN_OK,
                        Resp(503, {"detail": "prediction logging is unavailable, so the image was not classified"}))
    at = next(b for b in at.button if b.label == "Classify this example").click().run()
    assert len(posts) == 1
    assert any("logging is unavailable" in e.value for e in at.error)
    assert not any("Predicted" in s.value for s in at.success)


def test_a_result_the_service_did_not_confirm_as_logged_is_flagged(monkeypatch, tmp_path):
    at, _ = _try_it(monkeypatch, tmp_path, RUN_OK, Resp(200, _predict_body(logged=False, prediction_id=None)))
    at = next(b for b in at.button if b.label == "Classify this example").click().run()
    assert any("did not confirm that this prediction was logged" in w.value for w in at.warning)


def test_try_it_refuses_a_served_run_that_is_not_the_configured_fashion_model(monkeypatch, tmp_path):
    historical = Resp(200, {"run_name": "full_improved-seed0", "experiment_group": "fashion-test-1",
                            "config": {"n_rows": 2000, "noise": 1.0}})  # a synthetic-shape run: no dataset key
    at, posts = _try_it(monkeypatch, tmp_path, historical, Resp(200, _predict_body()))
    assert any("not the expected Fashion-MNIST model" in e.value for e in at.error)
    assert not [b for b in at.button if b.label == "Classify this example"] and posts == []


# ---- Upload warning (distribution shift) ------------------------------------------------------------------
EXPECTED_WARNING = (
    "This model expects Fashion-MNIST-style images: a centered, light clothing item on a dark background. "
    "Ordinary photos, especially those with light backgrounds, may receive incorrect high-confidence predictions."
)


def test_try_it_warns_before_upload_and_recommends_the_bundled_samples(monkeypatch, tmp_path):
    at, posts = _try_it(monkeypatch, tmp_path, RUN_OK, Resp(200, _predict_body()))
    assert not at.exception, [e.value for e in at.exception]
    assert EXPECTED_WARNING in [w.value for w in at.warning]
    assert any("bundled held-out examples" in c.value and "Option B" in c.value for c in at.caption)
    assert posts == []  # the warning is shown with nothing sent or logged, and uploads are not altered or rejected for it


def test_the_warning_is_not_a_background_detector():
    """Documented limitation only: the UI must not invert, crop or reject uploads on its own guess about the background."""
    import inspect
    sys_path = str(APP.parent)
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import tryit

    assert tryit.UPLOAD_WARNING == EXPECTED_WARNING
    src = inspect.getsource(tryit.validate_upload) + inspect.getsource(tryit.predict_upload)
    assert "invert" not in src.lower() and "background" not in src.lower()
