"""Run selection and error handling for the Try It tab (no network, no UI)."""
from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import pytest
import requests

_SPEC = importlib.util.spec_from_file_location(
    "ui_tryit", Path(__file__).parent.parent / "ui" / "tryit.py"
)
t = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(t)

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20
TOP5 = [{"label": 3, "class_name": "Dress", "prob": 0.6}, {"label": 6, "class_name": "Shirt", "prob": 0.2},
        {"label": 0, "class_name": "T-shirt/top", "prob": 0.1}, {"label": 4, "class_name": "Coat", "prob": 0.05},
        {"label": 2, "class_name": "Pullover", "prob": 0.03}]
OK = {"class_name": "Dress", "confidence": 0.6, "input_png": "AA==", "cam_png": "AA==",
      "image_sha256": "ab" * 32, "top5": TOP5, "probs": [0.1] * 10, "logged": True, "prediction_id": 5}


class FakeResp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class Poster:
    """Records calls; returns/raises what it is told to."""
    def __init__(self, result):
        self.result, self.calls = result, 0

    def __call__(self, url, **kw):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# --- served model: explicit configuration, verified -----------------------------------------
def _write(path, recs):
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def _cfg(tmp_path, **over):
    cfg = {"dataset": "fashion-mnist", "n_classes": 10, "experiment_group": "fashion-test-1",
           "run_name": "full_improved-seed0", "served_run_id": 41, **over}
    path = tmp_path / "served_model.json"
    path.write_text(json.dumps(cfg))
    return path


def _records(tmp_path, recs):
    path = tmp_path / "runs.jsonl"
    _write(path, recs)
    return path


SAVED = {"run_name": "full_improved-seed0", "run_id": 41, "status": "saved"}


def test_the_committed_configuration_is_valid_and_names_a_fashion_model_chosen_in_advance():
    cfg = t.load_served_config()
    assert cfg["dataset"] == "fashion-mnist" and cfg["n_classes"] == 10
    assert cfg["run_name"] == "full_improved-seed0" and cfg["chosen_before_official_results"] is True


def test_run_id_comes_from_configuration_and_must_match_saved_records(tmp_path):
    served = t.resolve_served_run(_cfg(tmp_path), _records(tmp_path, [SAVED]))
    assert served["run_id"] == 41 and served["run_name"] == "full_improved-seed0" and served["group"] == "fashion-test-1"


def test_there_is_no_hardcoded_run_id_in_the_ui_code():
    import re
    for name in ("tryit.py", "app.py"):
        code = (Path(__file__).parent.parent / "ui" / name).read_text()
        assert not re.search(r"run_id\s*=\s*\d+", code) and "run ID 4" not in code, name


def test_unconfigured_model_is_reported_not_substituted(tmp_path):
    with pytest.raises(t.TryItError, match="not been trained and configured"):
        t.resolve_served_run(_cfg(tmp_path, served_run_id=None), tmp_path / "nope.jsonl")


def test_the_committed_configuration_is_unset_until_the_official_run_exists():
    # Keeps Try It off (rather than serving a wrong or synthetic model) until the id is filled in.
    cfg = t.load_served_config()
    if cfg.get("served_run_id") is None:
        with pytest.raises(t.TryItError, match="not been trained"):
            t.resolve_served_run()


def test_record_mismatches_are_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(t.TryItError, match="missing"):
        t.resolve_served_run(cfg, tmp_path / "nope.jsonl")
    with pytest.raises(t.TryItError, match="does not match"):
        t.resolve_served_run(cfg, _records(tmp_path, [{**SAVED, "run_id": 7}]))
    with pytest.raises(t.TryItError, match="0 record"):
        t.resolve_served_run(cfg, _records(tmp_path, [{**SAVED, "status": "failed"}]))
    with pytest.raises(t.TryItError, match="2 record"):
        t.resolve_served_run(cfg, _records(tmp_path, [SAVED, SAVED]))


@pytest.mark.parametrize("over", [{"dataset": "synthetic-shapes"}, {"n_classes": 4}, {"run_name": ""}])
def test_configuration_must_describe_a_ten_class_fashion_model(tmp_path, over):
    with pytest.raises(t.TryItError):
        t.load_served_config(_cfg(tmp_path, **over))


def test_the_records_file_is_derived_from_the_configured_group_not_a_constant(tmp_path, monkeypatch):
    cfg = t.load_served_config()
    assert cfg["experiment_group"] != "official-supabase-1"  # the historical synthetic group is never the target
    assert t.runs_jsonl_for("fashion-test-1").name == "fashion-test-1-runs.jsonl"
    monkeypatch.setattr(t, "runs_jsonl_for", lambda g: tmp_path / f"{g}-runs.jsonl")
    _write(tmp_path / "fashion-test-1-runs.jsonl", [SAVED])
    assert t.resolve_served_run(_cfg(tmp_path))["run_id"] == 41


class Getter:
    def __init__(self, resp):
        self.resp, self.urls = resp, []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


SERVED = {"run_id": 41, "run_name": "full_improved-seed0", "group": "fashion-test-1"}
RUN_OK = {"run_name": "full_improved-seed0", "experiment_group": "fashion-test-1",
          "config": {"dataset": "fashion-mnist", "n_classes": 10}}


def test_served_run_is_verified_against_the_api():
    g = Getter(FakeResp(200, RUN_OK))
    t.verify_served_run("http://api", SERVED, get=g)
    assert g.urls == ["http://api/runs/41"]


@pytest.mark.parametrize("resp, fragment", [
    (FakeResp(200, {**RUN_OK, "config": {"dataset": "fashion-mnist", "n_classes": 4}}), "not the expected"),
    (FakeResp(200, {**RUN_OK, "config": {}}), "not the expected"),            # a historical synthetic run
    (FakeResp(200, {**RUN_OK, "run_name": "baseline-seed0"}), "not the expected"),
    (FakeResp(200, {**RUN_OK, "experiment_group": "official-supabase-1"}), "not the expected"),
    (FakeResp(404, {"detail": "run_id not found"}), "does not exist"),
    (FakeResp(500), "Couldn't verify"),
    (FakeResp(200, None), "unreadable"),
    (requests.exceptions.ConnectionError(), "waking up"),
])
def test_verification_failures_disable_try_it_with_a_clear_message(resp, fragment):
    with pytest.raises(t.TryItError, match=fragment):
        t.verify_served_run("http://api", SERVED, get=Getter(resp))


# --- held-out examples ----------------------------------------------------------------------
def test_bundled_samples_are_official_test_images_of_every_class_if_present():
    m = t.load_samples()
    if not m:
        pytest.skip("samples not exported yet (python tools/export_samples.py)")
    assert m["partition"].startswith("official test") and len(m["class_names"]) == 10
    assert {s["class_name"] for s in m["samples"]} == set(m["class_names"])
    assert all((t.SAMPLES_DIR / s["file"]).is_file() for s in m["samples"])
    assert "not chosen by model output" in m["rule"]


def test_missing_samples_give_an_empty_manifest(tmp_path):
    assert t.load_samples(tmp_path) == {}


# --- upload validation ------------------------------------------------------
@pytest.mark.parametrize("data", [b"", b"not an image", b"GIF89a....", PNG + b"0" * t.MAX_UPLOAD_BYTES])
def test_invalid_uploads_rejected_before_any_request(data):
    post = Poster(FakeResp(200, OK))
    with pytest.raises(t.TryItError):
        t.predict_upload("http://api", 4, "x.png", data, "image/png", post=post)
    assert post.calls == 0


def test_jpeg_and_png_accepted():
    for data in (PNG, b"\xff\xd8\xff\xe0" + b"0" * 10):
        t.validate_upload(data)


# --- error handling: friendly messages, exactly one POST --------------------
@pytest.mark.parametrize("result, fragment", [
    (requests.exceptions.ReadTimeout(), "waking up"),
    (requests.exceptions.ConnectionError(), "Couldn't reach"),
    (requests.exceptions.RequestException(), "failed"),
    (FakeResp(404, {"detail": "no artifact"}), "isn't available"),
    (FakeResp(413, {"detail": "file larger than 5 MB"}), "larger than 5 MB"),
    (FakeResp(503, {"detail": "prediction logging is unavailable, so the image was not classified"}), "logging is unavailable"),
    (FakeResp(422, {"detail": "could not decode image"}), "could not decode image"),
    (FakeResp(503), "unavailable"),
    (FakeResp(503, {"detail": "database unavailable"}), "nothing was classified or logged"),
    (FakeResp(500), "HTTP 500"),
    (FakeResp(200, None), "unreadable"),
    (FakeResp(200, {"class_name": "Dress"}), "incomplete"),
    (FakeResp(200, {k: v for k, v in OK.items() if k != "logged"}), "incomplete"),
])
def test_errors_are_friendly_and_post_is_never_retried(result, fragment):
    post = Poster(result)
    with pytest.raises(t.TryItError, match=fragment) as ei:
        t.predict_upload("http://api", 41, "x.png", PNG, "image/png", post=post)
    assert post.calls == 1
    assert "Traceback" not in str(ei.value)


def test_upload_sends_run_id_and_file_exactly_once_and_returns_the_response():
    seen = {"n": 0}

    def post(url, **kw):
        seen.update(url=url, **kw)
        seen["n"] += 1
        return FakeResp(200, OK)

    out = t.predict_upload("http://api", 41, "a.png", PNG, "image/png", post=post)
    assert out["class_name"] == "Dress" and seen["n"] == 1
    assert seen["url"] == "http://api/predict" and seen["data"] == {"run_id": 41}
    assert seen["files"]["file"][0] == "a.png" and seen["timeout"] == t.PREDICT_TIMEOUT


def test_top5_rows_come_from_the_api_in_order_and_are_limited_to_five():
    rows = t.top5_rows(OK)
    assert [r["Class"] for r in rows] == ["Dress", "Shirt", "T-shirt/top", "Coat", "Pullover"]
    assert rows[0]["Probability"] == pytest.approx(0.6) and len(rows) == 5
    assert len(t.top5_rows({"top5": TOP5 + [{"label": 1, "class_name": "Trouser", "prob": 0.01}]})) == 5
    assert t.top5_rows({}) == [] and t.top5_rows({"top5": [{"oops": 1}]}) == []


# --- cold-start warm-up (GET only) -----------------------------------------
def test_wake_api_retries_gets_until_healthy():
    answers = [requests.exceptions.ConnectionError(), FakeResp(503), FakeResp(200)]

    def get(url, timeout):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    assert t.wake_api("http://api", get=get, sleep=lambda s: None) is True
    assert answers == []


def test_wake_api_gives_up_without_raising():
    def get(url, timeout):
        raise requests.exceptions.ConnectionError()

    assert t.wake_api("http://api", get=get, sleep=lambda s: None) is False


def test_decode_png():
    assert t.decode_png(base64.b64encode(b"abc").decode()) == b"abc"
    with pytest.raises(t.TryItError):
        t.decode_png("%%%not base64%%%")
