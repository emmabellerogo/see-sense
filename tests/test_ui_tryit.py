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
OK = {"class_name": "block", "confidence": 0.9, "input_png": "AA==", "cam_png": "AA==",
      "image_sha256": "ab" * 32}


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


# --- run selection ----------------------------------------------------------
def test_default_run_resolves_from_saved_records():
    run = t.resolve_default_run()
    assert run == {"run_id": 4, "run_name": "full_improved-seed0", "group": "official-supabase-1"}


def test_default_run_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert t.resolve_default_run()["run_id"] == 4


def _write(path, recs):
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_run_chosen_by_name_not_by_position_or_score(tmp_path):
    f = tmp_path / "runs.jsonl"
    _write(f, [
        {"run_name": "baseline-seed0", "run_id": 3, "status": "saved", "test_accuracy": 0.99},
        {"run_name": "full_improved-seed0", "run_id": 41, "status": "saved"},
    ])
    assert t.resolve_default_run(f, tmp_path / "none.json")["run_id"] == 41


def test_missing_records_give_friendly_error(tmp_path):
    with pytest.raises(t.TryItError, match="missing"):
        t.resolve_default_run(tmp_path / "nope.jsonl")


def test_unsaved_or_duplicate_records_rejected(tmp_path):
    f = tmp_path / "runs.jsonl"
    _write(f, [{"run_name": "full_improved-seed0", "run_id": 4, "status": "failed"}])
    with pytest.raises(t.TryItError, match="found 0"):
        t.resolve_default_run(f, tmp_path / "none.json")
    _write(f, [{"run_name": "full_improved-seed0", "run_id": 4, "status": "saved"}] * 2)
    with pytest.raises(t.TryItError, match="found 2"):
        t.resolve_default_run(f, tmp_path / "none.json")


def test_conflicting_gradcam_source_is_reported(tmp_path):
    f = tmp_path / "runs.jsonl"
    _write(f, [{"run_name": "full_improved-seed0", "run_id": 4, "status": "saved"}])
    src = tmp_path / "source.json"
    src.write_text(json.dumps({"run_name": "full_improved-seed0", "run_id": 99}))
    with pytest.raises(t.TryItError, match="disagree"):
        t.resolve_default_run(f, src)


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
    (FakeResp(422, {"detail": "could not decode image"}), "could not decode image"),
    (FakeResp(503), "unavailable"),
    (FakeResp(503, {"detail": "database unavailable"}), "nothing was classified or logged"),
    (FakeResp(500), "HTTP 500"),
    (FakeResp(200, None), "unreadable"),
    (FakeResp(200, {"class_name": "block"}), "incomplete"),
])
def test_errors_are_friendly_and_post_is_never_retried(result, fragment):
    post = Poster(result)
    with pytest.raises(t.TryItError, match=fragment) as ei:
        t.predict_sample("http://api", 4, 0, None, post=post)
    assert post.calls == 1
    assert "Traceback" not in str(ei.value)


def test_success_returns_response_once_with_expected_payload():
    seen = {}

    def post(url, **kw):
        seen.update(url=url, **kw)
        return FakeResp(200, OK)

    assert t.predict_sample("http://api", 4, 7, 3, post=post)["class_name"] == "block"
    assert seen["url"] == "http://api/predict_sample"
    assert seen["json"] == {"run_id": 4, "seed": 7, "true_class": 3}
    assert seen["timeout"] == t.PREDICT_TIMEOUT


def test_upload_sends_run_id_and_file():
    seen = {}

    def post(url, **kw):
        seen.update(url=url, **kw)
        return FakeResp(200, OK)

    t.predict_upload("http://api", 4, "a.png", PNG, "image/png", post=post)
    assert seen["url"] == "http://api/predict" and seen["data"] == {"run_id": 4}


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
