"""Logic behind the Try It tab: default-run resolution and friendly API errors.

No streamlit import, so it is unit-testable offline. Prediction POSTs are sent
exactly once and never retried, because every successful call logs a row to
Supabase. Only idempotent GETs (health, classes) may be retried.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUP = "official-supabase-1"
DEFAULT_RUN_NAME = "full_improved-seed0"
RUNS_JSONL = REPO_ROOT / "experiments" / "results" / f"{GROUP}-runs.jsonl"
GRADCAM_SOURCE = REPO_ROOT / "experiments" / "results" / f"{GROUP}-gradcam" / "source.json"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
PREDICT_TIMEOUT = 60       # seconds; Render free tier cold start can take ~30-50 s
WARMUP_TIMEOUT = 5
WARMUP_ATTEMPTS = 6

_REQUIRED_RESPONSE_KEYS = ("class_name", "confidence", "input_png", "cam_png", "image_sha256")


class TryItError(Exception):
    """A problem with a user-friendly message (never contains a traceback)."""


# ---------------------------------------------------------------------------
# Default run
# ---------------------------------------------------------------------------
def resolve_default_run(runs_jsonl: Path = RUNS_JSONL,
                        gradcam_source: Path = GRADCAM_SOURCE) -> dict:
    """Return {'run_id', 'run_name', 'group'} for full_improved-seed0.

    The ID comes from the saved experiment records, not from user input or test
    performance. If the Grad-CAM source record also names this run, the two IDs
    must agree; a disagreement is reported rather than guessed around.
    """
    try:
        lines = runs_jsonl.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise TryItError("The saved experiment records are missing, so the default "
                         "model cannot be selected.") from None
    matches = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if (rec.get("run_name") == DEFAULT_RUN_NAME and rec.get("status") == "saved"
                and isinstance(rec.get("run_id"), int)):
            matches.append(rec["run_id"])
    if len(matches) != 1:
        raise TryItError(
            f"Expected exactly one saved record for {DEFAULT_RUN_NAME}, found {len(matches)}."
        )
    run_id = matches[0]
    try:
        src = json.loads(gradcam_source.read_text(encoding="utf-8"))
        if src.get("run_name") == DEFAULT_RUN_NAME and src.get("run_id") != run_id:
            raise TryItError("Saved records disagree about the default model's run ID.")
    except (OSError, ValueError):
        pass  # the cross-check is optional; the runs file alone is authoritative
    return {"run_id": run_id, "run_name": DEFAULT_RUN_NAME, "group": GROUP}


# ---------------------------------------------------------------------------
# Upload validation (client-side, before anything is sent or logged)
# ---------------------------------------------------------------------------
def validate_upload(data: bytes) -> None:
    if not data:
        raise TryItError("That file is empty. Please choose a PNG or JPG image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise TryItError("That file is larger than 5 MB. Please use a smaller image.")
    if not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")):
        raise TryItError("That doesn't look like a PNG or JPG image. Please upload one of those.")


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------
def wake_api(api_url: str, get=requests.get, sleep=None) -> bool:
    """Ping /healthz (safe to repeat) until the service answers. Returns whether
    it came up. Prediction requests are NOT sent from here."""
    import time
    sleep = sleep or time.sleep
    for attempt in range(WARMUP_ATTEMPTS):
        try:
            if get(f"{api_url}/healthz", timeout=WARMUP_TIMEOUT).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        if attempt < WARMUP_ATTEMPTS - 1:
            sleep(5)
    return False


def _detail(resp) -> str:
    try:
        d = resp.json().get("detail")
        return d if isinstance(d, str) else ""
    except Exception:  # noqa: BLE001
        return ""


def _post_once(post, url: str, **kwargs) -> dict:
    """Single POST with friendly errors. Deliberately no retry loop."""
    try:
        resp = post(url, timeout=PREDICT_TIMEOUT, **kwargs)
    except requests.exceptions.Timeout:
        raise TryItError(
            "The prediction service didn't answer in time. It may be waking up from "
            "sleep (the free tier can take up to a minute). Wait a moment and press the "
            "button again. Nothing is retried automatically, so nothing was logged twice."
        ) from None
    except requests.exceptions.ConnectionError:
        raise TryItError("Couldn't reach the prediction service. It may be starting up or "
                         "temporarily offline; please try again shortly.") from None
    except requests.exceptions.RequestException:
        raise TryItError("The request to the prediction service failed. Please try again.") from None

    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            raise TryItError("The prediction service returned an unreadable response.") from None
        if not isinstance(data, dict) or any(k not in data for k in _REQUIRED_RESPONSE_KEYS):
            raise TryItError("The prediction service returned an incomplete response.")
        return data
    if resp.status_code == 404:
        raise TryItError("The default model isn't available on the prediction service "
                         "(its saved checkpoint wasn't found). Try again later.")
    if resp.status_code == 422:
        raise TryItError("The service couldn't use that input" +
                         (f" ({_detail(resp)})." if _detail(resp) else ".") +
                         " Please upload a valid PNG or JPG image.")
    if resp.status_code == 503 and "database" in _detail(resp).lower():
        raise TryItError("The database behind the prediction service is temporarily "
                         "unavailable, so nothing was classified or logged. Please try "
                         "again in a few minutes.")
    if resp.status_code in (502, 503, 504):
        raise TryItError("The prediction service is unavailable right now (it may be "
                         "waking up). Please wait a moment and try again.")
    raise TryItError(f"The prediction service reported an error (HTTP {resp.status_code}). "
                     "Please try again later.")


def predict_sample(api_url: str, run_id: int, seed: int, true_class: int | None,
                   post=requests.post) -> dict:
    payload = {"run_id": int(run_id), "seed": int(seed)}
    if true_class is not None:
        payload["true_class"] = int(true_class)
    return _post_once(post, f"{api_url}/predict_sample", json=payload)


def predict_upload(api_url: str, run_id: int, filename: str, data: bytes,
                   content_type: str | None, post=requests.post) -> dict:
    validate_upload(data)
    return _post_once(
        post, f"{api_url}/predict",
        data={"run_id": int(run_id)},
        files={"file": (filename, data, content_type or "application/octet-stream")},
    )


def decode_png(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception:  # noqa: BLE001
        raise TryItError("The service returned an image that couldn't be decoded.") from None
