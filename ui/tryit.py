"""Logic behind the Try It tab: served-model resolution and friendly API errors.

No streamlit import, so it is unit-testable offline. Prediction POSTs are sent
exactly once and never retried, because every successful call logs a row to
Supabase. Only idempotent GETs (health, runs) may be retried.

The model Try It serves is NOT hardcoded and NOT picked by test score. It comes from
``config/served_model.json`` (chosen before any official result existed), is checked against the
saved experiment records, and is verified against the API (``GET /runs/{id}``) before use.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVED_MODEL_CONFIG = REPO_ROOT / "config" / "served_model.json"
SAMPLES_DIR = Path(__file__).resolve().parent / "assets" / "samples"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
PREDICT_TIMEOUT = 60       # seconds; Render free tier cold start can take ~30-50 s
WARMUP_TIMEOUT = 5
WARMUP_ATTEMPTS = 6

_REQUIRED_RESPONSE_KEYS = ("class_name", "confidence", "input_png", "cam_png", "image_sha256",
                           "top5", "probs", "logged")


class TryItError(Exception):
    """A problem with a user-friendly message (never contains a traceback)."""


# ---------------------------------------------------------------------------
# Served model (explicit configuration, verified)
# ---------------------------------------------------------------------------
def runs_jsonl_for(group: str) -> Path:
    return REPO_ROOT / "experiments" / "results" / f"{group}-runs.jsonl"


def load_served_config(path: Path | None = None) -> dict:
    """Read and sanity-check the served-model configuration."""
    path = path or SERVED_MODEL_CONFIG  # looked up at call time so tests can point it elsewhere
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise TryItError("The served-model configuration is missing or unreadable, so Try It "
                         "cannot choose a model.") from None
    for key in ("dataset", "n_classes", "experiment_group", "run_name"):
        if not cfg.get(key):
            raise TryItError(f"The served-model configuration has no '{key}'.")
    if cfg["dataset"] != "fashion-mnist" or cfg["n_classes"] != 10:
        raise TryItError("The served-model configuration does not name a 10-class Fashion-MNIST model.")
    return cfg


def resolve_served_run(config_path: Path | None = None, runs_jsonl: Path | None = None) -> dict:
    """Return {'run_id', 'run_name', 'group'} for the configured Fashion-MNIST model.

    The run ID is taken from ``served_run_id`` in the configuration and must match exactly one
    saved ``status: saved`` record of that run name in the group's experiment records. There is
    no fallback to any other run (in particular never to the historical synthetic-shape runs).
    """
    cfg = load_served_config(config_path)
    run_id = cfg.get("served_run_id")
    if run_id is None:
        raise TryItError("The Fashion-MNIST model has not been trained and configured yet, so Try It "
                         "is not available. (The earlier synthetic-shape model is deliberately not used.)")
    if not isinstance(run_id, int):
        raise TryItError("served_run_id in the served-model configuration must be an integer.")
    jsonl = runs_jsonl or runs_jsonl_for(cfg["experiment_group"])
    try:
        lines = jsonl.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise TryItError("The saved experiment records for the Fashion-MNIST group are missing, so "
                         "the configured model cannot be verified.") from None
    matches = []
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("run_name") == cfg["run_name"] and rec.get("status") == "saved":
            matches.append(rec.get("run_id"))
    if matches != [run_id]:
        raise TryItError(f"The configured run ID {run_id} does not match the saved records for "
                         f"{cfg['run_name']} ({len(matches)} record(s) found).")
    return {"run_id": run_id, "run_name": cfg["run_name"], "group": cfg["experiment_group"],
            "config": cfg}


def verify_served_run(api_url: str, served: dict, get=None) -> None:
    """Ask the API for the configured run and confirm it is the Fashion-MNIST model we expect.

    Raises TryItError if the API's run has a different name, group, dataset or class count. An
    unreachable API raises a different, retryable message; nothing is predicted or logged here.
    """
    get = get or requests.get  # looked up at call time
    try:
        resp = get(f"{api_url}/runs/{served['run_id']}", timeout=WARMUP_TIMEOUT * 3)
    except requests.exceptions.RequestException:
        raise TryItError("Couldn't reach the prediction service to verify the model. It may be "
                         "waking up; please check again in a moment.") from None
    if resp.status_code == 404:
        raise TryItError("The configured model does not exist on the prediction service.")
    if resp.status_code != 200:
        raise TryItError("Couldn't verify the configured model right now. Please try again shortly.")
    try:
        run = resp.json()
    except ValueError:
        raise TryItError("The prediction service returned an unreadable run record.") from None
    cfg = run.get("config") or {}
    if (run.get("run_name") != served["run_name"] or run.get("experiment_group") != served["group"]
            or cfg.get("dataset") != "fashion-mnist" or cfg.get("n_classes") != 10):
        raise TryItError("The configured run on the prediction service is not the expected "
                         "Fashion-MNIST model, so Try It is disabled instead of guessing.")


# ---------------------------------------------------------------------------
# Held-out example images (official Fashion-MNIST test partition)
# ---------------------------------------------------------------------------
def load_samples(samples_dir: Path | None = None) -> dict:
    """Manifest of bundled held-out examples, or {} if none are bundled."""
    samples_dir = samples_dir or SAMPLES_DIR
    try:
        m = json.loads((samples_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    items = [s for s in m.get("samples", []) if (samples_dir / s.get("file", "")).is_file()]
    return {**m, "samples": items} if items else {}


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
def wake_api(api_url: str, get=None, sleep=None) -> bool:
    """Ping /healthz (safe to repeat) until the service answers. Returns whether
    it came up. Prediction requests are NOT sent from here."""
    import time
    get = get or requests.get
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
        raise TryItError("The configured model isn't available on the prediction service "
                         "(its saved checkpoint wasn't found). Try again later.")
    if resp.status_code == 413:
        raise TryItError("That file is larger than 5 MB. Please use a smaller image.")
    if resp.status_code == 422:
        raise TryItError("The service couldn't use that input" +
                         (f" ({_detail(resp)})." if _detail(resp) else ".") +
                         " Please upload a valid PNG or JPG image.")
    if resp.status_code == 503 and "logging" in _detail(resp).lower():
        raise TryItError("Predictions are required to be logged, and logging is unavailable right "
                         "now, so nothing was classified. Please try again later.")
    if resp.status_code == 503 and "database" in _detail(resp).lower():
        raise TryItError("The database behind the prediction service is temporarily "
                         "unavailable, so nothing was classified or logged. Please try "
                         "again in a few minutes.")
    if resp.status_code in (502, 503, 504):
        raise TryItError("The prediction service is unavailable right now (it may be "
                         "waking up). Please wait a moment and try again.")
    raise TryItError(f"The prediction service reported an error (HTTP {resp.status_code}). "
                     "Please try again later.")


def predict_upload(api_url: str, run_id: int, filename: str, data: bytes,
                   content_type: str | None, post=None) -> dict:
    validate_upload(data)
    return _post_once(
        post or requests.post, f"{api_url}/predict",
        data={"run_id": int(run_id)},
        files={"file": (filename, data, content_type or "application/octet-stream")},
    )


def top5_rows(resp: dict) -> list[dict]:
    """The five most probable classes from a prediction response, best first.

    Uses the API's own ``top5`` (already ordered, from the served probabilities); returns []
    when the response carries no usable list. Probabilities are the temperature-scaled ones
    when the checkpoint has a temperature.
    """
    top = resp.get("top5")
    if not isinstance(top, list):
        return []
    try:
        rows = [{"Class": str(t["class_name"]), "Probability": float(t["prob"])} for t in top[:5]]
    except (KeyError, TypeError, ValueError):
        return []
    return rows


def decode_png(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception:  # noqa: BLE001
        raise TryItError("The service returned an image that couldn't be decoded.") from None
