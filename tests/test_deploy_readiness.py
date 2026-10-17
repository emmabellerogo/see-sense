"""Deployment-readiness checks: health modes, database outage, real uploads, tier isolation."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import re
import subprocess
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image
from postgrest.exceptions import APIError

from api import db

ROOT = Path(__file__).parent.parent


def _png(seed: int = 0) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (40, 40), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def run_id(client):
    ds = client.post("/datasets", json={"name": "d", "n_rows": 200, "noise": 0.1, "seed": 1}).json()
    r = client.post("/train", json={"dataset_id": ds["id"], "lr": 0.01, "batch_size": 32, "epochs": 1})
    assert r.status_code == 200
    return r.json()["run_id"]


# --- health: real Supabase vs offline vs outage ---------------------------------
def test_health_distinguishes_supabase_offline_and_outage(client, monkeypatch):
    assert client.get("/healthz").json() == {
        "status": "ok", "model_loader": True, "supabase": True, "storage": "supabase"}

    monkeypatch.setattr(db, "ping", lambda: False)                 # Supabase unreachable
    down = client.get("/healthz")
    assert down.status_code == 200                                  # host keeps routing here
    assert down.json()["status"] == "degraded" and down.json()["supabase"] is False

    monkeypatch.setattr(db, "ping", lambda: True)                   # offline store "pings" fine...
    monkeypatch.setattr(db, "LOCAL_DB_PATH", ".local_db.json")
    off = client.get("/healthz").json()
    assert off["status"] == "offline" and off["storage"] == "offline"
    assert off["supabase"] is False                                 # ...but is never reported as Supabase


# --- real image upload + Grad-CAM through /predict -------------------------------
def test_upload_returns_gradcam_and_logs_hash_only(client, run_id):
    raw = _png()
    r = client.post("/predict", data={"run_id": run_id}, files={"file": ("a.png", raw, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["true_class"] is None
    assert base64.b64decode(body["cam_png"]).startswith(b"\x89PNG")
    assert body["image_sha256"] == hashlib.sha256(raw).hexdigest()
    logged = client._store["image_metadata"][-1]
    assert logged["sha256"] == body["image_sha256"]
    assert set(logged) == {"id", "run_id", "sha256", "width", "height", "label", "class_name", "confidence"}


@pytest.mark.parametrize("payload", [b"", b"definitely not an image"])
def test_invalid_upload_is_422_and_logs_nothing(client, run_id, payload):
    r = client.post("/predict", data={"run_id": run_id}, files={"file": ("x.png", payload, "image/png")})
    assert r.status_code == 422
    assert client._store["image_metadata"] == []


def test_unknown_run_is_404(client):
    r = client.post("/predict", data={"run_id": 9999}, files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 404


# --- database outage -----------------------------------------------------------
@pytest.mark.parametrize("exc", [
    httpx.ConnectError("secret-host.supabase.co refused"),
    httpx.ReadTimeout("timed out"),
    APIError({"code": "PGRST002", "message": "schema cache", "details": None, "hint": None}),
])
def test_database_outage_is_503_without_leaks(client, run_id, monkeypatch, exc):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(db, "get_run_artifact", boom)
    r = client.post("/predict_sample", json={"run_id": run_id, "seed": 1})
    assert r.status_code == 503
    assert r.json() == {"detail": "database unavailable"}
    assert client._store["image_metadata"] == []


def test_other_database_errors_are_generic_500(client, run_id, monkeypatch):
    def boom(*a, **k):
        raise APIError({"code": "23505", "message": "duplicate key secret-detail", "details": None, "hint": None})
    monkeypatch.setattr(db, "insert_image_metadata", boom)
    r = client.post("/predict_sample", json={"run_id": run_id, "seed": 1})
    assert r.status_code == 500 and r.json() == {"detail": "database error"}


# --- tier isolation and secrets hygiene -----------------------------------------
def test_ui_tier_has_no_ml_or_server_dependencies():
    banned = {"torch", "numpy", "PIL", "api", "shared", "fastapi"}
    for f in (ROOT / "ui").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split(".")[0] in banned for n in names), (f.name, names)
    reqs = (ROOT / "ui" / "requirements.txt").read_text().lower()
    assert not re.search(r"^(torch|numpy|pillow|fastapi)\b", reqs, flags=re.M)
    assert re.search(r"^pyarrow==", reqs, flags=re.M)


def test_api_requirements_use_cpu_torch_index():
    reqs = (ROOT / "api" / "requirements.txt").read_text()
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in reqs
    assert re.search(r"^torch==2\.4\.1$", reqs, flags=re.M) and "numpy==1.26.4" in reqs


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_gitignore_covers_secrets_in_root_and_ui():
    def ignored(path):
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0
    for p in (".env", ".streamlit/secrets.toml", "ui/.streamlit/secrets.toml", ".local_db.json"):
        assert ignored(p), p
    for p in (".env.example", ".streamlit/secrets.toml.example", "ui/.streamlit/secrets.toml.example"):
        assert not ignored(p), p
