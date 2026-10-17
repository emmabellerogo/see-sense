"""Deployment-readiness checks: health modes, tier isolation, secrets hygiene.

Uploads, logging and database-outage behaviour are covered in tests/test_api_predict.py."""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from api import db

ROOT = Path(__file__).parent.parent


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
