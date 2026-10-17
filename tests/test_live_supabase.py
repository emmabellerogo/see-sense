"""LIVE Supabase checks: prediction-to-database linkage against the real project.

These are NOT mocked and they WRITE to the real project (one dataset row, one tiny training run, one
logged prediction). They are skipped unless explicitly requested:

    SEE_SENSE_RUN_LIVE_TESTS=1 pytest tests/test_live_supabase.py     # credentials from .env / environment

Requirements: migrations 001-005 applied (see db/migrations/*.md), and the official Fashion-MNIST files
downloaded locally (python tools/fetch_fashion_mnist.py). The training run is deliberately tiny
(500 training images, 1 epoch) and named so it is easy to find and delete afterwards.

Everything else in tests/ uses an in-memory fake database; only a passing run of this file shows that the
deployed schema, the `log_prediction` function and the anon views really behave as designed.
"""
from __future__ import annotations

import hashlib
import io
import os

import numpy as np
import pytest
from PIL import Image

from shared import fashion
from tests.conftest import has_supabase_creds

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not has_supabase_creds(),
        reason="Live tests write to the real Supabase project; set SEE_SENSE_RUN_LIVE_TESTS=1 "
               "(and SUPABASE_URL / SUPABASE_SERVICE_KEY) to run them.",
    ),
    pytest.mark.skipif(
        not (fashion.data_dir() / "train-images-idx3-ubyte.gz").is_file(),
        reason="official Fashion-MNIST files not downloaded (python tools/fetch_fashion_mnist.py)",
    ),
]


@pytest.fixture(scope="module")
def live_run():
    from fastapi.testclient import TestClient

    from api.main import app

    assert not os.environ.get("SEE_SENSE_LOCAL_DB"), "unset SEE_SENSE_LOCAL_DB: this test must hit Supabase"
    with TestClient(app) as c:
        assert c.get("/healthz").json()["supabase"] is True
        ds = c.post("/datasets", json={"name": "live-test-fashion-mnist"})
        assert ds.status_code == 200, ds.text
        r = c.post("/train", json={"dataset_id": ds.json()["id"], "lr": 0.001, "batch_size": 128, "epochs": 1,
                                   "train_limit": 500, "model_variant": "improved",
                                   "experiment_group": "live-test", "run_name": f"live-{os.getpid()}"})
        assert r.status_code == 200, r.text
        yield c, r.json()["run_id"], ds.json()["id"]


def test_run_is_stored_with_fashion_identity_and_calibration(live_run):
    c, run_id, ds_id = live_run
    run = c.get(f"/runs/{run_id}").json()
    assert run["dataset_id"] == ds_id and run["config"]["dataset"] == "fashion-mnist"
    assert run["calibration"]["temperature"] > 0 and len(run["test_per_class"]["precision"]) == 10
    assert len(run["train_accuracy_history"]) == 1


def test_prediction_is_written_with_its_run_and_returned_by_get_predictions(live_run):
    c, run_id, _ = live_run
    img = fashion.load(include_test=True).x_test[0]
    buf = io.BytesIO()
    Image.fromarray(img, "L").save(buf, format="PNG")
    raw = buf.getvalue()
    r = c.post("/predict", data={"run_id": run_id}, files={"file": ("live.png", raw, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["logged"] is True and body["image_sha256"] == hashlib.sha256(raw).hexdigest()

    listed = c.get("/predictions?limit=50").json()
    mine = next(p for p in listed if p["id"] == body["prediction_id"])
    assert mine["served_by_run_id"] == run_id and mine["run"]["run_name"].startswith("live-")
    assert np.allclose(mine["probabilities"], body["probs"], atol=1e-9)
    assert "image_sha256" not in mine
    drill = c.get(f"/predictions/{body['prediction_id']}").json()
    assert c.get(f"/runs/{drill['served_by_run_id']}").json()["id"] == run_id


def test_both_rows_exist_in_the_database_and_hold_no_pixels(live_run):
    from api import db

    c, run_id, _ = live_run
    img = (np.arange(784).reshape(28, 28) % 256).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, "L").save(buf, format="PNG")
    raw = buf.getvalue()
    pid = c.post("/predict", data={"run_id": run_id}, files={"file": ("live2.png", raw, "image/png")}).json()["prediction_id"]
    sha = hashlib.sha256(raw).hexdigest()
    cl = db.get_client()
    pred = cl.table("predictions").select("*").eq("id", pid).execute().data
    meta = cl.table("image_metadata").select("*").eq("sha256", sha).eq("run_id", run_id).execute().data
    assert len(pred) == 1 and len(meta) >= 1
    assert pred[0]["image_sha256"] == sha and pred[0]["served_by_run_id"] == run_id
    assert not any(isinstance(v, (bytes, bytearray)) for v in {**pred[0], **meta[0]}.values())
    assert set(pred[0]) == {"id", "image_sha256", "predicted_label", "predicted_class", "probabilities",
                            "confidence", "served_by_run_id", "created_at"}


def test_the_atomic_function_rejects_an_unknown_run_and_leaves_no_orphan_rows(live_run):
    from postgrest.exceptions import APIError

    from api import db

    cl = db.get_client()
    sha = "live-atomicity-" + hashlib.sha256(os.urandom(8)).hexdigest()[:16]
    with pytest.raises(APIError):
        db.log_prediction(-1, sha, 28, 28, 0, "T-shirt/top", [0.5, 0.5], 0.5)
    assert cl.table("image_metadata").select("id").eq("sha256", sha).execute().data == []
    assert cl.table("predictions").select("id").eq("image_sha256", sha).execute().data == []


def test_anon_key_reads_the_hash_free_view_but_not_the_tables():
    """Needs no fixture: reads only (rows from earlier live runs / real use must already exist)."""
    from postgrest.exceptions import APIError
    from supabase import create_client

    url, anon = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_ANON_KEY")
    if not anon:
        pytest.skip("set SUPABASE_ANON_KEY to check the anon grants live")
    cl = create_client(url, anon)
    rows = cl.table("recent_predictions").select("*").limit(5).execute().data
    assert rows and "image_sha256" not in rows[0] and "sha256" not in rows[0]
    for table in ("predictions", "image_metadata", "run_artifacts"):
        # No grant (42501 permission denied) or RLS with no policy (empty result) both mean "no access".
        try:
            assert cl.table(table).select("*").limit(1).execute().data == []
        except APIError as e:
            assert e.code == "42501", e


def test_existing_datasets_are_labelled_by_source_and_never_relabelled():
    """Read-only: migration 005 must mark every pre-existing dataset as the historical synthetic one."""
    from api import db

    rows = db.get_client().table("datasets").select("id,name,source,class_names").order("id").execute().data
    assert rows
    assert {r["source"] for r in rows} <= {"synthetic_shapes", "fashion_mnist"}
    old = [r for r in rows if not (r["name"] or "").lower().replace("_", "-").startswith(("fashion", "live-test"))]
    assert old and all(r["source"] == "synthetic_shapes" for r in old), old
    assert all(len(r["class_names"] or []) in (0, 4) for r in old)  # synthetic shapes: 4 classes, never 10
