"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Responsibilities:
  * expose the CNN behind HTTP so the Streamlit UI can call it over HTTPS,
  * read dataset parameters from Supabase before training,
  * write run rows, model artifacts, and image metadata (hash only) back.

There is NO UI code here and NO business logic in the UI -- separation of
concerns across the three clouds. This is "See-Sense": image
classification with a small CNN + Grad-CAM explainability.
"""
from __future__ import annotations

import hashlib
import io
import os
import subprocess

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api import db
from api.training import predict_with_cam, run_experiment
from shared.data import CLASS_NAMES, IMG_SIZE, N_CLASSES, generate_one
from shared.schemas import (
    ClassesResponse,
    ClassMetrics,
    Dataset,
    DatasetCreate,
    ExperimentFields,
    Health,
    PredictResponse,
    PredictSampleRequest,
    Run,
    TrainRequest,
    TrainResponse,
    Version,
)

app = FastAPI(
    title="See-Sense API",
    description="CNN image classifier + Grad-CAM for the three-cloud stack.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _project_ref() -> str | None:
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


def _artifact_or_404(run_id: int) -> str:
    model_b64 = db.get_run_artifact(run_id)
    if model_b64 is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return model_b64


def _to_gray_array(raw: bytes, img_size: int) -> np.ndarray:
    """Decode arbitrary image bytes to a grayscale [img_size, img_size] uint8 array."""
    img = Image.open(io.BytesIO(raw)).convert("L").resize((img_size, img_size))
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@app.post("/datasets", response_model=Dataset, tags=["datasets"])
def create_dataset(req: DatasetCreate) -> Dataset:
    """Register a synthetic image dataset (parameters only; pixels come from seed)."""
    row = db.insert_dataset(
        name=req.name,
        n_rows=req.n_rows,
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        noise=req.noise,
        seed=req.seed,
    )
    return Dataset(**{k: row[k] for k in Dataset.model_fields})


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse, tags=["training"])
def train(req: TrainRequest) -> TrainResponse:
    """Train with the registered dataset's settings; save the run, metadata and checkpoint."""
    dataset = db.get_dataset(req.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="dataset_id not found")
    if (
        req.experiment_group
        and req.run_name
        and db.run_name_exists(req.experiment_group, req.run_name)
    ):
        raise HTTPException(
            status_code=409,
            detail=f"run_name {req.run_name!r} already exists in group {req.experiment_group!r}",
        )

    try:
        result = run_experiment(
            n_rows=dataset["n_rows"],
            img_size=dataset["img_size"],
            noise=dataset["noise"],
            data_seed=dataset["seed"],
            split_seed=req.split_seed,
            train_seed=req.train_seed,
            lr=req.lr,
            batch_size=req.batch_size,
            epochs=req.epochs,
            test_size=req.test_size,
            val_size=req.val_size,
            arch=req.model_variant,
            use_batchnorm=req.use_batchnorm,
            dropout=req.dropout,
            augment=req.augment,
            augment_max_shift=req.augment_max_shift,
            scheduler=req.scheduler,
        )
    except ValueError as exc:  # e.g. split fractions leave an empty partition
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        experiment = db.experiment_columns(result, req.experiment_group, req.run_name)
    except ValueError as exc:  # our own result failed its consistency checks
        raise HTTPException(status_code=500, detail=f"inconsistent experiment result: {exc}")

    metrics = result["metrics"]
    run = db.insert_run(
        dataset_id=req.dataset_id,
        lr=req.lr,
        batch_size=req.batch_size,
        epochs=req.epochs,
        metrics=metrics,
        model_b64=result["model_b64"],
        experiment=experiment,
    )
    return TrainResponse(
        run_id=run["id"],
        metrics=ClassMetrics(**metrics),
        **{k: run.get(k) for k in ExperimentFields.model_fields},
    )


def _run_from_row(row: dict) -> Run:
    # Rows saved before migration 002 have no (or NULL) experiment fields.
    return Run(**{k: row.get(k) for k in Run.model_fields})


@app.get("/runs/{run_id}", response_model=Run, tags=["training"])
def get_run(run_id: int) -> Run:
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return _run_from_row(row)


@app.get("/runs", response_model=list[Run], tags=["training"])
def list_runs() -> list[Run]:
    return [_run_from_row(r) for r in db.latest_runs(limit=50)]


# ---------------------------------------------------------------------------
# prediction  (Grad-CAM)
# ---------------------------------------------------------------------------
def _predict_array(run_id: int, gray: np.ndarray, sha256: str, true_class: str | None):
    model_b64 = _artifact_or_404(run_id)
    result = predict_with_cam(model_b64, gray)
    db.insert_image_metadata(
        run_id=run_id,
        sha256=sha256,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
        label=result["label"],
        class_name=result["class_name"],
        confidence=result["confidence"],
    )
    return PredictResponse(
        run_id=run_id, image_sha256=sha256, true_class=true_class, **result
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(run_id: int = Form(...), file: UploadFile = File(...)) -> PredictResponse:
    """Classify an uploaded image and return a Grad-CAM overlay.

    Only the image's sha256 hash + shape + prediction are logged -- never pixels.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="empty file")
    try:
        gray = _to_gray_array(raw, IMG_SIZE)
    except Exception:
        raise HTTPException(status_code=422, detail="could not decode image")
    sha256 = hashlib.sha256(raw).hexdigest()
    return _predict_array(run_id, gray, sha256, true_class=None)


@app.post("/predict_sample", response_model=PredictResponse, tags=["prediction"])
def predict_sample(req: PredictSampleRequest) -> PredictResponse:
    """Generate a synthetic image, classify it, and return a Grad-CAM overlay.

    Lets the UI demonstrate the model without the user uploading a file.
    """
    kind = req.true_class if (req.true_class is None or req.true_class < N_CLASSES) else None
    gray, label = generate_one(img_size=IMG_SIZE, noise=0.1, seed=req.seed, kind=kind)
    sha256 = hashlib.sha256(gray.tobytes()).hexdigest()
    return _predict_array(req.run_id, gray, sha256, true_class=CLASS_NAMES[label])


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
@app.get("/classes", response_model=ClassesResponse, tags=["meta"])
def classes() -> ClassesResponse:
    """Return the class labels the model predicts over."""
    return ClassesResponse(classes=CLASS_NAMES, n_classes=N_CLASSES)


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    supabase_ok = db.ping()
    model_ok = torch.tensor([1.0]).sum().item() == 1.0
    status = "ok" if (supabase_ok and model_ok) else "degraded"
    return Health(status=status, model_loader=model_ok, supabase=supabase_ok)


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        supabase_project_ref=_project_ref(),
    )
