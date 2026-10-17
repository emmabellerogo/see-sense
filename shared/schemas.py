"""Pydantic request/response models shared by the API and (optionally) the UI.

The contract between the three clouds. The "See-Sense" product is image
classification with a small CNN plus Grad-CAM explainability. The *shape* of the
contract is identical to the base template; only the fields change.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets."""

    name: str = Field(..., min_length=1, max_length=100)
    n_rows: int = Field(800, ge=100, le=20_000)
    noise: float = Field(0.1, ge=0.0, le=1.0, description="Pixel noise std.")
    seed: int = Field(42, description="Seed for the synthetic image generator.")


class Dataset(BaseModel):
    """A dataset row as stored in Supabase."""

    id: int
    name: str
    n_rows: int
    img_size: int
    n_classes: int
    created_at: datetime


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train.

    Everything after ``test_size`` is optional: an old request that only sends
    ``dataset_id``/``lr``/``batch_size``/``epochs`` trains the baseline CNN with
    no augmentation and no scheduler. The images always come from the registered
    dataset's own generation settings (rows, noise, seed).
    """

    dataset_id: int
    lr: float = Field(0.01, gt=0.0, description="Initial Adam learning rate.")
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(15, ge=1, le=500)
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Held-out test fraction.")
    val_size: float = Field(0.1, gt=0.0, lt=1.0, description="Validation fraction.")

    # Model variant and technique switches (independent of each other).
    model_variant: Literal["baseline", "improved"] = "baseline"
    use_batchnorm: bool = Field(True, description="Improved model only; ignored by baseline.")
    dropout: float = Field(0.3, ge=0.0, lt=1.0, description="Improved model only; ignored by baseline.")
    augment: bool = False
    augment_max_shift: int = Field(2, ge=0, le=8)
    scheduler: Literal["none", "cosine", "step"] = "none"

    # Seeds: images come from the dataset's seed; these control the rest.
    split_seed: int = Field(0, ge=0, le=2**32 - 1)
    train_seed: int = Field(0, ge=0, le=2**32 - 1)

    # Experiment bookkeeping (a run name is unique within its group).
    experiment_group: Optional[str] = Field(None, min_length=1, max_length=100)
    run_name: Optional[str] = Field(None, min_length=1, max_length=100)

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def _fractions_fit(self) -> "TrainRequest":
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be less than 1")
        return self


class ClassMetrics(BaseModel):
    accuracy: float
    macro_f1: float
    test_loss: Optional[float] = None


class ExperimentFields(BaseModel):
    """Experiment metadata stored on a run (columns added by migration 002).

    All optional: runs saved before the migration have no values for them.
    """

    experiment_group: Optional[str] = None
    run_name: Optional[str] = None
    model_variant: Optional[Literal["baseline", "improved"]] = None
    use_batchnorm: Optional[bool] = None
    dropout: Optional[float] = None
    augment: Optional[bool] = None
    augment_max_shift: Optional[int] = None
    scheduler: Optional[Literal["none", "cosine", "step"]] = None
    data_seed: Optional[int] = None
    split_seed: Optional[int] = None
    train_seed: Optional[int] = None
    split_sha256: Optional[str] = None
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    optimizer: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    test_loss: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = Field(
        None, description="Rows = true class, columns = predicted class (test set)."
    )
    train_loss_history: Optional[List[float]] = None
    val_loss_history: Optional[List[float]] = None
    val_accuracy_history: Optional[List[float]] = None
    lr_history: Optional[List[float]] = None
    best_epoch: Optional[int] = None
    param_count: Optional[int] = None
    train_seconds: Optional[float] = None
    artifact_version: Optional[int] = None

    model_config = ConfigDict(protected_namespaces=())


class Run(ExperimentFields):
    """A training-run row as stored in Supabase (metrics + experiment metadata)."""

    id: int
    dataset_id: int
    lr: float
    batch_size: int
    epochs: int
    accuracy: float
    macro_f1: float
    created_at: datetime


class TrainResponse(ExperimentFields):
    run_id: int
    metrics: ClassMetrics


# ---------------------------------------------------------------------------
# Prediction  (Grad-CAM explainability)
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    run_id: int
    label: int
    class_name: str
    confidence: float = Field(..., description="Softmax probability of the top class.")
    probs: List[float]
    image_sha256: str = Field(..., description="Hash of the input image (no pixels stored).")
    input_png: str = Field(..., description="base64 PNG of the (preprocessed) input.")
    cam_png: str = Field(..., description="base64 PNG of the Grad-CAM overlay.")
    true_class: Optional[str] = Field(
        None, description="Ground-truth class when predicting a generated sample."
    )


class PredictSampleRequest(BaseModel):
    """Request body for POST /predict_sample (no file upload needed)."""

    run_id: int
    seed: int = Field(0, description="Seed for the generated sample.")
    true_class: Optional[int] = Field(
        None, ge=0, description="Force a class index; None picks one at random."
    )


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------
class ClassesResponse(BaseModel):
    classes: List[str]
    n_classes: int


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
class Health(BaseModel):
    status: str  # "ok" | "degraded" (Supabase unreachable) | "offline" (local JSON store)
    model_loader: bool
    supabase: bool  # True ONLY when the real Supabase project answered a query
    storage: str = "supabase"  # "supabase" or "offline" (SEE_SENSE_LOCAL_DB is set)


class Version(BaseModel):
    git_sha: str
    torch_version: str
    supabase_project_ref: Optional[str]
