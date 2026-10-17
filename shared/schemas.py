"""Pydantic request/response models shared by the API and (optionally) the UI.

The contract between the three clouds. The "See-Sense" product is image
classification with a small CNN plus Grad-CAM explainability. The *shape* of the
contract is identical to the base template; only the fields change.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetCreate(BaseModel):
    """Request body for POST /datasets: register the local Fashion-MNIST files.

    The server reads the files from its own disk (training machine only), verifies them,
    and stores the dataset's identity (source, file hashes, class names). No pixels are stored.
    """

    name: str = Field("fashion-mnist", min_length=1, max_length=100)


class Dataset(BaseModel):
    """A dataset row as stored in Supabase."""

    id: int
    name: str
    n_rows: int
    img_size: int
    n_classes: int
    created_at: datetime
    # Added by migration 005. Rows from before it are historical synthetic-shape datasets.
    source: Optional[str] = Field(None, description="'fashion_mnist' or 'synthetic_shapes' (historical).")
    class_names: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainRequest(BaseModel):
    """Request body for POST /train.

    Everything after ``val_size`` is optional: a request that only sends
    ``dataset_id``/``lr``/``batch_size``/``epochs`` trains the baseline CNN with
    no augmentation and no scheduler. The images come from the registered Fashion-MNIST
    dataset's local files; the official test images are the test set, and validation
    images are carved out of the official training images with ``split_seed``.
    """

    dataset_id: int
    lr: float = Field(0.01, gt=0.0, description="Initial Adam learning rate.")
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(15, ge=1, le=500)
    val_size: float = Field(0.1, gt=0.0, lt=0.5, description="Fraction of the official TRAINING images held out for validation.")
    train_limit: Optional[int] = Field(None, ge=100, description="Keep only this many training images (a documented subset). None = all.")

    # Model variant and technique switches (independent of each other).
    model_variant: Literal["baseline", "improved"] = "baseline"
    use_batchnorm: bool = Field(True, description="Improved model only; ignored by baseline.")
    dropout: float = Field(0.3, ge=0.0, lt=1.0, description="Improved model only; ignored by baseline.")
    augment: bool = False
    augment_max_shift: int = Field(2, ge=0, le=8)
    scheduler: Literal["none", "cosine", "step"] = "none"

    # Seeds: the split seed picks the validation images; the train seed controls the rest.
    split_seed: int = Field(0, ge=0, le=2**32 - 1)
    train_seed: int = Field(0, ge=0, le=2**32 - 1)

    # Experiment bookkeeping (a run name is unique within its group).
    experiment_group: Optional[str] = Field(None, min_length=1, max_length=100)
    run_name: Optional[str] = Field(None, min_length=1, max_length=100)

    model_config = ConfigDict(protected_namespaces=())

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
    train_accuracy_history: Optional[List[float]] = None  # migration 004; NULL on older runs
    calibration: Optional[Dict[str, Any]] = Field(
        None, description="Temperature scaling fitted on validation data, with measured ECE/NLL (migration 005).")
    test_per_class: Optional[Dict[str, Any]] = Field(
        None, description="Per-class test precision / recall / F1 / support (migration 005).")

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
class TopKEntry(BaseModel):
    label: int
    class_name: str
    prob: float


class PredictResponse(BaseModel):
    run_id: int
    label: int
    class_name: str
    confidence: float = Field(..., description="Probability of the top class (temperature-scaled when the checkpoint has a temperature).")
    probs: List[float]
    top5: List[TopKEntry] = Field(..., description="The five most probable classes, best first.")
    class_names: List[str] = Field(..., description="Ordered labels; probs[i] belongs to class_names[i].")
    temperature: float = Field(1.0, description="Temperature applied to the logits (1.0 = plain softmax).")
    calibrated: bool = Field(False, description="True if the checkpoint carries a fitted temperature. Says nothing about how well calibrated it is; see the run's calibration measures.")
    image_sha256: str = Field(..., description="Hash of the input image (no pixels stored).")
    input_png: str = Field(..., description="base64 PNG of the (preprocessed) input.")
    cam_png: str = Field(..., description="base64 PNG of the Grad-CAM overlay.")
    true_class: Optional[str] = Field(None, description="Reserved; the API does not know ground truth for uploads.")
    logged: bool = Field(..., description="True only if the prediction and image metadata were both written.")
    prediction_id: Optional[int] = Field(None, description="Row id in `predictions`.")


class PredictionRunInfo(BaseModel):
    """The run that served a prediction (joined from `runs`)."""

    run_name: Optional[str] = None
    experiment_group: Optional[str] = None
    model_variant: Optional[str] = None
    accuracy: Optional[float] = Field(None, description="That run's final test accuracy.")
    macro_f1: Optional[float] = None

    model_config = ConfigDict(protected_namespaces=())


class PredictionRecord(BaseModel):
    """One row of GET /predictions: a logged prediction joined to its serving run.

    The image hash is deliberately NOT part of this public record.
    """

    id: int
    predicted_label: int
    predicted_class: str
    probabilities: List[float]
    confidence: float
    served_by_run_id: int
    created_at: datetime
    run: Optional[PredictionRunInfo] = Field(
        None, description="Joined from `runs`; None only if the run row is gone."
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
