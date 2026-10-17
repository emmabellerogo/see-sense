"""Small CNN + Grad-CAM for Fashion-MNIST classification.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. It trains a compact
convolutional network on Fashion-MNIST (10 clothing classes, 28x28 grayscale) and,
at predict time, produces a **Grad-CAM** heatmap so a non-technical client can see
*where* the network looked.

Training data comes from ``shared.fashion`` (local files; no downloads here). The
official test images are touched ONLY by ``run_experiment``, once, after the
checkpoint has been chosen on validation data. ``train_validate`` (used by the
runtime pilot) cannot see them at all.

The fitted model is pickled + base64-encoded so it can be persisted to Supabase and
reloaded per request (survives Render free-tier restarts). Checkpoint format 3
stores everything needed to use it without guessing: ordered class names,
preprocessing, architecture, dataset identity and temperature-scaling parameters.
Formats 1 and 2 (the earlier four-class synthetic-shape models) still load.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn

from api import calibration
from api.experiments import metrics_from_confusion
from shared import fashion
from shared.fashion import FashionData

# Format of the checkpoint stored in run_artifacts.model_b64.
# 1 = original (state_dict, img_size, n_classes); 2 = adds arch, use_batchnorm, dropout
# (both: four synthetic shapes, inputs scaled to [0, 1]);
# 3 = adds class_names, preprocessing, dataset identity, calibration, selected epoch.
ARTIFACT_VERSION = 3

# Class names of the retired synthetic-shape dataset, used only to read format 1-2 checkpoints.
LEGACY_SYNTHETIC_CLASS_NAMES = ["horizontal", "vertical", "diagonal", "block"]


class SmallCNN(nn.Module):
    """Two conv blocks -> linear classifier. `features` is the Grad-CAM target."""

    def __init__(self, img_size: int, n_classes: int, channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # img_size / 2
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # img_size / 4
        )
        feat = img_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * feat * feat, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """SmallCNN's layout plus optional BatchNorm and dropout, each toggled on its own.

    With ``use_batchnorm=False`` and ``dropout=0.0`` this is the same network as
    SmallCNN (same layer shapes). ``features`` is still the Grad-CAM target.
    """

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        channels: int = 1,
        use_batchnorm: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        def block(c_in: int, c_out: int) -> List[nn.Module]:
            layers: List[nn.Module] = [nn.Conv2d(c_in, c_out, 3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(c_out))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            return layers

        self.features = nn.Sequential(*block(channels, 8), *block(8, 16))
        feat = img_size // 4
        head: List[nn.Module] = [nn.Flatten()]
        if dropout > 0.0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(16 * feat * feat, n_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(
    img_size: int,
    n_classes: int,
    arch: str = "baseline",
    use_batchnorm: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Create the baseline SmallCNN or the ImprovedCNN."""
    if arch == "baseline":
        return SmallCNN(img_size, n_classes)
    if arch == "improved":
        return ImprovedCNN(img_size, n_classes, use_batchnorm=use_batchnorm, dropout=dropout)
    raise ValueError(f"unknown arch: {arch!r}")


# ---------------------------------------------------------------------------
# preprocessing  (one definition for training, evaluation and serving)
# ---------------------------------------------------------------------------
def default_preprocessing() -> Dict[str, Any]:
    return {
        "mode": "L", "size": [fashion.IMG_SIZE, fashion.IMG_SIZE], "resize": "bicubic (PIL default)",
        "scale": 255.0, "mean": fashion.NORM_MEAN, "std": fashion.NORM_STD,
    }


def to01(images_uint8: np.ndarray) -> torch.Tensor:
    """[N, H, W] uint8 -> [N, 1, H, W] float32 in [0, 1]."""
    return torch.from_numpy(images_uint8.astype(np.float32) / 255.0).unsqueeze(1)


def normalize(x01: torch.Tensor, pre: Dict[str, Any]) -> torch.Tensor:
    return (x01 - float(pre["mean"])) / float(pre["std"])


def make_splits(
    n: int, test_size: float = 0.2, val_size: float = 0.1, split_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle ``range(n)`` with ``split_seed`` and cut it into (train, val, test) indices.

    Used by tests and tools that work on a single array; Fashion-MNIST training uses the
    official test partition and ``shared.fashion.split_train_val`` instead.
    """
    n_test, n_val = int(n * test_size), int(n * val_size)
    if n_val < 1 or n_test < 1 or n_test + n_val >= n:
        raise ValueError("test_size/val_size leave an empty train, validation or test split")
    idx = np.random.default_rng(split_seed).permutation(n)
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def _augment(xb: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    """Label-preserving augmentation for a [N, 1, H, W] batch of [0, 1] pixels.

    Random shift of up to ``max_shift`` pixels (zero = black background, no wrap-around)
    plus a random HORIZONTAL flip. Clothing and footwear keep their class when mirrored
    left-right; a vertical flip would turn a dress or a boot upside down (not a realistic
    garment), so there is none. Shift offsets and flip decisions are always drawn so the
    random stream does not depend on which images are flipped.
    """
    n, _, h, w = xb.shape
    padded = torch.nn.functional.pad(xb, (max_shift,) * 4)
    dy = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen)
    dx = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen)
    flip = torch.rand(n, generator=gen) < 0.5
    rows = dy[:, None] + torch.arange(h)[None, :]                              # [n, h]
    cols = dx[:, None] + torch.arange(w)[None, :]                              # [n, w]
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    out = padded[torch.arange(n)[:, None, None], 0, rows[:, :, None], cols[:, None, :]]
    return out.unsqueeze(1)


def _make_scheduler(optimizer, name: str, epochs: int):
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    raise ValueError(f"unknown scheduler: {name!r}")


def _logits(model: nn.Module, x01: torch.Tensor, pre: Dict[str, Any], chunk: int = 4096) -> torch.Tensor:
    """Raw class scores in eval mode, without gradients, in chunks."""
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, x01.shape[0], chunk):
            outs.append(model(normalize(x01[s : s + chunk], pre)))
    return torch.cat(outs)


def _loss_acc(logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    return (
        float(nn.functional.cross_entropy(logits, y).item()),
        float((logits.argmax(dim=1) == y).float().mean().item()),
    )


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[List[int]]:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm.tolist()


# ---------------------------------------------------------------------------
# training  (validation only)
# ---------------------------------------------------------------------------
def train_validate(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *,
    n_classes: int, img_size: int, lr: float, batch_size: int, epochs: int, train_seed: int,
    arch: str = "baseline", use_batchnorm: bool = True, dropout: float = 0.3,
    augment: bool = False, augment_max_shift: int = 2, scheduler: str = "none",
    preprocessing: Optional[Dict[str, Any]] = None,
    on_epoch: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Train on ``x_train`` and select the epoch with the lowest ``x_val`` loss.

    Takes NO test data, so it cannot leak it. Returns the model (best-epoch weights),
    per-epoch histories and the validation logits of the selected epoch.

    Recorded every epoch: ``train_loss`` (mean mini-batch loss in training mode, i.e. on
    augmented data with dropout/batch-norm batch statistics), ``train_accuracy`` (the whole
    un-augmented training split scored in eval mode), ``val_loss`` and ``val_accuracy``.
    """
    pre = preprocessing or default_preprocessing()
    if arch == "baseline":
        # SmallCNN has no BatchNorm or dropout and ignores these options, so record
        # what the model really uses rather than what the caller passed.
        use_batchnorm, dropout = False, 0.0
    xt, xv = to01(x_train), to01(x_val)
    yt, yv = torch.from_numpy(np.asarray(y_train, dtype=np.int64)), torch.from_numpy(np.asarray(y_val, dtype=np.int64))

    torch.manual_seed(train_seed)  # initial weights + dropout masks
    model = build_model(img_size, n_classes, arch, use_batchnorm, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _make_scheduler(optimizer, scheduler, epochs)
    loss_fn = nn.CrossEntropyLoss()
    # Separate generators so turning augmentation on/off never changes batch order.
    shuffle_gen = torch.Generator().manual_seed(train_seed)
    aug_gen = torch.Generator().manual_seed(train_seed + 1)

    n = xt.shape[0]
    hist: Dict[str, List[float]] = {k: [] for k in
                                    ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr_history", "epoch_seconds")}
    best: Dict[str, Any] = {"val_loss": float("inf"), "epoch": -1, "state": None}

    t0 = time.perf_counter()
    for epoch in range(epochs):
        e0 = time.perf_counter()
        hist["lr_history"].append(optimizer.param_groups[0]["lr"])
        model.train()
        perm = torch.randperm(n, generator=shuffle_gen)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            xb, yb = xt[b], yt[b]
            if augment:
                xb = _augment(xb, augment_max_shift, aug_gen)
            optimizer.zero_grad()
            loss = loss_fn(model(normalize(xb, pre)), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        if sched is not None:
            sched.step()
        hist["train_loss"].append(epoch_loss / n)
        # Eval-mode accuracy on the un-augmented training split: draws no random numbers
        # and changes no weights, so it does not affect any other result.
        hist["train_accuracy"].append(_loss_acc(_logits(model, xt, pre), yt)[1])
        v_loss, v_acc = _loss_acc(_logits(model, xv, pre), yv)
        hist["val_loss"].append(v_loss)
        hist["val_accuracy"].append(v_acc)
        if v_loss < best["val_loss"]:
            best = {"val_loss": v_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        hist["epoch_seconds"].append(time.perf_counter() - e0)
        if on_epoch is not None:
            on_epoch(epoch, {k: v[-1] for k, v in hist.items()})
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best["state"])  # model selection uses validation only
    return {
        "model": model, "preprocessing": pre, "best_epoch": best["epoch"],
        "val_logits": _logits(model, xv, pre).numpy(), "train_seconds": train_seconds,
        "arch": arch, "use_batchnorm": use_batchnorm, "dropout": dropout,
        "param_count": sum(p.numel() for p in model.parameters()),
        **hist,
    }


def run_experiment(
    split_seed: int,
    train_seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    val_size: float = 0.1,
    arch: str = "baseline",
    use_batchnorm: bool = True,
    dropout: float = 0.3,
    augment: bool = False,
    augment_max_shift: int = 2,
    scheduler: str = "none",
    train_limit: Optional[int] = None,
    data: Optional[FashionData] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Train one model on Fashion-MNIST and return everything we want to persist and compare.

    Partitions: the 60,000 official training images are split into train / validation by
    ``split_seed`` (class-stratified; ``val_size`` is a fraction, 0.1 = 6,000 images); the
    10,000 official test images are the test set. ``train_limit`` optionally keeps only that
    many training images (a documented subset; ``None`` = all of them).

    Two seeds: ``split_seed`` (validation selection) and ``train_seed`` (initial weights,
    batch order, augmentation, dropout). Optimizer (Adam), epochs, batch size and initial LR are
    the caller's choice and are identical for every compared run unless an ablation declares
    otherwise. The validation split picks the epoch whose weights we keep (lowest validation
    loss) and fits the calibration temperature; the test split is evaluated once, at the end.
    """
    data = data if data is not None else fashion.load(data_dir)
    if data.x_test is None or data.y_test is None:
        raise ValueError("run_experiment needs the official test partition")
    n_val = int(round(len(data.y_train) * val_size))
    train_idx, val_idx = fashion.split_train_val(data.y_train, n_val, split_seed)
    if train_limit is not None:
        if not 1 <= train_limit <= len(train_idx):
            raise ValueError("train_limit must be between 1 and the training split size")
        keep = np.sort(np.random.default_rng(split_seed + 1).permutation(len(train_idx))[:train_limit])
        train_idx = train_idx[keep]
    pre = default_preprocessing()
    n_classes, img_size = fashion.N_CLASSES, fashion.IMG_SIZE

    out = train_validate(
        data.x_train[train_idx], data.y_train[train_idx], data.x_train[val_idx], data.y_train[val_idx],
        n_classes=n_classes, img_size=img_size, lr=lr, batch_size=batch_size, epochs=epochs,
        train_seed=train_seed, arch=arch, use_batchnorm=use_batchnorm, dropout=dropout,
        augment=augment, augment_max_shift=augment_max_shift, scheduler=scheduler, preprocessing=pre,
    )
    model = out["model"]
    use_batchnorm, dropout = out["use_batchnorm"], out["dropout"]

    # ---- the one and only look at the official test set --------------------------------
    y_val = data.y_train[val_idx].astype(np.int64)
    y_test = data.y_test.astype(np.int64)
    test_logits = _logits(model, to01(data.x_test), pre).numpy()
    preds = test_logits.argmax(axis=1)
    cm = _confusion_matrix(y_test, preds, n_classes)
    per_class = metrics_from_confusion(cm)
    test_loss = float(nn.functional.cross_entropy(torch.from_numpy(test_logits), torch.from_numpy(y_test)).item())
    metrics = {
        "accuracy": float((preds == y_test).mean()),
        "macro_f1": per_class["macro_f1"],
        "test_loss": test_loss,
    }
    calib = calibration.calibration_report(out["val_logits"], y_val, test_logits, y_test)

    identity = fashion.dataset_identity(data, train_idx, val_idx, split_seed, n_val, train_limit)
    split_hash = fashion.index_sha256(train_idx, val_idx)
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "state_dict": model.state_dict(),
        "img_size": img_size,
        "n_classes": n_classes,
        "channels": 1,
        "arch": arch,
        "use_batchnorm": use_batchnorm,
        "dropout": dropout,
        "class_names": list(fashion.CLASS_NAMES),
        "preprocessing": pre,
        "dataset": identity,
        "calibration": {k: calib[k] for k in ("method", "temperature", "fit_split", "fit_n")},
        "selected_epoch": out["best_epoch"],
        "train_seed": train_seed,
        "torch_version": torch.__version__,
    }
    return {
        "artifact_version": ARTIFACT_VERSION,
        "metrics": metrics,
        "confusion_matrix": cm,
        "per_class": per_class,
        "calibration": calib,
        "train_loss": out["train_loss"],
        "val_loss": out["val_loss"],
        "val_accuracy": out["val_accuracy"],
        "train_accuracy": out["train_accuracy"],
        "lr_history": out["lr_history"],
        "epoch_seconds": out["epoch_seconds"],
        "best_epoch": out["best_epoch"],
        "param_count": out["param_count"],
        "train_seconds": out["train_seconds"],
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(y_test)},
        "split_sha256": split_hash,
        "dataset": identity,
        "config": {
            "dataset": fashion.DATASET_NAME, "n_classes": n_classes, "img_size": img_size,
            "class_names": list(fashion.CLASS_NAMES),
            "split_seed": split_seed, "train_seed": train_seed,
            "optimizer": "adam", "lr": lr, "batch_size": batch_size, "epochs": epochs,
            "val_size": val_size, "train_limit": train_limit,
            "arch": arch, "use_batchnorm": use_batchnorm, "dropout": dropout,
            "augment": augment, "augment_max_shift": augment_max_shift, "scheduler": scheduler,
            "preprocessing": pre,
        },
        "model_b64": base64.b64encode(pickle.dumps(artifact)).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# loading checkpoints
# ---------------------------------------------------------------------------
@dataclass
class LoadedModel:
    model: nn.Module
    img_size: int
    class_names: List[str]
    preprocessing: Dict[str, Any]
    calibration: Optional[Dict[str, Any]]
    artifact_version: int
    dataset: Dict[str, Any]

    @property
    def temperature(self) -> float:
        return float(self.calibration["temperature"]) if self.calibration else 1.0


def load_artifact(model_b64: str) -> LoadedModel:
    """Rebuild a model and its metadata from a stored checkpoint (formats 1-3)."""
    obj = pickle.loads(base64.b64decode(model_b64))
    version = int(obj.get("artifact_version", 1))
    # Artifacts saved before ImprovedCNN existed have no "arch" key -> baseline.
    model = build_model(
        obj["img_size"], obj["n_classes"], obj.get("arch", "baseline"),
        obj.get("use_batchnorm", True), obj.get("dropout", 0.3),
    )
    model.load_state_dict(obj["state_dict"])
    model.eval()
    if version >= 3:
        names = list(obj["class_names"])
        pre, calib, dataset = obj["preprocessing"], obj.get("calibration"), obj.get("dataset", {})
    else:  # historical four-class synthetic-shape checkpoints: [0, 1] inputs, no calibration
        names = LEGACY_SYNTHETIC_CLASS_NAMES if obj["n_classes"] == 4 else [str(i) for i in range(obj["n_classes"])]
        pre = {"mode": "L", "size": [obj["img_size"]] * 2, "scale": 255.0, "mean": 0.0, "std": 1.0}
        calib, dataset = None, {"name": "synthetic-shapes", "source": "synthetic_shapes", "legacy": True}
    if len(names) != obj["n_classes"]:
        raise ValueError("checkpoint class_names do not match n_classes")
    return LoadedModel(model, obj["img_size"], names, pre, calib, version, dataset)


def describe_artifact(model_b64: str) -> Dict[str, Any]:
    """Small, JSON-safe summary of a checkpoint (what it is, never its weights)."""
    m = load_artifact(model_b64)
    return {
        "artifact_version": m.artifact_version, "n_classes": len(m.class_names),
        "class_names": m.class_names, "dataset": m.dataset.get("name"),
        "dataset_source": m.dataset.get("source"), "preprocessing": m.preprocessing,
        "temperature": m.temperature, "calibrated": m.calibration is not None,
        "param_count": sum(p.numel() for p in m.model.parameters()),
    }


def _grad_cam(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Return a Grad-CAM map in [0, 1] at the conv feature resolution.

    Uses the gradient of the RAW class logit (before softmax and before any temperature
    scaling), so calibration never changes the heatmap.
    """
    captured: Dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, out):
        captured["act"] = out

    def bwd_hook(_m, _gi, gout):
        captured["grad"] = gout[0]

    h1 = model.features.register_forward_hook(fwd_hook)
    h2 = model.features.register_full_backward_hook(bwd_hook)
    try:
        logits = model(x)
        model.zero_grad()
        logits[0, class_idx].backward()
    finally:
        h1.remove()
        h2.remove()

    act = captured["act"][0]           # [C, h, w]
    grad = captured["grad"][0]          # [C, h, w]
    weights = grad.mean(dim=(1, 2))     # [C]
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy()


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay_png(gray_uint8: np.ndarray, cam01: np.ndarray) -> str:
    """Blend a red Grad-CAM heatmap over the grayscale input; return base64 PNG."""
    h, w = gray_uint8.shape
    cam_img = Image.fromarray((cam01 * 255).astype(np.uint8)).resize((w, h))
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0
    base = np.stack([gray_uint8] * 3, axis=-1).astype(np.float32)
    heat = np.zeros_like(base)
    heat[..., 0] = cam_arr * 255.0  # red channel
    blended = np.clip(0.55 * base + 0.45 * heat, 0, 255).astype(np.uint8)
    return _png_b64(Image.fromarray(blended, "RGB"))


def cam_for_class(loaded: LoadedModel, gray_uint8: np.ndarray, class_idx: int) -> np.ndarray:
    """Grad-CAM map (0..1, feature resolution) of ``class_idx`` for one preprocessed image."""
    x = normalize(to01(gray_uint8[None, :, :]), loaded.preprocessing)
    return _grad_cam(loaded.model, x, class_idx)


def predict_with_cam(model_b64: str, gray_uint8: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
    """Classify one preprocessed grayscale image and build a Grad-CAM overlay.

    ``gray_uint8`` must be [img_size, img_size] uint8. Returns a dict matching the
    PredictResponse fields (minus run_id / true_class, which main.py fills in).
    ``probs`` / ``confidence`` / ``top5`` are the temperature-scaled probabilities when the
    checkpoint carries a temperature (format 3), else plain softmax.
    """
    loaded = load_artifact(model_b64)
    x = normalize(to01(gray_uint8[None, :, :]), loaded.preprocessing)  # [1, 1, H, W]

    with torch.no_grad():
        logits = loaded.model(x)[0].double().numpy()
    probs = calibration.softmax(logits[None, :], loaded.temperature)[0]
    label = int(probs.argmax())
    order = np.argsort(-probs, kind="stable")[:top_k]
    cam = _grad_cam(loaded.model, x, label)
    return {
        "label": label,
        "class_name": loaded.class_names[label],
        "confidence": float(probs[label]),
        "probs": [float(p) for p in probs],
        "top5": [{"label": int(i), "class_name": loaded.class_names[int(i)], "prob": float(probs[i])} for i in order],
        "class_names": loaded.class_names,
        "temperature": loaded.temperature,
        "calibrated": loaded.calibration is not None,
        "input_png": _png_b64(Image.fromarray(gray_uint8, "L")),
        "cam_png": _overlay_png(gray_uint8, cam),
    }
