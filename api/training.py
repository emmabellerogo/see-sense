"""Small CNN + Grad-CAM for image classification.

This is the ONLY place model code lives. It is pulled in by the FastAPI service
on Render.com; it is never imported by Streamlit. It trains a compact
convolutional network on the synthetic image dataset and, at predict time,
produces a **Grad-CAM** heatmap so a non-technical client can see *where* the
network looked.

The fitted model is pickled + base64-encoded so it can be persisted to Supabase
and reloaded per request (survives Render free-tier restarts).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn

from shared.data import CLASS_NAMES, N_CLASSES, generate_images


# Format of the checkpoint stored in run_artifacts.model_b64.
# 1 = original (state_dict, img_size, n_classes); 2 = adds arch, use_batchnorm, dropout.
ARTIFACT_VERSION = 2


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


def _to_tensor(images_uint8: np.ndarray) -> torch.Tensor:
    """[N, H, W] uint8 -> [N, 1, H, W] float32 in [0, 1]."""
    x = images_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(1)


def make_splits(
    n: int, test_size: float = 0.2, val_size: float = 0.1, split_seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle ``range(n)`` with ``split_seed`` and cut it into (train, val, test) indices.

    The result depends only on ``(n, test_size, val_size, split_seed)``, so two
    models given the same arguments always see exactly the same examples.
    """
    n_test, n_val = int(n * test_size), int(n * val_size)
    if n_val < 1 or n_test < 1 or n_test + n_val >= n:
        raise ValueError("test_size/val_size leave an empty train, validation or test split")
    idx = np.random.default_rng(split_seed).permutation(n)
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def _augment(xb: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    """Label-preserving augmentation for a [N, 1, H, W] batch.

    Random shift of up to ``max_shift`` pixels (zero padded, no wrap-around) plus
    random horizontal / vertical flips. None of these turns a horizontal bar into
    a vertical one (unlike a 90-degree rotation). A flipped diagonal keeps its
    "diagonal" label but runs the other way (top-right to bottom-left instead of
    top-left to bottom-right): the generator only draws the second kind, so flips
    add training examples beyond its original orientation. All random numbers are always drawn so the stream is stable.
    """
    n, _, h, w = xb.shape
    padded = torch.nn.functional.pad(xb, (max_shift,) * 4)
    dy = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen)
    dx = torch.randint(0, 2 * max_shift + 1, (n,), generator=gen)
    flip_h = torch.rand(n, generator=gen) < 0.5
    flip_v = torch.rand(n, generator=gen) < 0.5
    out = torch.empty_like(xb)
    for i in range(n):
        a = padded[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w]
        if flip_h[i]:
            a = a.flip(2)
        if flip_v[i]:
            a = a.flip(1)
        out[i] = a
    return out


def _make_scheduler(optimizer, name: str, epochs: int):
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    raise ValueError(f"unknown scheduler: {name!r}")


def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, np.ndarray]:
    """Mean cross-entropy and predicted classes, in eval mode, without gradients."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y).item()
    return float(loss), logits.argmax(dim=1).numpy()


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> List[List[int]]:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm.tolist()


def run_experiment(
    n_rows: int,
    img_size: int,
    noise: float,
    data_seed: int,
    split_seed: int,
    train_seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
    val_size: float = 0.1,
    arch: str = "baseline",
    use_batchnorm: bool = True,
    dropout: float = 0.3,
    augment: bool = False,
    augment_max_shift: int = 2,
    scheduler: str = "none",
) -> Dict[str, object]:
    """Train one model and return everything we want to persist and compare.

    Three independent seeds:
      * ``data_seed``  -> which images are generated,
      * ``split_seed`` -> which images go to train / validation / test,
      * ``train_seed`` -> initial weights, batch order, augmentation, dropout.
    Optimizer (Adam), epochs, batch size and initial LR are the caller's choice and
    are the same for every compared run unless an ablation declares otherwise.

    Augmentation touches the training batches only. The validation split picks the
    epoch whose weights we keep (lowest validation loss); the test split is
    evaluated once, at the end, with those weights.
    """
    if arch == "baseline":
        # SmallCNN has no BatchNorm or dropout and ignores these options, so record
        # what the model really uses rather than what the caller passed.
        use_batchnorm, dropout = False, 0.0
    images, labels = generate_images(n_rows, img_size=img_size, noise=noise, seed=data_seed)
    y = np.asarray(labels, dtype=np.int64)
    train_idx, val_idx, test_idx = make_splits(len(images), test_size, val_size, split_seed)

    x_train, x_val, x_test = (_to_tensor(images[i]) for i in (train_idx, val_idx, test_idx))
    y_train, y_val, y_test = (torch.from_numpy(y[i]) for i in (train_idx, val_idx, test_idx))

    torch.manual_seed(train_seed)  # initial weights + dropout masks
    model = build_model(img_size, N_CLASSES, arch, use_batchnorm, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _make_scheduler(optimizer, scheduler, epochs)
    loss_fn = nn.CrossEntropyLoss()
    # Separate generators so turning augmentation on/off never changes batch order.
    shuffle_gen = torch.Generator().manual_seed(train_seed)
    aug_gen = torch.Generator().manual_seed(train_seed + 1)

    n = x_train.shape[0]
    train_loss: List[float] = []
    val_loss: List[float] = []
    val_accuracy: List[float] = []
    lr_history: List[float] = []
    best = {"val_loss": float("inf"), "epoch": -1, "state": None}

    t0 = time.perf_counter()
    for epoch in range(epochs):
        lr_history.append(optimizer.param_groups[0]["lr"])
        model.train()
        perm = torch.randperm(n, generator=shuffle_gen)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            b = perm[start : start + batch_size]
            xb, yb = x_train[b], y_train[b]
            if augment:
                xb = _augment(xb, augment_max_shift, aug_gen)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        if sched is not None:
            sched.step()
        train_loss.append(epoch_loss / n)

        v_loss, v_pred = _evaluate(model, x_val, y_val)
        val_loss.append(v_loss)
        val_accuracy.append(float((v_pred == y[val_idx]).mean()))
        if v_loss < best["val_loss"]:
            best = {"val_loss": v_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
    train_seconds = time.perf_counter() - t0

    model.load_state_dict(best["state"])  # model selection uses validation only
    test_loss, preds = _evaluate(model, x_test, y_test)
    y_test_np = y[test_idx]
    metrics = {
        "accuracy": float((preds == y_test_np).mean()),
        "macro_f1": _macro_f1(y_test_np, preds, N_CLASSES),
        "test_loss": test_loss,
    }

    artifact = {
        "state_dict": model.state_dict(),
        "img_size": img_size,
        "n_classes": N_CLASSES,
        "arch": arch,
        "use_batchnorm": use_batchnorm,
        "dropout": dropout,
        "artifact_version": ARTIFACT_VERSION,
    }
    return {
        "artifact_version": ARTIFACT_VERSION,
        "metrics": metrics,
        "confusion_matrix": _confusion_matrix(y_test_np, preds, N_CLASSES),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "lr_history": lr_history,
        "best_epoch": best["epoch"],
        "param_count": sum(p.numel() for p in model.parameters()),
        "train_seconds": train_seconds,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "split_sha256": hashlib.sha256(
            b"".join(i.astype(np.int64).tobytes() for i in (train_idx, val_idx, test_idx))
        ).hexdigest(),
        "config": {
            "n_rows": n_rows, "img_size": img_size, "noise": noise,
            "data_seed": data_seed, "split_seed": split_seed, "train_seed": train_seed,
            "optimizer": "adam", "lr": lr, "batch_size": batch_size, "epochs": epochs,
            "test_size": test_size, "val_size": val_size,
            "arch": arch, "use_batchnorm": use_batchnorm, "dropout": dropout,
            "augment": augment, "augment_max_shift": augment_max_shift, "scheduler": scheduler,
        },
        "model_b64": base64.b64encode(pickle.dumps(artifact)).decode("ascii"),
    }


def train_cnn(
    n_rows: int,
    img_size: int,
    noise: float,
    seed: int,
    lr: float,
    batch_size: int,
    epochs: int,
    test_size: float = 0.2,
    arch: str = "baseline",
    use_batchnorm: bool = True,
    dropout: float = 0.3,
) -> Tuple[Dict[str, float], str, List[float]]:
    """Original entry point used by the API: ``(metrics, model_b64, train_loss)``.

    A thin wrapper over ``run_experiment`` (no augmentation, no scheduler, split
    and training seeds fixed at 0). A 10% validation split is now carved out.
    """
    r = run_experiment(
        n_rows, img_size, noise, data_seed=seed, split_seed=0, train_seed=0,
        lr=lr, batch_size=batch_size, epochs=epochs, test_size=test_size,
        arch=arch, use_batchnorm=use_batchnorm, dropout=dropout,
    )
    m = r["metrics"]
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}, r["model_b64"], r["train_loss"]


def _load_model(model_b64: str) -> Tuple[nn.Module, int]:
    obj = pickle.loads(base64.b64decode(model_b64))
    # Artifacts saved before ImprovedCNN existed have no "arch" key -> baseline.
    model = build_model(
        obj["img_size"],
        obj["n_classes"],
        obj.get("arch", "baseline"),
        obj.get("use_batchnorm", True),
        obj.get("dropout", 0.3),
    )
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, obj["img_size"]


def _grad_cam(model: nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Return a Grad-CAM map in [0, 1] at the conv feature resolution."""
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


def predict_with_cam(model_b64: str, gray_uint8: np.ndarray) -> Dict[str, object]:
    """Classify one preprocessed grayscale image and build a Grad-CAM overlay.

    ``gray_uint8`` must be [img_size, img_size] uint8. Returns a dict matching the
    PredictResponse fields (minus run_id / true_class, which main.py fills in).
    """
    model, img_size = _load_model(model_b64)
    x = _to_tensor(gray_uint8[None, :, :])  # [1, 1, H, W]

    with torch.no_grad():
        probs_t = torch.softmax(model(x), dim=1)[0]
    label = int(probs_t.argmax().item())
    probs = [float(p) for p in probs_t.tolist()]

    cam = _grad_cam(model, x, label)
    return {
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": probs[label],
        "probs": probs,
        "input_png": _png_b64(Image.fromarray(gray_uint8, "L")),
        "cam_png": _overlay_png(gray_uint8, cam),
    }
