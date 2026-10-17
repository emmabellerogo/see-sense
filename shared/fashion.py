"""Fashion-MNIST: the project's dataset (Zalando Research, 28x28 grayscale, 10 classes).

Source: https://github.com/zalandoresearch/fashion-mnist (MIT licence). Official
partitions: 60,000 training images and 10,000 test images, stored as four gzip'd
IDX files. This module reads those files from a LOCAL directory only.

  * Downloading happens in exactly one place, ``download()``, which is called by
    ``tools/fetch_fashion_mnist.py``. The API, the UI and every test import this
    module without ever touching the network, so Render and Streamlit startup
    never download anything.
  * Every file is verified against its published MD5 before use; its SHA-256 is
    recorded in the dataset identity that is stored with each checkpoint.
  * Validation data is carved out of the OFFICIAL TRAINING images only, with a
    seeded, class-stratified split. The official test images are never used to
    train, to choose an epoch, or to fit calibration.

Plain numpy only (no torchvision) so the pinned API requirements do not change.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATASET_NAME = "fashion-mnist"
IMG_SIZE = 28
CLASS_NAMES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
N_CLASSES = len(CLASS_NAMES)

# Pixel statistics of the 60,000 official training images (labels and test data
# not involved), computed by ``measure_norm_stats`` and checked by a test when the
# data is present. Inputs are scaled to [0, 1] and then (x - MEAN) / STD.
NORM_MEAN = 0.2860
NORM_STD = 0.3530

FILES: Dict[str, str] = {  # file name -> published MD5 (same values torchvision pins)
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
MIRRORS = (
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
)
N_TRAIN_OFFICIAL, N_TEST_OFFICIAL = 60_000, 10_000

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "data" / "fashion-mnist"


class DatasetUnavailable(RuntimeError):
    """The local Fashion-MNIST files are missing or fail verification."""


def data_dir(path: str | os.PathLike | None = None) -> Path:
    return Path(path or os.environ.get("FASHION_MNIST_DIR") or DEFAULT_DIR)


# ---------------------------------------------------------------------------
# download (local training tools only)
# ---------------------------------------------------------------------------
def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: str | os.PathLike | None = None, mirrors=MIRRORS, log=print) -> Path:
    """Fetch the four official files into ``dest`` and verify them. Idempotent."""
    out = data_dir(dest)
    out.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        target = out / name
        if target.is_file() and _digest(target, "md5") == md5:
            log(f"ok (already verified): {name}")
            continue
        last: Optional[Exception] = None
        for base in mirrors:
            try:
                log(f"downloading {base}{name}")
                tmp = target.with_suffix(target.suffix + ".part")
                with urllib.request.urlopen(base + name, timeout=60) as resp, open(tmp, "wb") as fh:
                    fh.write(resp.read())
                if _digest(tmp, "md5") != md5:
                    tmp.unlink()
                    raise DatasetUnavailable(f"{name}: MD5 mismatch from {base}")
                tmp.replace(target)
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last = exc
        else:
            raise DatasetUnavailable(f"could not download {name}: {last}")
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    magic = struct.unpack(">I", raw[:4])[0]
    if magic == 2051:  # images
        n, h, w = struct.unpack(">III", raw[4:16])
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, h, w)
    if magic == 2049:  # labels
        (n,) = struct.unpack(">I", raw[4:8])
        return np.frombuffer(raw, dtype=np.uint8, offset=8).reshape(n)
    raise DatasetUnavailable(f"{path.name}: unexpected IDX magic number {magic}")


@dataclass(frozen=True)
class FashionData:
    """The official partitions, in their original order. ``x_test`` is None if not loaded."""

    x_train: np.ndarray  # [60000, 28, 28] uint8
    y_train: np.ndarray  # [60000] uint8
    x_test: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    file_sha256: Dict[str, str]


def load(path: str | os.PathLike | None = None, include_test: bool = True, verify: bool = True) -> FashionData:
    """Read the official files from disk. ``include_test=False`` never opens the test files."""
    root = data_dir(path)
    names = [n for n in FILES if include_test or n.startswith("train")]
    for name in names:
        f = root / name
        if not f.is_file():
            raise DatasetUnavailable(
                f"{f} is missing. Run `python tools/fetch_fashion_mnist.py` on the training machine "
                "(the API and UI never download the dataset)."
            )
    sha: Dict[str, str] = {}
    for name in names:
        if verify and _digest(root / name, "md5") != FILES[name]:
            raise DatasetUnavailable(f"{name} does not match the published MD5; delete and re-download it.")
        sha[name] = _digest(root / name, "sha256")
    x_tr, y_tr = _read_idx(root / "train-images-idx3-ubyte.gz"), _read_idx(root / "train-labels-idx1-ubyte.gz")
    if x_tr.shape != (N_TRAIN_OFFICIAL, IMG_SIZE, IMG_SIZE) or y_tr.shape != (N_TRAIN_OFFICIAL,):
        raise DatasetUnavailable("unexpected training-set shape")
    x_te = y_te = None
    if include_test:
        x_te, y_te = _read_idx(root / "t10k-images-idx3-ubyte.gz"), _read_idx(root / "t10k-labels-idx1-ubyte.gz")
        if x_te.shape != (N_TEST_OFFICIAL, IMG_SIZE, IMG_SIZE) or y_te.shape != (N_TEST_OFFICIAL,):
            raise DatasetUnavailable("unexpected test-set shape")
    return FashionData(x_tr, y_tr, x_te, y_te, sha)


def measure_norm_stats(x_train: np.ndarray) -> Tuple[float, float]:
    """Mean and (population) std of the training pixels scaled to [0, 1]."""
    x = x_train.astype(np.float64) / 255.0
    return float(x.mean()), float(x.std())


# ---------------------------------------------------------------------------
# validation split (official TRAIN images only)
# ---------------------------------------------------------------------------
def split_train_val(y_train: np.ndarray, val_size: int, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Class-stratified, seeded train/validation indices into the official training set.

    Each class contributes ``val_size / n_classes`` validation images (the remainder, if any,
    goes to the first classes), drawn by a Generator seeded with ``split_seed``. The result
    depends only on ``(y_train, val_size, split_seed)``. Both index arrays are sorted.
    """
    n = len(y_train)
    if not 0 < val_size < n:
        raise ValueError("val_size must be between 1 and len(y_train) - 1")
    rng = np.random.default_rng(split_seed)
    classes = np.unique(y_train)
    base, extra = divmod(val_size, len(classes))
    val: List[np.ndarray] = []
    for i, c in enumerate(classes):
        idx = np.flatnonzero(y_train == c)
        k = base + (1 if i < extra else 0)
        if k >= len(idx):
            raise ValueError("val_size leaves no training images for some class")
        val.append(rng.choice(idx, size=k, replace=False))
    val_idx = np.sort(np.concatenate(val))
    mask = np.ones(n, dtype=bool)
    mask[val_idx] = False
    return np.flatnonzero(mask), val_idx


def index_sha256(*arrays: np.ndarray) -> str:
    """Hash of partition identities (indices into the official files)."""
    return hashlib.sha256(b"".join(np.asarray(a, dtype=np.int64).tobytes() for a in arrays)).hexdigest()


def class_counts(y: np.ndarray) -> List[int]:
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=N_CLASSES).tolist()


def dataset_identity(
    data: FashionData, train_idx: np.ndarray, val_idx: np.ndarray, split_seed: int, val_size: int,
    train_limit: Optional[int] = None,
) -> dict:
    """Everything needed to say exactly which images a run trained/validated/tested on."""
    ident = {
        "name": DATASET_NAME,
        "source": "fashion_mnist",
        "url": MIRRORS[0],
        "license": "MIT (Zalando Research)",
        "class_names": list(CLASS_NAMES),
        "file_sha256": dict(data.file_sha256),
        "official_train_images": int(len(data.y_train)),
        "official_test_images": int(len(data.y_test)) if data.y_test is not None else None,
        "split_seed": int(split_seed),
        "val_size": int(val_size),
        "train_limit": train_limit,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_index_sha256": index_sha256(train_idx),
        "val_index_sha256": index_sha256(val_idx),
        "val_class_counts": class_counts(data.y_train[val_idx]),
        "normalization": {"scale": "x/255", "mean": NORM_MEAN, "std": NORM_STD},
    }
    if data.y_test is not None:
        ident["n_test"] = int(len(data.y_test))
        ident["test_class_counts"] = class_counts(data.y_test)
    return ident


def identity_json(ident: dict) -> str:
    return json.dumps(ident, sort_keys=True)
