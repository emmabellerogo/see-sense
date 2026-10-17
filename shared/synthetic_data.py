"""HISTORICAL: the synthetic four-shape dataset generator the project started with.

This is NOT the project's dataset. The assignment asks for CIFAR-10 or Fashion-MNIST, and the project's
dataset is now Fashion-MNIST (``shared/fashion.py``). The starter shipped this generator, and the first
version of the project (experiment group ``official-supabase-1``) was built on it. It is kept, unchanged,
only so that

  * those saved results and their four-class checkpoints stay reproducible, identifiable and loadable
    (``api.training.load_artifact`` labels them ``synthetic_shapes``), and
  * a few tests can build legacy-format checkpoints.

Nothing in training, serving or the UI generates these images any more. Only dataset *parameters*
(n_rows, img_size, noise, seed) were ever persisted to Supabase; no real images are involved.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

IMG_SIZE = 28
# Each class is a distinct shape a small CNN can localize with Grad-CAM.
CLASS_NAMES: List[str] = ["horizontal", "vertical", "diagonal", "block"]
N_CLASSES = len(CLASS_NAMES)


def _draw(kind: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a float32 [size, size] image in [0, 1] for the given class."""
    img = np.zeros((size, size), dtype=np.float32)
    if kind == 0:  # horizontal bar
        r = int(rng.integers(3, size - 3))
        t = int(rng.integers(1, 3))
        img[max(0, r - t) : r + t, :] = 1.0
    elif kind == 1:  # vertical bar
        c = int(rng.integers(3, size - 3))
        t = int(rng.integers(1, 3))
        img[:, max(0, c - t) : c + t] = 1.0
    elif kind == 2:  # diagonal line
        t = int(rng.integers(1, 3))
        for i in range(size):
            for dj in range(-t, t + 1):
                j = i + dj
                if 0 <= j < size:
                    img[i, j] = 1.0
    else:  # filled block
        s = int(rng.integers(6, 12))
        r = int(rng.integers(0, size - s))
        c = int(rng.integers(0, size - s))
        img[r : r + s, c : c + s] = 1.0
    return img


def generate_one(
    img_size: int = IMG_SIZE,
    noise: float = 0.1,
    seed: int = 0,
    kind: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Return ``(image_uint8 [H, W], label)`` for a single synthetic image."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(0, N_CLASSES)) if kind is None else int(kind)
    im = _draw(k, img_size, rng)
    im = np.clip(im + rng.normal(0.0, noise, im.shape).astype(np.float32), 0.0, 1.0)
    return (im * 255).astype(np.uint8), k


def generate_images(
    n_rows: int,
    img_size: int = IMG_SIZE,
    noise: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int]]:
    """Return ``(images_uint8 [N, H, W], labels)`` for a synthetic dataset."""
    rng = np.random.default_rng(seed)
    images = np.empty((n_rows, img_size, img_size), dtype=np.uint8)
    labels: List[int] = []
    for i in range(n_rows):
        k = int(rng.integers(0, N_CLASSES))
        im = _draw(k, img_size, rng)
        im = np.clip(im + rng.normal(0.0, noise, im.shape).astype(np.float32), 0.0, 1.0)
        images[i] = (im * 255).astype(np.uint8)
        labels.append(k)
    return images, labels
