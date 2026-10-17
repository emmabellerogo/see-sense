"""A tiny Fashion-MNIST-shaped dataset for tests (no download, no real data).

Ten classes of 28x28 uint8 images; class k is a bright patch at its own grid position plus a
little noise, so a small CNN learns it in a couple of epochs. It exercises the same code paths
as the real files (``FashionData``, the same split and identity functions) but says nothing about
Fashion-MNIST accuracy.
"""
from __future__ import annotations

import numpy as np

from shared.fashion import FashionData


def _patch_origin(k: int) -> tuple[int, int]:
    return 2 + 8 * (k % 3), 2 + 8 * (k // 3 % 3) + (k // 9) * 0


def make_images(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for k in range(10):
        r, c = _patch_origin(k)
        for _ in range(n_per_class):
            img = rng.integers(0, 25, (28, 28)).astype(np.uint8)
            img[r : r + 6, c : c + 6] = rng.integers(180, 255, (6, 6))
            if k == 9:  # make class 9 distinct from class 0 (same origin): add a second patch
                img[20:26, 20:26] = 255
            xs.append(img)
            ys.append(k)
    x, y = np.stack(xs), np.asarray(ys, dtype=np.uint8)
    order = rng.permutation(len(y))
    return x[order], y[order]


def tiny_fashion(n_train_per_class: int = 50, n_test_per_class: int = 10, seed: int = 0) -> FashionData:
    x_tr, y_tr = make_images(n_train_per_class, seed)
    x_te, y_te = make_images(n_test_per_class, seed + 1)
    return FashionData(x_tr, y_tr, x_te, y_te, {"tiny-train": "0" * 64, "tiny-test": "1" * 64})
