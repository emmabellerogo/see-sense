"""Arithmetic behind the Concepts tab (no streamlit import, unit-tested).

Layers are (name, kernel, stride, padding) tuples. The Concepts page builds its
tables from these functions so the numbers shown always follow the formulas.
"""
from __future__ import annotations

Layer = tuple  # (name, kernel, stride, padding)

# The project's CNN (SmallCNN / ImprovedCNN): two conv(3x3, pad 1) + 2x2 max-pool blocks.
OUR_CNN: list[Layer] = [
    ("Conv 3×3, pad 1 (1→8 channels)", 3, 1, 1),
    ("MaxPool 2×2, stride 2", 2, 2, 0),
    ("Conv 3×3, pad 1 (8→16 channels)", 3, 1, 1),
    ("MaxPool 2×2, stride 2", 2, 2, 0),
]

THREE_3X3: list[Layer] = [(f"Conv 3×3 #{i}", 3, 1, 0) for i in (1, 2, 3)]


def out_size(n: int, k: int, s: int = 1, p: int = 0) -> int:
    """floor((n + 2p - k) / s) + 1. Raises ValueError when the kernel does not fit."""
    if min(n, k, s) < 1 or p < 0:
        raise ValueError("n, k and s must be at least 1 and p at least 0")
    if n + 2 * p < k:
        raise ValueError("kernel larger than the padded input")
    return (n + 2 * p - k) // s + 1


def trace(layers: list[Layer], n: int) -> list[dict]:
    """Per-layer output size, receptive field r and jump j (input pixels per step).

    r_out = r_in + (k - 1) * j_in      j_out = j_in * s      (start: r = 1, j = 1)
    """
    rows = [{"layer": "Input", "size": n, "r": 1, "j": 1}]
    r, j = 1, 1
    for name, k, s, p in layers:
        n = out_size(n, k, s, p)
        r = r + (k - 1) * j
        j = j * s
        rows.append({"layer": name, "size": n, "r": r, "j": j})
    return rows


def conv_params(c_in: int, c_out: int, k: int) -> int:
    return c_in * c_out * k * k + c_out


def our_cnn_param_count(n_classes: int = 4, img: int = 28) -> int:
    """Parameters of SmallCNN (the baseline); matches the 4,388 stored in Supabase."""
    feat = img // 4
    return conv_params(1, 8, 3) + conv_params(8, 16, 3) + 16 * feat * feat * n_classes + n_classes
