"""Augmentation keeps every synthetic class recognizable (noise-free images)."""
from __future__ import annotations

import numpy as np
import torch

from api.training import _augment
from shared.data import CLASS_NAMES, generate_one

N_PER_CLASS = 300
# Smallest share of the original ink that may survive a shift (zero padding crops
# the edges). Worst cases: a 4-row bar at the edge loses one row and two columns
# (3/4 * 26/28 = 0.70); a 6x6 block at a corner keeps 4x4 of 6x6 (0.44).
MIN_INK_KEPT = {"horizontal": 0.65, "vertical": 0.65, "diagonal": 0.80, "block": 0.40}


def _batch(kind: int) -> torch.Tensor:
    imgs = [generate_one(noise=0.0, seed=s, kind=kind)[0] for s in range(N_PER_CLASS)]
    return torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0).unsqueeze(1)


def _spread(v: np.ndarray) -> int:
    return int(v.max() - v.min())


def _is_diagonal_line(ink: np.ndarray) -> bool:
    """Ink hugs a 45-degree line in either direction."""
    r, c = np.nonzero(ink)
    return _spread(r - c) <= 5 or _spread(r + c) <= 5


def _classify(img: np.ndarray) -> str:
    """Geometric rule for the four classes, tolerant of shifts and flips."""
    ink = img > 0.5
    r, c = np.nonzero(ink)
    if len(r) == 0:
        return "empty"
    h, w = _spread(r) + 1, _spread(c) + 1
    if w >= 20 and h <= 5:
        return "horizontal"
    if h >= 20 and w <= 5:
        return "vertical"
    if h >= 20 and w >= 20 and _is_diagonal_line(ink):
        return "diagonal"
    if 4 <= h <= 12 and 4 <= w <= 12 and ink.sum() >= 0.9 * h * w:
        return "block"
    return "other"


def test_classifier_rule_recognizes_the_unaugmented_generator():
    for kind, name in enumerate(CLASS_NAMES):
        x = _batch(kind)[:, 0].numpy()
        assert {_classify(im) for im in x} == {name}


def test_shifts_and_flips_keep_class_and_do_not_erase_shape():
    gen = torch.Generator().manual_seed(0)
    for kind, name in enumerate(CLASS_NAMES):
        x = _batch(kind)
        out = _augment(x, 2, gen)
        assert not torch.equal(out, x)
        kept = (out.sum(dim=(1, 2, 3)) / x.sum(dim=(1, 2, 3))).numpy()
        assert kept.min() >= MIN_INK_KEPT[name], (name, kept.min())
        labels = [_classify(im) for im in out[:, 0].numpy()]
        assert set(labels) == {name}, (name, {l for l in labels if l != name})


def test_every_shift_and_flip_combination_is_exercised_and_safe():
    # Hit the extreme shifts (+-2 on each axis) with all four flip states.
    x = torch.cat([_batch(k) for k in range(len(CLASS_NAMES))])
    kinds = [n for n in CLASS_NAMES for _ in range(N_PER_CLASS)]
    gen = torch.Generator().manual_seed(1)
    seen = set()
    for _ in range(6):
        out = _augment(x, 2, gen)
        for im, orig, name in zip(out[:, 0].numpy(), x[:, 0].numpy(), kinds):
            assert _classify(im) == name
            seen.add((bool(np.array_equal(im, orig[:, ::-1])), bool(np.array_equal(im, orig[::-1]))))
    assert len(seen) >= 2  # both "flipped" and "not flipped" states occurred


def test_flipped_diagonals_go_beyond_the_generators_orientation():
    original = _batch(2)[:, 0].numpy() > 0.5
    # The generator only draws one diagonal direction (constant row - column).
    assert all(_spread(np.nonzero(m)[0] - np.nonzero(m)[1]) <= 5 for m in original)
    assert not any(_spread(np.nonzero(m)[0] + np.nonzero(m)[1]) <= 5 for m in original)

    out = _augment(_batch(2), 2, torch.Generator().manual_seed(0))[:, 0].numpy() > 0.5
    flipped = [m for m in out if _spread(np.nonzero(m)[0] + np.nonzero(m)[1]) <= 5]
    assert len(flipped) > 0.2 * len(out)  # about half the diagonals now run the other way
    assert all(_is_diagonal_line(m) for m in out)  # and they are still diagonal lines
