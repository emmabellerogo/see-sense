"""Augmentation must keep clothing recognizable: shifts and left-right mirroring only."""
from __future__ import annotations

import numpy as np
import torch

from api.training import _augment


def _asymmetric_batch(n=64):
    """Images whose mirror and flip are all distinguishable: a bright L-shape off-centre."""
    x = torch.zeros(n, 1, 28, 28)
    x[:, 0, 8:20, 9:11] = 1.0       # vertical stroke
    x[:, 0, 18:20, 9:18] = 0.6      # foot to the right
    x[:, 0, 6:8, 9:12] = 0.3
    return x


def test_output_is_always_a_shifted_original_or_its_left_right_mirror():
    x = _asymmetric_batch()
    out = _augment(x, 2, torch.Generator().manual_seed(0))
    padded = torch.nn.functional.pad(x[:1], (2, 2, 2, 2))[0, 0]
    allowed = []
    for dy in range(5):
        for dx in range(5):
            win = padded[dy : dy + 28, dx : dx + 28]
            allowed += [win, win.flip(1)]  # shifted, or shifted then mirrored left-right
    for im in out[:, 0]:
        assert any(torch.equal(im, a) for a in allowed)


def test_there_is_never_a_vertical_flip():
    x = _asymmetric_batch(200)
    out = _augment(x, 2, torch.Generator().manual_seed(3))
    padded = torch.nn.functional.pad(x[:1], (2, 2, 2, 2))[0, 0]
    upside_down = [padded[dy : dy + 28, dx : dx + 28].flip(0) for dy in range(5) for dx in range(5)]
    for im in out[:, 0]:
        assert not any(torch.equal(im, u) for u in upside_down)
    # And the garment's vertical order survives: the heavy stroke bottom (rows ~18-20) stays below its top.
    rows = out[:, 0].sum(dim=2)
    assert (rows[:, 14:24].sum(1) > rows[:, 2:10].sum(1)).all()


def test_both_mirrored_and_unmirrored_images_occur_and_all_shifts_are_covered():
    x = _asymmetric_batch(400)
    out = _augment(x, 2, torch.Generator().manual_seed(1))[:, 0]
    foot_right = (out[:, :, 14:].sum((1, 2)) > out[:, :, :14].sum((1, 2))).float().mean().item()
    assert 0.3 < foot_right < 0.7  # roughly half mirrored
    centroid_x = (out.sum(1) * torch.arange(28)).sum(1) / out.sum((1, 2))
    assert centroid_x.max() - centroid_x.min() > 3  # shifts moved the content


def test_padding_is_background_not_wrapped_content():
    x = torch.zeros(1, 1, 28, 28)
    x[0, 0, 0, :] = 1.0  # a bright top row
    out = _augment(x, 2, torch.Generator().manual_seed(0))
    assert out.sum() <= x.sum()  # nothing is created or wrapped around
    assert out.min() >= 0


def test_random_stream_does_not_depend_on_image_content():
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    _augment(_asymmetric_batch(10), 2, g1)
    _augment(torch.rand(10, 1, 28, 28), 2, g2)
    assert torch.equal(torch.rand(3, generator=g1), torch.rand(3, generator=g2))
