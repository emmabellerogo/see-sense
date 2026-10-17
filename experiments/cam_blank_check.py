"""How often is the Grad-CAM map of the predicted class empty (no positive evidence) for one saved checkpoint?

    python -m experiments.cam_blank_check --supabase <run_name> <group> <out.csv>

Grad-CAM = ReLU(sum_k mean(grad_k) * A_k). If every cell is <= 0 before the final normalisation the map is
all zero and the overlay shows nothing. This counts that over ALL official test images, by predicted class.
Read-only (downloads the checkpoint into memory; nothing is saved to the cloud or retrained).
"""
from __future__ import annotations

import csv
import sys
from collections import Counter

import numpy as np

from api.training import load_artifact, normalize, to01
from experiments.gradcam_evidence import load_from_supabase
from shared import fashion


def main(run_name: str, group: str, out: str) -> None:
    _run, _ds, b64 = load_from_supabase(run_name, group)
    loaded = load_artifact(b64)
    model = loaded.model.eval()
    data = fashion.load(include_test=True)
    names = loaded.class_names
    blank, total = Counter(), Counter()
    for img in data.x_test:
        x = normalize(to01(img[None]), loaded.preprocessing)
        cap: dict = {}
        h1 = model.features.register_forward_hook(lambda _m, _i, o: cap.__setitem__("a", o))
        h2 = model.features.register_full_backward_hook(lambda _m, _gi, go: cap.__setitem__("g", go[0]))
        logits = model(x)
        pred = int(logits.argmax())
        model.zero_grad()
        logits[0, pred].backward()
        h1.remove(); h2.remove()
        raw = (cap["g"][0].mean((1, 2))[:, None, None] * cap["a"][0]).sum(0)
        total[names[pred]] += 1
        blank[names[pred]] += int(float(raw.max()) <= 0)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["predicted_class", "n_predictions", "n_blank_cam", "share_blank"])
        for n in names:
            w.writerow([n, total[n], blank[n], f"{blank[n] / max(total[n], 1):.4f}"])
            print(f"{n:12s} {blank[n]:4d}/{total[n]:4d}  {blank[n] / max(total[n], 1):.1%}")


if __name__ == "__main__":
    if sys.argv[1] != "--supabase":
        raise SystemExit(__doc__)
    main(*sys.argv[2:5])
