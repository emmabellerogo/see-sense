"""Export a small, fixed set of HELD-OUT Fashion-MNIST test images for the Try It tab.

    python tools/export_samples.py [--per-class 2] [--out ui/assets/samples]

Rule (fixed here, independent of any model): for each of the ten classes, the first
``--per-class`` images of that class in the OFFICIAL test partition, in file order. Images are
never chosen by whether a model gets them right. They are official test images, so no model
trained by this project has trained or validated on them. The UI sends them through the normal
POST /predict upload path, so the deployed UI needs no dataset download and no generator.
Writes 28x28 PNGs plus manifest.json (test index, label, PNG sha256, dataset file hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from shared import fashion  # noqa: E402


def slug(name: str) -> str:
    return name.lower().replace("/", "-").replace(" ", "-")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default=str(fashion.REPO_ROOT / "ui" / "assets" / "samples"))
    args = ap.parse_args()
    data = fashion.load()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    items = []
    for k, cname in enumerate(fashion.CLASS_NAMES):
        for n, idx in enumerate(np.flatnonzero(data.y_test == k)[: args.per_class], start=1):
            fname = f"{k}-{slug(cname)}-{n}.png"
            Image.fromarray(data.x_test[idx], "L").save(out / fname, optimize=True)
            items.append({
                "file": fname, "test_index": int(idx), "label": k, "class_name": cname,
                "png_sha256": hashlib.sha256((out / fname).read_bytes()).hexdigest(),
            })
    manifest = {
        "dataset": fashion.DATASET_NAME, "partition": "official test (t10k)",
        "rule": f"first {args.per_class} images of each class in official test order; not chosen by model output",
        "class_names": fashion.CLASS_NAMES, "dataset_file_sha256": data.file_sha256, "samples": items,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(items)} PNGs + manifest.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
