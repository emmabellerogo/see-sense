"""Download and verify Fashion-MNIST into ./data/fashion-mnist (git-ignored).

    python tools/fetch_fashion_mnist.py [--dest DIR]

The only place in the project that touches the network for data. Run it on the
training machine; the API (Render) and UI (Streamlit) never download anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import fashion  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=None, help="default: $FASHION_MNIST_DIR or data/fashion-mnist")
    args = ap.parse_args()
    out = fashion.download(args.dest)
    data = fashion.load(out)
    mean, std = fashion.measure_norm_stats(data.x_train)
    print(f"verified {len(fashion.FILES)} files in {out}")
    for name, digest in data.file_sha256.items():
        print(f"  sha256 {digest}  {name}")
    print(f"train {data.x_train.shape}, test {data.x_test.shape}")
    print(f"train pixel mean/std: {mean:.4f} / {std:.4f} (constants: {fashion.NORM_MEAN} / {fashion.NORM_STD})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
