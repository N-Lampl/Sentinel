#!/usr/bin/env python3
"""Generate a synthetic YOLO dataset of N images for scale / timing tests.

    python examples/make_scale_dataset.py 30000 scale-dataset
    time sentinel scan scale-dataset --cache scale-dataset/.cache --no-thumbnails -q
    time sentinel scan scale-dataset --cache scale-dataset/.cache --no-thumbnails -q   # cached

Layout: images/{train,val,test} (80/10/10) with labels/, 1% of the training
images copied byte-for-byte into test (exact cross-split duplicates). Images
are 160x120 by default; pass a third argument like 640x480 for larger ones.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from make_demo_dataset import make_image


def main(n: int, root: Path, size=(160, 120)) -> None:
    t0 = time.time()
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    seed = 0
    for split, frac in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
        for i in range(int(n * frac)):
            seed += 1
            make_image(seed, size).save(root / "images" / split / f"{split}_{i:06d}.jpg", quality=85)
            (root / "labels" / split / f"{split}_{i:06d}.txt").write_text(f"{seed % 3} 0.5 0.5 0.3 0.4\n")
    train_imgs = sorted((root / "images" / "train").glob("*.jpg"))
    for k, src in enumerate(train_imgs[: max(1, n // 100)]):
        (root / "images" / "test" / f"dup_{k:05d}.jpg").write_bytes(src.read_bytes())
        (root / "labels" / "test" / f"dup_{k:05d}.txt").write_text("0 0.5 0.5 0.3 0.4\n")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: [a, b, c]\n")
    print(f"generated {n} images (+{max(1, n // 100)} planted duplicates) in {root} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("scale-dataset")
    dims = tuple(int(v) for v in sys.argv[3].lower().split("x")) if len(sys.argv) > 3 else (160, 120)
    sys.path.insert(0, str(Path(__file__).parent))
    main(count, out, dims)  # type: ignore[arg-type]
