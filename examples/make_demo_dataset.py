#!/usr/bin/env python3
"""Generate a small synthetic COCO dataset with planted integrity problems.

    python examples/make_demo_dataset.py demo-dataset
    sentinel scan demo-dataset

Planted problems (all of which Sentinel should report):

* a byte-identical copy of a training image in ``test``;
* a pixel-identical PNG re-save of a training image in ``val`` with a
  different label (exact duplicate + conflicting labels);
* a brightness-shifted, re-compressed near-duplicate in ``test``;
* a horizontally flipped training image in ``test`` (derivative);
* a crop that declares ``derived_from`` its training source (lineage);
* two ``patient_id`` values shared between splits (group overlap);
* a degenerate, an out-of-bounds and an unknown-category annotation;
* a class that only appears in ``test`` (unseen class);
* a missing image file and an orphaned annotation.

Only numpy and Pillow are required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 160, 120


def make_image(seed: int, size=(W, H)) -> Image.Image:
    """Distinctive low-frequency texture plus random rectangles."""
    w, h = size
    r = np.random.default_rng(seed)
    small = r.uniform(0, 255, size=(6, 8, 3))
    base = np.asarray(Image.fromarray(small.astype(np.uint8), "RGB").resize((w, h), Image.Resampling.BICUBIC)).astype(np.float32)
    for _ in range(int(r.integers(4, 9))):
        x0, y0 = int(r.integers(0, w - 30)), int(r.integers(0, h - 30))
        bw, bh = int(r.integers(10, 70)), int(r.integers(10, 60))
        base[y0 : y0 + bh, x0 : x0 + bw] = r.integers(0, 255, size=3)
    noise = r.normal(0, 5, size=base.shape)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB")


def main(root: Path) -> None:
    cats = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}, {"id": 3, "name": "bird"}]
    data = {s: {"images": [], "annotations": []} for s in ("train", "val", "test")}
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    for s in data:
        (root / s).mkdir(exist_ok=True)
    ids = {"img": 0, "ann": 0}

    def add(split, name, img, boxes=None, extra=None, fmt="JPEG"):
        ids["img"] += 1
        img.save(root / split / name, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
        entry = {"id": ids["img"], "file_name": name, "width": img.width, "height": img.height}
        entry.update(extra or {})
        data[split]["images"].append(entry)
        for cat_id, bbox in boxes or [(1 + ids["img"] % 2, [10, 10, 50, 40])]:
            ids["ann"] += 1
            data[split]["annotations"].append({"id": ids["ann"], "image_id": ids["img"], "category_id": cat_id, "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
        return ids["img"]

    seed = 0
    train_images = []
    for split, n in (("train", 60), ("val", 20), ("test", 20)):
        for i in range(n):
            seed += 1
            pid = f"P{seed:03d}"
            if (split, i) == ("val", 0):
                pid = "P001"  # same patient as train image 1
            if (split, i) == ("test", 1):
                pid = "P002"
            img = make_image(seed)
            add(split, f"{split}_{i:03d}.jpg", img, extra={"date_captured": f"2024-{1 + seed % 3:02d}-{1 + seed % 27:02d} 10:00:00", "patient_id": pid})
            if split == "train":
                train_images.append(img)

    # exact byte duplicate
    (root / "test" / "dup_exact.jpg").write_bytes((root / "train" / "train_000.jpg").read_bytes())
    ids["img"] += 1
    data["test"]["images"].append({"id": ids["img"], "file_name": "dup_exact.jpg", "width": W, "height": H, "patient_id": "P900"})
    ids["ann"] += 1
    data["test"]["annotations"].append({"id": ids["ann"], "image_id": ids["img"], "category_id": 2, "bbox": [10, 10, 50, 40], "area": 2000, "iscrowd": 0})
    # pixel-identical re-save with different label
    add("val", "dup_pixels.png", Image.open(root / "train" / "train_001.jpg").convert("RGB"), boxes=[(1, [5, 5, 30, 30])], extra={"patient_id": "P901"}, fmt="PNG")
    # near duplicate
    arr = np.clip(np.asarray(Image.open(root / "train" / "train_002.jpg").convert("RGB")).astype(np.int16) + 8, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(root / "test" / "near_dup.jpg", quality=55)
    ids["img"] += 1
    data["test"]["images"].append({"id": ids["img"], "file_name": "near_dup.jpg", "width": W, "height": H, "patient_id": "P902"})
    ids["ann"] += 1
    data["test"]["annotations"].append({"id": ids["ann"], "image_id": ids["img"], "category_id": 1, "bbox": [10, 10, 50, 40], "area": 2000, "iscrowd": 0})
    # flipped derivative
    add("test", "flip.jpg", Image.open(root / "train" / "train_003.jpg").convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT), extra={"patient_id": "P903"})
    # lineage crop
    add("val", "crop_of_train_004.jpg", Image.open(root / "train" / "train_004.jpg").convert("RGB").crop((20, 20, 120, 100)),
        boxes=[(1, [1, 1, 20, 20])], extra={"patient_id": "P904", "derived_from": "train_004.jpg"})
    # label problems on train_005
    tid = data["train"]["images"][5]["id"]
    for cat_id, bbox in ((1, [10, 10, -5, 20]), (2, [140, 100, 50, 50]), (99, [10, 10, 20, 20])):
        ids["ann"] += 1
        data["train"]["annotations"].append({"id": ids["ann"], "image_id": tid, "category_id": cat_id, "bbox": bbox, "area": 0, "iscrowd": 0})
    # unseen class in test
    ids["ann"] += 1
    data["test"]["annotations"].append({"id": ids["ann"], "image_id": data["test"]["images"][0]["id"], "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0})
    # missing file + orphan annotation
    ids["img"] += 1
    data["train"]["images"].append({"id": ids["img"], "file_name": "missing.jpg", "width": W, "height": H})
    ids["ann"] += 1
    data["train"]["annotations"].append({"id": ids["ann"], "image_id": 99999, "category_id": 1, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0})

    for split, content in data.items():
        content["categories"] = cats
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(content))
    print(f"wrote demo dataset to {root} ({sum(len(c['images']) for c in data.values())} images)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "demo-dataset"))
