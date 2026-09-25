"""Shared fixtures: synthetic image generator and COCO / YOLO dataset builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
from PIL import Image

from dataset_sentinel import SentinelConfig, scan

W, H = 160, 120


def image_from_seed(seed: int, size: Tuple[int, int] = (W, H)) -> Image.Image:
    """Distinctive low-frequency texture plus random rectangles (photo-like structure)."""
    w, h = size
    r = np.random.default_rng(seed)
    small = r.uniform(0, 255, size=(6, 8, 3))
    base = np.asarray(Image.fromarray(small.astype(np.uint8), "RGB").resize((w, h), Image.Resampling.BICUBIC)).astype(np.float32)
    for _ in range(int(r.integers(4, 9))):
        x0, y0 = int(r.integers(0, max(1, w - 30))), int(r.integers(0, max(1, h - 30)))
        bw, bh = int(r.integers(10, 70)), int(r.integers(10, 60))
        base[y0 : y0 + bh, x0 : x0 + bw] = r.integers(0, 255, size=3)
    noise = r.normal(0, 5, size=base.shape)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB")


class CocoBuilder:
    """Builds a COCO dataset on disk: <root>/<split>/*.jpg + <root>/annotations/instances_<split>.json."""

    def __init__(self, root: Path, splits: Sequence[str] = ("train", "val", "test"), categories: Sequence[str] = ("cat", "dog")):
        self.root = root
        self.splits = list(splits)
        self.categories = [{"id": i + 1, "name": n} for i, n in enumerate(categories)]
        self.images: Dict[str, List[Dict[str, Any]]] = {s: [] for s in splits}
        self.annotations: Dict[str, List[Dict[str, Any]]] = {s: [] for s in splits}
        self._img_id = 0
        self._ann_id = 0
        self._seed = 0
        for s in splits:
            (root / s).mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)

    def path(self, split: str, name: str) -> Path:
        return self.root / split / name

    def add(
        self,
        split: str,
        name: Optional[str] = None,
        image: Optional[Image.Image] = None,
        seed: Optional[int] = None,
        boxes: Optional[Sequence[Tuple[int, Sequence[float]]]] = None,
        extra: Optional[Dict[str, Any]] = None,
        fmt: str = "JPEG",
        write_file: bool = True,
        declared_size: Optional[Tuple[int, int]] = None,
    ) -> int:
        self._img_id += 1
        self._seed += 1
        if image is None:
            image = image_from_seed(seed if seed is not None else self._seed)
        if name is None:
            name = f"{split}_{self._img_id:04d}.{'png' if fmt == 'PNG' else 'jpg'}"
        if write_file:
            image.save(self.path(split, name), format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
        w, h = declared_size or image.size
        entry = {"id": self._img_id, "file_name": name, "width": w, "height": h}
        entry.update(extra or {})
        self.images[split].append(entry)
        if boxes is None:
            boxes = [(1, [10, 10, 50, 40])]
        for cat_id, bbox in boxes:
            self.add_annotation(split, self._img_id, cat_id, list(bbox))
        return self._img_id

    def add_annotation(self, split: str, image_id: int, category_id: Any, bbox: Any, **kw: Any) -> int:
        self._ann_id += 1
        ann = {"id": self._ann_id, "image_id": image_id, "category_id": category_id, "bbox": bbox, "iscrowd": 0}
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            ann["area"] = bbox[2] * bbox[3]
        ann.update(kw)
        self.annotations[split].append(ann)
        return self._ann_id

    def fill(self, per_split: int = 12) -> "CocoBuilder":
        for split in self.splits:
            for _ in range(per_split):
                self.add(split, boxes=[(1 + (self._img_id % 2), [10, 10, 50, 40])])
        return self

    def write(self) -> Path:
        for split in self.splits:
            data = {"images": self.images[split], "annotations": self.annotations[split], "categories": self.categories}
            (self.root / "annotations" / f"instances_{split}.json").write_text(json.dumps(data))
        return self.root


class YoloBuilder:
    def __init__(self, root: Path, names: Sequence[str] = ("cat", "dog"), splits: Sequence[str] = ("train", "val", "test")):
        self.root = root
        self.names = list(names)
        self.splits = list(splits)
        self._seed = 500
        for s in splits:
            (root / "images" / s).mkdir(parents=True, exist_ok=True)
            (root / "labels" / s).mkdir(parents=True, exist_ok=True)

    def image_path(self, split: str, name: str) -> Path:
        return self.root / "images" / split / name

    def label_path(self, split: str, name: str) -> Path:
        return (self.root / "labels" / split / name).with_suffix(".txt")

    def add(
        self,
        split: str,
        name: Optional[str] = None,
        image: Optional[Image.Image] = None,
        seed: Optional[int] = None,
        labels: Optional[Sequence[str]] = None,
        write_label: bool = True,
    ) -> Path:
        self._seed += 1
        if image is None:
            image = image_from_seed(seed if seed is not None else self._seed)
        if name is None:
            name = f"{split}_{self._seed:04d}.jpg"
        p = self.image_path(split, name)
        image.save(p, quality=92)
        if write_label:
            lines = list(labels) if labels is not None else [f"{self._seed % len(self.names)} 0.5 0.5 0.3 0.4"]
            self.label_path(split, name).write_text("\n".join(lines) + ("\n" if lines else ""))
        return p

    def fill(self, per_split: int = 12) -> "YoloBuilder":
        for split in self.splits:
            for _ in range(per_split):
                self.add(split)
        return self

    def write(self, data_name: str = "data.yaml") -> Path:
        lines = ["path: .", *(f"{s}: images/{s}" for s in self.splits), "names:"]
        lines += [f"  {i}: {n}" for i, n in enumerate(self.names)]
        p = self.root / data_name
        p.write_text("\n".join(lines) + "\n")
        return p


def make_config(root: Path, fmt: Optional[str] = None, **overrides: Any) -> SentinelConfig:
    cfg = SentinelConfig({"dataset": {"root": str(root), "format": fmt}, "output": {"thumbnails": False}, "performance": {"cache": None, "workers": 2}})
    for key, value in overrides.items():
        cfg.set(key.replace("__", "."), value)
    return cfg


def run_scan(root: Path, fmt: Optional[str] = None, **overrides: Any):
    return scan(make_config(root, fmt, **overrides))


def findings_of(report, detector: Optional[str] = None, kind: Optional[str] = None):
    out = report.findings
    if detector:
        out = [f for f in out if f.detector == detector]
    if kind:
        out = [f for f in out if f.kind == kind]
    return out


@pytest.fixture
def coco_root(tmp_path: Path) -> Path:
    return CocoBuilder(tmp_path / "coco").fill(10).write()


@pytest.fixture
def yolo_root(tmp_path: Path) -> Path:
    b = YoloBuilder(tmp_path / "yolo").fill(10)
    b.write()
    return b.root


__all__ = ["CocoBuilder", "YoloBuilder", "image_from_seed", "make_config", "run_scan", "findings_of"]
