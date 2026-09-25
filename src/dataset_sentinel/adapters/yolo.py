"""YOLO (Ultralytics layout) adapter.

Configuration::

    dataset:
      format: yolo
      root: ./data
      data: data.yaml          # train/val/test + names, as used by Ultralytics

``data.yaml`` keys understood: ``path`` (dataset root), ``train``, ``val``,
``test`` (a directory of images, a ``.txt`` file listing images, or a list
of either), ``names`` (list or ``{id: name}`` mapping), ``nc``.

Label files are resolved the same way Ultralytics does: the last ``/images/``
path component is replaced by ``/labels/`` and the extension by ``.txt``.
Each line is ``class cx cy w h`` (normalised) or ``class x1 y1 x2 y2 ...``
for polygons. Absolute-pixel boxes are filled in later by the engine once
image sizes are known.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..model import Annotation, BBox, Confidence, Dataset, Finding, Sample, Severity
from ..registry import adapters
from .base import DatasetAdapter, LoadResult, is_image_file

log = logging.getLogger(__name__)

_DATA_FILES = ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")


def img2label_path(image_path: Path) -> Path:
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    text = str(image_path)
    if sa in text:
        head, tail = text.rsplit(sa, 1)
        text = head + sb + tail
    return Path(text).with_suffix(".txt")


@adapters.register("yolo")
class YoloAdapter(DatasetAdapter):
    name = "yolo"
    modality = "image"
    description = "YOLO / Ultralytics layout (data.yaml + images/ + labels/)"

    @classmethod
    def detect(cls, root: Path) -> Optional[Dict[str, Any]]:
        for candidate in _DATA_FILES:
            p = root / candidate
            if p.exists():
                try:
                    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                if isinstance(data, dict) and ("train" in data or "val" in data) and "names" in data:
                    return {"format": "yolo", "data": candidate}
        for p in sorted(root.glob("*.y*ml")):
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if isinstance(data, dict) and ("train" in data or "val" in data) and "names" in data:
                return {"format": "yolo", "data": p.name}
        return None

    # ------------------------------------------------------------------ load
    def load(self) -> LoadResult:
        data_cfg = self.config.get("dataset.data")
        if not data_cfg:
            detected = self.detect(self.root)
            if not detected:
                raise ValueError(f"No data.yaml found under {self.root}. Set dataset.data in the config.")
            data_cfg = detected["data"]
        data_path = self.resolve(data_cfg)
        assert data_path is not None
        if not data_path.exists():
            raise FileNotFoundError(f"YOLO data file not found: {data_path}")
        data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{data_path}: expected a mapping")

        base = data_path.parent
        if data.get("path"):
            p = Path(os.path.expanduser(str(data["path"])))
            base = p if p.is_absolute() else (self.root / p if (self.root / p).exists() else base / p)

        categories = self._parse_names(data.get("names"), data.get("nc"))
        kpt_shape = self._parse_kpt_shape(data.get("kpt_shape"))
        splits_present = [k for k in ("train", "val", "test") if data.get(k)]
        extra = self.config.section("dataset.splits")
        for k in extra:
            if k not in splits_present:
                splits_present.append(k)

        samples: List[Sample] = []
        findings: List[Finding] = []
        stats: Dict[str, Any] = {"splits": {}}
        limit = self.max_samples()
        source: Dict[str, Any] = {"format": "yolo", "root": str(self.root), "data": str(data_path), "base": str(base), "splits": {}}
        if kpt_shape:
            source["kpt_shape"] = list(kpt_shape)

        for split_name in splits_present:
            spec = extra.get(split_name, data.get(split_name))
            image_paths, listing = self._collect_images(spec, base, data_path.parent)
            source["splits"][split_name] = listing
            split_samples: List[Sample] = []
            seen_paths: set = set()
            missing_labels = 0
            for img in image_paths:
                if img in seen_paths:
                    continue
                seen_paths.add(img)
                rel = self.display_uri(img, self.root)
                sid = f"{split_name}:{rel}"
                label_path = img2label_path(img)
                sample = Sample(
                    id=sid,
                    split=split_name,
                    uri=rel,
                    path=img,
                    native_id=rel,
                    metadata={"label_file": str(label_path), "label_file_exists": label_path.exists()},
                )
                if label_path.exists():
                    self._parse_label_file(sample, label_path, categories, findings, kpt_shape)
                else:
                    missing_labels += 1
                split_samples.append(sample)
                if limit and len(split_samples) >= limit:
                    break
            samples.extend(split_samples)
            stats["splits"][split_name] = {
                "images": len(split_samples),
                "annotations": sum(len(s.annotations) for s in split_samples),
                "missing_label_files": missing_labels,
            }

        dataset = Dataset(
            name=self.config.get("dataset.name") or self.root.name or "dataset",
            samples=samples,
            splits=splits_present,
            modality="image",
            categories=categories,
            source=source,
        )
        return LoadResult(dataset=dataset, findings=findings, stats=stats)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _parse_names(names: Any, nc: Any) -> Dict[Any, str]:
        cats: Dict[Any, str] = {}
        if isinstance(names, dict):
            for k, v in names.items():
                try:
                    cats[int(k)] = str(v)
                except (TypeError, ValueError):
                    cats[k] = str(v)
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                cats[i] = str(v)
        elif nc:
            for i in range(int(nc)):
                cats[i] = str(i)
        return cats

    def _collect_images(self, spec: Any, base: Path, yaml_dir: Path) -> Tuple[List[Path], Any]:
        if spec is None:
            return [], None
        entries = spec if isinstance(spec, (list, tuple)) else [spec]
        images: List[Path] = []
        resolved: List[str] = []
        for entry in entries:
            p = Path(os.path.expanduser(str(entry)))
            candidates = [p] if p.is_absolute() else [base / p, yaml_dir / p, self.root / p]
            target = next((c for c in candidates if c.exists()), candidates[0])
            resolved.append(str(target))
            if target.is_dir():
                images.extend(sorted(q for q in target.rglob("*") if q.is_file() and is_image_file(q)))
            elif target.is_file() and target.suffix.lower() == ".txt":
                for line in target.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    lp = Path(os.path.expanduser(line))
                    if lp.is_absolute():
                        images.append(lp)
                        continue
                    for c in (base / lp, target.parent / lp, self.root / lp):
                        if c.exists():
                            images.append(c)
                            break
                    else:
                        images.append(base / lp)
            elif target.is_file() and is_image_file(target):
                images.append(target)
            else:
                log.warning("YOLO split entry %s does not exist (resolved to %s)", entry, target)
        return images, resolved

    @staticmethod
    def _parse_kpt_shape(value: Any) -> Optional[Tuple[int, int]]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                n, d = int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
            if n > 0 and d in (2, 3):
                return n, d
        return None

    def _parse_label_file(self, sample: Sample, label_path: Path, categories: Dict[Any, str], findings: List[Finding],
                          kpt_shape: Optional[Tuple[int, int]] = None) -> None:
        try:
            text = label_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            sample.metadata["label_file_error"] = str(exc)
            return
        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            attributes: Dict[str, Any] = {"line": line_no}
            cat_id: Any = None
            bbox_norm: Optional[List[float]] = None
            try:
                cat_id = int(float(tokens[0]))
                values = [float(t) for t in tokens[1:]]
            except ValueError:
                attributes["malformed"] = "non-numeric token"
                attributes["raw"] = line[:200]
                values = []
            if not attributes.get("malformed"):
                if len(values) == 4:
                    bbox_norm = values
                elif kpt_shape is not None and len(values) > 4:
                    # pose dataset: cx cy w h then n keypoints of dimension d
                    n, d = kpt_shape
                    bbox_norm = values[:4]
                    attributes["keypoints_norm"] = values[4:]
                    attributes["keypoint_dim"] = d
                    if len(values) - 4 != n * d:
                        attributes["keypoints_error"] = f"expected {n * d} keypoint values ({n}x{d}), got {len(values) - 4}"
                elif len(values) >= 6 and len(values) % 2 == 0:
                    xs, ys = values[0::2], values[1::2]
                    attributes["polygon_norm"] = values
                    bbox_norm = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
                elif len(values) > 5 and len(values) % 3 == 2:
                    # keypoints without kpt_shape: cx cy w h then (x y v)* -> box first
                    bbox_norm = values[:4]
                    attributes["keypoints_norm"] = values[4:]
                    attributes["keypoint_dim"] = 3
                else:
                    attributes["malformed"] = f"expected 4 box values or an even number (>=6) of polygon values, got {len(values)}"
                    attributes["raw"] = line[:200]
            if bbox_norm is not None:
                attributes["bbox_norm"] = bbox_norm
            sample.annotations.append(
                Annotation(
                    id=f"{sample.id}#{line_no}",
                    sample_id=sample.id,
                    category=categories.get(cat_id) if cat_id is not None else None,
                    category_id=cat_id,
                    bbox=None,
                    attributes=attributes,
                )
            )
        if not text.strip():
            sample.metadata["label_file_empty"] = True


def fill_absolute_bboxes(sample: Sample, width: int, height: int) -> None:
    """Convert normalised YOLO boxes to absolute pixels once image size is known."""
    for ann in sample.annotations:
        norm = ann.attributes.get("bbox_norm")
        if ann.bbox is None and norm and len(norm) == 4:
            cx, cy, w, h = norm
            ann.bbox = BBox((cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height)


def _unused(*_: Any) -> None:  # keep imports referenced for type checkers
    return None


_unused(Confidence, Severity)
