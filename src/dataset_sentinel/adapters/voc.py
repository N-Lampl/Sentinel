"""Pascal VOC adapter.

Layout::

    root/
      Annotations/<id>.xml
      JPEGImages/<id>.jpg          (or images/)
      ImageSets/Main/train.txt     one image id per line (per-class files such as
      ImageSets/Main/val.txt        cat_train.txt are ignored)

Configuration::

    dataset:
      format: voc
      root: ./VOC2012
      splits:                       # optional, auto-detected from ImageSets/Main
        train: ImageSets/Main/train.txt
        val: {list: ImageSets/Main/val.txt, annotations: Annotations, images: JPEGImages}
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..model import Annotation, BBox, Confidence, Dataset, Finding, Sample, Severity
from ..registry import adapters
from .base import IMAGE_EXTENSIONS, DatasetAdapter, LoadResult

log = logging.getLogger(__name__)

_SPLIT_FILES = ("train", "val", "test", "trainval")


def _text(node: Optional[ET.Element], default: Any = None) -> Any:
    return node.text.strip() if node is not None and node.text is not None else default


def _num(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@adapters.register("voc")
class VocAdapter(DatasetAdapter):
    name = "voc"
    modality = "image"
    description = "Pascal VOC (Annotations/*.xml, JPEGImages/, ImageSets/Main/<split>.txt)"

    @classmethod
    def detect(cls, root: Path) -> Optional[Dict[str, Any]]:
        if not (root / "Annotations").is_dir():
            return None
        main = root / "ImageSets" / "Main"
        if not main.is_dir():
            return None
        splits = {name: f"ImageSets/Main/{name}.txt" for name in _SPLIT_FILES if (main / f"{name}.txt").exists()}
        if "trainval" in splits and "train" in splits and "val" in splits:
            del splits["trainval"]
        if not splits:
            return None
        return {"format": "voc", "splits": splits}

    def load(self) -> LoadResult:
        splits_cfg = self.config.section("dataset.splits")
        if not splits_cfg:
            detected = self.detect(self.root)
            if not detected:
                raise ValueError(f"No VOC layout (Annotations/ + ImageSets/Main/*.txt) found under {self.root}.")
            splits_cfg = detected["splits"]

        samples: List[Sample] = []
        findings: List[Finding] = []
        categories: Dict[Any, str] = {}
        stats: Dict[str, Any] = {"splits": {}}
        limit = self.max_samples()
        source: Dict[str, Any] = {"format": "voc", "root": str(self.root), "splits": {}}

        for split_name, spec in splits_cfg.items():
            list_path, ann_dir, img_dir = self._resolve_split(spec)
            source["splits"][split_name] = {"list": str(list_path), "annotations": str(ann_dir), "images": str(img_dir)}
            if not list_path.exists():
                raise FileNotFoundError(f"VOC image-set file for split {split_name!r} not found: {list_path}")
            ids = []
            for line in list_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line.split()[0])
            split_samples: List[Sample] = []
            missing_xml = 0
            for image_id in ids:
                xml_path = ann_dir / f"{image_id}.xml"
                sid = f"{split_name}:{image_id}"
                metadata: Dict[str, Any] = {"voc_id": image_id}
                anns: List[Annotation] = []
                file_name: Optional[str] = None
                if xml_path.exists():
                    try:
                        file_name, size, objects, meta = self._parse_xml(xml_path)
                    except ET.ParseError as exc:
                        findings.append(self._malformed_xml(split_name, xml_path, str(exc)))
                        file_name, size, objects, meta = None, None, [], {}
                    metadata.update(meta)
                    if size:
                        metadata["width"], metadata["height"] = size
                    for k, (name, bbox, attrs, err) in enumerate(objects):
                        categories.setdefault(name, name)
                        if err:
                            attrs["bbox_error"] = err
                        anns.append(Annotation(id=f"{sid}#{k}", sample_id=sid, category=name, category_id=name, bbox=bbox, attributes=attrs))
                else:
                    missing_xml += 1
                    metadata["label_file"] = str(xml_path)
                    metadata["label_file_exists"] = False
                path = self._find_image(img_dir, image_id, file_name)
                uri = self.display_uri(path, self.root)
                sample = Sample(id=sid, split=split_name, uri=uri, path=path, native_id=image_id, annotations=anns, metadata=metadata)
                for a in anns:
                    a.sample_id = sample.id
                split_samples.append(sample)
                if limit and len(split_samples) >= limit:
                    break
            samples.extend(split_samples)
            stats["splits"][split_name] = {"images": len(split_samples), "annotations": sum(len(s.annotations) for s in split_samples), "missing_xml": missing_xml}

        dataset = Dataset(
            name=self.config.get("dataset.name") or self.root.name or "dataset",
            samples=samples,
            splits=list(splits_cfg.keys()),
            modality="image",
            categories=categories,
            source=source,
        )
        return LoadResult(dataset=dataset, findings=findings, stats=stats)

    # ------------------------------------------------------------------ helpers
    def _resolve_split(self, spec: Any) -> Tuple[Path, Path, Path]:
        ann_dir = self.root / "Annotations"
        img_dir = self.root / "JPEGImages"
        if not img_dir.is_dir() and (self.root / "images").is_dir():
            img_dir = self.root / "images"
        if isinstance(spec, dict):
            list_path = self.resolve(spec.get("list") or spec.get("file") or spec.get("ids"))
            if spec.get("annotations"):
                ann_dir = self.resolve(spec["annotations"]) or ann_dir
            if spec.get("images"):
                img_dir = self.resolve(spec["images"]) or img_dir
        else:
            list_path = self.resolve(spec)
        if list_path is None:
            raise ValueError("VOC split needs a path to an ImageSets list file")
        return list_path, ann_dir, img_dir

    @staticmethod
    def _find_image(img_dir: Path, image_id: str, file_name: Optional[str]) -> Path:
        if file_name:
            candidate = img_dir / file_name
            if candidate.exists():
                return candidate
        for ext in (".jpg", ".jpeg", ".png", *sorted(IMAGE_EXTENSIONS)):
            candidate = img_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        return img_dir / (file_name or f"{image_id}.jpg")

    @staticmethod
    def _parse_xml(path: Path) -> Tuple[Optional[str], Optional[Tuple[int, int]], List[Tuple[str, Optional[BBox], Dict[str, Any], Optional[str]]], Dict[str, Any]]:
        tree = ET.parse(path)
        root = tree.getroot()
        file_name = _text(root.find("filename"))
        size = None
        size_node = root.find("size")
        if size_node is not None:
            w, h = _num(_text(size_node.find("width"))), _num(_text(size_node.find("height")))
            if w and h and w > 0 and h > 0:
                size = (int(w), int(h))
        meta: Dict[str, Any] = {}
        for key in ("folder", "segmented"):
            v = _text(root.find(key))
            if v is not None:
                meta[key] = v
        src = root.find("source")
        if src is not None:
            for child in src:
                if child.text and child.text.strip():
                    meta[f"source_{child.tag}"] = child.text.strip()
        objects = []
        for obj in root.findall("object"):
            name = _text(obj.find("name"), "unknown")
            attrs: Dict[str, Any] = {}
            for key in ("difficult", "truncated", "occluded", "pose"):
                v = _text(obj.find(key))
                if v is not None:
                    attrs[key] = v
            box = obj.find("bndbox")
            bbox: Optional[BBox] = None
            err: Optional[str] = None
            if box is None:
                err = "object without bndbox"
            else:
                vals = [_num(_text(box.find(k))) for k in ("xmin", "ymin", "xmax", "ymax")]
                if any(v is None for v in vals):
                    err = "bndbox with missing or non-numeric coordinates"
                else:
                    xmin, ymin, xmax, ymax = vals  # type: ignore[misc]
                    bbox = BBox(xmin, ymin, xmax - xmin, ymax - ymin)
            objects.append((str(name), bbox, attrs, err))
        return file_name, size, objects, meta

    def _malformed_xml(self, split_name: str, path: Path, error: str) -> Finding:
        return Finding(
            detector="adapter:voc",
            kind="malformed_annotation",
            title=f"Malformed VOC annotation file {path.name}",
            severity=self.config.severity("policy.labels.malformed_label", Severity.ERROR) or Severity.INFO,
            confidence=Confidence.DETERMINISTIC,
            policy=self.config.policy_label("policy.labels.malformed_label"),
            policy_description="Annotation XML files must parse.",
            message=f"{path}: {error}",
            remediation="Fix or regenerate the XML file.",
            splits=[split_name],
            evidence={"file": str(path), "error": error},
        )
