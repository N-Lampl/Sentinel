"""COCO adapter.

Configuration::

    dataset:
      format: coco
      root: ./data
      splits:
        train: {annotations: annotations/instances_train.json, images: train}
        val:   annotations/instances_val.json        # images dir inferred

When ``splits`` is empty the adapter looks for ``annotations/*.json`` (or
``*.json`` in the root) and infers split names from the file names.

Every image entry becomes one :class:`Sample`; every annotation entry becomes
one :class:`Annotation` with ``bbox`` in absolute pixels. Extra keys on the
image entry (``date_captured``, ``license``, custom fields such as
``patient_id``) are copied to ``sample.metadata`` so group / time checks can
use them without a sidecar.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import split_role
from ..model import Annotation, BBox, Confidence, Dataset, Finding, Sample, SampleRef, Severity
from ..registry import adapters
from .base import DatasetAdapter, LoadResult, is_image_file

log = logging.getLogger(__name__)

_IMAGE_KNOWN = {"id", "file_name", "width", "height"}
_ANNOTATION_PREFIXES = ("instances_", "person_keypoints_", "captions_", "annotations_", "labels_", "stuff_", "panoptic_")


def _strip_prefix(stem: str) -> str:
    """``instances_train2017`` -> ``train2017`` (keeps the informative tail)."""
    for prefix in _ANNOTATION_PREFIXES:
        if stem.lower().startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@adapters.register("coco")
class CocoAdapter(DatasetAdapter):
    name = "coco"
    modality = "image"
    description = "COCO object detection / segmentation / keypoint JSON (one file per split)"

    # ------------------------------------------------------------------ detect
    @classmethod
    def detect(cls, root: Path) -> Optional[Dict[str, Any]]:
        candidates = cls._find_json_candidates(root)
        if not candidates:
            return None
        splits: Dict[str, Any] = {}
        for name, path in candidates:
            splits[name] = {"annotations": str(path.relative_to(root))}
        return {"format": "coco", "splits": splits}

    @staticmethod
    def _find_json_candidates(root: Path) -> List[Tuple[str, Path]]:
        found: List[Tuple[str, Path]] = []
        search_dirs = [root / "annotations", root]
        seen = set()
        for d in search_dirs:
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                if p in seen:
                    continue
                seen.add(p)
                try:
                    with p.open("rb") as fh:
                        head = fh.read(4096).decode("utf-8", "ignore")
                except OSError:
                    continue
                if '"images"' not in head and '"annotations"' not in head:
                    # cheap check first; fall back to a full parse for small files
                    if p.stat().st_size > 5_000_000:
                        continue
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not (isinstance(data, dict) and "images" in data):
                        continue
                found.append((_strip_prefix(p.stem), p))
        return found

    # ------------------------------------------------------------------ load
    def load(self) -> LoadResult:
        splits_cfg = self.config.section("dataset.splits")
        if not splits_cfg:
            detected = self.detect(self.root)
            if not detected:
                raise ValueError(
                    f"No COCO annotation files found under {self.root}. Set dataset.splits in the config."
                )
            splits_cfg = detected["splits"]
            log.info("auto-detected COCO splits: %s", ", ".join(splits_cfg))

        samples: List[Sample] = []
        findings: List[Finding] = []
        categories: Dict[Any, str] = {}
        category_keypoints: Dict[Any, int] = {}
        per_split_categories: Dict[str, Dict[Any, str]] = {}
        stats: Dict[str, Any] = {"splits": {}}
        limit = self.max_samples()
        source: Dict[str, Any] = {"format": "coco", "root": str(self.root), "splits": {}}

        for split_name, spec in splits_cfg.items():
            ann_path, img_dir = self._resolve_split(split_name, spec)
            source["splits"][split_name] = {"annotations": str(ann_path), "images": str(img_dir) if img_dir else None}
            if not ann_path.exists():
                raise FileNotFoundError(f"COCO annotation file for split {split_name!r} not found: {ann_path}")
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{ann_path}: expected a JSON object at the top level")

            split_categories: Dict[Any, str] = {}
            for cat in data.get("categories", []) or []:
                if isinstance(cat, dict) and "id" in cat:
                    name = str(cat.get("name", cat["id"]))
                    split_categories[cat["id"]] = name
                    if cat["id"] in categories and categories[cat["id"]] != name:
                        findings.append(self._category_conflict(split_name, cat["id"], categories[cat["id"]], name))
                    categories.setdefault(cat["id"], name)
                    if isinstance(cat.get("keypoints"), list) and cat["keypoints"]:
                        category_keypoints.setdefault(cat["id"], len(cat["keypoints"]))
            per_split_categories[split_name] = split_categories

            images = data.get("images", []) or []
            anns = data.get("annotations", []) or []
            if img_dir is None:
                img_dir = self._guess_image_dir(split_name, ann_path, images)
                source["splits"][split_name]["images"] = str(img_dir)

            by_native: Dict[Any, Sample] = {}
            names_seen: Dict[str, str] = {}
            split_samples: List[Sample] = []
            for entry in images:
                if not isinstance(entry, dict) or "id" not in entry or "file_name" not in entry:
                    findings.append(self._malformed_image_entry(split_name, entry))
                    continue
                native_id = entry["id"]
                file_name = str(entry["file_name"])
                sid = f"{split_name}:{native_id}"
                if native_id in by_native:
                    findings.append(self._duplicate_registration(split_name, sid, by_native[native_id], file_name, "id"))
                    continue
                path = img_dir / file_name
                uri = self.display_uri(path, self.root)
                metadata: Dict[str, Any] = {}
                if _is_number(entry.get("width")):
                    metadata["width"] = entry["width"]
                if _is_number(entry.get("height")):
                    metadata["height"] = entry["height"]
                for k, v in entry.items():
                    if k not in _IMAGE_KNOWN and v not in (None, ""):
                        metadata[k] = v
                sample = Sample(id=sid, split=split_name, uri=uri, path=path, native_id=native_id, metadata=metadata)
                if file_name in names_seen:
                    findings.append(
                        self._duplicate_registration(split_name, sid, by_native[names_seen[file_name]], file_name, "file_name")
                    )
                names_seen.setdefault(file_name, native_id)
                by_native[native_id] = sample
                split_samples.append(sample)
                if limit and len(split_samples) >= limit:
                    break

            ann_ids_seen: Dict[Any, int] = {}
            orphans = 0
            for raw in anns:
                if not isinstance(raw, dict):
                    findings.append(self._malformed_annotation(split_name, raw))
                    continue
                image_id = raw.get("image_id")
                sample = by_native.get(image_id)
                if sample is None:
                    orphans += 1
                    if orphans <= 50:
                        findings.append(self._orphan_annotation(split_name, raw))
                    continue
                ann_id = raw.get("id")
                key = ann_id if ann_id is not None else f"__missing__{len(ann_ids_seen)}"
                ann_ids_seen[key] = ann_ids_seen.get(key, 0) + 1
                cat_id = raw.get("category_id")
                bbox, bbox_error = self._parse_bbox(raw.get("bbox"))
                attributes: Dict[str, Any] = {}
                for k in ("area", "iscrowd", "attributes", "score", "num_keypoints", "keypoints"):
                    if k in raw:
                        attributes[k] = raw[k]
                if bbox_error:
                    attributes["bbox_error"] = bbox_error
                    attributes["bbox_raw"] = raw.get("bbox")
                if ann_id is None:
                    attributes["missing_id"] = True
                seg = raw.get("segmentation")
                if seg is not None:
                    attributes["segmentation_type"] = "rle" if isinstance(seg, dict) else "polygon"
                sample.annotations.append(
                    Annotation(
                        id=str(ann_id) if ann_id is not None else f"{sample.id}#{len(sample.annotations)}",
                        sample_id=sample.id,
                        category=categories.get(cat_id),
                        category_id=cat_id,
                        bbox=bbox,
                        segmentation=seg,
                        attributes=attributes,
                    )
                )
            dup_ann_ids = {k: v for k, v in ann_ids_seen.items() if v > 1 and not str(k).startswith("__missing__")}
            if dup_ann_ids:
                findings.append(self._duplicate_annotation_ids(split_name, dup_ann_ids))
            if orphans > 50:
                findings.append(
                    Finding(
                        detector="adapter:coco",
                        kind="orphan_annotation_summary",
                        title=f"{orphans} annotations reference missing images in split {split_name}",
                        severity=self.config.severity("policy.labels.orphan_annotation", Severity.ERROR) or Severity.INFO,
                        confidence=Confidence.DETERMINISTIC,
                        policy=self.config.policy_label("policy.labels.orphan_annotation"),
                        policy_description="Every annotation must reference an image that exists in the same split file.",
                        message=f"{orphans} annotations in {ann_path.name} have an image_id that is not present in 'images' (first 50 reported individually).",
                        remediation="Remove orphaned annotations or add the missing image entries.",
                        splits=[split_name],
                        evidence={"count": orphans, "file": str(ann_path)},
                    )
                )
            samples.extend(split_samples)
            stats["splits"][split_name] = {
                "images": len(split_samples),
                "annotations": sum(len(s.annotations) for s in split_samples),
                "orphan_annotations": orphans,
            }

        if len(per_split_categories) > 1:
            findings.extend(self._category_set_differences(per_split_categories))
        if category_keypoints:
            source["category_keypoints"] = {str(k): v for k, v in category_keypoints.items()}

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
    def _category_conflict(self, split_name: str, cat_id: Any, existing: str, new: str) -> Finding:
        return Finding(
            detector="adapter:coco",
            kind="category_definition_conflict",
            title=f"Category id {cat_id} is '{existing}' in one split file and '{new}' in {split_name}",
            severity=self.config.severity("policy.labels.category_mismatch", Severity.ERROR) or Severity.INFO,
            confidence=Confidence.DETERMINISTIC,
            policy=self.config.policy_label("policy.labels.category_mismatch"),
            policy_description="Category ids must map to the same name in every split file; otherwise labels mean different things per split.",
            message=f"categories[{cat_id}] is named {existing!r} in an earlier split file but {new!r} in the {split_name} file.",
            remediation="Use one shared category list for all split files (re-export or remap ids).",
            splits=[split_name],
            evidence={"category_id": cat_id, "names": [existing, new]},
        )

    def _category_set_differences(self, per_split: Dict[str, Dict[Any, str]]) -> List[Finding]:
        all_ids = set()
        for cats in per_split.values():
            all_ids |= set(cats)
        out: List[Finding] = []
        for split_name, cats in per_split.items():
            missing = sorted(all_ids - set(cats), key=str)
            if not missing:
                continue
            names = {cid: n for c in per_split.values() for cid, n in c.items()}
            out.append(
                Finding(
                    detector="adapter:coco",
                    kind="category_set_mismatch",
                    title=f"Split {split_name} declares {len(cats)} categories; {len(missing)} declared elsewhere are missing",
                    severity=self.config.severity("policy.labels.category_set_mismatch", Severity.WARNING) or Severity.INFO,
                    confidence=Confidence.DETERMINISTIC,
                    policy=self.config.policy_label("policy.labels.category_set_mismatch"),
                    policy_description="Every split file should declare the same category list.",
                    message=f"Categories missing from the {split_name} file: " + ", ".join(f"{cid} ({names.get(cid)})" for cid in missing[:15]) + (" ..." if len(missing) > 15 else ""),
                    remediation="Use one shared category list for all split files.",
                    splits=[split_name],
                    evidence={"missing_ids": [str(c) for c in missing], "declared": len(cats), "total": len(all_ids)},
                )
            )
        return out

    def _resolve_split(self, split_name: str, spec: Any) -> Tuple[Path, Optional[Path]]:
        if isinstance(spec, dict):
            ann = spec.get("annotations") or spec.get("json") or spec.get("file")
            if not ann:
                raise ValueError(f"dataset.splits.{split_name} needs an 'annotations' path")
            ann_path = self.resolve(ann)
            img = spec.get("images") or spec.get("image_dir") or spec.get("dir")
            img_dir = self.resolve(img) if img else None
        else:
            ann_path = self.resolve(spec)
            img_dir = None
        assert ann_path is not None
        return ann_path, img_dir

    def _guess_image_dir(self, split_name: str, ann_path: Path, images: List[Dict[str, Any]]) -> Path:
        sample_name: Optional[str] = None
        for entry in images:
            if isinstance(entry, dict) and entry.get("file_name"):
                sample_name = str(entry["file_name"])
                break
        base_dirs = [ann_path.parent.parent, ann_path.parent, self.root]
        names = [split_name, _strip_prefix(ann_path.stem), ann_path.stem, "images", "imgs", "data", ""]
        role = split_role(split_name)
        if role:
            names.insert(1, role)
        tried: List[Path] = []
        for base in base_dirs:
            for name in names:
                for sub in ("", "images"):
                    d = base / sub / name if sub else base / name
                    if d in tried:
                        continue
                    tried.append(d)
                    if d.is_dir() and (sample_name is None or (d / sample_name).exists()):
                        return d
        # fall back to the first existing directory candidate, else root/split
        for d in tried:
            if d.is_dir():
                return d
        return self.root / split_name

    @staticmethod
    def _parse_bbox(value: Any) -> Tuple[Optional[BBox], Optional[str]]:
        if value is None:
            return None, None
        if not isinstance(value, (list, tuple)):
            return None, f"bbox is not a list: {type(value).__name__}"
        if len(value) != 4:
            return None, f"bbox has {len(value)} values, expected 4"
        try:
            x, y, w, h = (float(v) for v in value)
        except (TypeError, ValueError):
            return None, "bbox contains non-numeric values"
        for v in (x, y, w, h):
            if v != v or v in (float("inf"), float("-inf")):
                return None, "bbox contains NaN or infinity"
        return BBox(x, y, w, h), None

    # ------------------------------------------------------------------ findings
    def _malformed_image_entry(self, split_name: str, entry: Any) -> Finding:
        return Finding(
            detector="adapter:coco",
            kind="malformed_image_entry",
            title=f"Malformed image entry in split {split_name}",
            severity=self.config.severity("policy.labels.malformed_label", Severity.ERROR) or Severity.INFO,
            confidence=Confidence.DETERMINISTIC,
            policy=self.config.policy_label("policy.labels.malformed_label"),
            policy_description="Image entries must be objects with 'id' and 'file_name'.",
            message=f"Skipped image entry without id/file_name: {str(entry)[:200]}",
            remediation="Fix the entry in the annotation file.",
            splits=[split_name],
            evidence={"entry": str(entry)[:500]},
        )

    def _malformed_annotation(self, split_name: str, raw: Any) -> Finding:
        return Finding(
            detector="adapter:coco",
            kind="malformed_annotation",
            title=f"Malformed annotation entry in split {split_name}",
            severity=self.config.severity("policy.labels.malformed_label", Severity.ERROR) or Severity.INFO,
            confidence=Confidence.DETERMINISTIC,
            policy=self.config.policy_label("policy.labels.malformed_label"),
            policy_description="Annotation entries must be JSON objects.",
            message=f"Skipped annotation that is not an object: {str(raw)[:200]}",
            remediation="Fix or remove the entry in the annotation file.",
            splits=[split_name],
            evidence={"entry": str(raw)[:500]},
        )

    def _orphan_annotation(self, split_name: str, raw: Dict[str, Any]) -> Finding:
        return Finding(
            detector="adapter:coco",
            kind="orphan_annotation",
            title=f"Annotation {raw.get('id')} references missing image {raw.get('image_id')}",
            severity=self.config.severity("policy.labels.orphan_annotation", Severity.ERROR) or Severity.INFO,
            confidence=Confidence.DETERMINISTIC,
            policy=self.config.policy_label("policy.labels.orphan_annotation"),
            policy_description="Every annotation must reference an image that exists in the same split file.",
            message=f"Annotation id={raw.get('id')} has image_id={raw.get('image_id')} which is not in 'images' for split {split_name}.",
            remediation="Remove the orphaned annotation or add the missing image entry.",
            splits=[split_name],
            evidence={"annotation_id": raw.get("id"), "image_id": raw.get("image_id"), "category_id": raw.get("category_id")},
        )

    def _duplicate_registration(self, split_name: str, sid: str, existing: Sample, file_name: str, by: str) -> Finding:
        return Finding(
            detector="adapter:coco",
            kind="duplicate_image_registration",
            title=f"Image registered twice by {by} in split {split_name}",
            severity=self.config.severity("policy.consistency.duplicate_registration", Severity.ERROR) or Severity.INFO,
            confidence=Confidence.DETERMINISTIC,
            policy=self.config.policy_label("policy.consistency.duplicate_registration"),
            policy_description="Each image id and file name must appear once per split.",
            message=f"'{file_name}' (id {sid.split(':', 1)[1]}) duplicates an existing image entry ({existing.uri}) by {by}.",
            remediation="Remove the duplicate image entry and merge its annotations.",
            samples=[SampleRef(existing.id, existing.split, existing.uri)],
            splits=[split_name],
            evidence={"file_name": file_name, "duplicate_by": by},
        )

    def _duplicate_annotation_ids(self, split_name: str, dups: Dict[Any, int]) -> Finding:
        sample_ids = list(dups.items())[:20]
        return Finding(
            detector="adapter:coco",
            kind="duplicate_annotation_id",
            title=f"{len(dups)} duplicate annotation ids in split {split_name}",
            severity=self.config.severity("policy.labels.duplicate_annotation_id", Severity.ERROR) or Severity.INFO,
            confidence=Confidence.DETERMINISTIC,
            policy=self.config.policy_label("policy.labels.duplicate_annotation_id"),
            policy_description="Annotation ids must be unique within a split file.",
            message=f"Annotation ids used more than once: {', '.join(str(k) for k, _ in sample_ids)}{' ...' if len(dups) > 20 else ''}",
            remediation="Re-number the annotations so that every id is unique.",
            splits=[split_name],
            evidence={"duplicate_ids": {str(k): v for k, v in sample_ids}, "count": len(dups)},
        )


def is_coco_root(root: Path) -> bool:
    return bool(CocoAdapter._find_json_candidates(root))


__all__ = ["CocoAdapter", "is_coco_root", "is_image_file"]
