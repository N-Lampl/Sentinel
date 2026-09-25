"""Label, annotation and file validity.

Deterministic structural checks on samples and their annotations:

* missing / unreadable / empty / blank image files;
* declared image size differs from the actual file (COCO);
* samples with no annotations, empty splits, missing YOLO label files;
* malformed, degenerate (zero-area / tiny), out-of-bounds bounding boxes;
* unknown category ids and categories outside ``allowed_categories``;
* invalid segmentation polygons, duplicated annotations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..model import Annotation, Confidence, Sample, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap

_P = "policy.labels."


@detectors.register("label_validity")
class LabelValidityDetector(Detector):
    name = "label_validity"
    description = "Malformed, missing, empty, out-of-bounds or out-of-policy labels and files"
    modalities = ("image",)
    needs_fingerprints = True

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        cfg = ctx.config
        sev: Dict[str, Optional[Severity]] = {
            k: cfg.severity(_P + k)
            for k in (
                "missing_file", "unreadable_image", "empty_image", "blank_image", "size_mismatch",
                "missing_annotations", "no_valid_annotations", "empty_split", "invalid_bbox", "out_of_bounds_bbox", "degenerate_bbox",
                "unknown_category", "duplicate_annotation", "invalid_segmentation", "malformed_label", "missing_label_file",
                "invalid_keypoints", "keypoints_out_of_bounds",
            )
        }
        category_keypoints = {str(k): int(v) for k, v in (ctx.dataset.source.get("category_keypoints") or {}).items()}
        kpt_shape = ctx.dataset.source.get("kpt_shape")
        min_px = float(cfg.get(_P + "min_bbox_size_px", 1.0) or 0)
        allowed = cfg.get(_P + "allowed_categories")
        allowed_set = {str(a) for a in allowed} if allowed else None
        cap = FindingCap(int(cfg.get("output.max_findings_per_kind") or 1000))
        sev_for: Dict[str, Severity] = {}
        counts: Dict[str, int] = defaultdict(int)
        store = ctx.fingerprints
        categories = ctx.dataset.categories

        def emit(kind: str, sample: Optional[Sample], title: str, message: str, remediation: str, desc: str, evidence: Dict[str, Any], policy_key: str, splits=None) -> None:
            s = sev.get(policy_key)
            counts[kind] += 1
            if s is None:
                return
            sev_for[kind] = s
            if not cap.allow(kind):
                return
            res.add(
                self.finding(
                    kind=kind,
                    title=title,
                    severity=s,
                    confidence=Confidence.DETERMINISTIC,
                    policy=cfg.policy_label(_P + policy_key),
                    policy_description=desc,
                    message=message,
                    remediation=remediation,
                    samples=[sample.ref()] if sample else [],
                    evidence=evidence,
                    splits=splits,
                )
            )

        # ---------------------------------------------------------------- splits
        sizes = ctx.dataset.split_sizes()
        for split, n in sizes.items():
            if n == 0:
                emit("empty_split", None, f"Split {split} contains no samples", f"Split {split!r} has zero samples.",
                     "Check the split configuration / annotation file.", "Every configured split must contain samples.",
                     {"split": split}, "empty_split", splits=[split])

        no_annotations: Dict[str, List[Sample]] = defaultdict(list)
        missing_labels: Dict[str, List[Sample]] = defaultdict(list)
        unknown_cats: Dict[Any, List[str]] = defaultdict(list)

        for s in ctx.dataset.samples:
            fp = store.get(s.id)
            # ---------------------------------------------------------- files
            if fp is not None and not fp.ok:
                err = fp.error or "unknown error"
                if err == "file not found":
                    emit("missing_file", s, f"Image file missing: {s.uri}", f"{s.split}:{s.uri} does not exist on disk ({fp.path}).",
                         "Restore the file or remove the sample from the annotation file.", "Every sample must point to an existing file.",
                         {"path": fp.path}, "missing_file")
                elif err.startswith("empty file"):
                    emit("empty_image", s, f"Empty image file: {s.uri}", f"{s.split}:{s.uri} is 0 bytes.",
                         "Re-export the image or remove the sample.", "Image files must contain data.", {"path": fp.path}, "empty_image")
                else:
                    emit("unreadable_image", s, f"Unreadable image: {s.uri}", f"{s.split}:{s.uri} could not be decoded: {err}.",
                         "Re-encode the file or remove the sample.", "Every image must decode with Pillow.", {"path": fp.path, "error": err}, "unreadable_image")
                continue
            if fp is not None and fp.is_blank:
                emit("blank_image", s, f"Blank / uniform image: {s.uri}", f"{s.split}:{s.uri} has no visual variation (luma std {fp.std_luma:.2f}).",
                     "Check the export pipeline; blank images usually indicate a failed conversion.", "Images should contain visual content.",
                     {"std_luma": fp.std_luma, "mean_luma": fp.mean_luma}, "blank_image")
            width = s.width or (fp.width if fp else None)
            height = s.height or (fp.height if fp else None)
            if fp is not None and s.width and s.height and (s.width != fp.width or s.height != fp.height):
                emit("size_mismatch", s, f"Declared size differs from file: {s.uri}",
                     f"{s.split}:{s.uri} is declared {s.width}x{s.height} but the file is {fp.width}x{fp.height}.",
                     "Fix width/height in the annotation file (boxes may be scaled wrongly).", "Declared image size must match the file.",
                     {"declared": [s.width, s.height], "actual": [fp.width, fp.height]}, "size_mismatch")
                width, height = fp.width, fp.height

            # ---------------------------------------------------------- annotations
            if s.metadata.get("label_file_exists") is False:
                missing_labels[s.split].append(s)
            elif not s.annotations:
                no_annotations[s.split].append(s)

            seen_sig: Dict[str, List[str]] = defaultdict(list)
            for ann in s.annotations:
                self._check_annotation(ann, s, width, height, min_px, categories, allowed_set, unknown_cats, emit)
                self._check_keypoints(ann, s, width, height, category_keypoints, kpt_shape, emit)
                seen_sig[ann.signature()].append(ann.id)
            if s.annotations and all(a.attributes.get("malformed") for a in s.annotations):
                emit("no_valid_annotations", s, f"No valid annotations after parsing: {s.uri}",
                     f"{s.split}:{s.uri} has {len(s.annotations)} label line(s) and every one is malformed; the sample is effectively unlabeled.",
                     "Fix the label file or remove the sample.", "A label file should contain at least one parsable annotation.",
                     {"lines": len(s.annotations)}, "no_valid_annotations")
            dups = {sig: ids for sig, ids in seen_sig.items() if len(ids) > 1}
            if dups:
                emit("duplicate_annotation", s, f"Duplicated annotations on {s.uri}",
                     f"{s.split}:{s.uri} has {sum(len(v) for v in dups.values())} annotations that repeat the same class and box.",
                     "Remove the repeated annotations.", "An object should be annotated once.",
                     {"duplicates": {k: v[:10] for k, v in list(dups.items())[:10]}}, "duplicate_annotation")

        # ---------------------------------------------------------------- aggregates
        for split, members in no_annotations.items():
            total = sizes.get(split, 0)
            rate = len(members) / total if total else 0.0
            emit("missing_annotations", None, f"{len(members)} of {total} samples in {split} have no annotations ({rate:.0%})",
                 f"{len(members)} samples ({rate:.1%}) in split {split} carry no labels: " + ", ".join(m.uri for m in members[:8]) + (" ..." if len(members) > 8 else ""),
                 "Confirm these are intentional background images; otherwise add the missing labels.",
                 "Samples are expected to carry at least one annotation unless they are deliberate negatives.",
                 {"count": len(members), "split_total": total, "rate": round(rate, 4), "samples": [m.id for m in members[:100]]}, "missing_annotations", splits=[split])
            if sev.get("missing_annotations") is not None and res.findings:
                res.findings[-1].samples = [m.ref() for m in members[:100]]
        for split, members in missing_labels.items():
            emit("missing_label_file", None, f"{len(members)} samples in {split} have no label file",
                 f"{len(members)} images in split {split} have no matching labels/*.txt (treated as background by YOLO): "
                 + ", ".join(m.uri for m in members[:8]) + (" ..." if len(members) > 8 else ""),
                 "Create empty label files for intentional negatives so the omission is explicit; otherwise label the images.",
                 "Every YOLO image should have a label file, even if empty.",
                 {"count": len(members), "samples": [m.id for m in members[:100]]}, "missing_label_file", splits=[split])
            if sev.get("missing_label_file") is not None and res.findings:
                res.findings[-1].samples = [m.ref() for m in members[:100]]
        for cat_id, ann_ids in unknown_cats.items():
            emit("unknown_category_summary", None, f"Unknown category id {cat_id!r} used by {len(ann_ids)} annotations",
                 f"category_id={cat_id!r} is not declared in the categories/names list.",
                 "Declare the category or fix the annotations.", "Every annotation must reference a declared category.",
                 {"category_id": cat_id, "annotations": ann_ids[:50], "count": len(ann_ids)}, "unknown_category")

        res.findings.extend(cap.summaries(self, sev_for))
        res.stats = dict(counts)
        return res

    # ------------------------------------------------------------------ per annotation
    @staticmethod
    def _check_annotation(ann: Annotation, s: Sample, width: Optional[int], height: Optional[int], min_px: float,
                          categories: Dict[Any, str], allowed: Optional[set], unknown_cats: Dict[Any, List[str]], emit) -> None:
        attrs = ann.attributes
        if attrs.get("malformed"):
            emit("malformed_label", s, f"Malformed label line on {s.uri}", f"{s.split}:{s.uri} line {attrs.get('line')}: {attrs['malformed']} ({attrs.get('raw', '')!r}).",
                 "Fix the label line (class cx cy w h, normalised).", "Label lines must follow the format.", {"annotation": ann.id, "reason": attrs["malformed"]}, "malformed_label")
            return
        if categories and ann.category_id is not None and ann.category_id not in categories and ann.category is None:
            unknown_cats[ann.category_id].append(ann.id)
            emit("unknown_category", s, f"Unknown category id {ann.category_id!r} on {s.uri}",
                 f"Annotation {ann.id} on {s.split}:{s.uri} uses category_id={ann.category_id!r}, which is not declared.",
                 "Declare the category or fix the annotation.", "Every annotation must reference a declared category.",
                 {"annotation": ann.id, "category_id": ann.category_id}, "unknown_category")
        if ann.category_id is None and ann.category is None and categories:
            emit("unknown_category", s, f"Annotation without category on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has no category.",
                 "Add a category_id.", "Every annotation must reference a declared category.", {"annotation": ann.id}, "unknown_category")
        if allowed is not None and ann.category is not None and str(ann.category) not in allowed:
            emit("unknown_category", s, f"Category {ann.category!r} is not allowed on {s.uri}",
                 f"Annotation {ann.id} on {s.split}:{s.uri} has category {ann.category!r}, outside policy.labels.allowed_categories.",
                 "Remove or re-map the annotation.", "Only the configured categories are allowed.", {"annotation": ann.id, "category": ann.category}, "unknown_category")

        if attrs.get("bbox_error"):
            emit("invalid_bbox", s, f"Invalid bbox on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri}: {attrs['bbox_error']} (raw: {attrs.get('bbox_raw')!r}).",
                 "Fix the bbox to [x, y, w, h] in pixels.", "Bounding boxes must be 4 finite numbers.", {"annotation": ann.id, "raw": attrs.get("bbox_raw")}, "invalid_bbox")
        norm = attrs.get("bbox_norm")
        if norm is not None:
            cx, cy, w, h = norm
            tol = 1e-3
            if w <= 0 or h <= 0 or (width and w * width < min_px) or (height and h * height < min_px):
                emit("degenerate_bbox", s, f"Degenerate bbox on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has normalised size {w:.4f}x{h:.4f}.",
                     "Remove or fix the zero-area box.", "Boxes must have positive width and height.", {"annotation": ann.id, "bbox_norm": norm}, "degenerate_bbox")
            elif any(v < -tol or v > 1 + tol for v in norm) or cx - w / 2 < -tol or cy - h / 2 < -tol or cx + w / 2 > 1 + tol or cy + h / 2 > 1 + tol:
                emit("out_of_bounds_bbox", s, f"Out-of-bounds bbox on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has normalised box {[round(v, 4) for v in norm]} outside [0, 1].",
                     "Clip or fix the box; check that coordinates are normalised.", "Normalised boxes must lie within [0, 1].", {"annotation": ann.id, "bbox_norm": norm}, "out_of_bounds_bbox")
        elif ann.bbox is not None:
            b = ann.bbox
            if b.w <= 0 or b.h <= 0 or b.w < min_px or b.h < min_px:
                emit("degenerate_bbox", s, f"Degenerate bbox on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has size {b.w:.2f}x{b.h:.2f} px.",
                     "Remove or fix the zero-area box.", "Boxes must have positive width and height.", {"annotation": ann.id, "bbox": b.as_list()}, "degenerate_bbox")
            elif width and height and (b.x < -0.5 or b.y < -0.5 or b.x2 > width + 0.5 or b.y2 > height + 0.5):
                emit("out_of_bounds_bbox", s, f"Out-of-bounds bbox on {s.uri}",
                     f"Annotation {ann.id} on {s.split}:{s.uri} box {[round(v, 1) for v in b.as_list()]} exceeds the {width}x{height} image.",
                     "Clip the box to the image or fix the coordinates.", "Boxes must lie inside the image.", {"annotation": ann.id, "bbox": b.as_list(), "image": [width, height]}, "out_of_bounds_bbox")
        seg = ann.segmentation
        poly = attrs.get("polygon_norm")
        if poly is not None and (len(poly) < 6 or len(poly) % 2):
            emit("invalid_segmentation", s, f"Invalid polygon on {s.uri}", f"Annotation {ann.id} has {len(poly)} polygon values.", "Polygons need at least 3 points.",
                 "Segmentation polygons must have >= 3 points.", {"annotation": ann.id}, "invalid_segmentation")
        elif seg is not None:
            bad = _segmentation_problem(seg)
            if bad:
                emit("invalid_segmentation", s, f"Invalid segmentation on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri}: {bad}.", "Fix the segmentation field.",
                     "Segmentations must be valid polygons or RLE.", {"annotation": ann.id, "problem": bad}, "invalid_segmentation")


    @staticmethod
    def _check_keypoints(ann: Annotation, s: Sample, width: Optional[int], height: Optional[int],
                         category_keypoints: Dict[str, int], kpt_shape: Any, emit) -> None:
        attrs = ann.attributes
        desc = "Keypoint annotations must have the declared number of points, visibility flags in {0, 1, 2} and visible points inside the image."
        if attrs.get("keypoints_error"):
            emit("invalid_keypoints", s, f"Invalid keypoints on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri}: {attrs['keypoints_error']}.",
                 "Fix the keypoint values (check kpt_shape in data.yaml).", desc, {"annotation": ann.id, "reason": attrs["keypoints_error"]}, "invalid_keypoints")
            return
        kps = attrs.get("keypoints")
        if kps is not None:  # COCO: absolute [x, y, v] triplets
            if not isinstance(kps, list) or len(kps) % 3 != 0 or any(not isinstance(v, (int, float)) for v in kps):
                emit("invalid_keypoints", s, f"Invalid keypoints on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has a keypoints list that is not [x, y, v] triplets.",
                     "Fix the keypoints list.", desc, {"annotation": ann.id}, "invalid_keypoints")
                return
            n = len(kps) // 3
            expected = category_keypoints.get(str(ann.category_id))
            if expected is not None and n != expected:
                emit("invalid_keypoints", s, f"Wrong keypoint count on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has {n} keypoints but category {ann.category!r} declares {expected}.",
                     "Pad missing keypoints with (0, 0, 0) or fix the category definition.", desc, {"annotation": ann.id, "count": n, "expected": expected}, "invalid_keypoints")
            vis = kps[2::3]
            if any(v not in (0, 1, 2) for v in vis):
                emit("invalid_keypoints", s, f"Invalid keypoint visibility on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has visibility flags outside {{0, 1, 2}}.",
                     "Use 0 (not labelled), 1 (labelled, hidden) or 2 (visible).", desc, {"annotation": ann.id, "flags": sorted({v for v in vis if v not in (0, 1, 2)})[:10]}, "invalid_keypoints")
            declared = attrs.get("num_keypoints")
            labelled = sum(1 for v in vis if v in (1, 2))
            if isinstance(declared, (int, float)) and int(declared) != labelled:
                emit("invalid_keypoints", s, f"num_keypoints mismatch on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} declares num_keypoints={int(declared)} but {labelled} keypoints are labelled.",
                     "Recompute num_keypoints from the visibility flags.", desc, {"annotation": ann.id, "declared": int(declared), "labelled": labelled}, "invalid_keypoints")
            if width and height:
                out = [(i, x, y) for i, (x, y, v) in enumerate(zip(kps[0::3], kps[1::3], vis)) if v in (1, 2) and not (-0.5 <= x <= width + 0.5 and -0.5 <= y <= height + 0.5)]
                if out:
                    emit("keypoints_out_of_bounds", s, f"Keypoints outside the image on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has {len(out)} labelled keypoints outside the {width}x{height} image.",
                         "Clip or fix the coordinates, or mark the points as not labelled.", desc, {"annotation": ann.id, "points": out[:10]}, "keypoints_out_of_bounds")
            return
        norm = attrs.get("keypoints_norm")
        if norm is not None:  # YOLO pose: normalised (x, y[, v]) tuples
            d = int(attrs.get("keypoint_dim") or 3)
            if len(norm) % d != 0:
                emit("invalid_keypoints", s, f"Invalid keypoints on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has {len(norm)} keypoint values, not a multiple of {d}.",
                     "Fix the label line.", desc, {"annotation": ann.id}, "invalid_keypoints")
                return
            n = len(norm) // d
            if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) == 2 and n != int(kpt_shape[0]):
                emit("invalid_keypoints", s, f"Wrong keypoint count on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has {n} keypoints but kpt_shape declares {int(kpt_shape[0])}.",
                     "Fix the label line or kpt_shape.", desc, {"annotation": ann.id, "count": n, "expected": int(kpt_shape[0])}, "invalid_keypoints")
            xs, ys = norm[0::d], norm[1::d]
            vis = norm[2::d] if d == 3 else [2] * n
            if d == 3 and any(v not in (0, 1, 2) for v in vis):
                emit("invalid_keypoints", s, f"Invalid keypoint visibility on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has visibility flags outside {{0, 1, 2}}.",
                     "Use 0, 1 or 2 for visibility.", desc, {"annotation": ann.id}, "invalid_keypoints")
            tol = 1e-3
            out = [i for i, (x, y, v) in enumerate(zip(xs, ys, vis)) if v in (1, 2) and not (-tol <= x <= 1 + tol and -tol <= y <= 1 + tol)]
            if out:
                emit("keypoints_out_of_bounds", s, f"Keypoints outside [0, 1] on {s.uri}", f"Annotation {ann.id} on {s.split}:{s.uri} has {len(out)} labelled keypoints outside the normalised range.",
                     "Clip or fix the coordinates, or mark the points as not labelled.", desc, {"annotation": ann.id, "points": out[:10]}, "keypoints_out_of_bounds")


def _segmentation_problem(seg: Any) -> Optional[str]:
    if isinstance(seg, dict):
        if "counts" not in seg or "size" not in seg:
            return "RLE without counts/size"
        return None
    if isinstance(seg, list):
        if not seg:
            return "empty polygon list"
        for poly in seg:
            if not isinstance(poly, list):
                return "polygon is not a list"
            if len(poly) < 6 or len(poly) % 2:
                return f"polygon with {len(poly)} values (need an even number >= 6)"
            if any(not isinstance(v, (int, float)) for v in poly):
                return "polygon contains non-numeric values"
        return None
    return f"unsupported segmentation type {type(seg).__name__}"


__all__ = ["LabelValidityDetector"]
_ = Tuple
