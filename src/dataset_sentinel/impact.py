"""Metric impact: how much do the flagged samples inflate the evaluation score?

Given predictions for an evaluation split, the split is scored three times:
on **all** samples, on the **clean** samples only (flagged leaked samples
removed) and on the **leaked** samples only. The difference between *all*
and *clean* is the inflation caused by leakage; a *leaked-only* score far
above *clean* is the smoking gun.

Tasks and prediction formats:

* **detection** (COCO / YOLO / VOC datasets): COCO results JSON
  (``[{image_id, category_id, bbox, score}]``, or a full COCO file with an
  ``annotations`` list) or a directory of YOLO ``.txt`` files
  (``class cx cy w h [conf]``, one per image, as written by
  ``yolo val save_txt=True save_conf=True``). Metrics: COCO-style mAP@[.5:.95],
  AP50, AP75 (101-point interpolation, up to 100 detections per image, crowd
  boxes ignored). Implemented in numpy; values track pycocotools closely but
  are not guaranteed bit-identical.
* **classification** (image-folder datasets or any dataset whose annotations
  carry no boxes): CSV / JSON rows with a sample column (``file_name``,
  ``sample``, ``id`` or ``uri``) and a ``label`` / ``prediction`` /
  ``predicted`` column, optional ``score``. Metrics: accuracy, macro-F1.

"Leaked" means: member of a policy-violating relationship group (the same
samples that make up the split integrity violation rate); "strict leaked"
restricts that to deterministic and high-confidence groups.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import SentinelConfig
from .metrics import STRICT
from .model import BBox, Dataset, RelationshipGroup, Sample

log = logging.getLogger(__name__)

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100


@dataclass
class Prediction:
    sample_id: str
    category: str
    score: float = 1.0
    bbox: Optional[BBox] = None


@dataclass
class PredictionSet:
    task: str  # detection | classification
    by_sample: Dict[str, List[Prediction]] = field(default_factory=dict)
    source: str = ""
    unmatched: int = 0
    unknown_categories: Set[str] = field(default_factory=set)

    def for_samples(self, ids: Iterable[str]) -> List[Prediction]:
        out: List[Prediction] = []
        for sid in ids:
            out.extend(self.by_sample.get(sid, []))
        return out


# --------------------------------------------------------------------------- loading predictions
def _category_name(dataset: Dataset, raw: Any) -> str:
    if raw in dataset.categories:
        return str(dataset.categories[raw])
    try:
        as_int = int(raw)
        if as_int in dataset.categories:
            return str(dataset.categories[as_int])
    except (TypeError, ValueError):
        pass
    return str(raw)


def _sample_lookup(samples: Sequence[Sample]) -> Dict[str, Sample]:
    table: Dict[str, Sample] = {}
    for s in samples:
        keys = {str(s.native_id), s.uri, Path(s.uri).name, Path(s.uri).stem, s.id}
        if s.path is not None:
            keys.add(str(s.path))
        for k in keys:
            table.setdefault(k, s)
    return table


def load_predictions(spec: Path | str, dataset: Dataset, split: str, task_hint: str = "auto") -> PredictionSet:
    path = Path(spec)
    samples = [s for s in dataset.samples if s.split == split]
    lookup = _sample_lookup(samples)
    if path.is_dir():
        return _load_yolo_dir(path, samples, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("annotations"), list):
            data = data["annotations"]
        if isinstance(data, list) and data and isinstance(data[0], dict) and "bbox" in data[0]:
            return _load_coco_results(data, dataset, lookup, str(path))
        if isinstance(data, list):
            return _load_classification_rows(data, dataset, lookup, str(path))
        if isinstance(data, dict):
            rows = [{"sample": k, **(v if isinstance(v, dict) else {"label": v})} for k, v in data.items()]
            return _load_classification_rows(rows, dataset, lookup, str(path))
        raise ValueError(f"{path}: unrecognised predictions structure")
    if suffix in {".csv", ".tsv"}:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = [dict(r) for r in csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ",")]
        return _load_classification_rows(rows, dataset, lookup, str(path))
    if suffix == ".txt":
        return _load_yolo_dir(path.parent, samples, dataset, single=path)
    raise ValueError(f"{path}: unsupported predictions format (use COCO results JSON, a YOLO txt directory or a CSV)")


def _load_coco_results(rows: List[Dict[str, Any]], dataset: Dataset, lookup: Dict[str, Sample], source: str) -> PredictionSet:
    ps = PredictionSet(task="detection", source=source)
    for r in rows:
        s = lookup.get(str(r.get("image_id")))
        if s is None:
            ps.unmatched += 1
            continue
        bbox_raw = r.get("bbox")
        if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            continue
        try:
            x, y, w, h = (float(v) for v in bbox_raw)
        except (TypeError, ValueError):
            continue
        cat = _category_name(dataset, r.get("category_id"))
        if dataset.categories and str(cat) not in {str(v) for v in dataset.categories.values()}:
            ps.unknown_categories.add(cat)
        ps.by_sample.setdefault(s.id, []).append(Prediction(s.id, cat, float(r.get("score", 1.0)), BBox(x, y, w, h)))
    return ps


def _load_yolo_dir(directory: Path, samples: Sequence[Sample], dataset: Dataset, single: Optional[Path] = None) -> PredictionSet:
    ps = PredictionSet(task="detection", source=str(directory))
    for s in samples:
        txt = single if single is not None else directory / (Path(s.uri).stem + ".txt")
        if single is not None and Path(s.uri).stem != single.stem:
            continue
        if not txt.exists():
            continue
        w, h = s.width or 0, s.height or 0
        for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                conf = float(parts[5]) if len(parts) > 5 else 1.0
            except ValueError:
                continue
            cat = _category_name(dataset, cls)
            if w and h:
                bbox = BBox((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)
            else:
                bbox = BBox(cx - bw / 2, cy - bh / 2, bw, bh)
            ps.by_sample.setdefault(s.id, []).append(Prediction(s.id, cat, conf, bbox))
    return ps


def _load_classification_rows(rows: List[Dict[str, Any]], dataset: Dataset, lookup: Dict[str, Sample], source: str) -> PredictionSet:
    ps = PredictionSet(task="classification", source=source)
    key_cols = ("file_name", "sample", "sample_id", "id", "uri", "image", "path", "image_id")
    label_cols = ("prediction", "predicted", "pred", "label", "class", "category")
    for r in rows:
        key = next((r[c] for c in key_cols if c in r and r[c] not in (None, "")), None)
        label = next((r[c] for c in label_cols if c in r and r[c] not in (None, "")), None)
        if key is None or label is None:
            continue
        s = lookup.get(str(key)) or lookup.get(Path(str(key)).name) or lookup.get(Path(str(key)).stem)
        if s is None:
            ps.unmatched += 1
            continue
        cat = _category_name(dataset, label)
        try:
            score = float(r.get("score", 1.0) or 1.0)
        except (TypeError, ValueError):
            score = 1.0
        ps.by_sample.setdefault(s.id, []).append(Prediction(s.id, cat, score))
    return ps


# --------------------------------------------------------------------------- scoring: detection
def _iou_matrix(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """preds (n,4) and gts (m,4) as x, y, w, h -> (n, m) IoU."""
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)), dtype=np.float64)
    px1, py1 = preds[:, 0], preds[:, 1]
    px2, py2 = px1 + preds[:, 2], py1 + preds[:, 3]
    gx1, gy1 = gts[:, 0], gts[:, 1]
    gx2, gy2 = gx1 + gts[:, 2], gy1 + gts[:, 3]
    ix1 = np.maximum(px1[:, None], gx1[None, :])
    iy1 = np.maximum(py1[:, None], gy1[None, :])
    ix2 = np.minimum(px2[:, None], gx2[None, :])
    iy2 = np.minimum(py2[:, None], gy2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_p = (preds[:, 2] * preds[:, 3])[:, None]
    area_g = (gts[:, 2] * gts[:, 3])[None, :]
    union = area_p + area_g - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _class_of(ann) -> str:
    return str(ann.category if ann.category is not None else ann.category_id)


def _average_precision(tp_flags: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO-style AP: 101-point interpolated precision over recall."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp_flags[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # make precision monotonically decreasing from the right
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    interp = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(interp.mean())


def score_detection(samples: Sequence[Sample], preds: PredictionSet, classes: Sequence[str]) -> Dict[str, Any]:
    """mAP@[.5:.95], AP50, AP75 and per-class AP over ``samples``."""
    sample_ids = [s.id for s in samples]
    by_class_gt: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)  # class -> sample -> (boxes, crowd)
    n_gt: Dict[str, int] = defaultdict(int)
    for s in samples:
        per_class: Dict[str, List[Tuple[List[float], bool]]] = defaultdict(list)
        for a in s.annotations:
            if a.bbox is None or a.attributes.get("malformed"):
                continue
            crowd = bool(a.attributes.get("iscrowd"))
            per_class[_class_of(a)].append((a.bbox.as_list(), crowd))
        for cls, items in per_class.items():
            boxes = np.array([b for b, _ in items], dtype=np.float64)
            crowd = np.array([c for _, c in items], dtype=bool)
            by_class_gt[cls][s.id] = (boxes, crowd)
            n_gt[cls] += int((~crowd).sum())
    # predictions per class per sample, capped at MAX_DETS per image
    by_class_pred: Dict[str, Dict[str, List[Prediction]]] = defaultdict(lambda: defaultdict(list))
    for sid in sample_ids:
        plist = sorted(preds.by_sample.get(sid, []), key=lambda p: -p.score)[:MAX_DETS]
        for p in plist:
            if p.bbox is not None:
                by_class_pred[p.category][sid].append(p)

    ap_per_class: Dict[str, Dict[str, float]] = {}
    classes_eval = [c for c in classes if n_gt.get(c, 0) > 0]
    for cls in classes_eval:
        aps = []
        for thr in IOU_THRESHOLDS:
            tp_all: List[np.ndarray] = []
            score_all: List[np.ndarray] = []
            for sid in sample_ids:
                plist = by_class_pred.get(cls, {}).get(sid, [])
                if not plist:
                    continue
                pboxes = np.array([p.bbox.as_list() for p in plist], dtype=np.float64)  # type: ignore[union-attr]
                pscores = np.array([p.score for p in plist], dtype=np.float64)
                gboxes, gcrowd = by_class_gt.get(cls, {}).get(sid, (np.zeros((0, 4)), np.zeros(0, dtype=bool)))
                iou = _iou_matrix(pboxes, gboxes)
                matched = np.zeros(len(gboxes), dtype=bool)
                tp = np.zeros(len(plist), dtype=bool)
                keep = np.ones(len(plist), dtype=bool)
                for pi in np.argsort(-pscores, kind="stable"):
                    best, best_j = thr, -1
                    for j in range(len(gboxes)):
                        if matched[j] and not gcrowd[j]:
                            continue
                        if iou[pi, j] >= best:
                            # prefer non-crowd matches
                            if best_j >= 0 and gcrowd[j] and not gcrowd[best_j]:
                                continue
                            best, best_j = iou[pi, j], j
                    if best_j >= 0:
                        if gcrowd[best_j]:
                            keep[pi] = False  # matched a crowd region: ignored
                        else:
                            matched[best_j] = True
                            tp[pi] = True
                tp_all.append(tp[keep])
                score_all.append(pscores[keep])
            tp_flags = np.concatenate(tp_all) if tp_all else np.zeros(0, dtype=bool)
            scores = np.concatenate(score_all) if score_all else np.zeros(0)
            aps.append(_average_precision(tp_flags, scores, n_gt[cls]))
        aps_arr = np.array(aps)
        ap_per_class[cls] = {"AP": float(np.nanmean(aps_arr)), "AP50": float(aps_arr[0]), "AP75": float(aps_arr[5]), "gt": n_gt[cls]}
    if not ap_per_class:
        return {"mAP": float("nan"), "AP50": float("nan"), "AP75": float("nan"), "per_class": {}, "classes": 0}
    return {
        "mAP": float(np.mean([v["AP"] for v in ap_per_class.values()])),
        "AP50": float(np.mean([v["AP50"] for v in ap_per_class.values()])),
        "AP75": float(np.mean([v["AP75"] for v in ap_per_class.values()])),
        "per_class": ap_per_class,
        "classes": len(ap_per_class),
    }


# --------------------------------------------------------------------------- scoring: classification
def score_classification(samples: Sequence[Sample], preds: PredictionSet) -> Dict[str, Any]:
    truth: Dict[str, str] = {}
    for s in samples:
        labels = [_class_of(a) for a in s.annotations if not a.attributes.get("malformed")]
        if labels:
            truth[s.id] = labels[0]
    if not truth:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "per_class": {}, "scored": 0}
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    correct = 0
    scored = 0
    for sid, gt in truth.items():
        plist = preds.by_sample.get(sid)
        if not plist:
            fn[gt] += 1
            continue
        scored += 1
        pred = max(plist, key=lambda p: p.score).category
        if pred == gt:
            correct += 1
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt] += 1
    classes = sorted(set(truth.values()) | set(fp))
    per_class = {}
    f1s = []
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": tp[c] + fn[c]}
        if tp[c] + fn[c]:
            f1s.append(f1)
    return {"accuracy": correct / len(truth), "macro_f1": float(np.mean(f1s)) if f1s else 0.0, "per_class": per_class, "scored": scored, "total": len(truth)}


# --------------------------------------------------------------------------- impact
def leaked_sample_ids(groups: Iterable[RelationshipGroup], split: str, dataset: Dataset, strict: bool = False) -> Set[str]:
    out: Set[str] = set()
    for g in groups:
        if not g.violates_policy:
            continue
        if strict and g.confidence not in STRICT:
            continue
        for m in g.member_ids:
            s = dataset.get_or_none(m)
            if s is not None and s.split == split:
                out.add(m)
    return out


def _primary(task: str, scores: Dict[str, Any]) -> Tuple[str, float]:
    if task == "detection":
        return "mAP", scores.get("mAP", float("nan"))
    return "accuracy", scores.get("accuracy", float("nan"))


def compute_impact(dataset: Dataset, groups: Sequence[RelationshipGroup], config: SentinelConfig) -> Optional[Dict[str, Any]]:
    spec = config.section("evaluation.predictions")
    if not spec:
        return None
    task_cfg = str(config.get("evaluation.task") or "auto")
    result: Dict[str, Any] = {"splits": {}, "notes": []}
    for split, path_spec in spec.items():
        if split not in dataset.splits:
            result["notes"].append(f"predictions given for unknown split {split!r}")
            continue
        samples = [s for s in dataset.samples if s.split == split]
        preds = load_predictions(config.resolve_path(path_spec), dataset, split, task_cfg)  # type: ignore[arg-type]
        task = task_cfg if task_cfg != "auto" else preds.task
        if task == "detection" and not any(a.bbox is not None for s in samples for a in s.annotations):
            task = "classification"
        classes = sorted({_class_of(a) for s in samples for a in s.annotations if not a.attributes.get("malformed")})
        leaked = leaked_sample_ids(groups, split, dataset)
        strict_leaked = leaked_sample_ids(groups, split, dataset, strict=True)
        subsets = {
            "all": [s for s in samples],
            "clean": [s for s in samples if s.id not in leaked],
            "leaked": [s for s in samples if s.id in leaked],
            "strict_clean": [s for s in samples if s.id not in strict_leaked],
        }
        scored: Dict[str, Dict[str, Any]] = {}
        for name, subset in subsets.items():
            if not subset:
                scored[name] = {"samples": 0}
                continue
            sc = score_detection(subset, preds, classes) if task == "detection" else score_classification(subset, preds)
            sc["samples"] = len(subset)
            sc["predictions"] = sum(len(preds.by_sample.get(s.id, [])) for s in subset)
            scored[name] = sc
        metric_name, m_all = _primary(task, scored["all"])
        _, m_clean = _primary(task, scored["clean"]) if scored["clean"].get("samples") else (metric_name, float("nan"))
        _, m_leaked = _primary(task, scored["leaked"]) if scored["leaked"].get("samples") else (metric_name, float("nan"))
        _, m_strict = _primary(task, scored["strict_clean"]) if scored["strict_clean"].get("samples") else (metric_name, float("nan"))
        per_class_delta: List[Dict[str, Any]] = []
        if task == "detection" and scored["clean"].get("per_class"):
            for cls, v in scored["all"].get("per_class", {}).items():
                c = scored["clean"]["per_class"].get(cls)
                if c is not None:
                    per_class_delta.append({"class": cls, "all": round(v["AP"], 4), "clean": round(c["AP"], 4), "delta": round(v["AP"] - c["AP"], 4)})
            per_class_delta.sort(key=lambda x: -abs(x["delta"]))
        entry = {
            "task": task,
            "metric": metric_name,
            "predictions_source": preds.source,
            "unmatched_predictions": preds.unmatched,
            "unknown_prediction_categories": sorted(preds.unknown_categories)[:20],
            "samples": len(samples),
            "leaked_samples": len(leaked),
            "strict_leaked_samples": len(strict_leaked),
            "all": m_all,
            "clean": m_clean,
            "leaked": m_leaked,
            "strict_clean": m_strict,
            "inflation": (m_all - m_clean) if not (np.isnan(m_all) or np.isnan(m_clean)) else None,
            "strict_inflation": (m_all - m_strict) if not (np.isnan(m_all) or np.isnan(m_strict)) else None,
            "subsets": {k: _jsonable_scores(v) for k, v in scored.items()},
            "per_class_delta": per_class_delta[:15],
        }
        if preds.unmatched:
            result["notes"].append(f"{split}: {preds.unmatched} predictions did not match any sample")
        if not leaked:
            result["notes"].append(f"{split}: no leaked samples flagged, so all and clean scores are identical")
        result["splits"][split] = entry
    return result


def _jsonable_scores(v: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, val in v.items():
        if k == "per_class":
            out[k] = {c: {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv) for kk, vv in d.items()} for c, d in list(val.items())[:100]}
        elif isinstance(val, float) and np.isnan(val):
            out[k] = None
        else:
            out[k] = val
    return out


def format_impact_line(split: str, entry: Dict[str, Any]) -> str:
    metric = entry["metric"]
    fmt = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"  # noqa: E731
    inflation = entry.get("inflation")
    delta = "" if inflation is None else f" ({inflation:+.3f})"
    return (
        f"{split} {metric}: all {fmt(entry['all'])} -> clean {fmt(entry['clean'])}{delta}; "
        f"leaked-only {fmt(entry['leaked'])} on {entry['leaked_samples']} of {entry['samples']} samples"
    )
