"""Label and metadata distribution checks.

* class imbalance inside a split (max/min class count ratio);
* class distribution shift between the training split and the others
  (Jensen-Shannon divergence, base 2, range 0..1);
* classes present in an evaluation split but absent from training
  (deterministic: the model cannot have learnt them), and vice versa;
* image size / aspect and relative bbox size shifts (heuristic);
* class co-occurrence anomalies: pairs that frequently co-occur in one
  split and never in another.

These findings never count toward the split integrity violation rate: they
describe distribution quality, not leakage.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import split_role
from ..model import Confidence, Sample
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult

_P = "policy.distribution."


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def wasserstein_1d(a: List[float], b: List[float]) -> float:
    """First Wasserstein distance (earth mover's distance) between two 1-D samples."""
    if not a or not b:
        return 0.0
    xa = np.sort(np.asarray(a, dtype=float))
    xb = np.sort(np.asarray(b, dtype=float))
    grid = np.linspace(0.0, 1.0, 512, endpoint=False) + 0.5 / 512
    qa = np.quantile(xa, grid)
    qb = np.quantile(xb, grid)
    return float(np.mean(np.abs(qa - qb)))


def normalized_wasserstein(a: List[float], b: List[float]) -> float:
    """Wasserstein distance divided by the reference sample's standard deviation
    (0 = identical, ~1 = shifted by one standard deviation)."""
    if not a or not b:
        return 0.0
    scale = float(np.std(np.asarray(a, dtype=float)))
    if scale < 1e-9:
        scale = float(np.std(np.asarray(b, dtype=float))) or 1.0
    return wasserstein_1d(a, b) / scale


def _class_of(ann) -> str:
    return str(ann.category if ann.category is not None else ann.category_id)


def _countable(ann, has_categories: bool) -> bool:
    """Skip malformed annotations and, when the dataset declares categories,
    annotations whose category is unknown (label_validity reports those)."""
    if ann.attributes.get("malformed"):
        return False
    if has_categories and ann.category is None:
        return False
    return ann.category is not None or ann.category_id is not None


@detectors.register("distribution")
class DistributionDetector(Detector):
    name = "distribution"
    description = "Class imbalance, label / size distribution shift, unseen classes, co-occurrence anomalies"
    modalities = ("*",)

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        cfg = ctx.config
        dataset = ctx.dataset
        by_split = dataset.by_split()
        min_samples = int(cfg.get(_P + "min_samples", 20) or 0)

        has_cats = bool(dataset.categories)
        class_counts: Dict[str, Counter] = {}
        image_counts: Dict[str, Counter] = {}
        for split, members in by_split.items():
            cc: Counter = Counter()
            ic: Counter = Counter()
            for s in members:
                classes = {_class_of(a) for a in s.annotations if _countable(a, has_cats)}
                for a in s.annotations:
                    if _countable(a, has_cats):
                        cc[_class_of(a)] += 1
                for c in classes:
                    ic[c] += 1
            class_counts[split] = cc
            image_counts[split] = ic
        all_classes = sorted({c for cc in class_counts.values() for c in cc} | {str(v) for v in dataset.categories.values()})
        res.stats["class_distribution"] = {split: dict(cc) for split, cc in class_counts.items()}
        res.stats["classes"] = all_classes
        if not all_classes:
            res.skipped = "no labels available"
            return res

        ref = self._reference_split(ctx, by_split)
        eval_splits = [s for s in dataset.splits if s != ref]
        res.stats["reference_split"] = ref

        # ------------------------------------------------------------- imbalance
        sev = cfg.severity(_P + "imbalance")
        ratio_limit = float(cfg.get(_P + "imbalance_ratio", 20) or 0)
        if sev is not None and ratio_limit > 0:
            for split, cc in class_counts.items():
                present = {c: n for c, n in cc.items() if n > 0}
                if len(present) < 2 or sum(present.values()) < min_samples:
                    continue
                top = max(present, key=present.get)  # type: ignore[arg-type]
                low = min(present, key=present.get)  # type: ignore[arg-type]
                ratio = present[top] / present[low]
                if ratio > ratio_limit:
                    res.add(
                        self.finding(
                            kind="class_imbalance",
                            title=f"Class imbalance in {split}: {top} is {ratio:.0f}x more frequent than {low}",
                            severity=sev,
                            confidence=Confidence.DETERMINISTIC,
                            policy=cfg.policy_label(_P + "imbalance_ratio"),
                            policy_description="The most frequent class should not exceed the rarest by more than the configured ratio.",
                            message=f"In split {split}, {top} has {present[top]} annotations and {low} has {present[low]} (ratio {ratio:.1f}, limit {ratio_limit:g}).",
                            remediation="Collect or oversample rare classes, use class-weighted losses, and report per-class metrics.",
                            evidence={"split": split, "ratio": round(ratio, 2), "counts": dict(sorted(present.items(), key=lambda kv: -kv[1])[:30])},
                            splits=[split],
                        )
                    )

        # ------------------------------------------------------------- unseen / missing classes
        sev_unseen = cfg.severity(_P + "unseen_class")
        sev_missing = cfg.severity(_P + "missing_class_in_eval")
        if ref is not None:
            ref_classes = {c for c, n in class_counts[ref].items() if n > 0}
            for split in eval_splits:
                classes = {c for c, n in class_counts[split].items() if n > 0}
                unseen = sorted(classes - ref_classes)
                missing = sorted(ref_classes - classes)
                if unseen and sev_unseen is not None:
                    res.add(
                        self.finding(
                            kind="unseen_class",
                            title=f"{len(unseen)} classes in {split} never appear in {ref}: {', '.join(unseen[:6])}{' ...' if len(unseen) > 6 else ''}",
                            severity=sev_unseen,
                            confidence=Confidence.DETERMINISTIC,
                            policy=cfg.policy_label(_P + "unseen_class"),
                            policy_description="Every class evaluated must be present in the training split.",
                            message=f"Split {split} contains classes with no training examples: {', '.join(unseen)}. Metrics on these classes will be zero or undefined.",
                            remediation="Add training examples for these classes or exclude them from evaluation.",
                            evidence={"split": split, "reference": ref, "classes": unseen, "counts": {c: class_counts[split][c] for c in unseen}},
                            splits=[ref, split],
                        )
                    )
                if missing and sev_missing is not None and sum(class_counts[split].values()) >= min_samples:
                    res.add(
                        self.finding(
                            kind="missing_class_in_eval",
                            title=f"{len(missing)} training classes have no examples in {split}",
                            severity=sev_missing,
                            confidence=Confidence.DETERMINISTIC,
                            policy=cfg.policy_label(_P + "missing_class_in_eval"),
                            policy_description="Evaluation splits should cover every training class.",
                            message=f"Classes trained on but absent from {split}: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}.",
                            remediation="Use a stratified split so that every class is represented in evaluation.",
                            evidence={"split": split, "reference": ref, "classes": missing},
                            splits=[ref, split],
                        )
                    )

        # ------------------------------------------------------------- label shift
        sev_shift = cfg.severity(_P + "shift")
        threshold = float(cfg.get(_P + "shift_threshold", 0.1) or 0)
        shifts: Dict[str, float] = {}
        if ref is not None and sev_shift is not None:
            ref_vec = np.array([class_counts[ref].get(c, 0) for c in all_classes], dtype=float)
            for split in eval_splits:
                vec = np.array([class_counts[split].get(c, 0) for c in all_classes], dtype=float)
                if ref_vec.sum() < min_samples or vec.sum() < min_samples:
                    continue
                d = js_divergence(ref_vec, vec)
                shifts[split] = round(d, 4)
                if d > threshold:
                    ref_p = ref_vec / ref_vec.sum()
                    p = vec / vec.sum()
                    deltas = sorted(zip(all_classes, (p - ref_p).tolist()), key=lambda kv: -abs(kv[1]))[:8]
                    res.add(
                        self.finding(
                            kind="label_shift",
                            title=f"Class distribution of {split} differs from {ref} (JS divergence {d:.3f})",
                            severity=sev_shift,
                            confidence=Confidence.HEURISTIC,
                            policy=cfg.policy_label(_P + "shift_threshold"),
                            policy_description="Class proportions should be similar across splits unless the shift is intentional.",
                            message=(
                                f"Jensen-Shannon divergence between {ref} and {split} class distributions is {d:.3f} (threshold {threshold:g}). "
                                "Largest proportion changes: " + ", ".join(f"{c} {delta:+.1%}" for c, delta in deltas) + "."
                            ),
                            remediation="Use stratified splitting, or document the intentional shift so metrics are interpreted correctly.",
                            evidence={"split": split, "reference": ref, "js_divergence": round(d, 4), "largest_changes": {c: round(v, 4) for c, v in deltas}},
                            splits=[ref, split],
                        )
                    )
        res.stats["label_shift_js"] = shifts

        # ------------------------------------------------------------- image size / bbox size shift
        self._size_shift(ctx, res, ref, eval_splits, by_split, threshold, min_samples)

        # ------------------------------------------------------------- co-occurrence
        self._cooccurrence(ctx, res, ref, eval_splits, image_counts, by_split, min_samples)
        return res

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _reference_split(ctx: DetectorContext, by_split: Dict[str, List[Sample]]) -> Optional[str]:
        for split in ctx.dataset.splits:
            if split_role(split) == "train":
                return split
        if not by_split:
            return None
        return max(by_split, key=lambda k: len(by_split[k]))

    def _size_shift(self, ctx: DetectorContext, res: DetectorResult, ref: Optional[str], eval_splits: List[str],
                    by_split: Dict[str, List[Sample]], threshold: float, min_samples: int) -> None:
        cfg = ctx.config
        sev_img = cfg.severity(_P + "image_size_shift")
        sev_box = cfg.severity(_P + "bbox_size_shift")
        if ref is None or (sev_img is None and sev_box is None):
            return
        store = ctx.fingerprints if ctx.has_fingerprints else None

        def dims(s: Sample) -> Optional[Tuple[int, int]]:
            if s.width and s.height:
                return s.width, s.height
            if store is not None:
                fp = store.get(s.id)
                if fp and fp.ok:
                    return fp.width, fp.height
            return None

        def log_area_hist(values: List[float], edges: np.ndarray) -> np.ndarray:
            return np.histogram(values, bins=edges)[0].astype(float)

        img_area: Dict[str, List[float]] = {}
        aspect: Dict[str, List[float]] = {}
        box_rel: Dict[str, List[float]] = {}
        box_aspect: Dict[str, List[float]] = {}
        objects_per_image: Dict[str, List[float]] = {}
        for split, members in by_split.items():
            areas: List[float] = []
            asp: List[float] = []
            rel: List[float] = []
            basp: List[float] = []
            opi: List[float] = []
            for s in members:
                d = dims(s)
                if d is None:
                    continue
                w, h = d
                areas.append(math.log10(max(w * h, 1)))
                asp.append(math.log2(max(w / h, 1e-6)))
                n_obj = 0
                for a in s.annotations:
                    bw = bh = None
                    if a.bbox is not None and a.bbox.area > 0:
                        rel.append(math.log10(max(a.bbox.area / (w * h), 1e-6)))
                        bw, bh = a.bbox.w, a.bbox.h
                    elif a.attributes.get("bbox_norm"):
                        nw, nh = a.attributes["bbox_norm"][2:4]
                        if nw > 0 and nh > 0:
                            rel.append(math.log10(max(nw * nh, 1e-6)))
                            bw, bh = nw * w, nh * h
                    if bw and bh:
                        basp.append(math.log2(max(bw / bh, 1e-6)))
                    if not a.attributes.get("malformed"):
                        n_obj += 1
                opi.append(float(n_obj))
            img_area[split] = areas
            aspect[split] = asp
            box_rel[split] = rel
            box_aspect[split] = basp
            objects_per_image[split] = opi

        def summary(values: Dict[str, List[float]], fmt) -> Dict[str, Any]:
            return {split: {"count": len(v), "median": fmt(float(np.median(v))) if v else None} for split, v in values.items()}

        res.stats["image_size"] = summary(img_area, lambda m: round(10 ** m))
        res.stats["image_aspect"] = summary(aspect, lambda m: round(2 ** m, 3))
        res.stats["objects_per_image"] = summary(objects_per_image, lambda m: round(m, 2))
        res.stats["box_relative_area"] = summary(box_rel, lambda m: round(10 ** m, 5))
        res.stats["box_aspect"] = summary(box_aspect, lambda m: round(2 ** m, 3))

        area_edges = np.arange(2, 9.01, 0.25)
        rel_edges = np.arange(-6, 0.01, 0.25)
        continuous: Dict[str, Dict[str, Dict[str, float]]] = {}
        for split in eval_splits:
            cont: Dict[str, Dict[str, float]] = {}
            for name, values in (("image_log10_area", img_area), ("image_log2_aspect", aspect), ("objects_per_image", objects_per_image),
                                 ("box_log10_relative_area", box_rel), ("box_log2_aspect", box_aspect)):
                a, b = values.get(ref, []), values.get(split, [])
                if len(a) >= min_samples and len(b) >= min_samples:
                    cont[name] = {"wasserstein": round(wasserstein_1d(a, b), 4), "wasserstein_normalized": round(normalized_wasserstein(a, b), 4)}
            continuous[split] = cont
            if sev_img is not None and len(img_area.get(ref, [])) >= min_samples and len(img_area.get(split, [])) >= min_samples:
                d = js_divergence(log_area_hist(img_area[ref], area_edges), log_area_hist(img_area[split], area_edges))
                if d > threshold:
                    res.add(
                        self.finding(
                            kind="image_size_shift",
                            title=f"Image sizes in {split} differ from {ref} (JS divergence {d:.3f})",
                            severity=sev_img,
                            confidence=Confidence.HEURISTIC,
                            policy=cfg.policy_label(_P + "image_size_shift"),
                            policy_description="Image resolution distributions should be similar across splits.",
                            message=(
                                f"Median image area: {ref} {10 ** np.median(img_area[ref]):,.0f} px vs {split} {10 ** np.median(img_area[split]):,.0f} px "
                                f"(JS {d:.3f}, normalised Wasserstein {cont.get('image_log10_area', {}).get('wasserstein_normalized', 0):.2f})."
                            ),
                            remediation="Check that the splits come from the same capture / export pipeline.",
                            evidence={"split": split, "reference": ref, "js_divergence": round(d, 4), "continuous": cont},
                            splits=[ref, split],
                        )
                    )
            if sev_box is not None and len(box_rel.get(ref, [])) >= min_samples and len(box_rel.get(split, [])) >= min_samples:
                d = js_divergence(log_area_hist(box_rel[ref], rel_edges), log_area_hist(box_rel[split], rel_edges))
                if d > threshold:
                    res.add(
                        self.finding(
                            kind="bbox_size_shift",
                            title=f"Relative object sizes in {split} differ from {ref} (JS divergence {d:.3f})",
                            severity=sev_box,
                            confidence=Confidence.HEURISTIC,
                            policy=cfg.policy_label(_P + "bbox_size_shift"),
                            policy_description="Object scale distributions should be similar across splits.",
                            message=(
                                f"Median relative box area: {ref} {10 ** np.median(box_rel[ref]):.4f} vs {split} {10 ** np.median(box_rel[split]):.4f} "
                                f"(JS {d:.3f}, normalised Wasserstein {cont.get('box_log10_relative_area', {}).get('wasserstein_normalized', 0):.2f})."
                            ),
                            remediation="Verify annotation guidelines and camera setups are consistent between splits.",
                            evidence={"split": split, "reference": ref, "js_divergence": round(d, 4), "continuous": cont},
                            splits=[ref, split],
                        )
                    )
        res.stats["continuous_shift"] = continuous

    def _cooccurrence(self, ctx: DetectorContext, res: DetectorResult, ref: Optional[str], eval_splits: List[str],
                      image_counts: Dict[str, Counter], by_split: Dict[str, List[Sample]], min_samples: int) -> None:
        cfg = ctx.config
        sev = cfg.severity(_P + "cooccurrence")
        if sev is None or ref is None:
            return
        has_cats = bool(ctx.dataset.categories)
        pair_counts: Dict[str, Counter] = {}
        for split, members in by_split.items():
            pc: Counter = Counter()
            for s in members:
                classes = sorted({_class_of(a) for a in s.annotations if _countable(a, has_cats)})
                for a, b in combinations(classes, 2):
                    pc[(a, b)] += 1
            pair_counts[split] = pc
        anomalies: List[Dict[str, Any]] = []
        for split in eval_splits:
            if len(by_split.get(split, [])) < min_samples or len(by_split.get(ref, [])) < min_samples:
                continue
            for (a, b), n in pair_counts[ref].items():
                base = min(image_counts[ref][a], image_counts[ref][b])
                if base < 10 or n / base < 0.5:
                    continue
                if image_counts[split][a] >= 5 and image_counts[split][b] >= 5 and pair_counts[split][(a, b)] == 0:
                    anomalies.append({"split": split, "classes": [a, b], "reference_rate": round(n / base, 3), "eval_pairs": 0})
        if anomalies:
            top = anomalies[:10]
            res.add(
                self.finding(
                    kind="cooccurrence_anomaly",
                    title=f"{len(anomalies)} class pairs co-occur in {ref} but never together in evaluation splits",
                    severity=sev,
                    confidence=Confidence.HEURISTIC,
                    policy=cfg.policy_label(_P + "cooccurrence"),
                    policy_description="Class co-occurrence structure should be similar across splits.",
                    message="; ".join(f"{a['classes'][0]}+{a['classes'][1]} co-occur in {a['reference_rate']:.0%} of {ref} images but never in {a['split']}" for a in top),
                    remediation="Check whether the evaluation split systematically lacks multi-object scenes.",
                    evidence={"anomalies": anomalies[:50], "reference": ref},
                )
            )
        res.stats["cooccurrence_anomalies"] = len(anomalies)


__all__ = ["DistributionDetector", "js_divergence"]
_ = defaultdict
