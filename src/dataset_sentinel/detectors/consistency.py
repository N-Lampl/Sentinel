"""Annotation consistency triage.

Surfaces *potential* inconsistencies between related samples for human
review. It does not claim which annotation is right:

* identical images (byte- or pixel-identical) that carry different labels;
* verified near-duplicate images whose class sets differ, or whose number of
  annotated objects differs substantially.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from ..model import Confidence, Sample, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap

_POLICY_DESC = "Identical or near-identical samples are expected to carry consistent labels; differences deserve review."


def _class_summary(s: Sample) -> Dict[str, int]:
    c: Counter = Counter()
    for a in s.annotations:
        if not a.attributes.get("malformed"):
            c[str(a.category if a.category is not None else a.category_id)] += 1
    return dict(sorted(c.items()))


def _object_count(s: Sample) -> int:
    return sum(1 for a in s.annotations if not a.attributes.get("malformed"))


@detectors.register("consistency")
class ConsistencyDetector(Detector):
    name = "consistency"
    description = "Potential label inconsistencies on identical or near-identical samples (review triage)"
    modalities = ("image",)
    needs_fingerprints = True

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        sev = ctx.severity("policy.consistency.conflicting_labels")
        if sev is None:
            res.skipped = "disabled by policy"
            return res
        store = ctx.fingerprints
        cap = FindingCap(int(ctx.config.get("output.max_findings_per_kind") or 1000))
        stats = {"identical_conflicts": 0, "near_duplicate_class_conflicts": 0, "near_duplicate_count_conflicts": 0}
        near_sev = Severity.WARNING if sev is Severity.ERROR else sev

        # ------------------------------------------------------------ identical images
        by_key: Dict[str, List[Sample]] = defaultdict(list)
        for s in ctx.dataset.samples:
            fp = store.get(s.id)
            if fp is None or not fp.ok:
                continue
            by_key[fp.pixel_hash or fp.content_sha256 or s.id].append(s)
        for members in by_key.values():
            if len(members) < 2:
                continue
            sigs = {m.label_signature() for m in members}
            if len(sigs) < 2:
                continue
            stats["identical_conflicts"] += 1
            if not cap.allow("conflicting_labels_identical"):
                continue
            splits = sorted({m.split for m in members})
            summaries = [_class_summary(m) for m in members]
            message = "Potential inconsistency: " + "; ".join(f"{m.split}:{m.uri} -> {s or 'no labels'}" for m, s in zip(members[:6], summaries))
            if len({tuple(sorted(s.items())) for s in summaries}) == 1:
                message += " (same classes, but the bounding boxes or other annotation details differ)."
            else:
                message += " (class sets differ)."
            message += " Review recommended; Sentinel does not decide which annotation is correct."
            res.add(
                self.finding(
                    kind="conflicting_labels_identical",
                    title=f"Identical images carry different labels ({len(members)} samples, {'/'.join(splits)})",
                    severity=sev,
                    confidence=Confidence.DETERMINISTIC,
                    policy=ctx.policy_label("policy.consistency.conflicting_labels"),
                    policy_description=_POLICY_DESC,
                    message=message,
                    remediation="Review the copies, decide on the correct annotation, apply it to all of them, then deduplicate.",
                    samples=[m.ref() for m in members],
                    evidence={"labels": {m.id: _class_summary(m) for m in members[:20]}, "object_counts": {m.id: _object_count(m) for m in members[:20]}},
                )
            )

        # ------------------------------------------------------------ near duplicates
        pairs: List[Tuple[str, str, int]] = ctx.shared.get("near_duplicate_pairs") or []
        for a_id, b_id, d in pairs:
            a, b = ctx.dataset.get_or_none(a_id), ctx.dataset.get_or_none(b_id)
            if a is None or b is None:
                continue
            ca, cb = set(_class_summary(a)), set(_class_summary(b))
            na, nb = _object_count(a), _object_count(b)
            if not ca and not cb:
                continue
            if ca != cb:
                stats["near_duplicate_class_conflicts"] += 1
                if not cap.allow("conflicting_labels_near_duplicate"):
                    continue
                res.add(
                    self.finding(
                        kind="conflicting_labels_near_duplicate",
                        title=f"Potential inconsistency: near-identical images have different class sets ({a.split}:{a.uri} vs {b.split}:{b.uri})",
                        severity=near_sev,
                        confidence=Confidence.HEURISTIC,
                        policy=ctx.policy_label("policy.consistency.conflicting_labels"),
                        policy_description=_POLICY_DESC,
                        message=(
                            f"{a.split}:{a.uri} has {sorted(ca) or 'no labels'} while {b.split}:{b.uri} (dHash distance {d}) has "
                            f"{sorted(cb) or 'no labels'}. Review recommended."
                        ),
                        remediation="Review both annotations; one of them is probably incomplete or wrong.",
                        samples=[a.ref(), b.ref()],
                        evidence={"distance": d, "labels": {a.id: _class_summary(a), b.id: _class_summary(b)}},
                    )
                )
            elif max(na, nb) >= 2 and (abs(na - nb) >= 2 and max(na, nb) >= 2 * min(na, nb)):
                stats["near_duplicate_count_conflicts"] += 1
                if not cap.allow("object_count_mismatch_near_duplicate"):
                    continue
                res.add(
                    self.finding(
                        kind="object_count_mismatch_near_duplicate",
                        title=f"Potential inconsistency: near-identical images have {na} vs {nb} annotated objects ({a.split}:{a.uri} vs {b.split}:{b.uri})",
                        severity=near_sev,
                        confidence=Confidence.HEURISTIC,
                        policy=ctx.policy_label("policy.consistency.conflicting_labels"),
                        policy_description=_POLICY_DESC,
                        message=(
                            f"{a.split}:{a.uri} has {na} annotations and {b.split}:{b.uri} (dHash distance {d}) has {nb} with the same classes. "
                            "One of them may be under-annotated. Review recommended."
                        ),
                        remediation="Compare the two images and complete the annotations of the one with fewer objects if they are missing.",
                        samples=[a.ref(), b.ref()],
                        evidence={"distance": d, "object_counts": {a.id: na, b.id: nb}},
                    )
                )
        res.findings.extend(
            cap.summaries(self, {"conflicting_labels_identical": sev, "conflicting_labels_near_duplicate": near_sev, "object_count_mismatch_near_duplicate": near_sev})
        )
        res.stats = stats
        return res
