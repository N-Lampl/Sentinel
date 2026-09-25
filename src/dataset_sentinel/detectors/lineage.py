"""Lineage leakage: derived samples (crops, tiles, augmentations, exports,
paraphrases, ...) whose declared parent lives in another split.

Relationships come from ``dataset.relationships`` (kind ``derived_from``),
resolved during enrichment from metadata keys such as ``derived_from`` /
``parent`` or from file-name patterns (``dataset.lineage.from_filename``).
Families are connected components of the lineage graph.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..fingerprints.index import union_find_groups
from ..model import Confidence, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap, sample_uri_list

_POLICY_DESC = (
    "A sample and everything derived from it (crops, tiles, augmentations, exports, "
    "paraphrases) must live in the same split."
)


@detectors.register("lineage")
class LineageDetector(Detector):
    name = "lineage"
    description = "Declared derived_from / parent relationships that cross splits"
    modalities = ("*",)

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        sev = ctx.severity("policy.lineage.cross_split")
        if sev is None:
            res.skipped = "disabled by policy"
            return res
        edges = [r for r in ctx.dataset.relationships if r.kind == "derived_from"]
        if not edges:
            res.add(
                self.finding(
                    kind="lineage_not_evaluated",
                    title="No lineage metadata found: derivative leakage via lineage not evaluated",
                    severity=Severity.INFO,
                    confidence=Confidence.INCONCLUSIVE,
                    policy=ctx.policy_label("policy.lineage.cross_split"),
                    policy_description=_POLICY_DESC,
                    message=(
                        "No sample declares a parent (derived_from / parent / source_image) and no "
                        "dataset.lineage.from_filename pattern matched. Content-based checks still cover flips, "
                        "rotations and near-duplicates, but crops and tiles need lineage information."
                    ),
                    remediation="Record the source sample when generating derived samples, or configure a file-name pattern.",
                )
            )
            res.stats = {"edges": 0}
            return res

        samples = ctx.dataset.samples
        pos = {s.id: i for i, s in enumerate(samples)}
        pairs = [(pos[e.source_id], pos[e.target_id]) for e in edges if e.source_id in pos and e.target_id in pos]
        comps = union_find_groups(len(samples), pairs)
        edges_by_member: Dict[str, List] = defaultdict(list)
        for e in edges:
            edges_by_member[e.source_id].append(e)
            edges_by_member[e.target_id].append(e)

        cap = FindingCap(int(ctx.config.get("output.max_findings_per_kind") or 1000))
        stats = {"edges": len(edges), "families": len(comps), "cross_split_families": 0}
        for comp in comps:
            members = [samples[i] for i in comp]
            member_ids = [m.id for m in members]
            splits = sorted({m.split for m in members})
            fam_edges = {id(e): e for m in members for e in edges_by_member.get(m.id, [])}.values()
            confidence = Confidence.DETERMINISTIC
            if any(e.confidence is not Confidence.DETERMINISTIC for e in fam_edges):
                confidence = Confidence.HEURISTIC
            evidence = {
                "edges": [
                    {"derived": e.source_id, "parent": e.target_id, "via": e.evidence.get("key"), "value": e.evidence.get("value"), "confidence": e.confidence.value}
                    for e in list(fam_edges)[:30]
                ],
                "family_size": len(members),
            }
            cross = len(splits) > 1
            violates = cross and sev in (Severity.ERROR, Severity.WARNING)
            group = self.group("lineage", member_ids, splits, confidence, violates, evidence)
            res.add_group(group)
            if not cross:
                continue
            stats["cross_split_families"] += 1
            if not cap.allow("lineage_cross_split"):
                continue
            cross_edges = [e for e in fam_edges if ctx.split_of(e.source_id) != ctx.split_of(e.target_id)]
            e0 = cross_edges[0] if cross_edges else next(iter(fam_edges))
            src, parent = ctx.dataset.get(e0.source_id), ctx.dataset.get(e0.target_id)
            res.add(
                self.finding(
                    kind="lineage_cross_split",
                    title=f"Derived sample and its source are in different splits ({'/'.join(splits)}, family of {len(members)})",
                    severity=sev,
                    confidence=confidence,
                    policy=ctx.policy_label("policy.lineage.cross_split"),
                    policy_description=_POLICY_DESC,
                    message=(
                        f"{src.split}:{src.uri} is derived from {parent.split}:{parent.uri} "
                        f"(via {e0.evidence.get('key')}). Family: {sample_uri_list(members)}."
                    ),
                    remediation="Assign whole lineage families to one split; split on the root samples, then carry derivatives along.",
                    samples=[m.ref() for m in members],
                    evidence=evidence,
                    group_id=group.id,
                    counts=violates,
                )
            )
        res.findings.extend(cap.summaries(self, {"lineage_cross_split": sev}))
        res.stats = stats
        return res
