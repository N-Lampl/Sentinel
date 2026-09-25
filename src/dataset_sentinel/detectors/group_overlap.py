"""Entity / source / batch group overlap across splits.

Samples that share an entity identifier (patient, subject, account, ...),
a data source (camera, device, site, video, session) or a batch must be
kept in the same split. Identifiers come from ``sample.groups``, filled by
the enrichment step from metadata keys and file-name patterns.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ..model import Confidence, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap, sample_uri_list

_POLICY_DESC = (
    "Samples that share an entity, source, session or batch identifier must be assigned "
    "to a single split; otherwise the model is evaluated on entities it trained on."
)


@detectors.register("group_overlap")
class GroupOverlapDetector(Detector):
    name = "group_overlap"
    description = "Shared entity / source / session / batch identifiers across splits"
    modalities = ("*",)

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        sev = ctx.severity("policy.group_overlap.cross_split")
        if sev is None:
            res.skipped = "disabled by policy"
            return res
        filename_keys = set((ctx.config.get("dataset.groups.from_filename") or {}).keys())

        by_key: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for s in ctx.dataset.samples:
            for key, value in s.groups.items():
                by_key[key][value].append(s.id)

        if not by_key:
            res.add(
                self.finding(
                    kind="group_overlap_not_evaluated",
                    title="No entity/source/batch identifiers found: group leakage not evaluated",
                    severity=Severity.INFO,
                    confidence=Confidence.INCONCLUSIVE,
                    policy=ctx.policy_label("policy.group_overlap.cross_split"),
                    policy_description=_POLICY_DESC,
                    message=(
                        "No sample carries a group identifier (e.g. patient_id, source, session, device). "
                        "If your data has such structure, entity leakage cannot be ruled out."
                    ),
                    remediation=(
                        "Provide identifiers via a metadata sidecar (dataset.metadata.file), COCO image fields, "
                        "or a file-name pattern (dataset.groups.from_filename)."
                    ),
                )
            )
            res.stats = {"keys": []}
            return res

        cap = FindingCap(int(ctx.config.get("output.max_findings_per_kind") or 1000))
        stats: Dict[str, Dict[str, int]] = {}
        for key, groups in by_key.items():
            overlapping = 0
            overlapping_samples = 0
            for value, member_ids in groups.items():
                members = [ctx.dataset.get(i) for i in member_ids]
                splits = sorted({m.split for m in members})
                if len(splits) < 2:
                    continue
                overlapping += 1
                overlapping_samples += len(members)
                confidence = Confidence.HIGH_CONFIDENCE if key in filename_keys else Confidence.DETERMINISTIC
                per_split = _count_by_split(members)
                evidence = {"group_key": key, "group_value": value, "per_split": per_split, "source": "filename_pattern" if key in filename_keys else "metadata"}
                violates = sev in (Severity.ERROR, Severity.WARNING)
                group = self.group(f"group_overlap:{key}", member_ids, splits, confidence, violates, evidence)
                res.add_group(group)
                kind = "group_overlap"
                if not cap.allow(kind):
                    continue
                res.add(
                    self.finding(
                        kind=kind,
                        title=f"{key}={value} appears in splits {'/'.join(splits)} ({len(members)} samples)",
                        severity=sev,
                        confidence=confidence,
                        policy=ctx.policy_label("policy.group_overlap.cross_split"),
                        policy_description=_POLICY_DESC,
                        message=(
                            f"Samples sharing {key}={value!r} are split as "
                            + ", ".join(f"{k}: {v}" for k, v in per_split.items())
                            + f". Examples: {sample_uri_list(members)}."
                        ),
                        remediation=(
                            f"Re-split with a group-aware strategy (e.g. GroupKFold / GroupShuffleSplit on '{key}') "
                            "so every value of the identifier lands in exactly one split."
                        ),
                        samples=[m.ref() for m in members],
                        evidence=evidence,
                        group_id=group.id,
                        counts=violates,
                    )
                )
            stats[key] = {"groups": len(groups), "overlapping_groups": overlapping, "overlapping_samples": overlapping_samples}
        res.findings.extend(cap.summaries(self, {"group_overlap": sev}))
        res.stats = {"keys": stats}
        return res


def _count_by_split(members) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for m in members:
        out[m.split] += 1
    return dict(sorted(out.items()))


__all__ = ["GroupOverlapDetector"]
_ = Tuple
