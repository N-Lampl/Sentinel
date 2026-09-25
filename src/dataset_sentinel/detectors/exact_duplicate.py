"""Exact duplicate detection.

Three deterministic signals, strongest first:

1. the same file path registered in more than one split (a very common YOLO
   mistake: ``train`` and ``val`` pointing at the same directory);
2. byte-identical files (same SHA-256 of the file contents);
3. pixel-identical images after decoding (same size and RGB pixels; catches
   PNG <-> BMP conversions and lossless re-saves).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..fingerprints.index import union_find_groups
from ..model import Confidence, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap, sample_uri_list

_POLICY_DESC = (
    "Identical samples must not appear in more than one split: a model that has "
    "seen the exact evaluation image during training produces inflated metrics."
)


@detectors.register("exact_duplicate")
class ExactDuplicateDetector(Detector):
    name = "exact_duplicate"
    description = "Byte-identical or pixel-identical samples (and the same file registered in several splits)"
    modalities = ("image",)
    needs_fingerprints = True

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        cross = ctx.severity("policy.exact_duplicate.cross_split")
        within = ctx.severity("policy.exact_duplicate.within_split")
        if cross is None and within is None:
            res.skipped = "disabled by policy"
            return res

        store = ctx.fingerprints
        samples = ctx.dataset.samples
        n = len(samples)
        index = {s.id: i for i, s in enumerate(samples)}
        edges: List[Tuple[int, int]] = []
        edge_kinds: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

        def connect(members: List[int], kind: str) -> None:
            for a, b in zip(members, members[1:]):
                key = (min(a, b), max(a, b))
                edges.append(key)
                edge_kinds[key].add(kind)

        by_path: Dict[str, List[int]] = defaultdict(list)
        by_content: Dict[str, List[int]] = defaultdict(list)
        by_pixels: Dict[str, List[int]] = defaultdict(list)
        for s in samples:
            i = index[s.id]
            if s.path is not None:
                by_path[os.path.normcase(os.path.abspath(str(s.path)))].append(i)
            fp = store.get(s.id)
            if fp is None:
                continue
            if fp.content_sha256 and fp.file_size > 0:
                by_content[fp.content_sha256].append(i)
            if fp.ok and fp.pixel_hash:
                by_pixels[fp.pixel_hash].append(i)

        for members in by_path.values():
            if len(members) > 1:
                connect(members, "same_path")
        for members in by_content.values():
            if len(members) > 1:
                connect(members, "byte_identical")
        for members in by_pixels.values():
            if len(members) > 1:
                connect(members, "pixel_identical")

        comps = union_find_groups(n, edges)
        cap = FindingCap(int(ctx.config.get("output.max_findings_per_kind") or 1000))
        sev_for: Dict[str, Severity] = {}
        stats = {"groups": 0, "cross_split_groups": 0, "within_split_groups": 0, "samples_in_cross_split_groups": 0}

        for comp in comps:
            members = [samples[i] for i in comp]
            member_ids = [s.id for s in members]
            splits = sorted({s.split for s in members})
            kinds: Set[str] = set()
            for a in comp:
                for b in comp:
                    if a < b and (a, b) in edge_kinds:
                        kinds |= edge_kinds[(a, b)]
            fps = [store.get(s.id) for s in members]
            content_hashes = {fp.content_sha256 for fp in fps if fp and fp.content_sha256}
            evidence = {
                "match_types": sorted(kinds),
                "byte_identical_all": len(content_hashes) == 1,
                "content_sha256": sorted(content_hashes)[:5],
                "file_sizes": [fp.file_size if fp else None for fp in fps][:20],
                "paths": [str(s.path) for s in members][:20],
            }
            cross_split = len(splits) > 1
            sev = cross if cross_split else within
            violates = cross_split and sev in (Severity.ERROR, Severity.WARNING)
            group = self.group("exact_duplicate", member_ids, splits, Confidence.DETERMINISTIC, violates, evidence)
            res.add_group(group)
            stats["groups"] += 1
            if cross_split:
                stats["cross_split_groups"] += 1
                stats["samples_in_cross_split_groups"] += len(members)
            else:
                stats["within_split_groups"] += 1
            if sev is None:
                continue
            kind = "exact_duplicate_cross_split" if cross_split else "exact_duplicate_within_split"
            sev_for[kind] = sev
            if not cap.allow(kind):
                continue
            how = "same file path" if "same_path" in kinds else ("byte-identical files" if "byte_identical" in kinds else "pixel-identical images")
            if cross_split:
                title = f"Exact duplicate across splits {'/'.join(splits)} ({len(members)} samples, {how})"
                message = f"The same image appears in splits {', '.join(splits)}: {sample_uri_list(members)}."
                remediation = (
                    "Keep a single copy in one split (normally train) and remove the others, or move the whole "
                    "group to one split. Re-evaluate afterwards: metrics on the current evaluation split are inflated."
                )
            else:
                title = f"Duplicate samples within split {splits[0]} ({len(members)} samples, {how})"
                message = f"Identical images inside split {splits[0]}: {sample_uri_list(members)}."
                remediation = "Deduplicate to avoid over-weighting these samples and wasting compute."
            res.add(
                self.finding(
                    kind=kind,
                    title=title,
                    severity=sev,
                    confidence=Confidence.DETERMINISTIC,
                    policy=ctx.policy_label("policy.exact_duplicate.cross_split" if cross_split else "policy.exact_duplicate.within_split"),
                    policy_description=_POLICY_DESC,
                    message=message,
                    remediation=remediation,
                    samples=[s.ref() for s in members],
                    evidence=evidence,
                    group_id=group.id,
                    counts=violates,
                )
            )
        res.findings.extend(cap.summaries(self, sev_for))
        res.stats = stats
        return res
