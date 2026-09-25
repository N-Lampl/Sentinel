"""Near-duplicate detection with 64-bit difference hashes.

Two images are near-duplicates when the Hamming distance between their
dHashes is at most ``policy.near_duplicate.threshold`` (default 6). This
catches re-encodes, resizes, small crops of the border, colour/brightness
tweaks and light compression artefacts. Exact duplicates are excluded here
because the exact-duplicate detector already reports them.

Confidence: distance <= 3 on the closest cross-split pair is
``high_confidence``; larger distances are ``heuristic`` and should be
reviewed (simple graphics and low-texture images collide more easily).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from ..fingerprints.image import normalized_correlation, thumb_array
from ..fingerprints.index import HammingIndex, union_find_groups
from ..model import Confidence, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap, sample_uri_list

_POLICY_DESC = (
    "Visually near-identical samples must not be split between training and evaluation: "
    "the model effectively memorises the evaluation sample."
)
HIGH_CONF_DISTANCE = 3
HIGH_CONF_CORRELATION = 0.9
LARGE_CLUSTER = 200


@detectors.register("near_duplicate")
class NearDuplicateDetector(Detector):
    name = "near_duplicate"
    description = "Perceptually near-identical images (dHash Hamming distance)"
    modalities = ("image",)
    needs_fingerprints = True

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        cross = ctx.severity("policy.near_duplicate.cross_split")
        within = ctx.severity("policy.near_duplicate.within_split")
        threshold = int(ctx.config.get("policy.near_duplicate.threshold", 6))
        if cross is None and within is None:
            res.skipped = "disabled by policy"
            return res

        store = ctx.fingerprints
        all_ids, all_codes = store.dhash_arrays()
        keep = [i for i, sid in enumerate(all_ids) if not (store.get(sid) or _NULL).is_blank]
        ids = [all_ids[i] for i in keep]
        codes = all_codes[keep]
        blank_skipped = len(all_ids) - len(ids)
        if len(ids) < 2:
            res.skipped = "fewer than two decodable images"
            return res

        exact_key: List[str] = []
        for sid in ids:
            fp = store.get(sid)
            exact_key.append((fp.pixel_hash or fp.content_sha256 or sid) if fp else sid)

        index = HammingIndex(codes, threshold)
        raw_pairs = index.pairs_within()
        min_corr = float(ctx.config.get("policy.near_duplicate.min_correlation", 0.8) or 0)
        pairs: List[Tuple[int, int, int]] = []
        correlations: Dict[Tuple[int, int], float] = {}
        rejected = 0
        thumbs: Dict[int, Any] = {}

        def thumb(i: int):
            if i not in thumbs:
                fp = store.get(ids[i])
                thumbs[i] = thumb_array(fp) if fp else None
            return thumbs[i]

        for a, b, d in raw_pairs:
            if exact_key[a] == exact_key[b]:
                continue
            ta, tb = thumb(a), thumb(b)
            corr = normalized_correlation(ta, tb) if ta is not None and tb is not None else 1.0
            if corr < min_corr:
                rejected += 1
                continue
            pairs.append((a, b, d))
            correlations[(a, b)] = corr
        ctx.shared["near_duplicate_pairs"] = [(ids[a], ids[b], d) for a, b, d in pairs]
        ctx.shared["near_duplicate_threshold"] = threshold

        comps = union_find_groups(len(ids), [(a, b) for a, b, _ in pairs])
        pair_lookup: Dict[Tuple[int, int], int] = {(a, b): d for a, b, d in pairs}
        cap = FindingCap(int(ctx.config.get("policy.near_duplicate.max_group_findings") or ctx.config.get("output.max_findings_per_kind") or 500))
        sev_for: Dict[str, Severity] = {}
        stats = {
            "threshold": threshold,
            "min_correlation": min_corr,
            "hash_candidate_pairs": len(raw_pairs),
            "rejected_by_verification": rejected,
            "candidate_pairs": len(pairs),
            "groups": 0,
            "cross_split_groups": 0,
            "within_split_groups": 0,
            "blank_images_skipped": blank_skipped,
            "distance_histogram": _hist(pairs, threshold),
        }

        for comp in comps:
            members = [ctx.dataset.get(ids[i]) for i in comp]
            member_ids = [s.id for s in members]
            splits = sorted({s.split for s in members})
            cross_split = len(splits) > 1
            comp_set = set(comp)
            comp_pairs = [(a, b, d) for (a, b), d in pair_lookup.items() if a in comp_set and b in comp_set]
            comp_pairs.sort(key=lambda t: t[2])
            cross_pairs = [(a, b, d) for a, b, d in comp_pairs if ctx.dataset.get(ids[a]).split != ctx.dataset.get(ids[b]).split]
            best_pair = cross_pairs[0] if cross_pairs else (comp_pairs[0] if comp_pairs else None)
            best = best_pair[2] if best_pair else threshold
            best_corr = correlations.get((best_pair[0], best_pair[1]), 1.0) if best_pair else 1.0
            confidence = Confidence.HIGH_CONFIDENCE if (best <= HIGH_CONF_DISTANCE and best_corr >= HIGH_CONF_CORRELATION) else Confidence.HEURISTIC
            if len(comp) > LARGE_CLUSTER:
                confidence = Confidence.HEURISTIC
            evidence = {
                "algorithm": "dhash64 + 16x16 thumbnail correlation",
                "threshold": threshold,
                "min_cross_split_distance": cross_pairs[0][2] if cross_pairs else None,
                "min_distance": comp_pairs[0][2] if comp_pairs else None,
                "best_pair_correlation": round(best_corr, 4),
                "pairs": [
                    {
                        "a": ids[a],
                        "b": ids[b],
                        "distance": d,
                        "correlation": round(correlations.get((a, b), 1.0), 4),
                        "cross_split": ctx.dataset.get(ids[a]).split != ctx.dataset.get(ids[b]).split,
                    }
                    for a, b, d in (cross_pairs[:10] + [p for p in comp_pairs if p not in cross_pairs][:10])
                ],
                "cluster_size": len(comp),
                "large_cluster": len(comp) > LARGE_CLUSTER,
            }
            sev = cross if cross_split else within
            violates = cross_split and sev in (Severity.ERROR, Severity.WARNING)
            group = self.group("near_duplicate", member_ids, splits, confidence, violates, evidence)
            res.add_group(group)
            stats["groups"] += 1
            stats["cross_split_groups" if cross_split else "within_split_groups"] += 1
            if sev is None:
                continue
            kind = "near_duplicate_cross_split" if cross_split else "near_duplicate_within_split"
            sev_for[kind] = sev
            if not cap.allow(kind):
                continue
            if cross_split:
                a, b, d = cross_pairs[0]
                title = f"Near-duplicate across splits {'/'.join(splits)} (distance {d}, {len(members)} samples)"
                message = (
                    f"{ctx.dataset.get(ids[a]).split}:{ctx.dataset.get(ids[a]).uri} and "
                    f"{ctx.dataset.get(ids[b]).split}:{ctx.dataset.get(ids[b]).uri} differ by {d} bits of 64 "
                    f"(threshold {threshold}; thumbnail correlation {correlations.get((a, b), 1.0):.2f}). Cluster: {sample_uri_list(members)}."
                )
                remediation = (
                    "Inspect the pair; if they show the same content, move the whole cluster into a single split "
                    "(or drop the evaluation copies). Tune policy.near_duplicate.threshold if the match is spurious."
                )
            else:
                d = comp_pairs[0][2] if comp_pairs else 0
                title = f"Near-duplicates within split {splits[0]} (distance {d}, {len(members)} samples)"
                message = f"Cluster of visually near-identical images inside {splits[0]}: {sample_uri_list(members)}."
                remediation = "Consider deduplicating to reduce redundancy; not a leakage problem by itself."
            res.add(
                self.finding(
                    kind=kind,
                    title=title,
                    severity=sev,
                    confidence=confidence,
                    policy=ctx.policy_label("policy.near_duplicate.cross_split" if cross_split else "policy.near_duplicate.within_split"),
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


class _Null:
    is_blank = True


_NULL = _Null()


def _hist(pairs: List[Tuple[int, int, int]], threshold: int) -> Dict[str, int]:
    h: Dict[int, int] = defaultdict(int)
    for _, _, d in pairs:
        h[d] += 1
    return {str(d): h.get(d, 0) for d in range(threshold + 1)}
