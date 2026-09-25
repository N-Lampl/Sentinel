"""Exact and near-duplicate detection for text samples.

* ``text_exact_duplicate``: identical text after normalisation (case,
  punctuation and whitespace ignored). Deterministic.
* ``text_near_duplicate``: MinHash / LSH candidates over word 2-gram
  shingles, verified with the exact Jaccard index (default >= 0.7: a
  one-word edit in a 13-17 word question scores 0.71-0.78, sibling items of a
  two-slot template about 0.68). Jaccard >= 0.85 is high confidence, below
  is heuristic. Benchmarks built from a one-slot template ("What is 2 + {n}?")
  are lexical near-duplicates of each other by any measure; raise
  ``min_jaccard`` or allowlist ``near_duplicate_within_split`` for those and
  rely on ``text_contamination``, which is unaffected.

Both reuse the image policies (``policy.exact_duplicate``,
``policy.near_duplicate``) and the same finding kinds, so allowlists,
split-pair overrides and the metric treat text and images alike.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ..fingerprints.text import jaccard, lsh_candidate_pairs, snippet
from ..model import Confidence, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap, sample_uri_list, scoped_components

_EXACT_DESC = "Identical texts must not appear in more than one split: the model has seen the evaluation item verbatim."
_NEAR_DESC = "Near-identical texts (light edits, reformatting) must not be split between training and evaluation."


def _snippets(ctx: DetectorContext, ids: List[str], limit: int = 12) -> Dict[str, str]:
    out = {}
    for sid in ids[:limit]:
        s = ctx.dataset.get_or_none(sid)
        if s is not None:
            out[sid] = snippet(s.metadata.get("text", ""))
    return out


@detectors.register("text_exact_duplicate")
class TextExactDuplicateDetector(Detector):
    name = "text_exact_duplicate"
    description = "Identical texts (after normalisation) across or within splits"
    modalities = ("text",)

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        cross = ctx.severity("policy.exact_duplicate.cross_split")
        within = ctx.severity("policy.exact_duplicate.within_split")
        if cross is None and within is None:
            res.skipped = "disabled by policy"
            return res
        store = ctx.text_fingerprints
        by_hash: Dict[str, List[str]] = defaultdict(list)
        for sid, fp in store.all().items():
            if fp.n_tokens > 0:
                by_hash[fp.exact_hash].append(sid)
        cap = FindingCap(int(ctx.config.get("output.max_findings_per_kind") or 1000))
        sev_for: Dict[str, Severity] = {}
        stats = {"groups": 0, "cross_split_groups": 0}
        for member_ids in by_hash.values():
            if len(member_ids) < 2:
                continue
            members = [ctx.dataset.get(i) for i in member_ids]
            splits = sorted({m.split for m in members})
            cross_split = len(splits) > 1
            sev = cross if cross_split else within
            violates = cross_split and sev in (Severity.ERROR, Severity.WARNING)
            evidence = {"match": "normalised text identical", "snippets": _snippets(ctx, member_ids)}
            group = self.group("exact_duplicate", member_ids, splits, Confidence.DETERMINISTIC, violates, evidence)
            res.add_group(group)
            stats["groups"] += 1
            stats["cross_split_groups"] += int(cross_split)
            if sev is None:
                continue
            kind = "exact_duplicate_cross_split" if cross_split else "exact_duplicate_within_split"
            sev_for[kind] = sev
            if not cap.allow(kind):
                continue
            res.add(
                self.finding(
                    kind=kind,
                    title=(f"Identical text across splits {'/'.join(splits)} ({len(members)} records)" if cross_split else f"Identical text repeated within {splits[0]} ({len(members)} records)"),
                    severity=sev,
                    confidence=Confidence.DETERMINISTIC,
                    policy=ctx.policy_label("policy.exact_duplicate.cross_split" if cross_split else "policy.exact_duplicate.within_split"),
                    policy_description=_EXACT_DESC,
                    message=f"\"{snippet(members[0].metadata.get('text', ''), 120)}\" appears in {', '.join(splits)}: {sample_uri_list(members)}.",
                    remediation=("Remove the evaluation copies (or move the whole group to one split) and re-evaluate." if cross_split else "Deduplicate to avoid over-weighting the record."),
                    samples=[m.ref() for m in members],
                    evidence=evidence,
                    group_id=group.id,
                    counts=violates,
                )
            )
        res.findings.extend(cap.summaries(self, sev_for))
        res.stats = stats
        return res


@detectors.register("text_near_duplicate")
class TextNearDuplicateDetector(Detector):
    name = "text_near_duplicate"
    description = "Near-identical texts (MinHash/LSH candidates verified by Jaccard on word shingles)"
    modalities = ("text",)

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        cross = ctx.severity("policy.near_duplicate.cross_split")
        within = ctx.severity("policy.near_duplicate.within_split")
        if cross is None and within is None:
            res.skipped = "disabled by policy"
            return res
        min_j = float(ctx.config.get("policy.text_near_duplicate.min_jaccard", 0.7) or 0)
        bands = int(ctx.config.get("policy.text_near_duplicate.bands", 32) or 32)
        store = ctx.text_fingerprints
        ids = [sid for sid, fp in store.all().items() if fp.n_tokens >= 3]
        if len(ids) < 2:
            res.skipped = "fewer than two texts with at least three tokens"
            return res
        exact = {sid: store.get(sid).exact_hash for sid in ids}  # type: ignore[union-attr]
        sigs = store.signature_matrix(ids)
        candidates = lsh_candidate_pairs(sigs, bands=bands)
        pairs: List[Tuple[int, int, float]] = []
        for a, b in candidates:
            if exact[ids[a]] == exact[ids[b]]:
                continue
            j = jaccard(store.get(ids[a]).shingles, store.get(ids[b]).shingles)  # type: ignore[union-attr]
            if j >= min_j:
                pairs.append((a, b, j))
        ctx.shared["text_near_duplicate_pairs"] = [(ids[a], ids[b], j) for a, b, j in pairs]
        split_of_idx = [ctx.dataset.get(sid).split for sid in ids]
        comps = scoped_components(len(ids), [(a, b) for a, b, _ in pairs], lambda i: split_of_idx[i])
        lookup = {(a, b): j for a, b, j in pairs}
        cap = FindingCap(int(ctx.config.get("output.max_findings_per_kind") or 1000))
        sev_for: Dict[str, Severity] = {}
        stats = {"lsh_candidates": len(candidates), "verified_pairs": len(pairs), "min_jaccard": min_j, "groups": 0, "cross_split_groups": 0}
        for comp, cross_split in comps:
            cs = set(comp)
            members = [ctx.dataset.get(ids[i]) for i in comp]
            member_ids = [m.id for m in members]
            splits = sorted({m.split for m in members})
            comp_pairs = sorted(((a, b, j) for (a, b), j in lookup.items() if a in cs and b in cs and (split_of_idx[a] != split_of_idx[b]) == cross_split), key=lambda t: -t[2])
            cross_pairs = comp_pairs if cross_split else []
            best = comp_pairs[0]
            confidence = Confidence.HIGH_CONFIDENCE if best[2] >= 0.85 else Confidence.HEURISTIC
            evidence = {
                "algorithm": f"minhash/lsh over word {store.shingle_n}-gram shingles, exact jaccard verification",
                "min_jaccard": min_j,
                "best_cross_split_jaccard": round(cross_pairs[0][2], 4) if cross_pairs else None,
                "pairs": [{"a": ids[a], "b": ids[b], "jaccard": round(j, 4), "cross_split": ctx.dataset.get(ids[a]).split != ctx.dataset.get(ids[b]).split} for a, b, j in comp_pairs[:20]],
                "snippets": _snippets(ctx, member_ids),
            }
            sev = cross if cross_split else within
            violates = cross_split and sev in (Severity.ERROR, Severity.WARNING)
            group = self.group("near_duplicate", member_ids, splits, confidence, violates, evidence)
            res.add_group(group)
            stats["groups"] += 1
            stats["cross_split_groups"] += int(cross_split)
            if sev is None:
                continue
            kind = "near_duplicate_cross_split" if cross_split else "near_duplicate_within_split"
            sev_for[kind] = sev
            if not cap.allow(kind):
                continue
            a, b = ctx.dataset.get(ids[best[0]]), ctx.dataset.get(ids[best[1]])
            res.add(
                self.finding(
                    kind=kind,
                    title=(f"Near-identical text across splits {'/'.join(splits)} (Jaccard {best[2]:.2f}, {len(members)} records)" if cross_split else f"Near-identical texts within {splits[0]} (Jaccard {best[2]:.2f}, {len(members)} records)"),
                    severity=sev,
                    confidence=confidence,
                    policy=ctx.policy_label("policy.near_duplicate.cross_split" if cross_split else "policy.near_duplicate.within_split"),
                    policy_description=_NEAR_DESC,
                    message=f"{a.split}:{a.uri} \"{snippet(a.metadata.get('text', ''), 90)}\" vs {b.split}:{b.uri} \"{snippet(b.metadata.get('text', ''), 90)}\" share {best[2]:.0%} of their word shingles.",
                    remediation=("Treat as the same item: drop the evaluation copies or move the group to one split." if cross_split else "Deduplicate to reduce redundancy."),
                    samples=[m.ref() for m in members],
                    evidence=evidence,
                    group_id=group.id,
                    counts=violates,
                )
            )
        res.findings.extend(cap.summaries(self, sev_for))
        res.stats = stats
        return res
