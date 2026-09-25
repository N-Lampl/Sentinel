"""Near-duplicate detection with 64-bit difference hashes.

Two images are near-duplicates when the Hamming distance between their
dHashes is at most ``policy.near_duplicate.threshold`` (default 6) and the
normalised correlation of their 16x16 thumbnails is at least
``min_correlation``. This catches re-encodes, resizes, small crops of the
border, colour/brightness tweaks and light compression artefacts. Exact
duplicates are excluded here because the exact-duplicate detector already
reports them.

Optional embedding re-ranking (``policy.near_duplicate.embedding.enabled``):
candidates are generated with a wider hash threshold and kept only when the
cosine similarity of their embeddings is at least ``min_cosine``. Embeddings
are computed for candidate samples only, never all-pairs.

Confidence: distance <= 3 and correlation >= 0.9 on the closest cross-split
pair is ``high_confidence``; larger distances are ``heuristic`` and should be
reviewed (simple graphics and low-texture images collide more easily).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..fingerprints.image import normalized_correlation, thumb_array
from ..fingerprints.index import HammingIndex
from ..model import Confidence, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap, sample_uri_list, scoped_components

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
    description = "Perceptually near-identical images (dHash Hamming distance, thumbnail-verified, optional embedding re-ranking)"
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

        # ---------------------------------------------------------- embeddings (optional)
        emb_cfg = ctx.config.section("policy.near_duplicate.embedding")
        emb_enabled = bool(emb_cfg.get("enabled", False))
        emb_store = None
        emb_stats: Dict[str, Any] = {"enabled": emb_enabled}
        query_threshold = threshold
        if emb_enabled:
            from ..embeddings.base import get_provider
            from ..embeddings.store import EmbeddingStore

            provider = get_provider(str(emb_cfg.get("provider") or "builtin"))
            if provider is None:
                emb_stats["error"] = "provider unavailable; re-ranking skipped"
                emb_enabled = False
            else:
                cache = ctx.config.get("performance.cache")
                cache_dir = ctx.config.resolve_path(cache) if cache else None
                emb_store = EmbeddingStore(ctx.dataset, provider, cache_dir=cache_dir)
                query_threshold = int(emb_cfg.get("threshold") or max(threshold, 10))
                emb_stats.update({"provider": provider.name, "version": provider.version, "candidate_threshold": query_threshold})
        min_cosine = float(emb_cfg.get("min_cosine", 0.9) or 0)

        index = HammingIndex(codes, query_threshold)
        raw_pairs = index.pairs_within()
        min_corr = float(ctx.config.get("policy.near_duplicate.min_correlation", 0.8) or 0)
        pairs: List[Tuple[int, int, int]] = []
        correlations: Dict[Tuple[int, int], float] = {}
        cosines: Dict[Tuple[int, int], float] = {}
        rejected = 0
        rejected_by_embedding = 0
        thumbs: Dict[int, Any] = {}

        def thumb(i: int):
            if i not in thumbs:
                fp = store.get(ids[i])
                thumbs[i] = thumb_array(fp) if fp else None
            return thumbs[i]

        verified: List[Tuple[int, int, int, float]] = []
        for a, b, d in raw_pairs:
            if exact_key[a] == exact_key[b]:
                continue
            ta, tb = thumb(a), thumb(b)
            corr = normalized_correlation(ta, tb) if ta is not None and tb is not None else 1.0
            if corr < min_corr:
                rejected += 1
                continue
            verified.append((a, b, d, corr))

        if emb_enabled and emb_store is not None and verified:
            from ..embeddings.base import cosine_similarity

            emb_store.ensure({ids[a] for a, _, _, _ in verified} | {ids[b] for _, b, _, _ in verified})
            for a, b, d, corr in verified:
                va, vb = emb_store.get(ids[a]), emb_store.get(ids[b])
                cos = cosine_similarity(va, vb) if va is not None and vb is not None else 0.0
                if cos < min_cosine:
                    rejected_by_embedding += 1
                    continue
                pairs.append((a, b, d))
                correlations[(a, b)] = corr
                cosines[(a, b)] = cos
            emb_stats.update({k: v for k, v in emb_store.stats.items()})
            emb_stats["rejected_by_embedding"] = rejected_by_embedding
        else:
            for a, b, d, corr in verified:
                pairs.append((a, b, d))
                correlations[(a, b)] = corr

        ctx.shared["near_duplicate_pairs"] = [(ids[a], ids[b], d) for a, b, d in pairs]
        ctx.shared["near_duplicate_threshold"] = query_threshold

        split_of_idx = [ctx.dataset.get(sid).split for sid in ids]
        comps = scoped_components(len(ids), [(a, b) for a, b, _ in pairs], lambda i: split_of_idx[i])
        pair_lookup: Dict[Tuple[int, int], int] = {(a, b): d for a, b, d in pairs}
        cap = FindingCap(int(ctx.config.get("policy.near_duplicate.max_group_findings") or ctx.config.get("output.max_findings_per_kind") or 500))
        sev_for: Dict[str, Severity] = {}
        stats: Dict[str, Any] = {
            "threshold": threshold,
            "min_correlation": min_corr,
            "hash_candidate_pairs": len(raw_pairs),
            "rejected_by_verification": rejected,
            "candidate_pairs": len(pairs),
            "groups": 0,
            "cross_split_groups": 0,
            "within_split_groups": 0,
            "blank_images_skipped": blank_skipped,
            "distance_histogram": _hist(pairs, query_threshold),
            "embedding": emb_stats,
        }

        for comp, cross_split in comps:
            members = [ctx.dataset.get(ids[i]) for i in comp]
            member_ids = [s.id for s in members]
            splits = sorted({s.split for s in members})
            comp_set = set(comp)
            comp_pairs = [(a, b, d) for (a, b), d in pair_lookup.items() if a in comp_set and b in comp_set and (split_of_idx[a] != split_of_idx[b]) == cross_split]
            comp_pairs.sort(key=lambda t: t[2])
            cross_pairs = comp_pairs if cross_split else []
            best_pair = cross_pairs[0] if cross_pairs else (comp_pairs[0] if comp_pairs else None)
            best = best_pair[2] if best_pair else threshold
            best_corr = correlations.get((best_pair[0], best_pair[1]), 1.0) if best_pair else 1.0
            best_cos: Optional[float] = cosines.get((best_pair[0], best_pair[1])) if best_pair else None
            strong = best <= HIGH_CONF_DISTANCE and best_corr >= HIGH_CONF_CORRELATION
            if best_cos is not None and best_cos >= 0.97 and best_corr >= HIGH_CONF_CORRELATION:
                strong = True  # embeddings confirm a wider-threshold candidate
            confidence = Confidence.HIGH_CONFIDENCE if strong else Confidence.HEURISTIC
            if len(comp) > LARGE_CLUSTER:
                confidence = Confidence.HEURISTIC
            evidence: Dict[str, Any] = {
                "algorithm": "dhash64 + 16x16 thumbnail correlation" + (" + embedding cosine" if emb_enabled else ""),
                "threshold": threshold,
                "min_cross_split_distance": cross_pairs[0][2] if cross_pairs else None,
                "min_distance": comp_pairs[0][2] if comp_pairs else None,
                "best_pair_correlation": round(best_corr, 4),
                "best_pair_cosine": round(best_cos, 4) if best_cos is not None else None,
                "pairs": [
                    {
                        "a": ids[a],
                        "b": ids[b],
                        "distance": d,
                        "correlation": round(correlations.get((a, b), 1.0), 4),
                        **({"cosine": round(cosines[(a, b)], 4)} if (a, b) in cosines else {}),
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
                cos_txt = f", embedding cosine {cosines[(a, b)]:.2f}" if (a, b) in cosines else ""
                title = f"Near-duplicate across splits {'/'.join(splits)} (distance {d}, {len(members)} samples)"
                message = (
                    f"{ctx.dataset.get(ids[a]).split}:{ctx.dataset.get(ids[a]).uri} and "
                    f"{ctx.dataset.get(ids[b]).split}:{ctx.dataset.get(ids[b]).uri} differ by {d} bits of 64 "
                    f"(threshold {threshold}; thumbnail correlation {correlations.get((a, b), 1.0):.2f}{cos_txt}). Cluster: {sample_uri_list(members)}."
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
