"""Derivative-sample detection based on image content.

Two families of derivations are detected:

* **Dihedral transforms**: horizontal / vertical flips, rotations by
  90 / 180 / 270 degrees, transposes. Every image carries the dHash of its 8
  dihedral transforms; the identity hashes are indexed and each non-identity
  transform is queried against the index. A match ``dhash(T(A)) ~ dhash(B)``
  means ``B`` is (close to) ``T(A)``.
* **Regular crops and tiles**: 2x2 tiles, left / right / top / bottom halves
  and 50% / 75% centre crops. Every image carries the dHash of those regions
  of its thumbnail; a sample whose identity hash matches a region hash of
  another sample is a crop / tile of it.

All candidates are verified with the correlation of 16x16 thumbnails. Pairs
already within threshold on the identity hash are left to the near-duplicate
detector.

Arbitrary crops, tiles from other grids, paraphrases and other derivations are
covered by the *lineage* detector when the dataset carries ``derived_from``
metadata or a file-name pattern (see ``dataset.lineage``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ..fingerprints.image import (
    DIHEDRAL_INVERSE,
    DIHEDRAL_NAMES,
    REGION_NAMES,
    normalized_correlation,
    region_correlation,
    region_slices,
    thumb_array,
    transform_thumb,
)
from ..fingerprints.index import HammingIndex, union_find_groups
from ..model import Confidence, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap, sample_uri_list

_POLICY_DESC = (
    "Transformed copies (flips, rotations, transposes, tiles, crops) of a sample must stay in the "
    "same split as their source; otherwise augmentation applied before splitting leaks training content."
)
HIGH_CONF_DISTANCE = 3
HIGH_CONF_CORRELATION = 0.9
HIGH_CONF_CROP_CORRELATION = 0.95


@detectors.register("derivative")
class DerivativeDetector(Detector):
    name = "derivative"
    description = "Flipped / rotated / transposed copies and regular tiles / crops of other samples"
    modalities = ("image",)
    needs_fingerprints = True

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        cross = ctx.severity("policy.derivative.cross_split")
        within = ctx.severity("policy.derivative.within_split")
        threshold = int(ctx.config.get("policy.derivative.threshold", 6))
        min_corr = float(ctx.config.get("policy.derivative.min_correlation", 0.8) or 0)
        detect_crops = bool(ctx.config.get("policy.derivative.crops", True))
        if cross is None and within is None:
            res.skipped = "disabled by policy"
            return res

        store = ctx.fingerprints
        all_ids, all_arr = store.dihedral_arrays()
        keep = [i for i, sid in enumerate(all_ids) if not store.get(sid).is_blank]  # type: ignore[union-attr]
        ids = [all_ids[i] for i in keep]
        arr = all_arr[keep]
        if len(ids) < 2:
            res.skipped = "fewer than two decodable images"
            return res
        pos = {sid: i for i, sid in enumerate(ids)}

        identity = arr[:, 0]
        index = HammingIndex(identity, threshold)
        close: set = set()
        shared_pairs = ctx.shared.get("near_duplicate_pairs")
        if shared_pairs is not None and ctx.shared.get("near_duplicate_threshold", threshold) >= threshold:
            for a, b, _ in shared_pairs:
                if a in pos and b in pos:
                    close.add((min(pos[a], pos[b]), max(pos[a], pos[b])))
        else:
            for a, b, _ in index.pairs_within():
                close.add((a, b))
        exact_key = []
        for sid in ids:
            fp = store.get(sid)
            exact_key.append((fp.pixel_hash or fp.content_sha256 or sid) if fp else sid)

        thumbs: Dict[int, Any] = {}

        def thumb(i: int):
            if i not in thumbs:
                fp = store.get(ids[i])
                thumbs[i] = thumb_array(fp) if fp else None
            return thumbs[i]

        split_rank = {name: r for r, name in enumerate(ctx.ordered_splits())}

        def rank(i: int) -> int:
            return split_rank.get(ctx.dataset.get(ids[i]).split, 99)

        rejected = 0
        # key -> (distance, transform name, source index, correlation, directed)
        best: Dict[Tuple[int, int], Tuple[int, str, int, float, bool]] = {}

        def consider(q: int, i: int, d: int, name: str, corr: float, directed: bool) -> None:
            key = (min(q, i), max(q, i))
            prev = best.get(key)
            if prev is None or (d, -corr) < (prev[0], -prev[3]):
                best[key] = (d, name, q, corr, directed)

        # ---------------------------------------------------------- dihedral transforms
        for t in range(1, 8):
            for q, i, d in index.query(arr[:, t]):
                if q == i:
                    continue
                key = (min(q, i), max(q, i))
                if key in close or exact_key[q] == exact_key[i]:
                    continue
                tq, ti = thumb(q), thumb(i)
                corr = normalized_correlation(transform_thumb(tq, t), ti) if tq is not None and ti is not None else 1.0
                if corr < min_corr:
                    rejected += 1
                    continue
                # statement: ids[i] == T(ids[q])
                consider(q, i, d, DIHEDRAL_NAMES[t], corr, directed=False)

        # ---------------------------------------------------------- tiles and crops
        # Region hashes are computed from the parent's thumbnail while the crop's
        # own hash comes from its full-resolution file, so the two pipelines
        # disagree by a few bits more than for whole-image transforms. Candidates
        # therefore use a wider Hamming threshold and rely on the (stricter)
        # thumbnail correlation for verification and confidence.
        crop_candidates = 0
        crop_note = ""
        crop_threshold = int(ctx.config.get("policy.derivative.crop_threshold", 8))
        crop_min_corr = float(ctx.config.get("policy.derivative.crop_min_correlation", 0.9) or 0)
        crops_max = int(ctx.config.get("policy.derivative.crops_max_images", 0) or 0)
        if detect_crops and crops_max and len(ids) > crops_max:
            detect_crops = False
            crop_note = f"crop/tile detection skipped: {len(ids)} images exceed policy.derivative.crops_max_images={crops_max}"
        if detect_crops:
            # Only cross-split crops matter for leakage, so every split's region
            # hashes are queried against an index of the OTHER splits only; flat
            # regions (sky, walls) are skipped because their hashes collide with
            # every low-texture image.
            region_ids, regions = store.region_arrays()
            rpos = np.array([pos.get(sid, -1) for sid in region_ids], dtype=np.int64)
            valid = rpos >= 0
            region_ids = [sid for sid, ok in zip(region_ids, valid) if ok]
            regions = regions[valid]
            rpos = rpos[valid]
            flat = _flat_region_mask(store, region_ids)
            split_of_idx = np.array([split_rank.get(ctx.dataset.get(sid).split, 99) for sid in ids])
            split_of_region = split_of_idx[rpos]
            for s_rank in np.unique(split_of_region):
                others = np.flatnonzero(split_of_idx != s_rank)
                mine = np.flatnonzero(split_of_region == s_rank)
                if len(others) == 0 or len(mine) == 0:
                    continue
                crop_index = HammingIndex(identity[others], crop_threshold)
                for r, name in enumerate(REGION_NAMES):
                    query_rows = mine[~flat[mine, r]]
                    if len(query_rows) == 0:
                        continue
                    for q_r, i_o, d in crop_index.query(regions[query_rows, r]):
                        q = int(rpos[query_rows[q_r]])
                        i = int(others[i_o])
                        if q == i:
                            continue
                        key = (min(q, i), max(q, i))
                        if key in close or exact_key[q] == exact_key[i]:
                            continue
                        crop_candidates += 1
                        tq, ti = thumb(q), thumb(i)
                        corr = region_correlation(tq, ti, name) if tq is not None and ti is not None else 1.0
                        if corr < crop_min_corr:
                            rejected += 1
                            continue
                        # statement: ids[i] is region `name` of ids[q]  (directed: parent -> crop)
                        consider(q, i, d, f"crop:{name}", corr, directed=True)

        # Orient undirected (dihedral) statements from the earlier split (train) to
        # the later one, inverting the transform when the direction is swapped.
        for key, (d, t_name, q, corr, directed) in list(best.items()):
            if directed:
                continue
            other = key[1] if q == key[0] else key[0]
            if (rank(q), ids[q]) > (rank(other), ids[other]):
                best[key] = (d, DIHEDRAL_INVERSE[t_name], other, corr, directed)

        comps = union_find_groups(len(ids), list(best.keys()))
        cap = FindingCap(int(ctx.config.get("output.max_findings_per_kind") or 1000))
        sev_for: Dict[str, Severity] = {}
        transform_counts: Dict[str, int] = defaultdict(int)
        for _d, t, _q, _c, _dir in best.values():
            transform_counts[t] += 1
        stats = {
            "threshold": threshold,
            "min_correlation": min_corr,
            "rejected_by_verification": rejected,
            "crop_candidates": crop_candidates,
            "crops": detect_crops,
            **({"note": crop_note} if crop_note else {}),
            "pairs": len(best),
            "groups": 0,
            "cross_split_groups": 0,
            "transforms": dict(transform_counts),
        }

        for comp in comps:
            comp_set = set(comp)
            members = [ctx.dataset.get(ids[i]) for i in comp]
            member_ids = [s.id for s in members]
            splits = sorted({s.split for s in members})
            cross_split = len(splits) > 1
            pairs = []
            for (a, b), (d, t, q, corr, _directed) in best.items():
                if a in comp_set and b in comp_set:
                    src, dst = (q, b if q == a else a)
                    pairs.append(
                        {
                            "source": ids[src],
                            "derived": ids[dst],
                            "transform": t,
                            "distance": d,
                            "correlation": round(corr, 4),
                            "cross_split": ctx.dataset.get(ids[src]).split != ctx.dataset.get(ids[dst]).split,
                        }
                    )
            pairs.sort(key=lambda p: (not p["cross_split"], -p["correlation"], p["distance"]))
            cross_pairs = [p for p in pairs if p["cross_split"]]
            p_best = cross_pairs[0] if cross_pairs else pairs[0]
            if p_best["transform"].startswith("crop:"):
                strong = p_best["correlation"] >= HIGH_CONF_CROP_CORRELATION
            else:
                strong = p_best["distance"] <= HIGH_CONF_DISTANCE and p_best["correlation"] >= HIGH_CONF_CORRELATION
            confidence = Confidence.HIGH_CONFIDENCE if strong else Confidence.HEURISTIC
            evidence = {
                "algorithm": "dhash64 over dihedral transforms and sub-regions + 16x16 thumbnail correlation",
                "threshold": threshold,
                "min_cross_split_distance": cross_pairs[0]["distance"] if cross_pairs else None,
                "pairs": pairs[:20],
                "transforms": sorted({p["transform"] for p in pairs}),
            }
            sev = cross if cross_split else within
            violates = cross_split and sev in (Severity.ERROR, Severity.WARNING)
            group = self.group("derivative_transform", member_ids, splits, confidence, violates, evidence)
            res.add_group(group)
            stats["groups"] += 1
            if cross_split:
                stats["cross_split_groups"] += 1
            if sev is None:
                continue
            kind = "derivative_cross_split" if cross_split else "derivative_within_split"
            sev_for[kind] = sev
            if not cap.allow(kind):
                continue
            p0 = cross_pairs[0] if cross_pairs else pairs[0]
            src, dst = ctx.dataset.get(p0["source"]), ctx.dataset.get(p0["derived"])
            how = _describe(p0["transform"])
            desc = (
                f"{dst.split}:{dst.uri} looks like {how} of {src.split}:{src.uri} "
                f"(distance {p0['distance']}, correlation {p0['correlation']:.2f})"
            )
            if cross_split:
                title = f"Transformed copy across splits {'/'.join(splits)} ({p0['transform']}, {len(members)} samples)"
                remediation = (
                    "Move augmented / transformed / cropped copies into the same split as their source image, or "
                    "generate augmentations at training time only. Verify visually if the distance is above 3."
                )
            else:
                title = f"Transformed copies within split {splits[0]} ({p0['transform']}, {len(members)} samples)"
                remediation = "Offline augmentation inside one split is not leakage; consider on-the-fly augmentation instead."
            res.add(
                self.finding(
                    kind=kind,
                    title=title,
                    severity=sev,
                    confidence=confidence,
                    policy=ctx.policy_label("policy.derivative.cross_split" if cross_split else "policy.derivative.within_split"),
                    policy_description=_POLICY_DESC,
                    message=f"{desc}. Cluster: {sample_uri_list(members)}.",
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


def _flat_region_mask(store: Any, region_ids: List[str], min_std: float = 2.0) -> np.ndarray:
    """(n, len(REGION_NAMES)) boolean mask: True where the parent's 16x16
    thumbnail region is (nearly) uniform, so its hash would collide with
    every low-texture image."""
    n = len(region_ids)
    mask = np.zeros((n, len(REGION_NAMES)), dtype=bool)
    if n == 0:
        return mask
    thumbs = np.zeros((n, 16, 16), dtype=np.float32)
    for k, sid in enumerate(region_ids):
        fp = store.get(sid)
        t = thumb_array(fp) if fp else None
        if t is not None:
            thumbs[k] = t
    slices = region_slices(16)
    for r, name in enumerate(REGION_NAMES):
        rows, cols = slices[name]
        part = thumbs[:, rows, cols]
        mask[:, r] = part.reshape(n, -1).std(axis=1) < min_std
    return mask


def _describe(transform: str) -> str:
    if transform.startswith("crop:"):
        region = transform.split(":", 1)[1]
        if region.startswith("tile_"):
            return f"the {region[5:].replace('_', '-')} tile"
        if region.endswith("_half"):
            return f"the {region.replace('_', ' ')}"
        return f"a {region.replace('center_crop_', '')}% centre crop"
    return transform.replace("_", " ")


__all__ = ["DerivativeDetector"]
_ = List
