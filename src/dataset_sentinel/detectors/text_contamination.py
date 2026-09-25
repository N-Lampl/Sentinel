"""Benchmark / evaluation contamination for text datasets.

For every evaluation item (samples in val / test splits) the detector
measures how much of it appears inside training documents: the fraction of
the item's word n-grams (default 8) that occur in a training sample
(*containment*). This is the standard contamination check used for LLM
benchmarks; unlike near-duplicate detection it catches a short question
embedded in a long document.

* containment 1.0: every n-gram of the item occurs in the document
  (deterministic: the item is present verbatim after normalisation);
* >= 0.8: high confidence; >= ``min_containment`` (default 0.5): heuristic.

Items shorter than the n-gram size are matched as a whole and only count at
containment 1.0. Only cross-split pairs are searched: evaluation items
against samples of earlier splits in ``policy.split_order`` (train).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ..config import split_role
from ..fingerprints.index import union_find_groups
from ..fingerprints.text import ngram_hashes, snippet
from ..model import Confidence, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap

_POLICY_DESC = (
    "Evaluation items must not appear inside the training data; a benchmark item contained in a training "
    "document measures memorisation, not capability."
)


@detectors.register("text_contamination")
class TextContaminationDetector(Detector):
    name = "text_contamination"
    description = "Evaluation items contained in training documents (word n-gram containment)"
    modalities = ("text",)

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        sev = ctx.severity("policy.contamination.cross_split")
        if sev is None:
            res.skipped = "disabled by policy"
            return res
        n = int(ctx.config.get("policy.contamination.ngram", 8) or 8)
        min_cont = float(ctx.config.get("policy.contamination.min_containment", 0.5) or 0)
        min_shared = int(ctx.config.get("policy.contamination.min_shared_ngrams", 2) or 1)
        store = ctx.text_fingerprints

        order = {role: i for i, role in enumerate(ctx.config.split_order)}
        ranked = sorted(ctx.dataset.splits, key=lambda s: (order.get(split_role(s) or "", 99), s))
        if len(ranked) < 2:
            res.skipped = "needs at least two splits (training data and evaluation items)"
            return res
        # the earliest split(s) are the training side; everything later is protected
        train_role = split_role(ranked[0])
        source_splits = [s for s in ranked if split_role(s) == train_role] if train_role else [ranked[0]]
        protected = [s for s in ranked if s not in source_splits]
        if not protected:
            res.skipped = "no evaluation split (val/test) to protect"
            return res

        # index: n-gram hash -> evaluation samples that contain it
        eval_ids: List[str] = [s.id for s in ctx.dataset.samples if s.split in protected and s.modality == "text"]
        eval_grams: Dict[str, np.ndarray] = {}
        owners: List[np.ndarray] = []
        hashes: List[np.ndarray] = []
        for k, sid in enumerate(eval_ids):
            g = ngram_hashes(store.tokens(sid), n)
            eval_grams[sid] = g
            if len(g):
                hashes.append(g)
                owners.append(np.full(len(g), k, dtype=np.int64))
        if not hashes:
            res.skipped = "evaluation items are empty"
            return res
        all_hashes = np.concatenate(hashes)
        all_owners = np.concatenate(owners)
        sort = np.argsort(all_hashes, kind="stable")
        all_hashes, all_owners = all_hashes[sort], all_owners[sort]

        # stream the training side
        edges: Dict[Tuple[str, str], Tuple[float, int]] = {}
        scanned = 0
        for s in ctx.dataset.samples:
            if s.split not in source_splits or s.modality != "text":
                continue
            scanned += 1
            grams = ngram_hashes(store.tokens(s.id), n)
            if len(grams) == 0:
                continue
            lo = np.searchsorted(all_hashes, grams, side="left")
            hi = np.searchsorted(all_hashes, grams, side="right")
            hit = hi > lo
            if not hit.any():
                continue
            spans = [(a, b) for a, b in zip(lo[hit].tolist(), hi[hit].tolist())]
            matched_owners = np.concatenate([all_owners[a:b] for a, b in spans])
            uniq, counts = np.unique(matched_owners, return_counts=True)
            for k, c in zip(uniq.tolist(), counts.tolist()):
                eid = eval_ids[k]
                total = len(eval_grams[eid])
                if total == 0:
                    continue
                short_item = store.get(eid).n_tokens < n  # type: ignore[union-attr]
                cont = c / total
                if short_item and cont < 1.0:
                    continue
                if c < min_shared and not short_item:
                    continue
                if cont >= min_cont:
                    edges[(eid, s.id)] = (cont, c)

        # groups: connected components over (eval item, training doc) edges
        ids = sorted({a for a, _ in edges} | {b for _, b in edges})
        pos = {sid: i for i, sid in enumerate(ids)}
        comps = union_find_groups(len(ids), [(pos[a], pos[b]) for a, b in edges])
        cap = FindingCap(int(ctx.config.get("output.max_findings_per_kind") or 1000))
        contaminated_by_split: Dict[str, set] = defaultdict(set)
        for (eid, _), _v in edges.items():
            contaminated_by_split[ctx.dataset.get(eid).split].add(eid)

        for comp in comps:
            member_ids = [ids[i] for i in comp]
            members = [ctx.dataset.get(m) for m in member_ids]
            splits = sorted({m.split for m in members})
            comp_set = set(member_ids)
            pairs = sorted(((a, b, c, sh) for (a, b), (c, sh) in edges.items() if a in comp_set and b in comp_set), key=lambda t: -t[2])
            best = pairs[0]
            confidence = Confidence.DETERMINISTIC if best[2] >= 0.999 else (Confidence.HIGH_CONFIDENCE if best[2] >= 0.8 else Confidence.HEURISTIC)
            violates = sev in (Severity.ERROR, Severity.WARNING)
            evidence = {
                "ngram": n,
                "min_containment": min_cont,
                "best_containment": round(best[2], 4),
                "pairs": [{"eval": a, "train": b, "containment": round(c, 4), "shared_ngrams": sh, "eval_ngrams": len(eval_grams[a])} for a, b, c, sh in pairs[:20]],
                "snippets": {sid: snippet(ctx.dataset.get(sid).metadata.get("text", "")) for sid in member_ids[:12]},
            }
            group = self.group("contamination", member_ids, splits, confidence, violates, evidence)
            res.add_group(group)
            if not cap.allow("contamination"):
                continue
            ev, tr = ctx.dataset.get(best[0]), ctx.dataset.get(best[1])
            n_eval = sum(1 for m in members if m.split in protected)
            res.add(
                self.finding(
                    kind="contamination",
                    title=f"{n_eval} evaluation item{'s' if n_eval != 1 else ''} from {'/'.join(s for s in splits if s in protected)} appear in {'/'.join(s for s in splits if s in source_splits)} (containment {best[2]:.0%})",
                    severity=sev,
                    confidence=confidence,
                    policy=ctx.policy_label("policy.contamination.cross_split"),
                    policy_description=_POLICY_DESC,
                    message=(
                        f"{best[2]:.0%} of the {n}-grams of {ev.split}:{ev.uri} \"{snippet(ev.metadata.get('text', ''), 100)}\" occur in "
                        f"{tr.split}:{tr.uri} \"{snippet(tr.metadata.get('text', ''), 100)}\"."
                    ),
                    remediation=(
                        "Remove the contaminated items from the evaluation set (or the matching documents from training) and "
                        "report the contamination rate next to the score."
                    ),
                    samples=[m.ref() for m in members],
                    evidence=evidence,
                    group_id=group.id,
                    counts=violates,
                )
            )
        res.findings.extend(cap.summaries(self, {"contamination": sev}))

        rates = {}
        for split in protected:
            total = sum(1 for s in ctx.dataset.samples if s.split == split)
            hit = len(contaminated_by_split.get(split, set()))
            rates[split] = {"items": total, "contaminated": hit, "rate": (hit / total) if total else 0.0}
            if total:
                res.add(
                    self.finding(
                        kind="contamination_rate",
                        title=f"{split}: {hit} of {total} items ({(hit / total):.1%}) appear in the training data",
                        severity=Severity.INFO if hit == 0 else (Severity.WARNING if sev is Severity.ERROR else sev),
                        confidence=Confidence.DETERMINISTIC,
                        policy=ctx.policy_label("policy.contamination.cross_split"),
                        policy_description=_POLICY_DESC,
                        message=f"Contamination rate of {split} against {', '.join(source_splits)}: {hit}/{total} items with at least {min_cont:.0%} of their {n}-grams found in a training document.",
                        remediation="Report this rate with every score on this benchmark; evaluate on the clean subset as well." if hit else "No contamination found at this n-gram size; try a smaller policy.contamination.ngram for a stricter check.",
                        evidence={"split": split, "items": total, "contaminated": hit, "rate": round((hit / total) if total else 0.0, 4), "training_splits": source_splits},
                        splits=[split],
                    )
                )
        res.stats = {"ngram": n, "min_containment": min_cont, "training_docs_scanned": scanned, "evaluation_items": len(eval_ids), "pairs": len(edges), "groups": len(comps), "contamination_rate": rates}
        return res
