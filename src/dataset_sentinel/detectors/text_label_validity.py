"""Structural checks for text samples: empty or very short texts, missing
labels where the split is otherwise labelled, labels outside
``policy.labels.allowed_categories``, duplicate ids, empty splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..fingerprints.text import snippet
from ..model import Confidence, Sample, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult, FindingCap

_P = "policy.labels."


@detectors.register("text_label_validity")
class TextLabelValidityDetector(Detector):
    name = "text_label_validity"
    description = "Empty / very short texts, missing or out-of-policy labels, duplicate ids"
    modalities = ("text",)

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        cfg = ctx.config
        sev_empty = cfg.severity(_P + "empty_text")
        sev_short = cfg.severity(_P + "short_text")
        min_tokens = int(cfg.get(_P + "min_tokens", 3) or 0)
        sev_missing = cfg.severity(_P + "missing_annotations")
        sev_unknown = cfg.severity(_P + "unknown_category")
        sev_empty_split = cfg.severity(_P + "empty_split")
        sev_dup = cfg.severity("policy.consistency.duplicate_registration")
        allowed = cfg.get(_P + "allowed_categories")
        allowed_set = {str(a) for a in allowed} if allowed else None
        cap = FindingCap(int(cfg.get("output.max_findings_per_kind") or 1000))
        store = ctx.text_fingerprints
        counts: Dict[str, int] = defaultdict(int)
        sev_for: Dict[str, Severity] = {}

        def emit(kind: str, sev, sample, title, message, remediation, desc, evidence, policy_key, splits=None) -> None:
            counts[kind] += 1
            if sev is None:
                return
            sev_for[kind] = sev
            if not cap.allow(kind):
                return
            res.add(self.finding(kind=kind, title=title, severity=sev, confidence=Confidence.DETERMINISTIC, policy=cfg.policy_label(policy_key),
                                 policy_description=desc, message=message, remediation=remediation, samples=[sample.ref()] if sample else [], evidence=evidence, splits=splits))

        by_split = ctx.dataset.by_split()
        for split, members in by_split.items():
            if not members:
                emit("empty_split", sev_empty_split, None, f"Split {split} contains no records", f"Split {split!r} has zero records.", "Check the split file.", "Every configured split must contain records.", {"split": split}, _P + "empty_split", splits=[split])
                continue
            labelled = sum(1 for m in members if m.annotations)
            seen_ids: Dict[str, List[Sample]] = defaultdict(list)
            unlabelled: List[Sample] = []
            for s in members:
                fp = store.get(s.id)
                n_tok = fp.n_tokens if fp else 0
                if n_tok == 0:
                    emit("empty_text", sev_empty, s, f"Empty text: {s.uri}", f"{s.split}:{s.uri} has no text after normalisation.", "Remove the record or fill in its text field.", "Every record must carry text.", {"fields": ctx.dataset.source.get("text_fields", {}).get(split)}, _P + "empty_text")
                elif n_tok < min_tokens:
                    emit("short_text", sev_short, s, f"Very short text ({n_tok} tokens): {s.uri}", f"{s.split}:{s.uri} is \"{snippet(s.metadata.get('text', ''), 80)}\" ({n_tok} tokens, minimum {min_tokens}).", "Check that the right text field is configured (dataset.text.fields).", "Records shorter than policy.labels.min_tokens are usually mis-parsed.", {"tokens": n_tok}, _P + "short_text")
                seen_ids[str(s.native_id)].append(s)
                if not s.annotations:
                    unlabelled.append(s)
                for a in s.annotations:
                    if allowed_set is not None and str(a.category) not in allowed_set:
                        emit("unknown_category", sev_unknown, s, f"Label {a.category!r} is not allowed: {s.uri}", f"{s.split}:{s.uri} has label {a.category!r}, outside policy.labels.allowed_categories.", "Re-map or remove the label.", "Only the configured labels are allowed.", {"label": a.category}, _P + "unknown_category")
            for native, dup in seen_ids.items():
                if len(dup) > 1:
                    emit("duplicate_registration", sev_dup, dup[0], f"Id {native!r} used by {len(dup)} records in {split}", f"{len(dup)} records in split {split} share the id {native!r}: " + ", ".join(d.uri for d in dup[:6]), "Make ids unique (or drop dataset.text.id_field to use record positions).", "Record ids must be unique within a split.", {"id": native, "records": [d.uri for d in dup[:20]]}, "policy.consistency.duplicate_registration", splits=[split])
            if unlabelled and labelled and len(unlabelled) < len(members):
                rate = len(unlabelled) / len(members)
                emit("missing_annotations", sev_missing, None, f"{len(unlabelled)} of {len(members)} records in {split} have no label ({rate:.0%})", f"{len(unlabelled)} records in split {split} carry no label while {labelled} do: " + ", ".join(u.uri for u in unlabelled[:8]) + (" ..." if len(unlabelled) > 8 else ""), "Fill in the missing labels or confirm these records are intentionally unlabelled.", "Records in a labelled split are expected to carry a label.", {"count": len(unlabelled), "split_total": len(members), "rate": round(rate, 4), "samples": [u.id for u in unlabelled[:100]]}, _P + "missing_annotations", splits=[split])
                if sev_missing is not None and res.findings:
                    res.findings[-1].samples = [u.ref() for u in unlabelled[:100]]
        res.findings.extend(cap.summaries(self, sev_for))
        res.stats = dict(counts)
        return res
