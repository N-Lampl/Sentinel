"""Split integrity violation rate, supporting breakdowns and fail conditions.

    violation_rate = |samples in at least one policy-violating relationship group|
                     / |evaluated samples|

A relationship group violates policy when it crosses split boundaries and
the policy for that relationship kind is ``error`` or ``warning``
(``info``/``allow`` groups are reported but do not count).

``strict_violation_rate`` restricts the numerator to groups whose
confidence is ``deterministic`` or ``high_confidence``; it is the number to
gate CI on when heuristic detectors are noisy for a particular dataset.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .model import (
    SEVERITY_ORDER,
    Confidence,
    Dataset,
    DetectorRunInfo,
    Finding,
    Metrics,
    RelationshipGroup,
    Severity,
)

STRICT = {Confidence.DETERMINISTIC, Confidence.HIGH_CONFIDENCE}


def compute_metrics(dataset: Dataset, groups: Iterable[RelationshipGroup], findings: Iterable[Finding]) -> Metrics:
    total = len(dataset)
    violating: Set[str] = set()
    strict_violating: Set[str] = set()
    by_detector: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"samples": set(), "groups": 0, "strict_samples": set()})
    by_conf: Dict[str, Set[str]] = defaultdict(set)
    n_groups = 0
    for g in groups:
        if not g.violates_policy:
            continue
        n_groups += 1
        members = set(g.member_ids)
        violating |= members
        by_conf[g.confidence.value] |= members
        entry = by_detector[g.detector]
        entry["samples"] |= members
        entry["groups"] = int(entry["groups"]) + 1
        if g.confidence in STRICT:
            strict_violating |= members
            entry["strict_samples"] |= members

    split_of = {s.id: s.split for s in dataset.samples}
    by_split: Dict[str, Dict[str, Any]] = {}
    sizes = dataset.split_sizes()
    counts: Dict[str, int] = defaultdict(int)
    strict_counts: Dict[str, int] = defaultdict(int)
    for sid in violating:
        counts[split_of.get(sid, "?")] += 1
    for sid in strict_violating:
        strict_counts[split_of.get(sid, "?")] += 1
    for split, n in sizes.items():
        by_split[split] = {
            "samples": n,
            "violating_samples": counts.get(split, 0),
            "violation_rate": (counts.get(split, 0) / n) if n else 0.0,
            "strict_violation_rate": (strict_counts.get(split, 0) / n) if n else 0.0,
        }

    severity_counts = {s.value: 0 for s in Severity}
    for f in findings:
        severity_counts[f.severity.value] += 1

    metrics = Metrics(
        evaluated_samples=total,
        violating_samples=len(violating),
        violation_rate=(len(violating) / total) if total else 0.0,
        by_split=by_split,
        by_detector={
            name: {
                "groups": int(v["groups"]),
                "violating_samples": len(v["samples"]),
                "violation_rate": (len(v["samples"]) / total) if total else 0.0,
                "strict_violating_samples": len(v["strict_samples"]),
            }
            for name, v in sorted(by_detector.items())
        },
        by_confidence={
            conf.value: {
                "violating_samples": len(by_conf.get(conf.value, set())),
                "violation_rate": (len(by_conf.get(conf.value, set())) / total) if total else 0.0,
            }
            for conf in Confidence
        },
        severity_counts=severity_counts,
        violating_groups=n_groups,
    )
    metrics.by_confidence["strict"] = {
        "violating_samples": len(strict_violating),
        "violation_rate": (len(strict_violating) / total) if total else 0.0,
        "includes": [c.value for c in STRICT],
    }
    return metrics


def evaluate_fail_conditions(
    config_fail_on: Dict[str, object],
    metrics: Metrics,
    findings: List[Finding],
    detectors: Sequence[DetectorRunInfo] = (),
    baseline: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return the list of reasons the run should fail (empty = pass).

    ``baseline`` is the dict produced by :func:`dataset_sentinel.baseline.load_baseline`
    (or ``None``). With ``new_only`` set and a baseline present, only findings
    marked ``new`` count for the severity condition, and the violation rate
    only fails when it exceeds the baseline's rate (no regression).
    """
    reasons: List[str] = []
    new_only = bool(config_fail_on.get("new_only", False)) and baseline is not None
    considered = [f for f in findings if f.new] if new_only else list(findings)

    sev_raw = str(config_fail_on.get("severity", "error") or "none").lower()
    if sev_raw not in {"none", "never", "off"}:
        threshold = Severity.parse(sev_raw)
        if threshold is not None:
            limit = SEVERITY_ORDER[threshold]
            hits = [f for f in considered if SEVERITY_ORDER[f.severity] <= limit]
            if hits:
                what = "new finding(s)" if new_only else "finding(s)"
                reasons.append(f"{len(hits)} {what} at severity {threshold.value} or above (fail_on.severity={threshold.value})")

    max_rate = config_fail_on.get("max_violation_rate")
    if max_rate is not None:
        try:
            limit_rate = float(max_rate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            limit_rate = 0.0
        if new_only:
            base_rate = float(baseline.get("violation_rate") or 0.0) if baseline else 0.0
            effective = max(limit_rate, base_rate)
            if metrics.violation_rate > effective + 1e-12:
                reasons.append(
                    f"split integrity violation rate {metrics.violation_rate:.2%} exceeds the baseline rate "
                    f"{base_rate:.2%} (fail_on.new_only=true, max_violation_rate={limit_rate:.2%})"
                )
        elif metrics.violation_rate > limit_rate:
            reasons.append(
                f"split integrity violation rate {metrics.violation_rate:.2%} exceeds fail_on.max_violation_rate={limit_rate:.2%}"
            )

    if config_fail_on.get("detector_errors", True):
        failed = [d for d in detectors if d.status == "error"]
        if failed:
            names = ", ".join(f"{d.name} ({d.message})" for d in failed)
            reasons.append(f"{len(failed)} detector(s) failed: {names} (fail_on.detector_errors=true)")
    return reasons
