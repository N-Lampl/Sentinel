"""Changes between the current report and a baseline JSON report.

Answers the questions a reviewer asks when a dataset version changes: did a
new cross-split leak appear, did invalid labels increase, did a class
disappear from an evaluation split, did split sizes change, did the
violation rate move, did a new group start spanning splits?
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .model import Report, Severity


def _delta(current: float, previous: float) -> Dict[str, Any]:
    return {"baseline": previous, "current": current, "delta": current - previous}


def build_diff(report: Report, baseline: Dict[str, Any]) -> Dict[str, Any]:
    data = baseline.get("data") or {}
    b_metrics = data.get("metrics") or {}
    b_dataset = data.get("dataset") or {}
    b_findings: List[Dict[str, Any]] = [f for f in data.get("findings", []) if isinstance(f, dict)]
    b_ids = {str(f.get("id")) for f in b_findings}

    new = [f for f in report.findings if f.new]
    resolved_ids = b_ids - {f.id for f in report.findings}
    resolved = [f for f in b_findings if str(f.get("id")) in resolved_ids]

    # severity counts
    b_sev = b_metrics.get("severity_counts") or {}
    sev = {
        s.value: _delta(report.metrics.severity_counts.get(s.value, 0), int(b_sev.get(s.value, 0) or 0))
        for s in Severity
    }

    # split sizes
    b_splits = b_dataset.get("splits") or {}
    cur_splits = report.dataset.get("splits") or {}
    split_sizes = {}
    for name in sorted(set(b_splits) | set(cur_splits)):
        split_sizes[name] = _delta(int((cur_splits.get(name) or {}).get("samples", 0)), int((b_splits.get(name) or {}).get("samples", 0)))

    # findings per detector / kind
    b_by_det: Counter = Counter(str(f.get("detector")) for f in b_findings)
    c_by_det: Counter = Counter(f.detector for f in report.findings)
    by_detector = {d: _delta(c_by_det.get(d, 0), b_by_det.get(d, 0)) for d in sorted(set(b_by_det) | set(c_by_det))}
    b_by_kind: Counter = Counter(str(f.get("kind")) for f in b_findings)
    c_by_kind: Counter = Counter(f.kind for f in report.findings)
    by_kind = {k: _delta(c_by_kind.get(k, 0), b_by_kind.get(k, 0)) for k in sorted(set(b_by_kind) | set(c_by_kind)) if c_by_kind.get(k, 0) != b_by_kind.get(k, 0)}

    # classes per split (from the distribution detector stats)
    def class_dist(detector_runs: List[Any]) -> Dict[str, Dict[str, int]]:
        for run in detector_runs:
            stats = run.get("stats") if isinstance(run, dict) else getattr(run, "stats", None)
            name = run.get("name") if isinstance(run, dict) else getattr(run, "name", None)
            if name == "distribution" and stats and stats.get("class_distribution"):
                return stats["class_distribution"]
        return {}

    b_classes = class_dist(data.get("detectors", []))
    c_classes = class_dist(report.detectors)  # type: ignore[arg-type]
    disappeared: Dict[str, List[str]] = {}
    appeared: Dict[str, List[str]] = {}
    for split in sorted(set(b_classes) | set(c_classes)):
        before = {c for c, n in (b_classes.get(split) or {}).items() if n}
        after = {c for c, n in (c_classes.get(split) or {}).items() if n}
        if before - after:
            disappeared[split] = sorted(before - after)
        if after - before:
            appeared[split] = sorted(after - before)

    new_leak_findings = [f for f in new if f.counts_toward_violation_rate]
    rate = _delta(report.metrics.violation_rate, float(b_metrics.get("split_integrity_violation_rate") or 0.0))
    strict_prev = float(((b_metrics.get("by_confidence") or {}).get("strict") or {}).get("violation_rate") or 0.0)
    strict = _delta(report.metrics.by_confidence.get("strict", {}).get("violation_rate", 0.0), strict_prev)

    headline: List[str] = []
    if new_leak_findings:
        headline.append(f"{len(new_leak_findings)} new cross-split relationship group(s)")
    if rate["delta"] > 1e-12:
        headline.append(f"violation rate up {rate['delta']:+.2%} to {rate['current']:.2%}")
    elif rate["delta"] < -1e-12:
        headline.append(f"violation rate down {rate['delta']:+.2%} to {rate['current']:.2%}")
    if sev["error"]["delta"]:
        headline.append(f"errors {sev['error']['delta']:+d}")
    if sev["warning"]["delta"]:
        headline.append(f"warnings {sev['warning']['delta']:+d}")
    for split, classes in disappeared.items():
        headline.append(f"classes gone from {split}: {', '.join(classes[:5])}")
    for split, d in split_sizes.items():
        if d["delta"]:
            headline.append(f"{split} {d['delta']:+d} samples")
    if not headline:
        headline.append("no material change since the baseline")

    return {
        "baseline_file": baseline.get("file"),
        "baseline_generated_at": baseline.get("generated_at"),
        "headline": headline,
        "violation_rate": rate,
        "strict_violation_rate": strict,
        "severity_counts": sev,
        "split_sizes": split_sizes,
        "findings": {
            "new": len(new),
            "resolved": len(resolved),
            "known": len(report.findings) - len(new),
            "new_errors": sum(1 for f in new if f.severity is Severity.ERROR),
            "new_warnings": sum(1 for f in new if f.severity is Severity.WARNING),
            "new_cross_split_groups": len(new_leak_findings),
        },
        "by_detector": by_detector,
        "by_kind": by_kind,
        "classes": {"disappeared": disappeared, "appeared": appeared},
        "new_finding_titles": [f.title for f in new[:50]],
        "resolved_finding_titles": [str(f.get("title", "")) for f in resolved[:50]],
    }
