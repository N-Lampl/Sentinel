"""Flat metrics file for DVC (``dvc metrics show`` / ``dvc metrics diff``).

Written as YAML (default) or JSON depending on the file extension.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import yaml

from ..model import Report, Severity
from ..registry import reporters
from .base import Reporter


def dvc_metrics(report: Report) -> Dict[str, Any]:
    m = report.metrics
    by_det = m.by_detector
    label_errors = sum(
        1 for f in report.findings if f.severity is Severity.ERROR and (f.detector == "label_validity" or f.detector.startswith("adapter:"))
    )
    integrity: Dict[str, Any] = {
        "passed": bool(report.passed),
        "split_integrity_violation_rate": round(m.violation_rate, 6),
        "strict_violation_rate": round(m.by_confidence.get("strict", {}).get("violation_rate", 0.0), 6),
        "violating_samples": m.violating_samples,
        "violating_groups": m.violating_groups,
        "cross_split_clusters": len(report.clusters),
        "exact_duplicate_groups": by_det.get("exact_duplicate", {}).get("groups", 0),
        "near_duplicate_groups": by_det.get("near_duplicate", {}).get("groups", 0),
        "derivative_groups": by_det.get("derivative", {}).get("groups", 0),
        "group_overlap_groups": by_det.get("group_overlap", {}).get("groups", 0),
        "lineage_groups": by_det.get("lineage", {}).get("groups", 0),
        "temporal_groups": by_det.get("temporal", {}).get("groups", 0),
        "invalid_annotation_findings": label_errors,
        "errors": m.severity_counts.get("error", 0),
        "warnings": m.severity_counts.get("warning", 0),
        "info": m.severity_counts.get("info", 0),
        "total_samples": m.evaluated_samples,
        "total_annotations": report.dataset.get("total_annotations", 0),
    }
    for split, row in m.by_split.items():
        integrity[f"samples_{split}"] = row["samples"]
        integrity[f"violation_rate_{split}"] = round(row["violation_rate"], 6)
    return {"integrity": integrity}


@reporters.register("dvc")
class DvcReporter(Reporter):
    name = "dvc"
    extension = ".yaml"

    def render(self, report: Report, fmt: str = "yaml") -> str:
        data = dvc_metrics(report)
        if fmt == "json":
            return json.dumps(data, indent=2) + "\n"
        return yaml.safe_dump(data, sort_keys=False)

    def write(self, report: Report, path):  # type: ignore[override]
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(report, "json" if p.suffix.lower() == ".json" else "yaml"), encoding="utf-8")
        return p
