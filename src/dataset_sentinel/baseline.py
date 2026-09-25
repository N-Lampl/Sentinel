"""Baseline support: compare a run against a previous JSON report.

Finding ids are stable hashes of (detector, kind, sample ids, group, title),
so a finding that was already present in a previous run keeps its id. With a
baseline, every finding is marked ``new`` (not in the baseline) or not, and
``fail_on.new_only`` lets CI fail only on regressions while a legacy dataset
is cleaned up incrementally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .model import Finding


def load_baseline(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline report not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "findings" not in data:
        raise ValueError(f"{p} is not a Dataset Sentinel JSON report")
    ids: Set[str] = {str(f.get("id")) for f in data.get("findings", []) if isinstance(f, dict) and f.get("id")}
    titles: Dict[str, str] = {str(f.get("id")): str(f.get("title", "")) for f in data.get("findings", []) if isinstance(f, dict)}
    metrics = data.get("metrics") or {}
    return {
        "file": str(p),
        "ids": ids,
        "titles": titles,
        "violation_rate": float(metrics.get("split_integrity_violation_rate") or 0.0),
        "generated_at": data.get("generated_at"),
        "version": (data.get("tool") or {}).get("version"),
        "data": data,
    }


def apply_baseline(findings: List[Finding], baseline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Mark findings as new / known and return summary stats."""
    if baseline is None:
        return {"applied": False}
    ids = baseline["ids"]
    current = {f.id for f in findings}
    new = 0
    for f in findings:
        f.new = f.id not in ids
        new += int(f.new)
    resolved = sorted(ids - current)
    return {
        "applied": True,
        "file": baseline["file"],
        "baseline_generated_at": baseline.get("generated_at"),
        "baseline_version": baseline.get("version"),
        "baseline_findings": len(ids),
        "baseline_violation_rate": baseline["violation_rate"],
        "new": new,
        "known": len(findings) - new,
        "resolved": len(resolved),
        "resolved_titles": [baseline["titles"].get(i, i) for i in resolved[:50]],
    }
