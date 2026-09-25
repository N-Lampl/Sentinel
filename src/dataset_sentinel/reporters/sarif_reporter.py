"""SARIF 2.1.0 export so findings can be uploaded to GitHub code scanning.

Mapping: error -> ``error``, warning -> ``warning``, info -> ``note``.
Every result carries a stable ``partialFingerprints`` entry (the finding id)
so repeated uploads do not create duplicate alerts, and one location per
involved sample (relative to the dataset root) so alerts can be tied to
files when the dataset lives in the repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..model import Finding, Report, Severity
from ..registry import reporters
from .base import Reporter

_LEVEL = {Severity.ERROR: "error", Severity.WARNING: "warning", Severity.INFO: "note"}


def _uri(uri: str) -> str:
    return Path(uri).as_posix()


@reporters.register("sarif")
class SarifReporter(Reporter):
    name = "sarif"
    extension = ".sarif"

    def render(self, report: Report, max_locations: int = 10) -> str:
        rules: Dict[str, Dict[str, Any]] = {}
        results: List[Dict[str, Any]] = []
        root = report.dataset.get("root")
        for f in report.findings:
            rule = rules.setdefault(
                f.kind,
                {
                    "id": f.kind,
                    "name": f.kind.replace("_", " ").title().replace(" ", ""),
                    "shortDescription": {"text": f.kind.replace("_", " ")},
                    "fullDescription": {"text": f.policy_description},
                    "help": {"text": f.remediation},
                    "defaultConfiguration": {"level": _LEVEL[f.severity]},
                    "properties": {"detector": f.detector},
                },
            )
            _ = rule
            results.append(self._result(f, max_locations))
        run = {
            "tool": {
                "driver": {
                    "name": "dataset-sentinel",
                    "version": report.version,
                    "informationUri": "https://github.com/dataset-sentinel/dataset-sentinel",
                    "rules": list(rules.values()),
                }
            },
            "originalUriBaseIds": {"DATASET": {"uri": Path(root).as_uri() + "/" if root else "file:///"}},
            "results": results,
            "properties": {
                "dataset": report.dataset.get("name"),
                "passed": report.passed,
                "split_integrity_violation_rate": report.metrics.violation_rate,
                "strict_violation_rate": report.metrics.by_confidence.get("strict", {}).get("violation_rate", 0.0),
            },
        }
        doc = {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [run],
        }
        return json.dumps(doc, indent=2) + "\n"

    @staticmethod
    def _result(f: Finding, max_locations: int) -> Dict[str, Any]:
        locations = [
            {"physicalLocation": {"artifactLocation": {"uri": _uri(s.uri), "uriBaseId": "DATASET"}}, "logicalLocations": [{"name": s.split, "kind": "split"}]}
            for s in f.samples[:max_locations]
        ]
        result: Dict[str, Any] = {
            "ruleId": f.kind,
            "level": _LEVEL[f.severity],
            "message": {"text": f"{f.title}. {f.message} Remediation: {f.remediation}"},
            "partialFingerprints": {"datasetSentinel/findingId/v1": f.id},
            "properties": {
                "detector": f.detector,
                "confidence": f.confidence.value,
                "policy": f.policy,
                "splits": list(f.splits),
                "group_id": f.group_id,
                "counts_toward_violation_rate": f.counts_toward_violation_rate,
                "sample_count": len(f.samples),
                "new": f.new,
            },
        }
        if locations:
            result["locations"] = locations
        if f.suppressed:
            result["suppressions"] = [{"kind": "external", "justification": str(f.suppressed.get("reason", ""))}]
        return result
