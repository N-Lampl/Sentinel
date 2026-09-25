"""Plain-text console summary (ANSI colours when writing to a TTY)."""

from __future__ import annotations

import os
import sys
from typing import Dict, List

from ..model import SEVERITY_ORDER, Finding, Report, Severity
from ..registry import reporters
from .base import Reporter


class _Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)

    def yellow(self, t: str) -> str:
        return self._wrap("33", t)

    def blue(self, t: str) -> str:
        return self._wrap("34", t)

    def green(self, t: str) -> str:
        return self._wrap("32", t)

    def magenta(self, t: str) -> str:
        return self._wrap("35", t)


def use_color(stream=None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


@reporters.register("console")
class ConsoleReporter(Reporter):
    name = "console"
    extension = ".txt"

    def __init__(self, config=None, color: bool | None = None, per_detector: int = 5):
        super().__init__(config)
        self.style = _Style(use_color() if color is None else color)
        self.per_detector = per_detector

    def render(self, report: Report) -> str:
        s = self.style
        m = report.metrics
        d = report.dataset
        out: List[str] = []
        out.append(s.bold(f"Dataset Sentinel {report.version}") + s.dim(f"  {report.generated_at}"))
        splits = d.get("splits", {})
        out.append(
            f"{d.get('name')}  ({d.get('format')}, {d.get('total_samples'):,} samples, "
            f"{d.get('total_annotations'):,} annotations, {len(splits)} splits: {', '.join(splits)})"
        )
        for w in report.stats.get("config_warnings") or []:
            out.append(s.yellow(f"config warning: {w}"))
        al = report.stats.get("allowlist") or {}
        for w in al.get("warnings") or []:
            out.append(s.yellow(f"allowlist warning: {w}"))
        out.append("")
        rate = f"{m.violation_rate:.2%}"
        rate_txt = s.red(s.bold(rate)) if m.violating_samples else s.green(s.bold(rate))
        strict = m.by_confidence.get("strict", {}).get("violation_rate", 0.0)
        out.append(
            f"Split integrity violation rate: {rate_txt}  "
            f"({m.violating_samples:,} of {m.evaluated_samples:,} samples in {m.violating_groups} violating groups; "
            f"strict {strict:.2%}; {len(report.clusters)} cross-split clusters)"
        )
        for split, row in m.by_split.items():
            out.append(s.dim(f"  {split:<12} {row['samples']:>8,} samples  {row['violating_samples']:>6,} violating  ({row['violation_rate']:.2%})"))
        if m.by_detector:
            out.append("")
            out.append("By detector:")
            for name, row in m.by_detector.items():
                out.append(s.dim(f"  {name:<18} {row['groups']:>5} groups  {row['violating_samples']:>6,} samples  ({row['violation_rate']:.2%})"))

        diff = report.stats.get("diff")
        if diff:
            out.append("")
            out.append(s.bold("Since baseline: ") + "; ".join(diff.get("headline", [])))
            fd = diff.get("findings", {})
            out.append(s.dim(f"  {fd.get('new', 0)} new ({fd.get('new_errors', 0)} errors, {fd.get('new_warnings', 0)} warnings), {fd.get('resolved', 0)} resolved, {fd.get('known', 0)} known"))

        if report.clusters:
            out.append("")
            out.append(s.bold(f"Cross-split clusters: {len(report.clusters)}"))
            for c in report.clusters[:3]:
                splits_txt = ", ".join(f"{k} ({v})" for k, v in c["splits"].items())
                out.append(f"  #{c['id']}  {c['size']} samples: {splits_txt}")
                for line in c["evidence"][:4]:
                    out.append(s.dim(f"      - {line}"))
                out.append(s.dim(f"      -> {c['recommendation']}"))
            if len(report.clusters) > 3:
                out.append(s.dim(f"  ... {len(report.clusters) - 3} more clusters in the HTML/JSON report"))

        plan = report.fix_plan
        if plan and plan["summary"]["samples_to_remove"]:
            ps = plan["summary"]
            by_split = ", ".join(f"{k}: {v}" for k, v in ps["by_split"].items())
            path = self.config.get("output.fix_plan")
            out.append("")
            out.append(
                s.bold("Suggested fix: ")
                + f"remove (or move) {ps['samples_to_remove']} samples ({by_split})"
                + (f"; plan written to {path}" if path else "; use --fix-plan plan.json to export the actions")
            )

        out.append("")
        sc = m.severity_counts
        suppressed = sum(1 for f in report.findings if f.suppressed)
        out.append(
            "Findings: "
            + s.red(f"{sc.get('error', 0)} errors") + ", "
            + s.yellow(f"{sc.get('warning', 0)} warnings") + ", "
            + s.blue(f"{sc.get('info', 0)} info")
            + (s.dim(f" ({suppressed} suppressed by allowlist)") if suppressed else "")
        )
        out.extend(self._grouped_findings(report.findings))
        not_ok = [x for x in report.detectors if x.status != "ok"]
        if not_ok:
            out.append("")
            out.append("Detectors skipped or failed:")
            for x in not_ok:
                line = f"  {x.name}: {x.status} {x.message}"
                out.append(s.red(line) if x.status == "error" else s.dim(line))
        out.append("")
        if report.passed:
            out.append(s.green(s.bold("RESULT: PASS")))
        else:
            out.append(s.red(s.bold("RESULT: FAIL")) + "  " + "; ".join(report.failure_reasons))
        return "\n".join(out) + "\n"

    def _grouped_findings(self, findings: List[Finding]) -> List[str]:
        """Findings grouped by detector (worst severity first), a few per detector."""
        s = self.style
        groups: Dict[str, List[Finding]] = {}
        for f in findings:  # already sorted by severity/confidence
            groups.setdefault(f.detector, []).append(f)
        ordered = sorted(groups.items(), key=lambda kv: (SEVERITY_ORDER[kv[1][0].severity], -len(kv[1]), kv[0]))
        out: List[str] = []
        for detector, items in ordered:
            counts = {sev: sum(1 for f in items if f.severity is sev) for sev in Severity}
            summary = ", ".join(f"{n} {sev.value}" for sev, n in counts.items() if n)
            out.append(s.bold(f"  {detector}") + s.dim(f"  ({summary})"))
            # new findings first within a detector
            shown = sorted(items, key=lambda f: (not f.new, SEVERITY_ORDER[f.severity]))[: self.per_detector]
            for f in shown:
                out.append(self._finding_line(f))
            if len(items) > self.per_detector:
                out.append(s.dim(f"      ... {len(items) - self.per_detector} more {detector} findings in the JSON/HTML report"))
        return out

    def _finding_line(self, f: Finding) -> str:
        s = self.style
        tag = {Severity.ERROR: s.red("ERROR"), Severity.WARNING: s.yellow("WARN "), Severity.INFO: s.blue("INFO ")}[f.severity]
        splits = ",".join(f.splits) if f.splits else "-"
        flags = ""
        if f.new:
            flags += s.magenta(" NEW")
        if f.suppressed:
            flags += s.dim(" (suppressed)")
        return f"    [{tag}]{flags} {f.title}  {s.dim(f'[{f.confidence.value}; {splits}]')}"


__all__ = ["ConsoleReporter", "use_color"]
