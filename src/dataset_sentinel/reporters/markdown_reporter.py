"""Markdown summary, sized for a GitHub Actions job summary or a PR comment."""

from __future__ import annotations

from ..model import Report, Severity
from ..registry import reporters
from .base import Reporter

_ICON = {Severity.ERROR: "🔴", Severity.WARNING: "🟠", Severity.INFO: "🔵"}


@reporters.register("markdown")
class MarkdownReporter(Reporter):
    name = "markdown"
    extension = ".md"

    def render(self, report: Report, max_findings: int = 40) -> str:
        m = report.metrics
        d = report.dataset
        status = "✅ PASS" if report.passed else "❌ FAIL"
        base = report.stats.get("baseline") or {}
        lines = [
            f"## Dataset Sentinel: {status}",
            "",
            f"**{d.get('name')}** ({d.get('format')}, {d.get('total_samples'):,} samples, {len(d.get('splits', {}))} splits)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Split integrity violation rate | **{m.violation_rate:.2%}** ({m.violating_samples:,} of {m.evaluated_samples:,} samples, {m.violating_groups} groups, {len(report.clusters)} clusters) |",
            f"| Strict rate (deterministic + high confidence) | {m.by_confidence.get('strict', {}).get('violation_rate', 0):.2%} |",
            f"| Errors / warnings / info | {m.severity_counts.get('error', 0)} / {m.severity_counts.get('warning', 0)} / {m.severity_counts.get('info', 0)} |",
        ]
        if base.get("applied"):
            lines.append(f"| New / resolved / known findings | {base.get('new', 0)} / {base.get('resolved', 0)} / {base.get('known', 0)} |")
        suppressed = sum(1 for f in report.findings if f.suppressed)
        if suppressed:
            lines.append(f"| Suppressed by allowlist | {suppressed} |")
        if report.failure_reasons:
            lines += ["", "**Failure reasons**", ""] + [f"- {r}" for r in report.failure_reasons]
        for w in report.stats.get("config_warnings") or []:
            lines += ["", f"> ⚠️ config: {w}"]
        diff = report.stats.get("diff")
        if diff:
            lines += ["", "**Since baseline:** " + "; ".join(diff.get("headline", []))]
        if m.by_split:
            lines += ["", "| Split | Samples | Violating | Rate |", "|---|---:|---:|---:|"]
            for split, row in m.by_split.items():
                lines.append(f"| {split} | {row['samples']:,} | {row['violating_samples']:,} | {row['violation_rate']:.2%} |")
        if report.clusters:
            lines += ["", f"### Cross-split clusters (top {min(5, len(report.clusters))} of {len(report.clusters)})", ""]
            for c in report.clusters[:5]:
                splits = ", ".join(f"{k} ({v})" for k, v in c["splits"].items())
                lines.append(f"- **{c['size']} samples** across {splits}: {'; '.join(c['evidence'][:3])}. _{c['recommendation']}_")
        plan = report.fix_plan
        if plan and plan["summary"]["samples_to_remove"]:
            ps = plan["summary"]
            lines += ["", f"**Suggested fix:** remove or move {ps['samples_to_remove']} samples (" + ", ".join(f"{k}: {v}" for k, v in ps["by_split"].items()) + ")."]
        if m.by_detector:
            lines += ["", "| Detector | Violating groups | Violating samples | Rate |", "|---|---:|---:|---:|"]
            for name, row in m.by_detector.items():
                lines.append(f"| {name} | {row['groups']} | {row['violating_samples']:,} | {row['violation_rate']:.2%} |")
        if report.findings:
            shown = sorted(report.findings, key=lambda f: (not f.new, 0))[:max_findings] if base.get("applied") else report.findings[:max_findings]
            lines += ["", f"### Findings (top {len(shown)} of {len(report.findings)})", ""]
            lines += ["| | Detector | Finding | Confidence | Splits |", "|---|---|---|---|---|"]
            for f in shown:
                title = f.title.replace("|", "\\|")
                flag = " 🆕" if f.new else ""
                flag += " (suppressed)" if f.suppressed else ""
                lines.append(f"| {_ICON[f.severity]} | `{f.detector}` | {title}{flag} | {f.confidence.value} | {', '.join(f.splits)} |")
        skipped = [x for x in report.detectors if x.status != "ok"]
        if skipped:
            lines += ["", "<details><summary>Detectors not run or skipped</summary>", ""]
            lines += [f"- `{x.name}`: {x.status} {('(' + x.message + ')') if x.message else ''}" for x in skipped]
            lines += ["", "</details>"]
        lines += ["", f"<sub>dataset-sentinel {report.version} · {report.generated_at}</sub>", ""]
        return "\n".join(lines)
