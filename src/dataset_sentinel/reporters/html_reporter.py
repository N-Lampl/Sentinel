"""Self-contained HTML report (no external assets, works offline, light/dark)."""

from __future__ import annotations

import html
import json
from collections import Counter
from typing import Any, Dict, List, Optional

from ..fingerprints.image import make_thumbnail_data_uri
from ..model import Confidence, Finding, Report, Severity
from ..registry import reporters
from .base import Reporter

_LEAK_KINDS_PREFIX = ("exact_duplicate", "near_duplicate", "derivative", "group_overlap", "lineage", "temporal", "conflicting_labels")
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

_SEV_LABEL = {Severity.ERROR: ("critical", "Error", "&#9679;"), Severity.WARNING: ("warning", "Warning", "&#9650;"), Severity.INFO: ("info", "Info", "&#9679;")}
_CONF_LABEL = {
    Confidence.DETERMINISTIC: "Deterministic",
    Confidence.HIGH_CONFIDENCE: "High confidence",
    Confidence.HEURISTIC: "Heuristic",
    Confidence.INCONCLUSIVE: "Inconclusive",
}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _signed(value: float, pct: bool = False) -> str:
    if pct:
        return f"{value:+.2%}"
    return f"{value:+d}" if isinstance(value, int) else f"{value:+.2f}"


_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --seq-200: #9ec5f4; --seq-450: #2a78d6; --seq-600: #184f95;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b; --info: #2a78d6;
  --good-text: #006300; --tag-bg: #f0efec; --new: #4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --seq-200: #6da7ec; --seq-450: #3987e5; --seq-600: #86b6ef; --info: #3987e5; --good-text: #0ca30c; --tag-bg: #383835; --new: #9085e9;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --seq-200: #6da7ec; --seq-450: #3987e5; --seq-600: #86b6ef; --info: #3987e5; --good-text: #0ca30c; --tag-bg: #383835; --new: #9085e9;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0; font-weight: 650; }
h2 { font-size: 16px; margin: 32px 0 12px; font-weight: 650; }
h3 { font-size: 14px; margin: 16px 0 8px; font-weight: 600; }
a { color: var(--seq-450); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
pre { background: var(--tag-bg); border-radius: 6px; padding: 10px; overflow: auto; max-height: 320px; margin: 8px 0; white-space: pre-wrap; word-break: break-word; }
.header { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px; }
.header .meta { color: var(--ink-2); }
.status { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px; font-weight: 650; border: 1px solid var(--ring); }
.status.pass { color: var(--good-text); background: color-mix(in srgb, var(--good) 12%, transparent); }
.status.fail { color: var(--critical); background: color-mix(in srgb, var(--critical) 12%, transparent); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 20px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 16px; }
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 26px; font-weight: 650; line-height: 1.2; margin-top: 4px; }
.tile .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
.tile.hero .value { font-size: 34px; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 16px; }
.banner { border-left: 4px solid var(--warning); padding: 8px 12px; margin: 12px 0; background: var(--surface); border-radius: 6px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.bar { position: relative; height: 10px; background: var(--grid); border-radius: 4px; min-width: 90px; }
.bar > span { position: absolute; left: 0; top: 0; bottom: 0; background: var(--seq-450); border-radius: 4px; }
.bars { display: grid; grid-template-columns: minmax(110px, 180px) 1fr; gap: 4px 12px; align-items: center; }
.bars .row { display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 2px; height: 22px; }
.bars .seg { position: relative; height: 100%; }
.bars .seg > span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 0 4px 4px 0; min-width: 2px; }
.bars .seg:hover > span { filter: brightness(1.12); }
.bars .lbl { font-size: 12px; color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--ink-2); margin: 8px 0 12px; }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -2px; margin-right: 6px; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0; }
.filters input, .filters select { font: inherit; padding: 6px 8px; border: 1px solid var(--axis); border-radius: 6px; background: var(--surface); color: var(--ink); }
.filters input[type=search] { min-width: 200px; flex: 1; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--axis); background: var(--surface); cursor: pointer; font: inherit; color: var(--ink); }
.chip[aria-pressed="true"] { background: var(--ink); color: var(--surface); border-color: var(--ink); }
.finding { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; margin-bottom: 8px; }
.finding summary { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 14px; cursor: pointer; list-style: none; }
.finding summary::-webkit-details-marker { display: none; }
.finding[hidden] { display: none; }
.finding .body { padding: 0 14px 14px; border-top: 1px solid var(--grid); }
.finding .title { font-weight: 600; flex: 1 1 320px; }
.finding.suppressed .title { color: var(--muted); text-decoration: line-through; }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; letter-spacing: .02em; text-transform: uppercase; border: 1px solid var(--ring); background: var(--tag-bg); color: var(--ink-2); white-space: nowrap; }
.badge.sev-critical { color: var(--critical); border-color: var(--critical); background: transparent; }
.badge.sev-warning { color: #8a5a00; border-color: var(--warning); background: color-mix(in srgb, var(--warning) 18%, transparent); }
:root[data-theme="dark"] .badge.sev-warning { color: var(--warning); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .badge.sev-warning { color: var(--warning); } }
.badge.sev-info { color: var(--info); border-color: var(--info); background: transparent; }
.badge.new { color: var(--new); border-color: var(--new); background: transparent; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 10px 0; }
.kv dt { color: var(--ink-2); font-size: 12px; margin: 0; }
.kv dd { margin: 0; }
.samples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.sample { width: 132px; font-size: 11px; color: var(--ink-2); overflow: hidden; }
.sample img { display: block; width: 100%; height: 96px; object-fit: cover; border-radius: 6px; border: 1px solid var(--ring); background: var(--tag-bg); }
.sample .ph { display: flex; align-items: center; justify-content: center; width: 100%; height: 96px; border-radius: 6px; border: 1px dashed var(--axis); color: var(--muted); }
.sample .split { font-weight: 600; color: var(--ink); }
.sample .uri { word-break: break-all; }
.muted { color: var(--muted); }
.small { font-size: 12px; }
.up { color: var(--critical); } .down { color: var(--good-text); }
.toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
.toolbar button { font: inherit; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--axis); background: var(--surface); color: var(--ink); cursor: pointer; }
details.section > summary { cursor: pointer; font-weight: 600; }
.reasons { margin: 12px 0 0; padding-left: 18px; color: var(--critical); }
ul.plain { margin: 6px 0; padding-left: 18px; }
@media (max-width: 640px) { .bars { grid-template-columns: 1fr; } .bars .lbl { margin-top: 6px; } .tile .value { font-size: 22px; } }
"""

_JS = """
(function () {
  const findings = Array.from(document.querySelectorAll('.finding'));
  const sevChips = Array.from(document.querySelectorAll('[data-sev-chip]'));
  const conf = document.getElementById('f-conf');
  const det = document.getElementById('f-det');
  const split = document.getElementById('f-split');
  const q = document.getElementById('f-q');
  const count = document.getElementById('f-count');
  const newOnly = document.getElementById('f-new');
  const hideSupp = document.getElementById('f-supp');
  const activeSev = new Set(sevChips.map(c => c.dataset.sevChip));
  function pressed(el) { return el && el.getAttribute('aria-pressed') === 'true'; }
  function apply() {
    const text = (q.value || '').toLowerCase();
    let shown = 0;
    for (const f of findings) {
      const ok = activeSev.has(f.dataset.severity)
        && (!conf.value || f.dataset.confidence === conf.value)
        && (!det.value || f.dataset.detector === det.value)
        && (!split.value || (' ' + f.dataset.splits + ' ').includes(' ' + split.value + ' '))
        && (!pressed(newOnly) || f.dataset.new === '1')
        && (!pressed(hideSupp) || f.dataset.suppressed !== '1')
        && (!text || f.dataset.text.includes(text));
      f.hidden = !ok;
      if (ok) shown++;
    }
    count.textContent = shown + ' of ' + findings.length + ' findings';
  }
  function toggle(el) {
    el.addEventListener('click', () => { el.setAttribute('aria-pressed', pressed(el) ? 'false' : 'true'); apply(); });
  }
  for (const c of sevChips) {
    c.addEventListener('click', () => {
      const s = c.dataset.sevChip;
      if (activeSev.has(s)) { activeSev.delete(s); c.setAttribute('aria-pressed', 'false'); }
      else { activeSev.add(s); c.setAttribute('aria-pressed', 'true'); }
      apply();
    });
  }
  if (newOnly) toggle(newOnly);
  if (hideSupp) toggle(hideSupp);
  for (const el of [conf, det, split]) el.addEventListener('change', apply);
  q.addEventListener('input', apply);
  document.getElementById('expand-all').addEventListener('click', () => findings.forEach(f => { if (!f.hidden) f.open = true; }));
  document.getElementById('collapse-all').addEventListener('click', () => findings.forEach(f => f.open = false));
  document.getElementById('theme-toggle').addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.getAttribute('data-theme') === 'dark' || (!root.getAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    root.setAttribute('data-theme', dark ? 'light' : 'dark');
  });
  apply();
})();
"""


@reporters.register("html")
class HtmlReporter(Reporter):
    name = "html"
    extension = ".html"

    # ------------------------------------------------------------------ render
    def render(self, report: Report) -> str:
        m = report.metrics
        d = report.dataset
        thumbs = self._thumbnails(report)
        parts: List[str] = [
            self._header(report),
            self._warnings(report),
            self._tiles(report),
            self._impact(report),
            self._diff(report),
            self._clusters(report),
            self._fix_plan(report),
            self._splits_table(report),
            self._detector_table(report),
            self._class_distribution(report),
            self._findings(report, thumbs),
            self._runs(report),
            self._config(report),
        ]
        title = f"Dataset Sentinel report: {d.get('name')}"
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{_e(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
            + "\n".join(p for p in parts if p)
            + f"\n<p class=\"muted small\">dataset-sentinel {_e(report.version)} · generated {_e(report.generated_at)} · "
            f"violation rate {_pct(m.violation_rate)}</p>\n</main>\n<script>{_JS}</script>\n</body>\n</html>\n"
        )

    # ------------------------------------------------------------------ sections
    def _header(self, report: Report) -> str:
        d = report.dataset
        status = "pass" if report.passed else "fail"
        icon = "&#10003;" if report.passed else "&#10007;"
        label = "PASS" if report.passed else "FAIL"
        splits = d.get("splits", {})
        reasons = "".join(f"<li>{_e(r)}</li>" for r in report.failure_reasons)
        return (
            "<div class=\"header\">"
            f"<h1>Dataset Sentinel</h1>"
            f"<span class=\"status {status}\" role=\"status\">{icon} {label}</span>"
            f"<span class=\"meta\">{_e(d.get('name'))} · {_e(d.get('format'))} · {d.get('total_samples', 0):,} samples · "
            f"{d.get('total_annotations', 0):,} annotations · {len(splits)} splits ({_e(', '.join(splits))})</span>"
            "<button id=\"theme-toggle\" class=\"chip\" type=\"button\" style=\"margin-left:auto\">Toggle theme</button>"
            "</div>"
            + (f"<ul class=\"reasons\">{reasons}</ul>" if reasons else "")
        )

    def _warnings(self, report: Report) -> str:
        items: List[str] = []
        for w in report.stats.get("config_warnings") or []:
            items.append(f"Config: {_e(w)}")
        al = report.stats.get("allowlist") or {}
        for w in al.get("warnings") or []:
            items.append(f"Allowlist: {_e(w)}")
        if al.get("expired_entries"):
            items.append(f"Allowlist: entries {_e(al['expired_entries'])} have expired and were ignored")
        if al.get("unused_entries"):
            items.append(f"Allowlist: entries {_e(al['unused_entries'])} matched nothing (stale?)")
        failed = [x for x in report.detectors if x.status == "error"]
        for x in failed:
            items.append(f"Detector <code>{_e(x.name)}</code> failed: {_e(x.message)}")
        if not items:
            return ""
        return "<div class=\"banner\"><ul class=\"plain\">" + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>"

    def _tiles(self, report: Report) -> str:
        m = report.metrics
        sc = m.severity_counts
        strict = m.by_confidence.get("strict", {}).get("violation_rate", 0.0)
        suppressed = sum(1 for f in report.findings if f.suppressed)
        base = report.stats.get("baseline") or {}
        tiles = [
            ("hero", "Split integrity violation rate", _pct(m.violation_rate), f"{m.violating_samples:,} of {m.evaluated_samples:,} samples in {m.violating_groups} violating groups"),
            ("", "Strict rate", _pct(strict), "deterministic + high-confidence groups only"),
            ("", "Cross-split clusters", f"{len(report.clusters):,}", "connected sets of leaking samples"),
            ("", "Errors", f"{sc.get('error', 0):,}", "fail the run by default"),
            ("", "Warnings", f"{sc.get('warning', 0):,}", "review recommended"),
            ("", "Info", f"{sc.get('info', 0):,}", f"{suppressed} suppressed by allowlist" if suppressed else "context and inconclusive checks"),
        ]
        if base.get("applied"):
            tiles.insert(3, ("", "New since baseline", f"{base.get('new', 0):,}", f"{base.get('resolved', 0):,} resolved · {base.get('known', 0):,} known"))
        if report.impact and report.impact.get("splits"):
            split, e = next(iter(report.impact["splits"].items()))
            infl = e.get("inflation")
            if infl is not None:
                tiles.insert(1, ("", f"{e['metric']} inflation ({split})", f"{infl:+.3f}", f"{e['all']:.3f} all vs {e['clean']:.3f} clean"))
        out = ["<div class=\"tiles\">"]
        for cls, label, value, sub in tiles:
            out.append(f"<div class=\"tile {cls}\"><div class=\"label\">{_e(label)}</div><div class=\"value\">{_e(value)}</div><div class=\"sub\">{_e(sub)}</div></div>")
        out.append("</div>")
        return "".join(out)

    def _impact(self, report: Report) -> str:
        impact = report.impact
        if not impact or not impact.get("splits"):
            return ""

        def fmt(x: Any) -> str:
            return "n/a" if x is None else f"{float(x):.3f}"

        blocks = []
        for split, e in impact["splits"].items():
            metric = e["metric"]
            infl = e.get("inflation")
            cls = "up" if infl and infl > 0.005 else ""
            rows = "".join(
                f"<tr><td>{_e(label)}</td><td class=\"num\">{e['subsets'][key].get('samples', 0):,}</td><td class=\"num\">{fmt(e.get(val_key))}</td></tr>"
                for label, key, val_key in (
                    ("All evaluation samples", "all", "all"),
                    ("Clean samples only (flagged removed)", "clean", "clean"),
                    ("Flagged (leaked) samples only", "leaked", "leaked"),
                    ("Clean, strict (deterministic + high-confidence flags only)", "strict_clean", "strict_clean"),
                )
            )
            per_class = ""
            if e.get("per_class_delta"):
                per_class = (
                    "<details class=\"section\"><summary>Per-class AP change (all vs clean)</summary><table><thead><tr><th>Class</th><th class=\"num\">All</th><th class=\"num\">Clean</th><th class=\"num\">Delta</th></tr></thead><tbody>"
                    + "".join(f"<tr><td>{_e(d['class'])}</td><td class=\"num\">{d['all']:.3f}</td><td class=\"num\">{d['clean']:.3f}</td><td class=\"num {'up' if d['delta'] > 0 else ''}\">{d['delta']:+.3f}</td></tr>" for d in e["per_class_delta"])
                    + "</tbody></table></details>"
                )
            headline = (
                f"<p><b>{_e(split)}</b> ({_e(e['task'])}, {_e(metric)}): {fmt(e['all'])} on all samples, "
                f"<b class=\"{cls}\">{fmt(e['clean'])}</b> once the {e['leaked_samples']:,} flagged samples are removed"
                + (f" (inflation <b class=\"{cls}\">{infl:+.3f}</b>)" if infl is not None else "")
                + f"; the flagged samples alone score {fmt(e['leaked'])}.</p>"
            )
            blocks.append(
                headline
                + f"<table><thead><tr><th>Subset</th><th class=\"num\">Samples</th><th class=\"num\">{_e(metric)}</th></tr></thead><tbody>{rows}</tbody></table>"
                + per_class
                + f"<p class=\"small muted\">Predictions: {_e(e['predictions_source'])}"
                + (f" · {e['unmatched_predictions']} predictions did not match a sample" if e.get("unmatched_predictions") else "")
                + "</p>"
            )
        notes = "".join(f"<li>{_e(n)}</li>" for n in impact.get("notes", []))
        return (
            "<h2>Metric impact of the flagged samples</h2><div class=\"card\">"
            + "".join(blocks)
            + (f"<ul class=\"plain small muted\">{notes}</ul>" if notes else "")
            + "</div>"
        )

    def _diff(self, report: Report) -> str:
        diff = report.stats.get("diff")
        if not diff:
            return ""
        head = "".join(f"<li>{_e(h)}</li>" for h in diff.get("headline", []))
        rows = []
        for sev, d in diff["severity_counts"].items():
            cls = "up" if d["delta"] > 0 else ("down" if d["delta"] < 0 else "")
            rows.append(f"<tr><td>{_e(sev)}</td><td class=\"num\">{d['baseline']}</td><td class=\"num\">{d['current']}</td><td class=\"num {cls}\">{d['delta']:+d}</td></tr>")
        for split, d in diff["split_sizes"].items():
            rows.append(f"<tr><td>samples in {_e(split)}</td><td class=\"num\">{d['baseline']:,}</td><td class=\"num\">{d['current']:,}</td><td class=\"num\">{d['delta']:+,d}</td></tr>")
        r = diff["violation_rate"]
        cls = "up" if r["delta"] > 0 else ("down" if r["delta"] < 0 else "")
        rows.append(f"<tr><td>violation rate</td><td class=\"num\">{_pct(r['baseline'])}</td><td class=\"num\">{_pct(r['current'])}</td><td class=\"num {cls}\">{r['delta']:+.2%}</td></tr>")
        classes = diff.get("classes", {})
        cls_html = ""
        for label, key in (("Classes gone from", "disappeared"), ("Classes new in", "appeared")):
            for split, names in (classes.get(key) or {}).items():
                cls_html += f"<li>{label} <b>{_e(split)}</b>: {_e(', '.join(names))}</li>"
        new_titles = "".join(f"<li>{_e(t)}</li>" for t in diff.get("new_finding_titles", [])[:20])
        resolved_titles = "".join(f"<li>{_e(t)}</li>" for t in diff.get("resolved_finding_titles", [])[:20])
        return (
            "<h2>Changes since baseline</h2><div class=\"card\">"
            f"<p class=\"small muted\">Baseline: {_e(diff.get('baseline_file'))} ({_e(diff.get('baseline_generated_at'))})</p>"
            f"<ul class=\"plain\">{head}</ul>"
            "<table><thead><tr><th>Metric</th><th class=\"num\">Baseline</th><th class=\"num\">Current</th><th class=\"num\">Delta</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + (f"<ul class=\"plain\">{cls_html}</ul>" if cls_html else "")
            + (f"<details class=\"section\"><summary>New findings ({diff['findings']['new']})</summary><ul class=\"plain\">{new_titles}</ul></details>" if new_titles else "")
            + (f"<details class=\"section\"><summary>Resolved findings ({diff['findings']['resolved']})</summary><ul class=\"plain\">{resolved_titles}</ul></details>" if resolved_titles else "")
            + "</div>"
        )

    def _clusters(self, report: Report) -> str:
        if not report.clusters:
            return ""
        rows = []
        for c in report.clusters[:50]:
            splits = ", ".join(f"{s} ({n})" for s, n in c["splits"].items())
            evidence = "".join(f"<li>{_e(x)}</li>" for x in c["evidence"])
            rows.append(
                f"<tr><td><code>{_e(c['id'])}</code></td><td class=\"num\">{c['size']:,}</td><td>{_e(splits)}</td>"
                f"<td><ul class=\"plain\">{evidence}</ul></td><td>{_e(_CONF_LABEL[Confidence(c['confidence'])])}</td><td class=\"small\">{_e(c['recommendation'])}</td></tr>"
            )
        more = f"<p class=\"muted small\">Showing 50 of {len(report.clusters)} clusters; all are in the JSON report.</p>" if len(report.clusters) > 50 else ""
        return (
            "<h2>Cross-split clusters</h2><div class=\"card\">"
            "<p class=\"small muted\">Samples connected by any combination of duplicate, transform, entity, lineage or time evidence that spans splits. One cluster is one remediation decision.</p>"
            "<table><thead><tr><th>Cluster</th><th class=\"num\">Samples</th><th>Splits</th><th>Evidence</th><th>Confidence</th><th>Recommendation</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>" + more + "</div>"
        )

    def _fix_plan(self, report: Report) -> str:
        plan = report.fix_plan
        if not plan or not plan["summary"]["samples_to_remove"]:
            return ""
        s = plan["summary"]
        by_split = ", ".join(f"{k}: {v}" for k, v in s["by_split"].items())
        sizes = ", ".join(f"{k}: {s['current_split_sizes'][k]:,} → {v:,}" for k, v in s["resulting_split_sizes"].items())
        path = self.config.get("output.fix_plan")
        return (
            "<h2>Suggested fix plan</h2><div class=\"card\">"
            f"<p><b>{s['samples_to_remove']:,} samples</b> to remove (or move) across {s['violating_groups']} violating groups: {_e(by_split)}.</p>"
            f"<p class=\"small muted\">Strategy: {_e(s['strategy'])}. Resulting split sizes: {_e(sizes)}."
            + (f" Full plan written to <code>{_e(path)}</code>." if path else " Run with <code>--fix-plan plan.json</code> to export the per-sample actions.")
            + "</p></div>"
        )

    def _splits_table(self, report: Report) -> str:
        m = report.metrics
        d = report.dataset
        rows = []
        for split, row in m.by_split.items():
            ann = d.get("splits", {}).get(split, {}).get("annotations", 0)
            width = min(100.0, row["violation_rate"] * 100)
            rows.append(
                f"<tr><td>{_e(split)}</td><td class=\"num\">{row['samples']:,}</td><td class=\"num\">{ann:,}</td>"
                f"<td class=\"num\">{row['violating_samples']:,}</td><td class=\"num\">{_pct(row['violation_rate'])}</td>"
                f"<td><div class=\"bar\" title=\"{_e(split)}: {_pct(row['violation_rate'])} violating\"><span style=\"width:{width:.1f}%\"></span></div></td></tr>"
            )
        return (
            "<h2>Splits</h2><div class=\"card\"><table><thead><tr><th>Split</th><th class=\"num\">Samples</th><th class=\"num\">Annotations</th>"
            "<th class=\"num\">Violating</th><th class=\"num\">Rate</th><th>Violation rate</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    def _detector_table(self, report: Report) -> str:
        m = report.metrics
        counts: Dict[str, Counter] = {}
        for f in report.findings:
            counts.setdefault(f.detector, Counter())[f.severity.value] += 1
        names = sorted(set(m.by_detector) | set(counts))
        rows = []
        for name in names:
            row = m.by_detector.get(name, {})
            c = counts.get(name, Counter())
            rows.append(
                f"<tr><td><code>{_e(name)}</code></td><td class=\"num\">{row.get('groups', 0):,}</td>"
                f"<td class=\"num\">{row.get('violating_samples', 0):,}</td><td class=\"num\">{_pct(row.get('violation_rate', 0.0))}</td>"
                f"<td class=\"num\">{c.get('error', 0)}</td><td class=\"num\">{c.get('warning', 0)}</td><td class=\"num\">{c.get('info', 0)}</td></tr>"
            )
        return (
            "<h2>By detector</h2><div class=\"card\"><table><thead><tr><th>Detector</th><th class=\"num\">Violating groups</th>"
            "<th class=\"num\">Violating samples</th><th class=\"num\">Rate</th><th class=\"num\">Errors</th><th class=\"num\">Warnings</th><th class=\"num\">Info</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    def _class_distribution(self, report: Report) -> str:
        dist: Optional[Dict[str, Dict[str, int]]] = None
        for run in report.detectors:
            if run.name == "distribution" and run.stats.get("class_distribution"):
                dist = run.stats["class_distribution"]
        if not dist:
            return ""
        splits = list(dist.keys())
        totals = {s: sum(v.values()) for s, v in dist.items()}
        classes = sorted({c for v in dist.values() for c in v}, key=lambda c: -sum(dist[s].get(c, 0) for s in splits))
        top = classes[:30]
        other = classes[30:]
        colors = _SERIES
        legend = "".join(f"<span><i style=\"background:{colors[i % 8]}\"></i>{_e(s)} ({totals[s]:,})</span>" for i, s in enumerate(splits))
        rows = []
        table_rows = []
        for cls in top + (["Other"] if other else []):
            segs = []
            cells = []
            for i, s in enumerate(splits):
                n = dist[s].get(cls, 0) if cls != "Other" else sum(dist[s].get(c, 0) for c in other)
                share = (n / totals[s]) if totals[s] else 0.0
                segs.append(
                    f"<div class=\"seg\" title=\"{_e(cls)} in {_e(s)}: {n:,} ({share:.1%})\"><span style=\"width:{min(100, share * 100):.2f}%;background:{colors[i % 8]}\"></span></div>"
                )
                cells.append(f"<td class=\"num\">{n:,} <span class=\"muted\">({share:.1%})</span></td>")
            rows.append(f"<div class=\"lbl\" title=\"{_e(cls)}\">{_e(cls)}</div><div class=\"row\">{''.join(segs)}</div>")
            table_rows.append(f"<tr><td>{_e(cls)}</td>{''.join(cells)}</tr>")
        head = "".join(f"<th class=\"num\">{_e(s)}</th>" for s in splits)
        return (
            "<h2>Class distribution</h2><div class=\"card\">"
            "<p class=\"small muted\">Share of each split's annotations per class (bars are per-split proportions, so splits of different sizes are comparable).</p>"
            f"<div class=\"legend\">{legend}</div><div class=\"bars\">{''.join(rows)}</div>"
            f"<details class=\"section\" style=\"margin-top:12px\"><summary>Table view</summary><table><thead><tr><th>Class</th>{head}</tr></thead><tbody>{''.join(table_rows)}</tbody></table></details>"
            "</div>"
        )

    def _findings(self, report: Report, thumbs: Dict[str, Optional[str]]) -> str:
        limit = int(self.config.get("output.max_findings_html") or 2000)
        findings = report.findings[:limit]
        dets = sorted({f.detector for f in findings})
        splits = sorted({s for f in findings for s in f.splits})
        chips = "".join(
            f"<button type=\"button\" class=\"chip\" data-sev-chip=\"{sev.value}\" aria-pressed=\"true\">{_SEV_LABEL[sev][2]} {_SEV_LABEL[sev][1]} ({report.metrics.severity_counts.get(sev.value, 0)})</button>"
            for sev in Severity
        )
        if (report.stats.get("baseline") or {}).get("applied"):
            chips += f"<button type=\"button\" class=\"chip\" id=\"f-new\" aria-pressed=\"false\">New only ({sum(1 for f in findings if f.new)})</button>"
        if any(f.suppressed for f in findings):
            chips += f"<button type=\"button\" class=\"chip\" id=\"f-supp\" aria-pressed=\"false\">Hide suppressed ({sum(1 for f in findings if f.suppressed)})</button>"
        conf_opts = "".join(f"<option value=\"{c.value}\">{_CONF_LABEL[c]}</option>" for c in Confidence)
        det_opts = "".join(f"<option value=\"{_e(x)}\">{_e(x)}</option>" for x in dets)
        split_opts = "".join(f"<option value=\"{_e(x)}\">{_e(x)}</option>" for x in splits)
        cards = "".join(self._finding_card(f, thumbs) for f in findings)
        if not findings:
            cards = "<p class=\"muted\">No findings. Every enabled detector ran without reporting a problem.</p>"
        note = f"<p class=\"muted small\">Showing the first {limit:,} of {len(report.findings):,} findings (output.max_findings_html); the JSON report has all of them.</p>" if len(report.findings) > limit else ""
        return (
            f"<h2>Findings <span class=\"muted small\" id=\"f-count\"></span></h2>{note}"
            f"<div class=\"filters\">{chips}"
            f"<select id=\"f-conf\" aria-label=\"confidence\"><option value=\"\">All confidence levels</option>{conf_opts}</select>"
            f"<select id=\"f-det\" aria-label=\"detector\"><option value=\"\">All detectors</option>{det_opts}</select>"
            f"<select id=\"f-split\" aria-label=\"split\"><option value=\"\">All splits</option>{split_opts}</select>"
            "<input id=\"f-q\" type=\"search\" placeholder=\"Search title, message, sample path\" aria-label=\"search findings\">"
            "</div><div class=\"toolbar\"><button id=\"expand-all\" type=\"button\">Expand all</button><button id=\"collapse-all\" type=\"button\">Collapse all</button></div>"
            f"<div id=\"findings\">{cards}</div>"
        )

    def _finding_card(self, f: Finding, thumbs: Dict[str, Optional[str]]) -> str:
        sev_cls, sev_label, sev_icon = _SEV_LABEL[f.severity]
        text = " ".join([f.title, f.message, f.kind, f.detector] + [s.uri for s in f.samples[:200]]).lower()
        samples_html = ""
        if f.samples:
            items = []
            for s in f.samples[:24]:
                uri = thumbs.get(s.id)
                img = f"<img src=\"{uri}\" alt=\"\" loading=\"lazy\">" if uri else "<div class=\"ph\">no preview</div>"
                items.append(f"<div class=\"sample\">{img}<div class=\"split\">{_e(s.split)}</div><div class=\"uri\" title=\"{_e(s.uri)}\">{_e(s.uri)}</div></div>")
            more = f"<div class=\"muted small\">+{len(f.samples) - 24} more samples (see JSON report)</div>" if len(f.samples) > 24 else ""
            samples_html = f"<div class=\"samples\">{''.join(items)}</div>{more}"
        evidence = json.dumps(f.evidence, indent=1, default=str)
        if len(evidence) > 6000:
            evidence = evidence[:6000] + "\n... (truncated; full evidence in the JSON report)"
        extra_badges = ""
        if f.new:
            extra_badges += "<span class=\"badge new\">new</span>"
        if f.suppressed:
            extra_badges += "<span class=\"badge\">suppressed</span>"
        suppressed_row = ""
        if f.suppressed:
            suppressed_row = f"<dt>Suppressed</dt><dd>allowlist entry {_e(f.suppressed.get('entry'))}: {_e(f.suppressed.get('reason') or 'no reason given')}" + (f" (expires {_e(f.suppressed.get('expires'))})" if f.suppressed.get("expires") else "") + "</dd>"
        return (
            f"<details class=\"finding{' suppressed' if f.suppressed else ''}\" data-severity=\"{f.severity.value}\" data-confidence=\"{f.confidence.value}\" "
            f"data-detector=\"{_e(f.detector)}\" data-splits=\"{_e(' '.join(f.splits))}\" data-new=\"{'1' if f.new else '0'}\" data-suppressed=\"{'1' if f.suppressed else '0'}\" data-text=\"{_e(text)}\">"
            f"<summary><span class=\"badge sev-{sev_cls}\">{sev_icon} {sev_label}</span><span class=\"title\">{_e(f.title)}</span>{extra_badges}"
            f"<span class=\"badge\">{_e(f.detector)}</span><span class=\"badge\">{_CONF_LABEL[f.confidence]}</span>"
            f"<span class=\"badge\">{_e(', '.join(f.splits) or 'n/a')}</span></summary>"
            "<div class=\"body\">"
            f"<p>{_e(f.message)}</p>"
            "<dl class=\"kv\">"
            f"<dt>Policy</dt><dd><code>{_e(f.policy)}</code> <span class=\"muted\">{_e(f.policy_description)}</span></dd>"
            f"<dt>Remediation</dt><dd>{_e(f.remediation)}</dd>"
            f"{suppressed_row}"
            f"<dt>Counts toward rate</dt><dd>{'yes' if f.counts_toward_violation_rate else 'no'}"
            + (f" <span class=\"muted\">(group <code>{_e(f.group_id)}</code>)</span>" if f.group_id else "")
            + f"</dd><dt>Finding id</dt><dd><code>{_e(f.id)}</code> <span class=\"muted small\">stable across runs; use it in the allowlist</span></dd></dl>"
            f"{samples_html}"
            f"<details class=\"section\"><summary>Evidence</summary><pre>{_e(evidence)}</pre></details>"
            "</div></details>"
        )

    def _runs(self, report: Report) -> str:
        rows = []
        for run in report.detectors:
            stats = json.dumps(run.stats, default=str)
            if len(stats) > 400:
                stats = stats[:400] + "…"
            rows.append(
                f"<tr><td><code>{_e(run.name)}</code></td><td>{_e(run.status)}</td><td class=\"num\">{run.duration_s:.2f}s</td>"
                f"<td class=\"num\">{run.findings}</td><td class=\"num\">{run.groups}</td><td class=\"small\">{_e(run.message)} <span class=\"muted\">{_e(stats)}</span></td></tr>"
            )
        fp = report.stats.get("fingerprints") or {}
        extra = ""
        if fp:
            extra = f"<p class=\"small muted\">Fingerprinted {fp.get('images', 0):,} images ({fp.get('cache_hits', 0):,} from cache, {fp.get('failed', 0):,} failed) in {fp.get('seconds', 0)}s. Total scan {report.stats.get('duration_s', 0)}s.</p>"
        return (
            "<h2>Detector runs</h2><div class=\"card\">" + extra + "<table><thead><tr><th>Detector</th><th>Status</th><th class=\"num\">Time</th>"
            "<th class=\"num\">Findings</th><th class=\"num\">Groups</th><th>Notes / stats</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    def _config(self, report: Report) -> str:
        cfg = json.dumps({k: v for k, v in report.config.items() if k in ("dataset", "policy", "detectors", "fail_on", "allowlist", "baseline")}, indent=1, default=str)
        ds = json.dumps(report.dataset, indent=1, default=str)
        return (
            "<details class=\"section\" style=\"margin-top:24px\"><summary>Dataset summary and effective configuration</summary>"
            f"<h3>Dataset</h3><pre>{_e(ds)}</pre><h3>Policy</h3><pre>{_e(cfg)}</pre></details>"
        )

    # ------------------------------------------------------------------ thumbnails
    def _thumbnails(self, report: Report) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {}
        if not self.config.get("output.thumbnails", True):
            return out
        dataset = report.dataset_ref
        if dataset is None:
            return out
        max_findings = int(self.config.get("output.max_thumbnails") or 0)
        size = int(self.config.get("output.thumbnail_size") or 96)
        budget = max_findings
        for f in report.findings:
            if budget <= 0:
                break
            if not (f.counts_toward_violation_rate or f.kind.startswith(_LEAK_KINDS_PREFIX)):
                continue
            budget -= 1
            for ref in f.samples[:8]:
                if ref.id in out:
                    continue
                s = dataset.get_or_none(ref.id)
                out[ref.id] = make_thumbnail_data_uri(s.path, size) if s is not None and s.path is not None else None
        return out


__all__ = ["HtmlReporter"]
