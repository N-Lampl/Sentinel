"""Scan orchestration: load -> enrich -> fingerprint -> detect -> post-process -> metrics -> report."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .adapters.base import DatasetAdapter, LoadResult
from .adapters.yolo import fill_absolute_bboxes
from .baseline import apply_baseline, load_baseline
from .config import SentinelConfig
from .detectors.base import Detector, DetectorContext
from .diff import build_diff
from .enrich import enrich
from .fixplan import build_fix_plan
from .metrics import compute_metrics, evaluate_fail_conditions
from .model import Dataset, DetectorRunInfo, Finding, RelationshipGroup, Report, sorted_findings
from .postprocess import apply_allowlist, apply_split_pair_overrides, build_clusters
from .registry import adapters, detectors

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]


def detect_format(config: SentinelConfig) -> Optional[Dict[str, Any]]:
    """Ask every registered adapter whether the root looks like its format."""
    root = config.root
    for name, cls in adapters.items():
        try:
            detected = cls.detect(root)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("adapter %s detect() failed: %s", name, exc)
            continue
        if detected:
            detected.setdefault("format", name)
            return detected
    return None


def load_dataset(config: SentinelConfig) -> LoadResult:
    fmt = config.get("dataset.format")
    if not fmt or fmt == "auto":
        detected = detect_format(config)
        if not detected:
            raise ValueError(
                f"Could not detect the dataset format under {config.root}. "
                f"Pass --format ({', '.join(adapters.names())}) or set dataset.format in the config."
            )
        for key, value in detected.items():
            if config.get(f"dataset.{key}") in (None, {}, [], "auto"):
                config.set(f"dataset.{key}", value)
        fmt = detected["format"]
        log.info("auto-detected dataset format: %s", fmt)
    adapter: DatasetAdapter = adapters.get(str(fmt))(config)
    result = adapter.load()
    if not result.dataset.samples:
        raise ValueError("The dataset contains no samples; check the split configuration.")
    return result


def _dataset_summary(dataset: Dataset, config: SentinelConfig, enrich_stats: Dict[str, Any]) -> Dict[str, Any]:
    by_split = dataset.by_split()
    return {
        "name": dataset.name,
        "format": config.get("dataset.format"),
        "root": str(config.root),
        "modality": dataset.modality,
        "splits": {
            name: {"samples": len(members), "annotations": sum(len(s.annotations) for s in members)}
            for name, members in by_split.items()
        },
        "total_samples": len(dataset),
        "total_annotations": sum(len(s.annotations) for s in dataset.samples),
        "categories": {"count": len(dataset.categories), "names": dataset.category_names()[:300]},
        "group_keys": enrich_stats.get("groups", {}).get("keys", []),
        "lineage_edges": enrich_stats.get("lineage", {}).get("edges", 0),
        "timestamp_coverage": (
            enrich_stats.get("time", {}).get("with_timestamp", 0) / len(dataset) if len(dataset) else 0.0
        ),
        "source": dataset.source,
    }


def scan(config: SentinelConfig, progress: Optional[ProgressFn] = None) -> Report:
    """Run a full validation and return the :class:`Report`."""
    from . import __version__

    t0 = time.time()
    config_warnings = config.validate(extra_policy_keys=detectors.names())
    for w in config_warnings:
        log.warning("config: %s", w)

    baseline = None
    if config.get("baseline.file"):
        baseline = load_baseline(config.resolve_path(config.get("baseline.file")))  # type: ignore[arg-type]

    load = load_dataset(config)
    dataset = load.dataset
    enrich_stats = enrich(dataset, config)

    names = config.detectors_enabled(detectors.names())
    instances: List[Detector] = []
    run_infos: List[DetectorRunInfo] = []
    for name in names:
        det = detectors.get(name)(config)
        if not det.supports(dataset.modality):
            run_infos.append(DetectorRunInfo(name=name, status="skipped", message=f"does not support modality {dataset.modality}"))
            continue
        instances.append(det)

    def fp_progress(done: int, total: int) -> None:
        if progress:
            progress("fingerprint", done, total)

    ctx = DetectorContext(dataset, config, progress=fp_progress)
    fingerprint_stats: Dict[str, Any] = {}
    if any(d.needs_fingerprints for d in instances):
        store = ctx.fingerprints
        fingerprint_stats = store.stats
        for s in dataset.samples:
            fp = store.get(s.id)
            if fp is not None and fp.ok:
                s.metadata.setdefault("width", fp.width)
                s.metadata.setdefault("height", fp.height)
                fill_absolute_bboxes(s, fp.width, fp.height)

    findings: List[Finding] = list(load.findings)
    groups: List[RelationshipGroup] = []
    for i, det in enumerate(instances):
        if progress:
            progress(f"detector:{det.name}", i, len(instances))
        t = time.time()
        try:
            result = det.run(ctx)
        except Exception as exc:
            log.exception("detector %s failed", det.name)
            run_infos.append(DetectorRunInfo(name=det.name, status="error", duration_s=time.time() - t, message=f"{type(exc).__name__}: {exc}"))
            continue
        findings.extend(result.findings)
        groups.extend(result.groups)
        run_infos.append(
            DetectorRunInfo(
                name=det.name,
                status="skipped" if result.skipped else "ok",
                duration_s=time.time() - t,
                findings=len(result.findings),
                groups=len(result.groups),
                message=result.skipped or "",
                stats=result.stats,
            )
        )
    if progress:
        progress("detector:done", len(instances), len(instances))

    # ------------------------------------------------------------- post-processing
    override_stats = apply_split_pair_overrides(findings, groups, config)
    allowlist_stats = apply_allowlist(findings, groups, config)
    for w in allowlist_stats.get("warnings", []):
        log.warning("allowlist: %s", w)
    findings = sorted_findings(findings)
    baseline_stats = apply_baseline(findings, baseline)
    clusters = build_clusters(dataset, groups, config)
    metrics = compute_metrics(dataset, groups, findings)
    reasons = evaluate_fail_conditions(config.section("fail_on"), metrics, findings, run_infos, baseline)

    report = Report(
        dataset=_dataset_summary(dataset, config, enrich_stats),
        findings=findings,
        groups=groups,
        metrics=metrics,
        detectors=run_infos,
        config=config.to_dict(),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        version=__version__,
        passed=not reasons,
        failure_reasons=reasons,
        stats={
            "duration_s": round(time.time() - t0, 3),
            "load": load.stats,
            "enrich": enrich_stats,
            "fingerprints": fingerprint_stats,
            "config_warnings": config_warnings,
            "split_pair_overrides": override_stats,
            "allowlist": allowlist_stats,
            "baseline": baseline_stats,
        },
        clusters=clusters,
        dataset_ref=dataset,
    )
    report.fix_plan = build_fix_plan(report, dataset, config)
    if config.section("evaluation.predictions"):
        from .impact import compute_impact

        report.impact = compute_impact(dataset, groups, config)
    if baseline is not None:
        report.stats["diff"] = build_diff(report, baseline)
    report.stats["duration_s"] = round(time.time() - t0, 3)
    return report
