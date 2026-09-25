"""Time leakage.

When splits are meant to be chronological (train before val before test),
any evaluation sample captured before the last training sample leaks future
information into training. The check uses ``sample.metadata["_timestamp"]``
(filled during enrichment from keys such as ``captured_at`` or COCO's
``date_captured``).

``policy.temporal.enabled``:

* ``auto`` (default): overlapping time ranges are reported as an
  *informational, inconclusive* finding because a random split legitimately
  overlaps in time. Nothing counts toward the violation rate.
* ``true``: splits must be chronological; overlaps are violations.
* ``false``: skipped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..config import split_role
from ..model import Confidence, Sample, Severity
from ..registry import detectors
from .base import Detector, DetectorContext, DetectorResult

_POLICY_DESC = (
    "For chronological splits every sample of a later split must be captured after every "
    "sample of an earlier split; overlapping time ranges leak future information."
)


def _fmt(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return str(ts)


@detectors.register("temporal")
class TemporalDetector(Detector):
    name = "temporal"
    description = "Overlapping capture-time ranges between splits that should be chronological"
    modalities = ("*",)

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        enabled = ctx.config.get("policy.temporal.enabled", "auto")
        if enabled in (False, "false", "off", "no", 0):
            res.skipped = "disabled by policy"
            return res
        strict = enabled in (True, "true", "on", "yes", 1, "strict")
        sev = ctx.severity("policy.temporal.cross_split", Severity.ERROR) or Severity.INFO
        min_cov = float(ctx.config.get("policy.temporal.min_coverage", 0.5) or 0)

        samples = ctx.dataset.samples
        with_ts = [s for s in samples if isinstance(s.metadata.get("_timestamp"), (int, float))]
        coverage = len(with_ts) / len(samples) if samples else 0.0
        res.stats = {"coverage": round(coverage, 4), "with_timestamp": len(with_ts), "strict": strict}
        if not with_ts:
            if strict:
                res.add(
                    self.finding(
                        kind="temporal_not_evaluated",
                        title="Temporal policy enabled but no timestamps were found",
                        severity=Severity.WARNING,
                        confidence=Confidence.INCONCLUSIVE,
                        policy=ctx.policy_label("policy.temporal.enabled"),
                        policy_description=_POLICY_DESC,
                        message="policy.temporal.enabled is true but no sample carries a parsable timestamp.",
                        remediation="Provide timestamps via metadata (dataset.time.keys) or a sidecar file.",
                    )
                )
            else:
                res.skipped = "no timestamps available"
            return res

        ordered = [s for s in ctx.ordered_splits() if split_role(s) in set(ctx.config.split_order)]
        if len(ordered) < 2:
            res.skipped = "fewer than two splits with a known role (train/val/test)"
            return res
        by_split: Dict[str, List[Sample]] = {name: [] for name in ordered}
        for s in with_ts:
            if s.split in by_split:
                by_split[s.split].append(s)
        ranges = {}
        for name, members in by_split.items():
            if members:
                ts = [float(m.metadata["_timestamp"]) for m in members]
                ranges[name] = {"min": min(ts), "max": max(ts), "count": len(ts)}
        res.stats["ranges"] = {k: {"start": _fmt(v["min"]), "end": _fmt(v["max"]), "count": v["count"]} for k, v in ranges.items()}

        low_coverage = coverage < min_cov
        for i, earlier in enumerate(ordered):
            for later in ordered[i + 1 :]:
                if earlier not in ranges or later not in ranges:
                    continue
                e_max = ranges[earlier]["max"]
                l_min = ranges[later]["min"]
                late_in_earlier = [s for s in by_split[earlier] if float(s.metadata["_timestamp"]) >= l_min]
                early_in_later = [s for s in by_split[later] if float(s.metadata["_timestamp"]) <= e_max]
                if not early_in_later:
                    continue
                members = early_in_later + late_in_earlier
                member_ids = [m.id for m in members]
                evidence = {
                    "earlier_split": earlier,
                    "later_split": later,
                    "earlier_range": [_fmt(ranges[earlier]["min"]), _fmt(e_max)],
                    "later_range": [_fmt(l_min), _fmt(ranges[later]["max"])],
                    "overlap_window": [_fmt(l_min), _fmt(e_max)],
                    "later_samples_before_earlier_end": len(early_in_later),
                    "earlier_samples_after_later_start": len(late_in_earlier),
                    "timestamp_coverage": round(coverage, 4),
                }
                if strict and not low_coverage:
                    confidence = Confidence.DETERMINISTIC
                    severity = sev
                    counts = severity in (Severity.ERROR, Severity.WARNING)
                elif strict and low_coverage:
                    confidence = Confidence.HEURISTIC
                    severity = sev
                    counts = severity in (Severity.ERROR, Severity.WARNING)
                else:
                    confidence = Confidence.INCONCLUSIVE
                    severity = Severity.INFO
                    counts = False
                group = self.group("temporal_overlap", member_ids, [earlier, later], confidence, counts, evidence)
                res.add_group(group)
                title = f"Time ranges of {earlier} and {later} overlap ({len(early_in_later)} {later} samples precede the last {earlier} sample)"
                message = (
                    f"{earlier} spans {evidence['earlier_range'][0]} to {evidence['earlier_range'][1]}; "
                    f"{later} spans {evidence['later_range'][0]} to {evidence['later_range'][1]}. "
                    f"{len(early_in_later)} {later} samples were captured before the last {earlier} sample and "
                    f"{len(late_in_earlier)} {earlier} samples after the first {later} sample."
                )
                if not strict:
                    message += (
                        " This is only a problem if the splits are meant to be chronological; set "
                        "policy.temporal.enabled: true to enforce it."
                    )
                elif low_coverage:
                    message += f" Only {coverage:.0%} of samples have timestamps (policy.temporal.min_coverage={min_cov})."
                res.add(
                    self.finding(
                        kind="temporal_overlap" if strict else "temporal_overlap_info",
                        title=title,
                        severity=severity,
                        confidence=confidence,
                        policy=ctx.policy_label("policy.temporal.enabled") if not strict else ctx.policy_label("policy.temporal.cross_split"),
                        policy_description=_POLICY_DESC,
                        message=message,
                        remediation=(
                            f"Re-split chronologically: choose a cut-off so that all {later} samples are captured after "
                            f"all {earlier} samples, or drop the overlapping window."
                        ),
                        samples=[m.ref() for m in members[:200]],
                        evidence=evidence,
                        group_id=group.id,
                        counts=counts,
                        splits=[earlier, later],
                    )
                )
        return res


__all__ = ["TemporalDetector"]
_ = Optional
