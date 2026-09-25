"""Detector interface and shared context.

A detector receives a :class:`DetectorContext` (dataset + config + lazily
computed fingerprints) and returns a :class:`DetectorResult` with findings,
relationship groups and stats. Detectors never mutate the dataset.

Writing a detector::

    from dataset_sentinel.detectors.base import Detector, DetectorContext, DetectorResult
    from dataset_sentinel.registry import detectors

    @detectors.register("my_check")
    class MyDetector(Detector):
        name = "my_check"
        description = "..."
        modalities = ("image",)

        def run(self, ctx: DetectorContext) -> DetectorResult:
            result = DetectorResult()
            ...
            return result
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import SentinelConfig, split_role
from ..fingerprints.store import FingerprintStore
from ..model import Confidence, Dataset, Finding, RelationshipGroup, Sample, SampleRef, Severity

log = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    findings: List[Finding] = field(default_factory=list)
    groups: List[RelationshipGroup] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    skipped: Optional[str] = None

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def add_group(self, group: RelationshipGroup) -> None:
        self.groups.append(group)


class DetectorContext:
    """Everything a detector may need. Fingerprints are computed on first use
    and shared between detectors."""

    def __init__(
        self,
        dataset: Dataset,
        config: SentinelConfig,
        fingerprint_store: Optional[FingerprintStore] = None,
        progress: Optional[Any] = None,
    ):
        self.dataset = dataset
        self.config = config
        self._store = fingerprint_store
        self.progress = progress
        self.log = log
        #: scratch space for detectors to share intermediate results
        #: (e.g. near-duplicate pairs reused by the consistency detector)
        self.shared: Dict[str, Any] = {}

    @property
    def fingerprints(self) -> FingerprintStore:
        if self._store is None:
            workers = int(self.config.get("performance.workers") or 0)
            cache = self.config.get("performance.cache")
            cache_dir = self.config.resolve_path(cache) if cache else None
            mode = str(self.config.get("performance.mode") or "exact")
            self._store = FingerprintStore(self.dataset, workers=workers, cache_dir=cache_dir, mode=mode)
        if not self._store._computed:
            self._store.compute(progress=self.progress)
        return self._store

    @property
    def has_fingerprints(self) -> bool:
        return self._store is not None and self._store._computed

    # ------------------------------------------------------------------ helpers
    def severity(self, dotted: str, default: Optional[Severity] = None) -> Optional[Severity]:
        return self.config.severity(dotted, default)

    def policy_label(self, dotted: str) -> str:
        return self.config.policy_label(dotted)

    def refs(self, sample_ids: Iterable[str]) -> List[SampleRef]:
        out = []
        for sid in sample_ids:
            s = self.dataset.get_or_none(sid)
            if s is not None:
                out.append(s.ref())
        return out

    def split_of(self, sample_id: str) -> str:
        return self.dataset.get(sample_id).split

    def role_of(self, split_name: str) -> Optional[str]:
        return split_role(split_name)

    def ordered_splits(self) -> List[str]:
        """Dataset splits ordered by policy.split_order roles (unknown roles last)."""
        order = {role: i for i, role in enumerate(self.config.split_order)}
        return sorted(self.dataset.splits, key=lambda s: (order.get(split_role(s) or "", 99), s))


class Detector(ABC):
    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    modalities: ClassVar[Tuple[str, ...]] = ("image",)
    #: set when the detector needs image fingerprints (used for planning)
    needs_fingerprints: ClassVar[bool] = False

    def __init__(self, config: SentinelConfig):
        self.config = config

    def supports(self, modality: str) -> bool:
        return "*" in self.modalities or modality in self.modalities

    @abstractmethod
    def run(self, ctx: DetectorContext) -> DetectorResult:
        ...

    # ------------------------------------------------------------------ helpers
    def finding(
        self,
        *,
        kind: str,
        title: str,
        severity: Severity,
        confidence: Confidence,
        policy: str,
        policy_description: str,
        message: str,
        remediation: str,
        samples: Sequence[SampleRef] = (),
        evidence: Optional[Dict[str, Any]] = None,
        group_id: Optional[str] = None,
        counts: bool = False,
        splits: Optional[Sequence[str]] = None,
    ) -> Finding:
        return Finding(
            detector=self.name,
            kind=kind,
            title=title,
            severity=severity,
            confidence=confidence,
            policy=policy,
            policy_description=policy_description,
            message=message,
            remediation=remediation,
            samples=list(samples),
            splits=list(splits) if splits else [],
            evidence=dict(evidence or {}),
            group_id=group_id,
            counts_toward_violation_rate=counts,
        )

    def group(
        self,
        kind: str,
        member_ids: Sequence[str],
        splits: Sequence[str],
        confidence: Confidence,
        violates: bool,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> RelationshipGroup:
        gid = group_id(self.name, kind, member_ids)
        return RelationshipGroup(
            id=gid,
            kind=kind,
            member_ids=sorted(member_ids),
            splits=sorted(set(splits)),
            confidence=confidence,
            detector=self.name,
            violates_policy=violates,
            evidence=dict(evidence or {}),
        )


class FindingCap:
    """Limits the number of findings emitted per kind and emits one summary
    finding per overflowing kind so nothing is silently dropped."""

    def __init__(self, limit: int):
        self.limit = max(1, int(limit))
        self.counts: Dict[str, int] = {}
        self.overflow: Dict[str, int] = {}

    def allow(self, kind: str) -> bool:
        n = self.counts.get(kind, 0) + 1
        self.counts[kind] = n
        if n > self.limit:
            self.overflow[kind] = self.overflow.get(kind, 0) + 1
            return False
        return True

    def summaries(self, detector: "Detector", severity_for: Dict[str, Severity]) -> List[Finding]:
        out = []
        for kind, extra in self.overflow.items():
            out.append(
                detector.finding(
                    kind=f"{kind}_overflow",
                    title=f"{extra} more '{kind}' findings not listed individually",
                    severity=severity_for.get(kind, Severity.INFO),
                    confidence=Confidence.DETERMINISTIC,
                    policy="output.max_findings_per_kind",
                    policy_description="Findings of one kind are capped in the report; all are still counted in metrics.",
                    message=f"{self.limit} '{kind}' findings are shown; {extra} more were found. Raise output.max_findings_per_kind to list them all.",
                    remediation="Fix the listed instances first, then re-run to see the remainder.",
                    evidence={"kind": kind, "shown": self.limit, "hidden": extra},
                )
            )
        return out


def group_id(detector: str, kind: str, member_ids: Sequence[str]) -> str:
    key = f"{detector}|{kind}|{','.join(sorted(member_ids))}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def crosses_splits(samples: Iterable[Sample]) -> bool:
    return len({s.split for s in samples}) > 1


def sample_uri_list(samples: Sequence[Sample], limit: int = 6) -> str:
    uris = [f"{s.split}:{s.uri}" for s in samples[:limit]]
    if len(samples) > limit:
        uris.append(f"... (+{len(samples) - limit} more)")
    return ", ".join(uris)


def relative_display(path: Optional[Path], root: Path) -> str:
    if path is None:
        return "<no path>"
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
