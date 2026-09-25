"""Core data model for Dataset Sentinel.

Every dataset, regardless of modality, is modelled as:

* :class:`Sample` objects that belong to exactly one split;
* :class:`Annotation` objects (labels / targets) attached to samples;
* free-form ``metadata`` on each sample (timestamps, capture device, ...);
* ``groups``: resolved entity / source / batch identifiers per sample;
* :class:`Relationship` edges between samples (lineage, derivation, ...);
* an :class:`~dataset_sentinel.policy.IntegrityPolicy` (see ``policy.py``)
  stating which relationships may cross split boundaries.

Detectors consume a :class:`Dataset` and emit :class:`Finding` objects plus
:class:`RelationshipGroup` objects. Groups are what the primary metric, the
*split integrity violation rate*, is computed from.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "Confidence",
    "Severity",
    "BBox",
    "Annotation",
    "Sample",
    "SampleRef",
    "Relationship",
    "Dataset",
    "Finding",
    "RelationshipGroup",
    "DetectorRunInfo",
    "Metrics",
    "Report",
    "SEVERITY_ORDER",
]


class Confidence(str, Enum):
    """How sure a detector is that a finding is real.

    * ``deterministic``: follows directly from data that cannot be wrong
      (byte-identical files, explicit lineage metadata, a bbox with
      negative width).
    * ``high_confidence``: a strong signal with a very low false-positive
      rate (tiny perceptual-hash distance, exact pixel match after decode).
    * ``heuristic``: a plausible signal that should be reviewed by a human
      (filename patterns, moderate hash distances, distribution shifts).
    * ``inconclusive``: the detector could not decide, usually because the
      required metadata was missing; reported so the gap is visible.
    """

    DETERMINISTIC = "deterministic"
    HIGH_CONFIDENCE = "high_confidence"
    HEURISTIC = "heuristic"
    INCONCLUSIVE = "inconclusive"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @classmethod
    def parse(cls, value: Any, default: "Severity | None" = None) -> "Severity | None":
        """Parse a policy value. ``ignore``/``off``/``false``/``None`` -> ``None``."""
        if value is None:
            return default
        if isinstance(value, Severity):
            return value
        text = str(value).strip().lower()
        if text in {"ignore", "off", "false", "none", "disabled", "skip"}:
            return None
        if text in {"error", "fail", "forbid", "forbidden"}:
            return cls.ERROR
        if text in {"warning", "warn"}:
            return cls.WARNING
        if text in {"info", "allow", "allowed", "note"}:
            return cls.INFO
        raise ValueError(f"Unknown severity/policy value: {value!r}")


SEVERITY_ORDER: Dict[Severity, int] = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates (x, y, w, h)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]


@dataclass
class Annotation:
    """A single label / target attached to a sample.

    ``category`` is the resolved human-readable class name when known.
    ``category_id`` is the native identifier from the source format.
    ``bbox`` is in absolute pixel coordinates when the adapter could resolve
    it (YOLO adapters keep normalized coordinates in ``attributes["bbox_norm"]``
    and fill ``bbox`` only if the image size is known).
    """

    id: str
    sample_id: str
    category: Optional[str] = None
    category_id: Optional[Any] = None
    bbox: Optional[BBox] = None
    segmentation: Optional[Any] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """A stable string that identifies the label content (used for
        detecting conflicting labels on identical samples)."""
        parts = [str(self.category if self.category is not None else self.category_id)]
        if self.bbox is not None:
            parts.append(",".join(f"{v:.2f}" for v in self.bbox.as_list()))
        if "bbox_norm" in self.attributes:
            parts.append(",".join(f"{float(v):.4f}" for v in self.attributes["bbox_norm"]))
        return "|".join(parts)


@dataclass
class Relationship:
    """A directed edge between two samples.

    ``kind`` is free-form but the built-in detectors understand:
    ``derived_from`` (target is the parent of source), ``same_group:<key>``.
    """

    kind: str
    source_id: str
    target_id: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: Confidence = Confidence.DETERMINISTIC


@dataclass
class Sample:
    """One unit of data (for the image modality: one image file)."""

    id: str
    split: str
    uri: str
    path: Optional[Path] = None
    modality: str = "image"
    native_id: Optional[Any] = None
    annotations: List[Annotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    groups: Dict[str, str] = field(default_factory=dict)

    @property
    def width(self) -> Optional[int]:
        w = self.metadata.get("width")
        return int(w) if isinstance(w, (int, float)) and w > 0 else None

    @property
    def height(self) -> Optional[int]:
        h = self.metadata.get("height")
        return int(h) if isinstance(h, (int, float)) and h > 0 else None

    def ref(self) -> "SampleRef":
        return SampleRef(id=self.id, split=self.split, uri=self.uri)

    def label_signature(self) -> str:
        return ";".join(sorted(a.signature() for a in self.annotations))


@dataclass(frozen=True)
class SampleRef:
    """Lightweight pointer to a sample used inside findings and reports."""

    id: str
    split: str
    uri: str

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.id, "split": self.split, "uri": self.uri}


@dataclass
class Dataset:
    """An in-memory, modality-agnostic view of a dataset."""

    name: str
    samples: List[Sample]
    splits: List[str]
    modality: str = "image"
    categories: Dict[Any, str] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    source: Dict[str, Any] = field(default_factory=dict)
    _index: Dict[str, Sample] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.reindex()

    def reindex(self) -> None:
        self._index = {s.id: s for s in self.samples}
        seen = set()
        ordered: List[str] = []
        for split in self.splits:
            if split not in seen:
                ordered.append(split)
                seen.add(split)
        for s in self.samples:
            if s.split not in seen:
                ordered.append(s.split)
                seen.add(s.split)
        self.splits = ordered

    def get(self, sample_id: str) -> Sample:
        return self._index[sample_id]

    def get_or_none(self, sample_id: str) -> Optional[Sample]:
        return self._index.get(sample_id)

    def __len__(self) -> int:
        return len(self.samples)

    def by_split(self) -> Dict[str, List[Sample]]:
        out: Dict[str, List[Sample]] = {name: [] for name in self.splits}
        for s in self.samples:
            out.setdefault(s.split, []).append(s)
        return out

    def split_sizes(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.by_split().items()}

    def category_names(self) -> List[str]:
        return sorted({str(v) for v in self.categories.values()})

    def annotations(self) -> Iterable[Annotation]:
        for s in self.samples:
            yield from s.annotations


@dataclass
class Finding:
    """One reported problem.

    The product requirements say every finding must state: the violated
    policy, the evidence used, the confidence class, the samples and splits
    involved, and the recommended remediation. Those are all mandatory
    fields here.
    """

    detector: str
    kind: str
    title: str
    severity: Severity
    confidence: Confidence
    policy: str
    policy_description: str
    message: str
    remediation: str
    samples: List[SampleRef] = field(default_factory=list)
    splits: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    group_id: Optional[str] = None
    counts_toward_violation_rate: bool = False
    id: str = ""
    #: set when a baseline report was given: True = not present in the baseline
    new: Optional[bool] = None
    #: set when an allowlist entry suppressed the finding: {"reason": ..., "entry": ...}
    suppressed: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.splits:
            self.splits = sorted({s.split for s in self.samples})
        if not self.id:
            self.id = self.compute_id()

    def compute_id(self) -> str:
        key = "|".join(
            [
                self.detector,
                self.kind,
                ",".join(sorted(s.id for s in self.samples)),
                self.group_id or "",
                self.title,
            ]
        )
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "detector": self.detector,
            "kind": self.kind,
            "title": self.title,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "policy": self.policy,
            "policy_description": self.policy_description,
            "message": self.message,
            "remediation": self.remediation,
            "samples": [s.to_dict() for s in self.samples],
            "splits": list(self.splits),
            "evidence": _jsonable(self.evidence),
            "group_id": self.group_id,
            "counts_toward_violation_rate": self.counts_toward_violation_rate,
            "new": self.new,
            "suppressed": _jsonable(self.suppressed),
        }


@dataclass
class RelationshipGroup:
    """A set of samples tied together by a relationship (identity, similarity,
    shared entity, lineage, time window, ...).

    ``violates_policy`` is set when the group spans splits in a way the
    policy forbids. The violation rate counts the members of such groups.
    """

    id: str
    kind: str
    member_ids: List[str]
    splits: List[str]
    confidence: Confidence
    detector: str
    violates_policy: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "detector": self.detector,
            "members": list(self.member_ids),
            "splits": list(self.splits),
            "confidence": self.confidence.value,
            "violates_policy": self.violates_policy,
            "evidence": _jsonable(self.evidence),
        }


@dataclass
class DetectorRunInfo:
    name: str
    status: str  # "ok" | "skipped" | "error"
    duration_s: float = 0.0
    findings: int = 0
    groups: int = 0
    message: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "duration_s": round(self.duration_s, 3),
            "findings": self.findings,
            "groups": self.groups,
            "message": self.message,
            "stats": _jsonable(self.stats),
        }


@dataclass
class Metrics:
    """The primary metric plus supporting breakdowns.

    ``violation_rate`` = samples in at least one policy-violating
    relationship group / all evaluated samples.
    """

    evaluated_samples: int
    violating_samples: int
    violation_rate: float
    by_split: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_detector: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_confidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    violating_groups: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_integrity_violation_rate": self.violation_rate,
            "evaluated_samples": self.evaluated_samples,
            "violating_samples": self.violating_samples,
            "violating_groups": self.violating_groups,
            "by_split": self.by_split,
            "by_detector": self.by_detector,
            "by_confidence": self.by_confidence,
            "severity_counts": self.severity_counts,
        }


@dataclass
class Report:
    dataset: Dict[str, Any]
    findings: List[Finding]
    groups: List[RelationshipGroup]
    metrics: Metrics
    detectors: List[DetectorRunInfo]
    config: Dict[str, Any]
    generated_at: str
    version: str
    passed: bool = True
    failure_reasons: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    #: suggested remediation actions (see ``fixplan.py``); serialised separately
    fix_plan: Optional[Dict[str, Any]] = None
    #: cross-split clusters: connected components over all violating groups
    clusters: List[Dict[str, Any]] = field(default_factory=list)
    #: metric impact of the flagged samples (see ``impact.py``), when predictions were given
    impact: Optional[Dict[str, Any]] = None
    #: in-memory dataset for reporters that need file paths (thumbnails); never serialised
    dataset_ref: Optional["Dataset"] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "tool": {"name": "dataset-sentinel", "version": self.version},
            "generated_at": self.generated_at,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "dataset": _jsonable(self.dataset),
            "metrics": self.metrics.to_dict(),
            "detectors": [d.to_dict() for d in self.detectors],
            "findings": [f.to_dict() for f in self.findings],
            "groups": [g.to_dict() for g in self.groups],
            "clusters": _jsonable(self.clusters),
            "impact": _jsonable(self.impact),
            "stats": _jsonable(self.stats),
            "fix_plan_summary": _jsonable(self.fix_plan.get("summary")) if self.fix_plan else None,
            "config": _jsonable(self.config),
        }

    def findings_by_severity(self) -> Dict[Severity, List[Finding]]:
        out: Dict[Severity, List[Finding]] = {s: [] for s in Severity}
        for f in self.findings:
            out[f.severity].append(f)
        return out


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, paths, enums, sets and numpy scalars
    into plain JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def sorted_findings(findings: Iterable[Finding]) -> List[Finding]:
    """Deterministic ordering: severity, then confidence, then detector, id."""
    conf_order = {
        Confidence.DETERMINISTIC: 0,
        Confidence.HIGH_CONFIDENCE: 1,
        Confidence.HEURISTIC: 2,
        Confidence.INCONCLUSIVE: 3,
    }

    def key(f: Finding) -> Tuple[int, int, str, str]:
        return (SEVERITY_ORDER[f.severity], conf_order[f.confidence], f.detector, f.id)

    return sorted(findings, key=key)
