"""Dataset Sentinel: validate ML dataset split integrity before training.

Public API::

    from dataset_sentinel import scan, load_config

    report = scan(load_config("sentinel.yaml"))
    print(report.metrics.violation_rate)
"""

from __future__ import annotations

from .config import SentinelConfig, load_config
from .engine import scan
from .model import (
    Annotation,
    BBox,
    Confidence,
    Dataset,
    Finding,
    Metrics,
    RelationshipGroup,
    Report,
    Sample,
    Severity,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "scan",
    "load_config",
    "SentinelConfig",
    "Annotation",
    "BBox",
    "Confidence",
    "Dataset",
    "Finding",
    "Metrics",
    "RelationshipGroup",
    "Report",
    "Sample",
    "Severity",
]
