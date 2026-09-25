"""Configuration and integrity-policy handling.

A Sentinel run is described by one nested mapping (usually ``sentinel.yaml``).
``DEFAULT_CONFIG`` documents every key. User config is deep-merged over the
defaults, then CLI overrides are applied on top.

Policy values for relationship checks are severities: ``error``, ``warning``,
``info`` or ``ignore``. ``forbid`` is an alias of ``error`` and ``allow`` an
alias of ``info`` so policies read naturally::

    policy:
      near_duplicate:
        cross_split: forbid
        within_split: allow
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .model import Severity

__all__ = [
    "DEFAULT_CONFIG",
    "SentinelConfig",
    "load_config",
    "deep_merge",
    "split_role",
    "SPLIT_ROLE_ALIASES",
    "DEFAULT_GROUP_KEYS",
    "DEFAULT_TIME_KEYS",
    "DEFAULT_LINEAGE_KEYS",
]

DEFAULT_GROUP_KEYS: List[str] = [
    "patient_id", "patient", "subject_id", "subject", "person_id", "entity_id", "entity",
    "source", "source_id", "batch", "batch_id", "session", "session_id",
    "device", "device_id", "camera", "camera_id", "location", "location_id", "site", "site_id",
    "author", "author_id", "account", "account_id", "user_id", "video_id", "video",
    "sequence", "sequence_id", "seq_id", "capture_id", "scene", "scene_id", "study_id",
    "case_id", "group", "group_id", "flickr_url_owner", "collection",
]

DEFAULT_TIME_KEYS: List[str] = [
    "captured_at", "timestamp", "date_captured", "datetime", "capture_time", "acquired_at",
    "acquisition_time", "time", "date", "created_at", "recorded_at",
]

DEFAULT_LINEAGE_KEYS: List[str] = [
    "derived_from", "parent", "parent_id", "parent_file", "source_image", "source_file",
    "original", "original_id", "original_file", "base_image",
]

SPLIT_ROLE_ALIASES: Dict[str, List[str]] = {
    "train": ["train", "training", "trn", "fit"],
    "val": ["val", "valid", "validation", "dev", "development"],
    "test": ["test", "testing", "eval", "evaluation", "holdout", "hold_out", "final"],
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 1,
    "dataset": {
        "name": None,
        # coco | yolo | image-folder | (any registered adapter name)
        "format": None,
        "root": ".",
        # COCO: {split_name: {"annotations": path, "images": dir}}
        # image-folder: {split_name: dir}
        "splits": {},
        # YOLO: path to data.yaml (relative to root)
        "data": None,
        # Optional sidecar with per-sample metadata (CSV or JSON list / mapping).
        "metadata": {
            "file": None,
            # column/field that identifies the sample; matched against the
            # file name, stem, relative path or native id (see "match")
            "key": "file_name",
            # name | stem | path | id
            "match": "name",
        },
        "groups": {
            # metadata keys that identify entities / sources / batches
            "keys": [],
            # when true, well-known keys found in metadata are used automatically
            "auto": True,
            # {group_name: regex with a named group "value" applied to the file name}
            "from_filename": {},
            # Python callable "package.module:function" receiving a Sample and
            # returning {group_name: value} (or None); for custom logic
            "resolver": None,
        },
        "lineage": {
            "keys": list(DEFAULT_LINEAGE_KEYS),
            # list of regexes applied to the file name; a named group "parent"
            # gives the stem/name of the parent sample
            "from_filename": [],
        },
        "time": {
            "keys": list(DEFAULT_TIME_KEYS),
            # optional strptime format; ISO-8601, epoch seconds/millis and
            # common formats are parsed automatically
            "format": None,
        },
    },
    "policy": {
        # role order used by the temporal check: earlier roles must precede later ones
        "split_order": ["train", "val", "test"],
        # severity overrides for leakage groups by the pair of split roles they
        # span, e.g. {"train/val": "warning"} keeps train/val overlap a warning
        # while train/test stays an error. Keys are role names joined by "/".
        "split_pair_overrides": {},
        "exact_duplicate": {"cross_split": "error", "within_split": "warning"},
        "near_duplicate": {
            "cross_split": "error",
            "within_split": "info",
            # Hamming distance on 64-bit dHash. 0-3 catches re-encodes and
            # resizes; 4-8 catches mild edits; >10 gets noisy on simple images.
            "threshold": 6,
            # Candidate pairs are verified with the normalised correlation of
            # 16x16 grayscale thumbnails; below this value the pair is dropped.
            "min_correlation": 0.8,
            "max_group_findings": 500,
            # Optional embedding re-ranking of hash candidates (never all-pairs).
            # provider: builtin (numpy descriptor) | torchvision (extra) | plugin
            "embedding": {
                "enabled": False,
                "provider": "builtin",
                # wider hash threshold used to generate candidates when enabled
                "threshold": 10,
                # candidates below this cosine similarity are dropped
                "min_cosine": 0.9,
            },
        },
        "derivative": {
            "cross_split": "error",
            "within_split": "info",
            "threshold": 6,
            "min_correlation": 0.8,
            # also look for regular tiles (2x2), halves and centre crops
            "crops": True,
            # crop candidates use a slightly wider hash threshold and a stricter
            # thumbnail-correlation check; only cross-split crops are searched
            "crop_threshold": 8,
            "crop_min_correlation": 0.9,
            # skip the (comparatively expensive) crop search above this many
            # images; the report says so. 0 = no limit
            "crops_max_images": 100000,
        },
        "group_overlap": {"cross_split": "error"},
        "lineage": {"cross_split": "error"},
        "temporal": {
            # auto: report overlapping time ranges as an informational,
            #       inconclusive finding (a random split legitimately overlaps)
            # true: the splits MUST be chronological (train before val before
            #       test); overlaps are violations with the severity below
            # false: skip
            "enabled": "auto",
            "cross_split": "error",
            # fraction of samples that must have a timestamp before the check
            # is considered conclusive
            "min_coverage": 0.5,
        },
        "labels": {
            "missing_file": "error",
            "unreadable_image": "error",
            "empty_image": "error",
            "blank_image": "warning",
            "size_mismatch": "warning",
            "missing_annotations": "warning",
            # label file present but every line malformed
            "no_valid_annotations": "warning",
            "empty_split": "error",
            "invalid_bbox": "error",
            "out_of_bounds_bbox": "error",
            "degenerate_bbox": "error",
            "min_bbox_size_px": 1.0,
            "unknown_category": "error",
            "allowed_categories": None,
            "duplicate_annotation_id": "error",
            "duplicate_annotation": "warning",
            "orphan_annotation": "error",
            "invalid_segmentation": "warning",
            # keypoints: wrong count, bad visibility flag, visible point outside the image
            "invalid_keypoints": "error",
            "keypoints_out_of_bounds": "warning",
            "malformed_label": "error",
            "missing_label_file": "warning",
            # COCO: the same category id named differently in two split files
            "category_mismatch": "error",
            # COCO: split files declaring different category sets
            "category_set_mismatch": "warning",
        },
        "distribution": {
            "imbalance": "warning",
            "imbalance_ratio": 20.0,
            "shift": "warning",
            # Jensen-Shannon divergence (base 2, in [0, 1]) between class
            # distributions of train and another split
            "shift_threshold": 0.1,
            "unseen_class": "error",
            "missing_class_in_eval": "warning",
            "image_size_shift": "info",
            "bbox_size_shift": "info",
            "cooccurrence": "info",
            "min_samples": 20,
            # warn when a class has fewer than this many annotated samples in a
            # split where it is expected (0 disables)
            "min_samples_per_class_per_split": 5,
            "rare_class": "warning",
            # class mix conditioned on metadata groups (camera, location,
            # session, ...): a group value whose class distribution diverges from
            # its split's overall distribution by more than shift_threshold
            "conditioned": "info",
            "conditioned_min_samples": 30,
            # per class: boxes cut off at the image border, size shift vs training
            "truncated_boxes": "info",
            "class_box_size_shift": "info",
            "min_boxes_per_class": 10,
        },
        "consistency": {
            "conflicting_labels": "error",
            "duplicate_registration": "error",
        },
    },
    "detectors": {
        # null = all registered detectors
        "enabled": None,
        "disabled": [],
    },
    "output": {
        "json": None,
        "html": None,
        "markdown": None,
        # SARIF 2.1.0 for GitHub code scanning
        "sarif": None,
        # flat metrics file (YAML/JSON) for DVC
        "dvc": None,
        # JSON (or .csv) file with suggested remove/move actions per sample
        "fix_plan": None,
        "thumbnails": True,
        "max_thumbnails": 200,
        "thumbnail_size": 96,
        "max_findings_per_kind": 1000,
        # findings rendered in the HTML report (the JSON report always has all)
        "max_findings_html": 2000,
    },
    "baseline": {
        # JSON report of a previous run; findings whose id appears there are
        # marked as not new
        "file": None,
    },
    # Known, permitted findings. Each entry needs a reason and matches by
    # finding id, group id, kind, detector, or sample uri globs (all samples
    # of the finding must match). Optional "expires: YYYY-MM-DD".
    #   allowlist:
    #     - id: 2cc9116ec30d
    #       reason: "duplicate kept on purpose for the calibration set"
    #     - samples: ["test/calib_*.jpg", "train/calib_*.jpg"]
    #       reason: "calibration images appear in every split by design"
    #       expires: 2027-01-01
    "allowlist": [],
    "fail_on": {
        # error | warning | info | none
        "severity": "error",
        "max_violation_rate": 0.0,
        # a detector that crashes fails the run (a silent gap is worse than a failure)
        "detector_errors": True,
        # with a baseline: only NEW findings fail the severity check, and the
        # violation rate only fails when it exceeds the baseline's rate
        "new_only": False,
    },
    "performance": {
        # exact: full decode + pixel-identity hash (default)
        # fast:  reduced-resolution JPEG decode, no pixel hash; 3-5x faster on
        #        large photos, byte-identical and perceptual checks unaffected
        "mode": "exact",
        # 0 = auto (cpu count, capped at 8)
        "workers": 0,
        # directory for the fingerprint cache (null disables caching)
        "cache": None,
        # limit samples per split (debugging only)
        "max_samples_per_split": None,
    },
}


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a new dict with ``override`` merged into ``base`` recursively."""
    result = copy.deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def split_role(name: str) -> Optional[str]:
    """Map a split name such as ``valid``, ``test2017`` or
    ``instances_train2017`` to train/val/test."""
    lowered = name.lower()
    for role, aliases in SPLIT_ROLE_ALIASES.items():
        for alias in aliases:
            if lowered == alias or lowered.startswith(alias) or lowered.endswith(alias):
                return role
    # substring match, longest alias first so "evaluation" wins over "val"
    pairs = sorted(((alias, role) for role, aliases in SPLIT_ROLE_ALIASES.items() for alias in aliases), key=lambda p: -len(p[0]))
    for alias, role in pairs:
        if alias in lowered:
            return role
    return None


#: config paths whose children are free-form (not validated against defaults)
FREE_FORM_PATHS = {
    "dataset.splits",
    "dataset.groups.from_filename",
    "dataset.source",
    "policy.split_pair_overrides",
}


def find_unknown_keys(user: Dict[str, Any], defaults: Dict[str, Any] = DEFAULT_CONFIG, path: str = "",
                      extra_policy_keys: Iterable[str] = ()) -> List[str]:
    """Return warnings for keys in ``user`` that do not exist in ``defaults``.

    ``policy.<name>`` is accepted for any registered detector name passed in
    ``extra_policy_keys`` (plugins define their own policy sections).
    """
    import difflib

    warnings: List[str] = []
    extra = set(extra_policy_keys)
    for key, value in user.items():
        dotted = f"{path}.{key}" if path else str(key)
        if path in FREE_FORM_PATHS:
            continue
        if key not in defaults:
            if path == "policy" and key in extra:
                continue
            suggestion = difflib.get_close_matches(str(key), [str(k) for k in defaults], n=1, cutoff=0.6)
            hint = f" (did you mean '{path + '.' if path else ''}{suggestion[0]}'?)" if suggestion else ""
            warnings.append(f"unknown config key '{dotted}'{hint}")
            continue
        if isinstance(value, dict) and isinstance(defaults[key], dict):
            warnings.extend(find_unknown_keys(value, defaults[key], dotted, extra_policy_keys))
    return warnings


class SentinelConfig:
    """Thin wrapper over the merged config dict with typed helpers."""

    def __init__(self, data: Optional[Dict[str, Any]] = None, source_path: Optional[Path] = None):
        self.data: Dict[str, Any] = deep_merge(DEFAULT_CONFIG, data or {})
        self.source_path = source_path
        self.base_dir: Path = source_path.parent if source_path else Path.cwd()
        self.warnings: List[str] = []

    def validate(self, extra_policy_keys: Iterable[str] = ()) -> List[str]:
        """Warn about keys that are not part of the schema (typos)."""
        user = {k: v for k, v in self.data.items()}
        self.warnings = find_unknown_keys(user, DEFAULT_CONFIG, "", extra_policy_keys)
        return self.warnings

    # ------------------------------------------------------------------ access
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def section(self, dotted: str) -> Dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}

    def severity(self, dotted: str, default: Optional[Severity] = None) -> Optional[Severity]:
        """Policy value -> Severity (or ``None`` when the check is disabled)."""
        raw = self.get(dotted, "__missing__")
        if raw == "__missing__":
            return default
        return Severity.parse(raw, default)

    def policy_label(self, dotted: str) -> str:
        return f"{dotted} = {self.get(dotted)}"

    # ------------------------------------------------------------------ paths
    def resolve_path(self, value: Any, relative_to: Optional[Path] = None) -> Optional[Path]:
        if value is None:
            return None
        p = Path(os.path.expanduser(str(value)))
        if p.is_absolute():
            return p
        return (relative_to or self.base_dir) / p

    @property
    def root(self) -> Path:
        root = self.resolve_path(self.get("dataset.root", "."))
        return root if root is not None else self.base_dir

    @property
    def split_order(self) -> List[str]:
        order = self.get("policy.split_order") or ["train", "val", "test"]
        return [str(x) for x in order]

    def detectors_enabled(self, available: Iterable[str]) -> List[str]:
        enabled = self.get("detectors.enabled")
        disabled = set(self.get("detectors.disabled") or [])
        names = list(available)
        if enabled:
            names = [n for n in names if n in set(enabled)]
        return [n for n in names if n not in disabled]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SentinelConfig({json.dumps(self.data, default=str)[:200]}...)"


def load_config(path: Optional[str | os.PathLike] = None, overrides: Optional[Dict[str, Any]] = None) -> SentinelConfig:
    """Load ``sentinel.yaml`` / ``.yml`` / ``.json`` (or defaults when ``path`` is None)."""
    data: Dict[str, Any] = {}
    source: Optional[Path] = None
    if path is not None:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Config file not found: {source}")
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            data = json.loads(text) or {}
        else:
            data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file {source} must contain a mapping at the top level")
    merged = deep_merge(data, overrides) if overrides else data
    return SentinelConfig(merged, source_path=source)


def config_template(fmt: str = "coco") -> str:
    """Return a commented YAML template for ``sentinel init``."""
    if fmt == "yolo":
        dataset_block = """dataset:
  name: my-dataset
  format: yolo
  root: .
  # Ultralytics-style data.yaml with train/val/test image dirs and class names
  data: data.yaml
"""
    elif fmt == "image-folder":
        dataset_block = """dataset:
  name: my-dataset
  format: image-folder
  root: .
  # each split is a directory; class sub-directories are optional
  splits:
    train: train
    val: val
    test: test
"""
    else:
        dataset_block = """dataset:
  name: my-dataset
  format: coco
  root: .
  splits:
    train:
      annotations: annotations/instances_train.json
      images: images/train
    val:
      annotations: annotations/instances_val.json
      images: images/val
    test:
      annotations: annotations/instances_test.json
      images: images/test
"""
    return (
        "# Dataset Sentinel configuration (https://github.com/N-Lampl/Sentinel)\n"
        "version: 1\n\n"
        + dataset_block
        + """
  # Optional per-sample metadata sidecar (CSV or JSON). Columns such as
  # patient_id, source, session, device, captured_at are used for group and
  # time leakage checks.
  # metadata:
  #   file: metadata.csv
  #   key: file_name
  #   match: name        # name | stem | path | id

  groups:
    auto: true           # use well-known keys (patient_id, source, session, ...) when present
    keys: []             # extra metadata keys that identify an entity / source / batch
    from_filename: {}    # e.g. {patient: "^(?P<value>P\\\\d+)_"}

  lineage:
    # keys: [derived_from, parent, source_image]
    from_filename: []    # e.g. ["^(?P<parent>.+?)_(?:aug|crop|tile|flip|rot)\\\\d*$"]

  time:
    # keys: [captured_at, timestamp, date_captured]
    format: null         # strptime format; auto-detected when null

policy:
  split_order: [train, val, test]
  # severity per pair of split roles a leakage group spans (optional)
  # split_pair_overrides: {"train/val": warning, "train/test": error, "val/test": error}
  exact_duplicate:   {cross_split: error, within_split: warning}
  near_duplicate:    {cross_split: error, within_split: info, threshold: 6, min_correlation: 0.8}
  #   embedding: {enabled: false, provider: builtin, min_cosine: 0.9}   # optional re-ranking
  derivative:        {cross_split: error, within_split: info, threshold: 6, crops: true}
  group_overlap:     {cross_split: error}
  lineage:           {cross_split: error}
  temporal:          {enabled: auto, cross_split: error, min_coverage: 0.5}
  labels:
    missing_file: error
    unreadable_image: error
    missing_annotations: warning
    empty_split: error
    out_of_bounds_bbox: error
    degenerate_bbox: error
    unknown_category: error
    allowed_categories: null
  distribution:
    imbalance: warning
    imbalance_ratio: 20
    shift: warning
    shift_threshold: 0.1
    unseen_class: error
  consistency:
    conflicting_labels: error

# Known, permitted findings: kept in the report as "suppressed", excluded from the rate.
# allowlist:
#   - id: 2cc9116ec30d
#     reason: "duplicate kept on purpose"
#     expires: 2027-01-01

# Compare against an approved run (sentinel baseline create -o .sentinel/baseline.json)
# baseline:
#   file: .sentinel/baseline.json

output:
  json: sentinel-report.json
  html: sentinel-report.html
  # sarif: sentinel-report.sarif
  # fix_plan: fix-plan.csv
  thumbnails: true

fail_on:
  severity: error
  max_violation_rate: 0.0
  detector_errors: true
  new_only: false        # true: with a baseline, fail only on regressions

performance:
  mode: exact            # fast = reduced JPEG decode, no pixel-identity hash
  workers: 0
  cache: .sentinel-cache
"""
    )
