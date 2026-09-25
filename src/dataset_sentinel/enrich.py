"""Post-load enrichment shared by all adapters.

* apply a metadata sidecar (CSV / JSON) to samples;
* resolve entity / source / batch *groups* from metadata and file names;
* resolve *lineage* (``derived_from``) relationships from metadata and
  file-name patterns;
* parse timestamps into ``sample.metadata["_timestamp"]``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DEFAULT_GROUP_KEYS, SentinelConfig
from .model import Confidence, Dataset, Relationship, Sample

log = logging.getLogger(__name__)

_TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y:%m:%d %H:%M:%S",  # EXIF
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%Y%m%d_%H%M%S",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y",
]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "n/a", "na"}:
        return True
    return False


# --------------------------------------------------------------------------- timestamps
def parse_timestamp(value: Any, fmt: Optional[str] = None) -> Optional[float]:
    """Return epoch seconds (UTC) for common timestamp representations."""
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e14:  # microseconds
            v /= 1e6
        elif v > 1e11:  # milliseconds
            v /= 1e3
        return v
    text = str(value).strip()
    if fmt:
        try:
            dt = datetime.strptime(text, fmt)
            return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
        except ValueError:
            return None
    if re.fullmatch(r"\d{8}", text) and "1900" <= text[:4] <= "2100":
        try:
            return datetime.strptime(text, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    if re.fullmatch(r"\d{14}", text) and "1900" <= text[:4] <= "2100":
        try:
            return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return parse_timestamp(float(text))
    iso = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso)
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        pass
    for f in _TS_FORMATS:
        try:
            dt = datetime.strptime(text, f)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def resolve_timestamps(dataset: Dataset, config: SentinelConfig) -> Dict[str, Any]:
    keys = [str(k) for k in (config.get("dataset.time.keys") or [])]
    fmt = config.get("dataset.time.format")
    found = 0
    used_keys: Dict[str, int] = {}
    unparsed = 0
    for s in dataset.samples:
        for key in keys:
            if key in s.metadata and not _is_missing(s.metadata[key]):
                ts = parse_timestamp(s.metadata[key], fmt)
                if ts is None:
                    unparsed += 1
                    continue
                s.metadata["_timestamp"] = ts
                s.metadata["_timestamp_key"] = key
                used_keys[key] = used_keys.get(key, 0) + 1
                found += 1
                break
    return {"with_timestamp": found, "keys": used_keys, "unparsed": unparsed, "total": len(dataset.samples)}


# --------------------------------------------------------------------------- sidecar
def _load_parquet(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(str(path))
        return table.to_pylist()
    except ImportError:
        pass
    try:
        import pandas as pd  # type: ignore

        return pd.read_parquet(path).to_dict(orient="records")
    except ImportError:
        raise ImportError(
            f"Reading {path.name} requires pyarrow or pandas (pip install pyarrow); or export the manifest as CSV/JSON."
        ) from None


def _load_sidecar(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return _load_parquet(path)
    if suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            rows = []
            for key, value in data.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("__key__", key)
                    rows.append(row)
            return rows
        raise ValueError(f"{path}: unsupported JSON structure for metadata sidecar")
    # CSV / TSV
    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter=delimiter)]


def _sample_keys(sample: Sample, match: str) -> List[str]:
    keys: List[str] = []
    p = Path(sample.uri)
    if match == "id":
        keys.append(str(sample.native_id))
    elif match == "stem":
        keys.append(p.stem)
    elif match == "path":
        keys.append(sample.uri)
        if sample.path is not None:
            keys.append(str(sample.path))
    else:  # name
        keys.append(p.name)
    return keys


def apply_metadata_sidecar(dataset: Dataset, config: SentinelConfig) -> Dict[str, Any]:
    file = config.get("dataset.metadata.file")
    if not file:
        return {"applied": False}
    path = config.resolve_path(file)
    assert path is not None
    if not path.exists():
        raise FileNotFoundError(f"Metadata sidecar not found: {path}")
    key = str(config.get("dataset.metadata.key") or "file_name")
    match = str(config.get("dataset.metadata.match") or "name").lower()
    rows = _load_sidecar(path)
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        raw = row.get(key, row.get("__key__"))
        if _is_missing(raw):
            continue
        value = str(raw)
        variants = {value, Path(value).name, Path(value).stem} if match != "id" else {value}
        for v in variants:
            index.setdefault(v, row)
    matched = 0
    for s in dataset.samples:
        row = None
        for k in _sample_keys(s, match):
            row = index.get(k)
            if row is not None:
                break
        if row is None:
            continue
        matched += 1
        for k, v in row.items():
            if k in (key, "__key__") or _is_missing(v):
                continue
            if k in ("width", "height") and k in s.metadata:
                continue
            s.metadata[k] = v
    stats = {"applied": True, "file": str(path), "rows": len(rows), "matched_samples": matched, "unmatched_samples": len(dataset.samples) - matched}
    if matched == 0:
        log.warning("metadata sidecar %s matched no samples (key=%s, match=%s)", path, key, match)
    return stats


# --------------------------------------------------------------------------- groups
def resolve_groups(dataset: Dataset, config: SentinelConfig) -> Dict[str, Any]:
    explicit = [str(k) for k in (config.get("dataset.groups.keys") or [])]
    auto = bool(config.get("dataset.groups.auto", True))
    keys: List[str] = list(explicit)
    if auto:
        present = set()
        for s in dataset.samples:
            present.update(k for k in s.metadata.keys() if not k.startswith("_"))
        for k in DEFAULT_GROUP_KEYS:
            if k in present and k not in keys:
                keys.append(k)
    patterns: Dict[str, re.Pattern] = {}
    for name, pattern in (config.get("dataset.groups.from_filename") or {}).items():
        try:
            patterns[str(name)] = re.compile(str(pattern))
        except re.error as exc:
            raise ValueError(f"dataset.groups.from_filename[{name}] is not a valid regex: {exc}") from exc

    resolver = _load_resolver(config.get("dataset.groups.resolver"))
    resolver_keys: set = set()

    counts: Dict[str, int] = {}
    for s in dataset.samples:
        for k in keys:
            v = s.metadata.get(k)
            if not _is_missing(v):
                s.groups[k] = str(v)
                counts[k] = counts.get(k, 0) + 1
        if patterns:
            name = Path(s.uri).name
            for gname, rx in patterns.items():
                m = rx.search(name)
                if m:
                    value = m.groupdict().get("value") or (m.group(1) if m.groups() else m.group(0))
                    if value:
                        s.groups[gname] = str(value)
                        counts[gname] = counts.get(gname, 0) + 1
        if resolver is not None:
            try:
                extra = resolver(s)
            except Exception as exc:
                raise ValueError(f"dataset.groups.resolver failed for sample {s.id}: {type(exc).__name__}: {exc}") from exc
            if extra:
                if not isinstance(extra, dict):
                    raise ValueError(f"dataset.groups.resolver must return a dict or None, got {type(extra).__name__}")
                for gname, value in extra.items():
                    if not _is_missing(value):
                        s.groups[str(gname)] = str(value)
                        counts[str(gname)] = counts.get(str(gname), 0) + 1
                        resolver_keys.add(str(gname))
    return {"keys": keys + list(patterns) + sorted(resolver_keys), "samples_with_group": counts}


def _load_resolver(spec: Any):
    """``"package.module:function"`` -> callable, or None."""
    if not spec:
        return None
    if callable(spec):
        return spec
    text = str(spec)
    if ":" not in text:
        raise ValueError(f"dataset.groups.resolver must look like 'package.module:function', got {text!r}")
    module_name, func_name = text.rsplit(":", 1)
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"dataset.groups.resolver: cannot import {module_name!r}: {exc}") from exc
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ValueError(f"dataset.groups.resolver: {text!r} is not a callable")
    return func


# --------------------------------------------------------------------------- lineage
def resolve_lineage(dataset: Dataset, config: SentinelConfig) -> Dict[str, Any]:
    keys = [str(k) for k in (config.get("dataset.lineage.keys") or [])]
    patterns: List[re.Pattern] = []
    for pattern in config.get("dataset.lineage.from_filename") or []:
        try:
            patterns.append(re.compile(str(pattern)))
        except re.error as exc:
            raise ValueError(f"dataset.lineage.from_filename entry is not a valid regex: {exc}") from exc

    by_name: Dict[str, List[Sample]] = {}
    by_stem: Dict[str, List[Sample]] = {}
    by_uri: Dict[str, Sample] = {}
    by_native: Dict[str, List[Sample]] = {}
    for s in dataset.samples:
        p = Path(s.uri)
        by_name.setdefault(p.name, []).append(s)
        by_stem.setdefault(p.stem, []).append(s)
        by_uri[s.uri] = s
        by_native.setdefault(str(s.native_id), []).append(s)

    def lookup(ref: str) -> List[Sample]:
        ref = str(ref).strip()
        if ref in by_uri:
            return [by_uri[ref]]
        p = Path(ref)
        out = by_name.get(p.name) or by_stem.get(p.stem) or by_native.get(ref) or []
        return out

    relationships: List[Relationship] = []
    unresolved = 0
    from_meta = 0
    from_name = 0
    seen = set()
    for s in dataset.samples:
        refs: List[tuple] = []
        for key in keys:
            v = s.metadata.get(key)
            if _is_missing(v):
                continue
            values = v if isinstance(v, (list, tuple)) else [v]
            for value in values:
                refs.append((key, str(value), Confidence.DETERMINISTIC))
        if patterns:
            stem = Path(s.uri).stem
            for rx in patterns:
                m = rx.search(stem) or rx.search(Path(s.uri).name)
                if m:
                    parent = m.groupdict().get("parent") or (m.group(1) if m.groups() else None)
                    if parent and parent != stem:
                        refs.append((f"filename:{rx.pattern}", parent, Confidence.HEURISTIC))
        for key, value, conf in refs:
            parents = [p for p in lookup(value) if p.id != s.id]
            if not parents:
                unresolved += 1
                continue
            for parent in parents:
                edge = (s.id, parent.id)
                if edge in seen:
                    continue
                seen.add(edge)
                relationships.append(
                    Relationship(kind="derived_from", source_id=s.id, target_id=parent.id, evidence={"key": key, "value": value}, confidence=conf)
                )
                if conf is Confidence.DETERMINISTIC:
                    from_meta += 1
                else:
                    from_name += 1
    dataset.relationships.extend(relationships)
    return {"edges": len(relationships), "from_metadata": from_meta, "from_filename": from_name, "unresolved": unresolved}


def enrich(dataset: Dataset, config: SentinelConfig) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    stats["metadata"] = apply_metadata_sidecar(dataset, config)
    stats["groups"] = resolve_groups(dataset, config)
    stats["lineage"] = resolve_lineage(dataset, config)
    stats["time"] = resolve_timestamps(dataset, config)
    return stats
