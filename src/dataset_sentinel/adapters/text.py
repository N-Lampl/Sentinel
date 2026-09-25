"""Text adapter: JSONL / JSON / CSV / TSV / Parquet / plain-text records per split.

Built for LLM training and evaluation data: a training corpus or SFT set on
one side, benchmark / eval items on the other.

Configuration::

    dataset:
      format: text
      splits:
        train: data/sft.jsonl          # a file, or a directory of files
        test: benchmarks/mmlu.jsonl
      text:
        fields: [question, choices]    # concatenated; auto-detected when null
        id_field: null                 # auto: id, _id, idx, uid, ...; else the record index
        label_field: null              # auto: label, target, category, class
        lowercase: true
        strip_punctuation: true

Every record becomes one :class:`Sample` (``modality="text"``, no file path)
whose ``metadata["text"]`` holds the joined text, plus every other scalar
field as metadata (so ``source``, ``session_id``, ``captured_at`` and the
like feed the group / lineage / time checks). A label field becomes one
:class:`Annotation`. Chat-style ``messages`` lists are flattened to
``role: content`` lines.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..config import split_role
from ..model import Annotation, Dataset, Finding, Sample
from ..registry import adapters
from .base import DatasetAdapter, LoadResult

log = logging.getLogger(__name__)

TEXT_FIELD_CANDIDATES = ("text", "content", "prompt", "question", "input", "instruction", "document", "body", "sentence", "query", "messages")
ID_FIELD_CANDIDATES = ("id", "_id", "idx", "uid", "sample_id", "example_id", "doc_id")
LABEL_FIELD_CANDIDATES = ("label", "target", "category", "class")
RECORD_EXTENSIONS = {".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet", ".pq", ".txt"}


def _read_records(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    yield {"__malformed__": line[:200]}
                    continue
                yield rec if isinstance(rec, dict) else {"text": rec}
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("data", "rows", "examples", "records", "items", "samples"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, dict):
            data = [{"id": k, **(v if isinstance(v, dict) else {"text": v})} for k, v in data.items()]
        for rec in data:
            yield rec if isinstance(rec, dict) else {"text": rec}
    elif suffix in {".csv", ".tsv"}:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for rec in csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ","):
                yield dict(rec)
    elif suffix in {".parquet", ".pq"}:
        from ..enrich import _load_parquet

        for rec in _load_parquet(path):
            yield rec
    elif suffix == ".txt":
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    yield {"text": line}
    else:
        raise ValueError(f"{path}: unsupported text record format")


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and not (isinstance(value, str) and len(value) > 500)


def _join_text(rec: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    for f in fields:
        v = rec.get(f)
        if v is None:
            continue
        if f == "messages" and isinstance(v, list):
            for m in v:
                if isinstance(m, dict):
                    parts.append(f"{m.get('role', '')}: {m.get('content', '')}".strip())
                else:
                    parts.append(str(m))
        elif isinstance(v, (list, tuple)):
            parts.append("\n".join(str(x) for x in v))
        elif isinstance(v, dict):
            parts.append(json.dumps(v, ensure_ascii=False, sort_keys=True))
        else:
            parts.append(str(v))
    return "\n".join(p for p in parts if p)


@adapters.register("text")
class TextAdapter(DatasetAdapter):
    name = "text"
    modality = "text"
    description = "Text records per split (JSONL/JSON/CSV/TSV/Parquet/txt): LLM training, SFT and eval data"

    @classmethod
    def detect(cls, root: Path) -> Optional[Dict[str, Any]]:
        splits: Dict[str, str] = {}
        for p in sorted(root.iterdir()) if root.is_dir() else []:
            if p.suffix.lower() in RECORD_EXTENSIONS and p.is_file() and split_role(p.stem):
                if p.suffix.lower() == ".json":
                    try:
                        head = p.read_text(encoding="utf-8")[:200].lstrip()
                    except OSError:
                        continue
                    if not head.startswith("["):
                        continue  # a COCO-style object, not a record list
                splits.setdefault(p.stem, p.name)
        if len(splits) >= 2:
            return {"format": "text", "splits": splits}
        return None

    def load(self) -> LoadResult:
        splits_cfg = self.config.section("dataset.splits")
        if not splits_cfg:
            detected = self.detect(self.root)
            if not detected:
                raise ValueError(f"No text split files found under {self.root}. Set dataset.splits (e.g. train: train.jsonl).")
            splits_cfg = detected["splits"]
        tcfg = self.config.section("dataset.text")
        fields_cfg = tcfg.get("fields")
        if isinstance(fields_cfg, str):
            fields_cfg = [fields_cfg]
        id_field = tcfg.get("id_field")
        label_field = tcfg.get("label_field")
        limit = self.max_samples()

        samples: List[Sample] = []
        findings: List[Finding] = []
        categories: Dict[Any, str] = {}
        stats: Dict[str, Any] = {"splits": {}}
        source: Dict[str, Any] = {"format": "text", "root": str(self.root), "splits": {}, "text_fields": {}}

        for split_name, spec in splits_cfg.items():
            target = self.resolve(spec if not isinstance(spec, dict) else spec.get("file") or spec.get("path"))
            assert target is not None
            if not target.exists():
                raise FileNotFoundError(f"Text split {split_name!r} not found: {target}")
            files = sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in RECORD_EXTENSIONS) if target.is_dir() else [target]
            source["splits"][split_name] = [str(f) for f in files[:50]]
            split_samples: List[Sample] = []
            malformed = 0
            chosen_fields: Optional[List[str]] = list(fields_cfg) if fields_cfg else None
            chosen_label: Optional[str] = label_field
            chosen_id: Optional[str] = id_field
            for file in files:
                for index, rec in enumerate(_read_records(file), start=1):
                    if "__malformed__" in rec:
                        malformed += 1
                        continue
                    if chosen_fields is None:
                        chosen_fields = [f for f in TEXT_FIELD_CANDIDATES if f in rec][:1] or [next((k for k, v in rec.items() if isinstance(v, str)), "text")]
                    if chosen_label is None and label_field is None:
                        chosen_label = next((f for f in LABEL_FIELD_CANDIDATES if f in rec), "")
                    if chosen_id is None and id_field is None:
                        chosen_id = next((f for f in ID_FIELD_CANDIDATES if f in rec), "")
                    text = _join_text(rec, chosen_fields)
                    native = rec.get(chosen_id) if chosen_id else None
                    native_id = str(native) if native not in (None, "") else f"{file.stem}:{index}"
                    sid = f"{split_name}:{native_id}"
                    uri = f"{self.display_uri(file, self.root)}:{index}"
                    # underscore keys are internal: never picked up as group / lineage / time metadata
                    metadata: Dict[str, Any] = {"text": text, "n_chars": len(text), "_file": str(file), "_record": index}
                    for k, v in rec.items():
                        if k in chosen_fields or k == chosen_id or k == chosen_label:
                            continue
                        if _scalar(v):
                            metadata[k] = v
                    sample = Sample(id=sid, split=split_name, uri=uri, path=None, modality="text", native_id=native_id, metadata=metadata)
                    if chosen_label and rec.get(chosen_label) not in (None, ""):
                        label = rec[chosen_label]
                        label_str = str(label) if not isinstance(label, (list, dict)) else json.dumps(label, ensure_ascii=False)
                        label_str = label_str[:200]
                        categories.setdefault(label_str, label_str)
                        sample.annotations.append(Annotation(id=f"{sid}#label", sample_id=sid, category=label_str, category_id=label_str))
                    split_samples.append(sample)
                    if limit and len(split_samples) >= limit:
                        break
                if limit and len(split_samples) >= limit:
                    break
            source["text_fields"][split_name] = {"text": chosen_fields, "id": chosen_id or None, "label": chosen_label or None}
            samples.extend(split_samples)
            stats["splits"][split_name] = {"records": len(split_samples), "malformed": malformed, "files": len(files)}
            if malformed:
                findings.append(self._malformed(split_name, malformed))

        dataset = Dataset(
            name=self.config.get("dataset.name") or self.root.name or "dataset",
            samples=samples,
            splits=list(splits_cfg.keys()),
            modality="text",
            categories=categories,
            source=source,
        )
        return LoadResult(dataset=dataset, findings=findings, stats=stats)

    def _malformed(self, split_name: str, count: int) -> Finding:
        from ..model import Confidence, Severity

        return Finding(
            detector="adapter:text",
            kind="malformed_record",
            title=f"{count} unparsable records skipped in split {split_name}",
            severity=self.config.severity("policy.labels.malformed_label", Severity.ERROR) or Severity.INFO,
            confidence=Confidence.DETERMINISTIC,
            policy=self.config.policy_label("policy.labels.malformed_label"),
            policy_description="Every record must be valid JSON / CSV.",
            message=f"{count} lines in split {split_name} could not be parsed and were skipped.",
            remediation="Fix or remove the malformed lines.",
            splits=[split_name],
            evidence={"count": count},
        )


__all__ = ["TextAdapter", "Tuple"]
