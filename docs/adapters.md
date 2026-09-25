# Adapters and modalities

An adapter reads one on-disk dataset format and produces the modality-agnostic
`Dataset` model. Detectors are written against that model, so a new format or
modality gets every applicable check for free.

## Status

| Modality | Formats | Status |
|----------|---------|--------|
| image | COCO, YOLO / Ultralytics, Pascal VOC, image-folder | **implemented (v0.1)** |
| text | JSONL / JSON / CSV / TSV / Parquet / txt records per split (LLM training, SFT and eval data); see `text.md` | **implemented (v0.1)**: contamination, exact / near duplicates, label validity, metadata checks |
| video | frame / clip manifests, per-video annotation files | *planned, not implemented* |
| tabular | CSV / Parquet with split column or split files | *planned, not implemented* |
| audio | manifests of audio files with transcripts / labels | *planned, not implemented* |
| time series | windows over one or more series with timestamps | *planned, not implemented* |
| multimodal | paired image + text, video + audio, ... | *planned, not implemented* |

Sentinel does not ship or advertise any of the planned adapters. The rest of
this page describes the interface they will implement so that contributors can
build them, and so that the current design does not paint itself into a corner.

## The adapter interface (implemented)

```python
from dataset_sentinel.adapters.base import DatasetAdapter, LoadResult
from dataset_sentinel.registry import adapters

@adapters.register("my-format")
class MyAdapter(DatasetAdapter):
    name = "my-format"
    modality = "image"                       # a detector runs only if it supports this modality
    description = "..."

    @classmethod
    def detect(cls, root: Path) -> dict | None:
        """Return a partial `dataset` config when root looks like this format."""

    def load(self) -> LoadResult:
        """Read the dataset described by self.config and return Dataset + load findings."""
```

Rules an adapter follows:

1. One `Sample` per unit of data with `id` unique across the dataset
   (convention: `f"{split}:{native id or relative path}"`), `split`, `uri`
   (display path, relative to the root when possible), `path` (file on disk if
   there is one), `modality`.
2. Labels become `Annotation` objects. Boxes go in `bbox` as absolute pixels
   when known; format-specific representations go in `attributes`
   (`bbox_norm`, `polygon_norm`, ...). Category names are resolved when the
   format declares them; unknown ids stay in `category_id` with `category=None`
   so the label validity detector can report them.
3. Everything else the format knows about a sample goes into `metadata`
   (`width`, `height`, timestamps, licence, custom fields). The enrichment step
   turns well-known keys into groups / lineage / timestamps.
4. Structural problems found while reading (orphan annotations, duplicate ids,
   malformed entries) are returned as `Finding` objects with
   `detector="adapter:<name>"`. Semantic checks belong in detectors.
5. Adapters do not compute fingerprints and do not run checks.

## What a new modality needs

Each modality provides three things. Only the first is strictly required to get
the metadata-based detectors (group overlap, lineage, temporal, distribution);
the other two unlock the content-based ones.

### 1. An adapter (required)

As above, with `modality` set (`"video"`, `"tabular"`, `"text"`, `"audio"`,
`"timeseries"`, `"multimodal"`). Samples may have no file path (a row of a
table, a line of a JSONL file); put the payload or a reference to it in
`metadata`.

### 2. A fingerprinter (for exact / near-duplicate / derivative checks)

The image fingerprinter lives in `fingerprints/image.py` and produces
content hashes plus perceptual hashes. Other modalities provide the same
shape: a per-sample record with

* a **content hash** (exact identity: file bytes, canonicalised row, normalised text);
* one or more **similarity codes** for near-duplicate search (a 64-bit code
  searchable with `HammingIndex`, or a dense vector for an ANN index), and
* an optional **verification payload** used to confirm hash candidates
  (the image adapter uses a 16x16 thumbnail; text could use a token multiset).

Expected shapes per modality (design intent, not implemented):

| Modality | Content hash | Similarity code | Derivative signals |
|----------|--------------|-----------------|--------------------|
| video | hash of file; per-frame hashes | per-clip sequence of frame dHashes / a temporal hash | sub-clips, re-encodes, frame-rate changes, flips |
| tabular | hash of canonicalised row | MinHash / SimHash over feature values | rows differing only in a derived column, scaled copies |
| text (implemented) | SHA-256 of normalised text | MinHash over word 2-gram shingles + LSH, word 8-gram containment | reformatting, light edits, items embedded in longer documents; paraphrases need embeddings (not implemented) |
| audio | hash of decoded PCM | chromaprint-style landmark hash / spectrogram dHash | resamples, re-encodes, trims, gain changes |
| time series | hash of the window values | SAX / piecewise-aggregate hash | overlapping windows (window leakage), resampled copies |
| multimodal | per-component hashes | per-component codes | any component shared across splits |

### 3. Modality-specific detectors (optional)

Detectors declare `modalities`. `("*",)` means metadata-only detectors that
work everywhere (`group_overlap`, `lineage`, `temporal`, `distribution`).
Content detectors (`exact_duplicate`, `near_duplicate`, `derivative`,
`consistency`, `label_validity`) are image-only today; a modality contributor
either generalises them over a common fingerprint interface or adds a sibling
detector (e.g. `text_near_duplicate`). Semantic (embedding-based) near-duplicate
detection is deliberately left to such modality detectors, since a
"modality-specific detector" is what the product concept requires before a
near-duplicate claim can be made for that modality.

## Registering a third-party adapter

```toml
[project.entry-points."dataset_sentinel.adapters"]
my-format = "my_package.adapters:MyAdapter"
```

`sentinel plugins` lists it once the package is installed; `--format my-format`
selects it; `detect()` lets `sentinel scan <path>` pick it automatically.
