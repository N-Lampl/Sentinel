# Architecture

Dataset Sentinel is a pipeline of small, replaceable parts:

```
 on-disk dataset ──► Adapter ──► Dataset (samples, annotations, metadata)
                                    │
                                    ▼
                               Enrichment  (manifest metadata, groups, lineage, timestamps)
                                    │
                                    ▼
                            Fingerprinting (per modality; images: hashes + thumbnails, cached)
                                    │
                                    ▼
                     Detectors ──► Findings + RelationshipGroups
                                    │
                                    ▼
                        Post-processing  (split-pair overrides, allowlist, baseline, clusters)
                                    │
                                    ▼
                          Metrics + fail conditions + fix plan + diff
                                    │
                                    ▼
          Reporters (console, JSON, HTML, Markdown, SARIF, DVC) · GitHub annotations
```

## Data model (`dataset_sentinel/model.py`)

| Type | Meaning |
|------|---------|
| `Sample` | one unit of data; `id`, `split`, `uri`, `path`, `modality`, `annotations`, `metadata`, `groups` |
| `Annotation` | one label / target; category, optional `BBox` (absolute pixels), segmentation, `attributes` (format-specific extras such as normalised YOLO boxes) |
| `Relationship` | directed edge between samples (`derived_from` today) with evidence and confidence |
| `Dataset` | samples + ordered splits + categories + relationships + source description |
| `Finding` | a reported problem: detector, kind, severity, **confidence**, **policy**, message, **evidence**, **samples/splits**, **remediation**, optional `group_id`, `new` (baseline), `suppressed` (allowlist); the `id` is a stable hash |
| `RelationshipGroup` | a set of samples tied together by one relationship; `violates_policy` drives the metric |
| `Metrics`, `Report` | aggregated results, clusters, fix plan and the serialisable report |

Splits are plain strings. `config.split_role()` maps common names
(`train2017`, `valid`, `holdout`, ...) onto the roles `train` / `val` / `test`
used by the temporal and distribution detectors, the split-pair overrides and
the fix plan.

## Modules

| Module | Responsibility |
|--------|----------------|
| `config.py` | defaults, deep-merge, YAML/JSON loading, typed accessors, policy -> severity, unknown-key validation |
| `registry.py` | plugin registries for adapters, detectors, reporters (built-ins + entry points) |
| `adapters/` | `coco.py`, `yolo.py`, `voc.py`, `image_folder.py`; `base.py` defines the interface |
| `enrich.py` | manifest metadata (CSV/TSV/JSON/JSONL/Parquet), entity/group resolution, lineage edges, timestamp parsing |
| `fingerprints/` | `image.py` (hashes, thumbnails, region hashes), `index.py` (multi-index Hamming search, union-find), `store.py` (parallel computation + sqlite cache keyed by path/size/mtime/version/mode) |
| `detectors/` | one module per detector; `base.py` defines `Detector`, `DetectorContext`, `DetectorResult`, `FindingCap` |
| `postprocess.py` | split-pair severity overrides, allowlist suppression, cross-split clusters |
| `baseline.py`, `diff.py` | load a previous JSON report, mark new / known findings, compute the changes |
| `metrics.py` | violation rate, breakdowns, fail conditions (severity, rate, detector errors, new-only) |
| `fixplan.py` | per-sample remove / move suggestions for every violating group |
| `engine.py` | orchestration (`scan()`), format auto-detection |
| `reporters/` | console, JSON, HTML, Markdown, SARIF, DVC |
| `cli.py` | `sentinel scan / baseline create / diff / init / plugins / version`, GitHub annotations |

## Detector contract

```python
class Detector:
    name: str                      # registry name and Finding.detector
    description: str
    modalities: tuple[str, ...]    # ("image",), ("*",) ...
    needs_fingerprints: bool       # engine computes fingerprints up-front when True

    def run(self, ctx: DetectorContext) -> DetectorResult: ...
```

`DetectorContext` gives access to the dataset, the config, the shared
`FingerprintStore` (computed once for all detectors) and a `shared` dict for
passing intermediate results between detectors (the near-duplicate detector
publishes its verified pairs; the derivative and consistency detectors reuse
them).

Detectors never mutate the dataset. They return findings and relationship
groups; the engine sorts findings, applies post-processing, computes metrics
and evaluates the fail conditions. A detector that raises is reported as
`status: error` and, by default, fails the run (`fail_on.detector_errors`),
because a silent gap in coverage is worse than a failed job.

## Relationship groups, clusters and the metric

Every leakage detector emits one `RelationshipGroup` per connected set of
samples (connected components over its evidence edges). The group's
`violates_policy` flag is set when the group spans more than one split and the
configured policy for that relationship is `error` or `warning`; split-pair
overrides and the allowlist can clear it afterwards. The metric is the size of
the union of violating groups' members divided by the dataset size.

Clusters are connected components over the members of *all* violating groups,
across detectors: a training image, its byte-identical copy in test and every
other image of the same patient form one cluster with all evidence listed. The
fix plan keeps each violating group's members in the earliest split of
`policy.split_order` and proposes to remove (or move) the others.

## Fingerprints (image modality)

For each image: file SHA-256, decoded-pixel BLAKE2b-128 (skipped in fast
mode), size / mode / format, a 64-bit difference hash (dHash) of a 64x64 float
grayscale thumbnail, the dHash of all 8 dihedral transforms, the dHash of 10
regular sub-regions (2x2 tiles, halves, centre crops), a 16x16 thumbnail (256
bytes) for verification, mean / std luma (blank detection). Large images are
box-reduced by an integer factor before the grayscale conversion; the factor
depends on the shorter side only, so rotated copies reduce identically.
Computation runs in a thread pool and is cached in
`<cache>/fingerprints.sqlite` keyed by path, size, mtime, schema version and
mode (an `exact` entry also serves `fast` requests).

Near-duplicate search uses multi-index hashing: a 64-bit code is split into
`threshold + 1` chunks; codes within the threshold must share at least one
chunk, so candidates come from hash buckets and are verified with a vectorised
popcount. Candidate pairs are then verified with the normalised correlation of
the 16x16 thumbnails, which removes most collisions on low-texture images.
Region (crop) candidates use a wider hash threshold and a stricter correlation
because the parent-thumbnail and child-file pipelines differ by a few bits.

## Extending

* **New format**: subclass `DatasetAdapter`, register it (`registry.adapters.register`
  or an entry point), produce `Sample` objects with `split`, `path`,
  `annotations` and `metadata`.
* **New check**: subclass `Detector`, emit findings through `self.finding(...)`
  and groups through `self.group(...)`, register it.
* **New output**: subclass `Reporter`, implement `render(report) -> str`.
* **New modality**: see `adapters.md`.
