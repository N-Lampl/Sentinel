# Changelog

## 0.1.0 (unreleased)

First release. Computer-vision focus.

### Detection
- Modality-agnostic data model: samples, annotations, splits, metadata, groups, relationships, integrity policies.
- Adapters: COCO (one JSON per split, auto-detection, image directory inference, category-list consistency across split files), YOLO / Ultralytics (`data.yaml`, directory or `.txt` lists, polygon and keypoint lines), Pascal VOC (`Annotations/`, `JPEGImages/`, `ImageSets/Main`), image-folder (optional class sub-directories).
- Enrichment: CSV / TSV / JSON / JSONL / YAML / Parquet manifests, entity / source / batch group resolution (well-known keys, explicit keys, file-name patterns, Python resolver), lineage edges from metadata and file-name patterns, timestamp parsing.
- Image fingerprints: file SHA-256, pixel BLAKE2b, 64-bit dHash, dHash of all 8 dihedral transforms and of 10 regular sub-regions (tiles, halves, centre crops; hashed from a 128px intermediate so they agree with the crop's own hash within 6 bits), 16x16 verification thumbnail, blank detection; symmetric under rotation; threaded computation with batched sqlite cache lookups keyed by path / size / mtime / version / mode; `fast` mode with reduced-resolution JPEG decoding.
- Scale: crop search restricted to cross-split pairs with flat regions skipped and a size guard; vectorised Hamming index build. 30k images: 132 s cold, 41 s cached, 478 MB peak (see `docs/performance.md`).
- Multi-index Hamming search (exact recall for the configured threshold) and union-find grouping; every hash candidate verified by thumbnail correlation.
- Detectors: `exact_duplicate`, `near_duplicate` (optional embedding re-ranking of hash candidates: provider interface, numpy `builtin` descriptor, experimental `torchvision` ResNet-18 extra, entry-point plugins), `derivative` (flips, rotations, transposes, tiles, crops), `group_overlap` (metadata, manifests, file-name patterns, Python resolver plugin), `lineage`, `temporal`, `label_validity` (incl. COCO keypoints and YOLO pose, unlabeled rate, label files without valid annotations), `distribution` (JS divergence, Wasserstein distances, unseen and rare classes, co-occurrence, metadata-conditioned class mix, per-class truncated boxes and size shift), `consistency` (review triage).

### Metric impact
- `--predictions SPLIT=PATH` / `evaluation.predictions`: scores an evaluation split with and without the flagged samples (COCO-style mAP@[.5:.95] / AP50 / AP75 with per-class deltas, or accuracy / macro-F1) from COCO results JSON, YOLO txt predictions or classification CSV; reported as `impact` in JSON, a tile and section in HTML, and lines in the console and Markdown summaries. Action input `predictions`.

### Policy and CI workflow
- Split integrity violation rate with per-split, per-detector and per-confidence breakdowns plus a strict rate; cross-split clusters that unify all evidence per connected set of samples.
- Baselines (`sentinel baseline create`), diffs (`sentinel diff`, `--baseline`, `fail_on.new_only`): findings marked new / known, changes since baseline (rate, severities, split sizes, classes, new cross-split groups).
- Allowlist with reasons and expiry (by finding id, group, kind, detector or sample globs); suppressed findings stay visible and leave the rate.
- Split-pair severity overrides (`policy.split_pair_overrides`).
- Fix plan (`--fix-plan plan.json|csv`) with per-sample remove / move actions.
- Fail conditions: severity, violation rate, detector errors, new-only mode.
- Config validation with typo suggestions.

### Outputs and integrations
- Reporters: console, JSON (schema version 1), self-contained HTML (filters incl. new / suppressed, thumbnails, clusters, baseline diff, class distribution, light / dark), Markdown (GitHub job summary), SARIF 2.1.0 (stable fingerprints, suppressions), DVC metrics.
- CLI (`sentinel scan / baseline create / diff / init / plugins / version`) with config discovery, `--set` overrides, `--fast`, GitHub workflow annotations and CI exit codes.
- Composite GitHub Action (`action.yml`) with job summary, annotations, artifact upload, baseline / new-only inputs, SARIF / fix-plan / DVC outputs.
- Plugin registries with entry points for adapters, detectors and reporters.
- Test-suite with synthetic COCO / YOLO / VOC datasets and planted problems; 5,000-image scale check.

### Not included
- Video, tabular, text, audio, time-series and multimodal adapters (interface documented in `docs/adapters.md`).
- Arbitrary (non-grid) crop detection by content and embedding-based semantic similarity.
- Hosted reporting platform.
