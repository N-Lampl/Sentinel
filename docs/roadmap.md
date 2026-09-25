# Roadmap and status

Guiding principle: build only what advances **trust** (fewer false positives,
explainable findings, reproducible results), **repeat use** (runs on every
dataset version or pull request), **scale** (hundreds of thousands of samples)
or, later, **workflow value** a local HTML file cannot provide.

## Two wedges, one engine

| Wedge | Status |
|-------|--------|
| Computer vision: COCO / YOLO / VOC / image-folder split integrity, label validity, distribution diagnostics, metric impact | done |
| Text / LLM data: benchmark contamination, exact and near-duplicate records, label validity, metadata checks | done (`docs/text.md`) |

Both share the policy file, the split integrity violation rate, clusters, fix
plans, baselines / diffs, allowlists, SARIF / DVC / HTML outputs and the
GitHub Action.

## Phase 1: credible v0.1 — done

| Item | Status |
|------|--------|
| Multi-stage similarity: byte hash, pixel hash, dHash candidates, thumbnail verification | done |
| Evidence per match: paths, splits, distance, threshold, relationship type, side-by-side thumbnails / text snippets | done |
| Configurable severity per relationship and per split pair | done (`policy.*`, `split_pair_overrides`) |
| Allowlist for permitted overlaps, with reason and expiry | done |
| Annotation validation (missing images / labels, malformed rows, class ids, degenerate / out-of-bounds boxes, duplicate ids, polygons, unlabeled rate, category lists across split files, keypoints / visibility) | done |
| Baselines and diffs | done (`baseline create`, `diff`, `--baseline`, `fail_on.new_only`) |
| YAML policy file, strict defaults for deterministic problems, warnings for statistical shift | done |
| **Metric impact**: score an evaluation split with and without the flagged samples | done (`--predictions`) |

## Phase 2: useful in CI — done

| Item | Status |
|------|--------|
| GitHub Action with job summary, annotations, artifact upload, baseline input, configurable failure, cache dir, predictions input | done |
| SARIF export with stable fingerprints | done |
| DVC metrics output and `dvc.yaml` example | done (`docs/ci.md`) |
| PyPI release workflow (trusted publishing) | done; first release pending the one-time PyPI setup |
| Scheduled scans | documented (same job on a cron trigger) |

## Phase 3: scale and depth — done for the local tool

| Item | Status |
|------|--------|
| Incremental scanning: cache keyed by path / size / mtime / version / mode | done |
| Fast mode (reduced JPEG decode) | done |
| Manifests: CSV / TSV / JSON / JSONL / YAML / Parquet, COCO fields, file-name patterns, `--group-by`, Python resolver plugin | done |
| Relationship graph and cluster reporting | done |
| Source lineage: declared `derived_from`, file-name patterns, tiles / halves / centre crops by content | done; arbitrary crops by content are out of scope for hashing |
| Embedding-based re-ranking (candidate pairs only, optional extra, off by default) | done for images: provider interface, `builtin` numpy descriptor, `torchvision` ResNet-18 (experimental), entry-point plugins |
| Scale benchmark and memory profile | done: see `docs/performance.md` |

## Phase 4: higher-value support — done except video

| Item | Status |
|------|--------|
| JS divergence for classes, Wasserstein for image size / aspect / box size / objects per image, category coverage, rare class / split combinations | done |
| Metadata-conditioned distributions (class mix by camera / location / session) | done (`conditioned_class_mix`) |
| Annotation consistency triage phrased as review, not correctness | done |
| Task support: detection, classification, segmentation structural checks, keypoints / pose | done |
| Text: contamination, duplicates, labels | done |
| Text: paraphrase / semantic contamination via sentence embeddings | not started: same provider interface as images; add when n-gram results prove useful |
| Object tracking, video decoding | not started: needs a video adapter (see `adapters.md`) and evidence of demand |

## Phase 5: monetise after repeat usage — not started

Hosted report history, PR / release gates with approvals, private
metadata-only cloud mode. Nothing in this repository depends on a hosted
service; the JSON report (stable finding ids, clusters, fix plan, diff,
impact) is the interface such a service would consume, and the allowlist
with expiry is the local form of suppressions.

## Not planned until demand is explicit

Labeling platform, dataset storage, model training or inference, annotation
UI, full video decoding, automatic label fixing, enterprise SSO / audit logs,
a generic multimodal platform, a vector-database service.
