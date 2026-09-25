# Roadmap and status

Guiding principle: build only what advances **trust** (fewer false positives,
explainable findings, reproducible results), **repeat use** (runs on every
dataset version or pull request), **scale** (hundreds of thousands of samples)
or, later, **workflow value** a local HTML file cannot provide. Modalities
beyond images, a vector database, a hosted platform and enterprise features
wait until the core scan is used consistently.

## Phase 1: credible v0.1 — done

| Item | Status |
|------|--------|
| Multi-stage similarity: byte hash, pixel hash, dHash candidates, thumbnail verification | done |
| Evidence per match: paths, splits, distance, threshold, relationship type, side-by-side thumbnails | done |
| Configurable severity per relationship and per split pair | done (`policy.*`, `split_pair_overrides`) |
| Allowlist for permitted overlaps, with reason and expiry | done |
| Annotation validation (missing images / labels, malformed rows, class ids, degenerate / out-of-bounds boxes, duplicate ids, polygons, unlabeled rate, category lists across split files) | done; keypoint / visibility validation later |
| Baselines and diffs | done (`baseline create`, `diff`, `--baseline`, `fail_on.new_only`) |
| YAML policy file, strict defaults for deterministic problems, warnings for statistical shift | done |

## Phase 2: useful in CI — done

| Item | Status |
|------|--------|
| GitHub Action with job summary, annotations, artifact upload, baseline input, configurable failure, cache dir | done |
| SARIF export with stable fingerprints | done |
| DVC metrics output and `dvc.yaml` example | done (`docs/ci.md`) |
| Scheduled scans | documented (same job on a cron trigger) |

## Phase 3: scale and depth — partly done

| Item | Status |
|------|--------|
| Incremental scanning: cache keyed by path / size / mtime / version / mode | done |
| Fast mode (reduced JPEG decode) | done |
| Manifests: CSV / Parquet / COCO fields / file-name patterns / `--group-by` | done; Python-plugin group resolvers via adapters or detectors |
| Relationship graph and cluster reporting | done (cross-split clusters with combined evidence and one recommendation) |
| Source lineage: declared `derived_from`, file-name patterns, tiles / halves / centre crops by content | done; arbitrary crops by content not done |
| Embedding-based reranking (optional extra, candidate pairs only) | not started, by design: only after hash-based results prove stable |
| Scale benchmark beyond 5k images, memory profile for 200k+ | to do |

## Phase 4: higher-value CV support — partly done

| Item | Status |
|------|--------|
| JS divergence for classes, Wasserstein for image size / aspect / box size / objects per image, category coverage | done |
| Metadata-conditioned distributions (class mix by camera / location / session) | to do |
| Annotation consistency triage phrased as review, not correctness | done |
| Task support: detection (P0), classification (P1), segmentation structural checks (P1) | done; keypoints / pose (P2), tracking / video (P2) to do |

## Phase 5: monetise after repeat usage — not started

Hosted report history, PR / release gates with approvals and suppressions
with expiry, private metadata-only cloud mode. Nothing in this repository
depends on a hosted service; the JSON report is the interface such a service
would consume.

## Not planned until demand is explicit

Labeling platform, dataset storage, model training or inference, annotation
UI, full video decoding, automatic label fixing, enterprise SSO / audit logs,
a generic multimodal platform, a vector-database service.
