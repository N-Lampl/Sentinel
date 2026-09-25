# Dataset Sentinel

[![CI](https://github.com/N-Lampl/Sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/N-Lampl/Sentinel/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Validate ML dataset split integrity before you train.** Dataset Sentinel scans a
dataset, finds samples that leak between train / validation / test splits, checks
that labels are well-formed, and reports distribution problems that make metrics
misleading. It runs locally, needs no account, and produces a JSON report, a
self-contained HTML report, a CI-friendly exit code and, optionally, SARIF for
GitHub code scanning, DVC metrics and a per-sample fix plan.

The first release is focused on **computer vision**: image datasets in **COCO**,
**YOLO (Ultralytics)**, **Pascal VOC** and plain image-folder layouts. The core
is modality-agnostic and built around adapters, so video, tabular, text, audio,
time-series and multimodal support can be added later without changing the
detectors or the report (see [docs/adapters.md](docs/adapters.md) for the
interface; those adapters are **not** part of this release).

```
$ sentinel scan ./my-dataset

Dataset Sentinel 0.1.0
my-dataset  (coco, 106 samples, 109 annotations, 3 splits: train, val, test)

Split integrity violation rate: 11.32%  (12 of 106 samples in 7 violating groups; strict 11.32%; 5 cross-split clusters)
  train              61 samples       5 violating  (8.20%)
  val                22 samples       3 violating  (13.64%)
  test               23 samples       4 violating  (17.39%)

Cross-split clusters: 5
  #a41c…  4 samples: test (2), train (2)
      - 1 exact duplicate group
      - shared patient_id: P002
      -> Move every sample with patient_id in {P002} into one split, then rescan. Keep the 2 samples in train and remove (or move to train) the 2 samples in test (2)

Suggested fix: remove (or move) 7 samples (test: 4, val: 3); use --fix-plan plan.json to export the actions

Findings: 16 errors, 1 warnings, 2 info
  exact_duplicate  (2 error)
    [ERROR] Exact duplicate across splits test/train (2 samples, byte-identical files)  [deterministic; test,train]
  near_duplicate  (1 error)
    [ERROR] Near-duplicate across splits test/train (distance 1, 2 samples)  [high_confidence; test,train]
  derivative  (1 error)
    [ERROR] Transformed copy across splits test/train (flip_horizontal, 2 samples)  [high_confidence; test,train]
  ...
RESULT: FAIL  16 finding(s) at severity error or above (fail_on.severity=error)
Reports: sentinel-report.json, sentinel-report.html
```

## What it detects

| # | Problem | Detector | How | Confidence |
|---|---------|----------|-----|------------|
| 1 | Exact duplicates across splits | `exact_duplicate` | same file path, byte-identical files (SHA-256), pixel-identical decoded images (BLAKE2) | deterministic |
| 2 | Near-duplicates across splits | `near_duplicate` | 64-bit difference hash within a Hamming threshold, every candidate verified by thumbnail correlation; optional embedding re-ranking of candidates | high confidence / heuristic |
| 3 | Entity, source or batch overlap | `group_overlap` | shared `patient_id`, `subject`, `source`, `session`, `device`, `camera`, `batch`, ... from metadata, manifests (CSV / JSON / Parquet), file-name patterns or a Python resolver | deterministic / high confidence |
| 4 | Derivative samples: flips, rotations, transposes, 2x2 tiles, halves, centre crops | `derivative` | dHash of the 8 dihedral transforms and of regular sub-regions, verified by thumbnail correlation | high confidence / heuristic |
| 4 | Derivative samples via lineage (arbitrary crops, augmentations, exports) | `lineage` | `derived_from` / `parent` metadata or file-name patterns | deterministic / heuristic |
| 5 | Time leakage | `temporal` | overlapping capture-time ranges between splits that must be chronological | deterministic (when the policy is enabled) |
| 6 | Invalid, missing, empty or out-of-policy labels | `label_validity` | missing / unreadable / blank images, degenerate or out-of-bounds boxes, unknown categories, malformed YOLO lines, invalid polygons, keypoint counts / visibility / bounds, duplicate annotations, category lists that differ between COCO split files | deterministic |
| 7 | Imbalance and distribution shift | `distribution` | class imbalance ratio, rare class / split combinations, Jensen-Shannon divergence of class distributions, Wasserstein distance of image size / aspect / box size / objects per image, classes unseen in training, co-occurrence anomalies, class mix conditioned on camera / session / location metadata | deterministic / heuristic |
| 8 | Inconsistent labels on related samples (review triage) | `consistency` | identical or near-identical images with different class sets or very different object counts | deterministic / heuristic |

Every finding states the **violated policy**, the **evidence** used, a
**confidence class** (`deterministic`, `high_confidence`, `heuristic`,
`inconclusive`), the **samples and splits** involved, and a **remediation**.
Checks that could not run because metadata is missing (no entity ids, no
timestamps, no lineage) are reported as *inconclusive* rather than silently
skipped. Findings have **stable ids**, so they can be diffed between runs and
allowlisted.

## Installation

```bash
pip install git+https://github.com/N-Lampl/Sentinel
```

Requires Python 3.9+, numpy, Pillow and PyYAML. No cloud upload, no account.
Once the package is published on PyPI, `pip install dataset-sentinel` will work
too. Extras: `[parquet]` for Parquet manifests, `[embeddings]` for the optional
torchvision re-ranking provider, e.g.
`pip install "dataset-sentinel[parquet] @ git+https://github.com/N-Lampl/Sentinel"`.

## Quick start

```bash
# 1. scan a dataset; the format (coco / yolo / voc / image-folder) is auto-detected
sentinel scan path/to/dataset

# 2. open the report
open sentinel-report.html          # JSON in sentinel-report.json

# 3. try it on a demo dataset with planted problems (and demo predictions)
python examples/make_demo_dataset.py demo-dataset
sentinel scan demo-dataset --fix-plan plan.csv --predictions test=demo-dataset/predictions/test.json
```

Exit codes: `0` pass, `1` a fail condition was hit (by default any
error-severity finding, a violation rate above 0, or a detector that crashed),
`2` usage or runtime error.

### Supported layouts

**COCO**: one JSON per split. Auto-detected from `annotations/*.json` or `*.json`
in the root; image directories are inferred (`train2017/`, `images/train/`, ...)
or configured explicitly:

```bash
sentinel scan data --format coco \
  --split train=annotations/instances_train.json:images/train \
  --split val=annotations/instances_val.json:images/val
```

Extra fields on COCO image entries (`date_captured`, `patient_id`, ...) are
picked up as metadata automatically.

**YOLO / Ultralytics**: a `data.yaml` with `train` / `val` / `test` (directories,
`.txt` lists or lists of either), `names` and optional `path`. Labels are found
the same way Ultralytics finds them (`images/` -> `labels/`, `.txt`).

**Pascal VOC**: `Annotations/*.xml`, `JPEGImages/`, `ImageSets/Main/<split>.txt`.

**Image folder**: one directory per split, optional class sub-directories
(`train/cat/*.jpg`) for classification datasets.

## How much does the leak inflate your metric?

Give Sentinel your model's predictions for an evaluation split and it scores
the split three times: on all samples, on the clean samples only, and on the
flagged samples only.

```bash
sentinel scan data --predictions test=runs/val/predictions.json
```

```
Metric impact of the flagged samples:
  test mAP: all 0.712 -> clean 0.634 (+0.078); leaked-only 0.951 on 12 of 100 samples
```

Accepted inputs: COCO results JSON (`[{image_id, category_id, bbox, score}]`),
a directory of YOLO `.txt` predictions (`yolo val save_txt=True save_conf=True`),
or a CSV / JSON of classification predictions. Detection is scored with a
COCO-style mAP@[.5:.95] (plus AP50 / AP75 and per-class deltas), classification
with accuracy and macro-F1. No extra dependencies. The difference between
*all* and *clean* is the number to put in front of whoever owns the dataset.

## Dataset CI: baselines, diffs and allowlists

The scan is most useful when it runs on every dataset change and only fails on
regressions.

```bash
# approve the current state once
sentinel baseline create ./data -o .sentinel/baseline.json

# on every change: fail only on NEW errors or a higher violation rate
sentinel diff ./data --baseline .sentinel/baseline.json --fail-on new-error
```

`diff` (or `scan --baseline FILE --new-only`) answers: did a new cross-split
group appear, did invalid labels increase, did a class disappear from an
evaluation split, did split sizes change, did the leak rate move? New findings
are tagged in every report.

Known, permitted overlaps go into an **allowlist** with a reason and an optional
expiry; matching findings are kept in the report as suppressed, drop to `info`
and leave the violation rate:

```yaml
allowlist:
  - id: 2cc9116ec30d                       # finding id from the report
    reason: "calibration image intentionally present in train and test"
  - samples: ["test/calib_*.jpg", "train/calib_*.jpg"]
    reason: "calibration set appears in every split by design"
    expires: 2027-01-01
```

Split-pair policies keep train/val overlap a warning while train/test stays an
error:

```yaml
policy:
  split_pair_overrides: {"train/val": warning, "train/test": error, "val/test": error}
```

## Configuration

`sentinel init` writes a commented `sentinel.yaml`; `sentinel scan` picks it up
automatically from the dataset root or the current directory. Policies are
severities (`error`, `warning`, `info`, `ignore`; `forbid` / `allow` are
aliases). Unknown keys (typos) are reported with a suggestion.

```yaml
dataset:
  format: coco
  root: .
  metadata:            # optional manifest (CSV/TSV/JSON/JSONL/Parquet) joined on file name
    file: metadata.csv
    key: file_name
  groups:
    auto: true         # use patient_id, source, session, device, ... when present
    keys: [hospital]   # extra entity keys
    from_filename:     # or derive them from the file name
      subject: "^(?P<value>S\\d+)_"
  lineage:
    from_filename: ["^(?P<parent>.+?)_(?:tile|crop|aug)\\d*$"]

policy:
  exact_duplicate: {cross_split: error, within_split: warning}
  near_duplicate:  {cross_split: error, threshold: 6, min_correlation: 0.8}
  derivative:      {cross_split: error, crops: true}
  group_overlap:   {cross_split: error}
  lineage:         {cross_split: error}
  temporal:        {enabled: auto}     # true = splits must be chronological
  labels:
    missing_annotations: warning
    out_of_bounds_bbox: error
    allowed_categories: null
  distribution:
    imbalance_ratio: 20
    shift_threshold: 0.1
    unseen_class: error

fail_on:
  severity: error
  max_violation_rate: 0.0
  detector_errors: true
```

Any key can be overridden on the command line:
`--set policy.near_duplicate.threshold=4 --fail-on warning --disable distribution`.
See [docs/configuration.md](docs/configuration.md) for the full reference.

## The metric: split integrity violation rate

```
violation rate = samples that belong to a relationship group violating split policy
                 -----------------------------------------------------------------
                 all evaluated samples
```

A relationship group is a set of samples tied together by exact identity,
near-duplicate similarity, a transform, a shared entity / source / batch
identifier, a lineage relationship or a temporal window. A group *violates*
policy when it crosses split boundaries and the policy for that relationship is
`error` or `warning`. Samples are counted once even if several detectors flag
them. The report also gives a **strict** rate (deterministic and high-confidence
groups only), per-split and per-detector breakdowns, and **cross-split
clusters**: connected sets of leaking samples with all their evidence combined
into one remediation decision. Label and distribution findings never count
toward the rate. Details in [docs/metric.md](docs/metric.md).

## Outputs

* **Console**: summary, clusters, suggested fix, findings grouped by detector.
* **JSON** (`schema_version: 1`): everything, with stable finding ids.
* **HTML**: single self-contained file (offline, light and dark theme) with
  filters by severity / confidence / detector / split / new / suppressed,
  thumbnails for leakage findings, clusters, baseline diff and class distribution.
* **Markdown**: summary for `$GITHUB_STEP_SUMMARY` or a PR comment.
* **SARIF 2.1.0** (`--sarif`): upload with `github/codeql-action/upload-sarif`
  to see findings in code scanning; results carry stable fingerprints.
* **DVC metrics** (`--dvc metrics.yaml`): flat integrity metrics for
  `dvc metrics diff`.
* **Fix plan** (`--fix-plan plan.json|csv`): per-sample remove / move actions
  that resolve every violating group while keeping training data.

## GitHub Action

```yaml
- uses: N-Lampl/Sentinel@main          # pin to a tag once releases exist
  with:
    path: data
    baseline: .sentinel/baseline.json   # optional: fail only on regressions
    new-only: "true"
    fail-on: error
    sarif: "true"
```

The action installs the package from its own checkout (or a PyPI version via
the `version` input), runs the scan, writes a job summary, emits
workflow annotations for the top findings, uploads the report directory as an
artifact and fails the job when the policy is violated. Outputs: `passed`,
`violation-rate`, `errors`, `new-findings`, `json-report`, `html-report`,
`sarif-report`. See [examples/github-workflow.yml](examples/github-workflow.yml)
and [docs/ci.md](docs/ci.md).

## Python API

```python
from dataset_sentinel import load_config, scan

config = load_config("sentinel.yaml")          # or SentinelConfig({...})
report = scan(config)

print(report.metrics.violation_rate, len(report.clusters))
for finding in report.findings:
    print(finding.severity.value, finding.confidence.value, finding.new, finding.title)

from dataset_sentinel.reporters.html_reporter import HtmlReporter
HtmlReporter(config).write(report, "report.html")
```

## Performance

Fingerprinting is multi-threaded and cached (`--cache DIR`, keyed by path, size
and mtime), so re-scans only touch changed files. Near-duplicate and derivative
search use multi-index hashing, not all-pairs comparison. On a laptop, 30,000
small images scan in about 130 s cold and 40 s from cache at under 500 MB of
memory. For large photos use `--fast` (reduced-resolution JPEG decoding, no
pixel-identity hash): 3-5x faster; byte-identical and perceptual detection are
unaffected. Details and tuning in [docs/performance.md](docs/performance.md).

## Plugins

Adapters, detectors and reporters are registered through entry points
(`dataset_sentinel.adapters`, `dataset_sentinel.detectors`,
`dataset_sentinel.reporters`). `sentinel plugins` lists what is installed.
See [docs/architecture.md](docs/architecture.md) and
[docs/detectors.md](docs/detectors.md).

## Scope of v0.1 and known limitations

Implemented: image datasets (COCO, YOLO, VOC, image-folder), the nine
detectors above, baseline / diff / allowlist workflow, clusters and fix plans,
JSON / HTML / Markdown / SARIF / DVC / console outputs, CLI, GitHub Action,
fingerprint cache, fast mode.

Not implemented (and not claimed): video, tabular, text, audio, time-series and
multimodal adapters; arbitrary (non-grid) crop detection by content (use lineage
metadata or file-name patterns); semantic duplicate detection (the optional
embedding re-ranking only filters hash candidates and is off by default); a
hosted reporting platform. Near-duplicate and transform detection use perceptual
hashes: low-texture or very simple graphics can produce heuristic matches, which
is why every such finding carries its distance, its correlation and a confidence
class, and why the strict rate exists. See [docs/roadmap.md](docs/roadmap.md)
and [docs/performance.md](docs/performance.md).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

Apache-2.0. See [CHANGELOG.md](CHANGELOG.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
