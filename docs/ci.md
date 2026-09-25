# Dataset CI

Dataset Sentinel is most valuable when it runs on every dataset change and
fails only on regressions. This page shows the pieces: baselines, the diff
command, allowlists, the GitHub Action, SARIF upload and DVC metrics.

## 1. Approve a baseline

```bash
sentinel baseline create ./data -o .sentinel/baseline.json
git add .sentinel/baseline.json
```

`baseline create` runs a normal scan with all fail conditions disabled and
stores the JSON report. Re-create it whenever the team accepts the current
state (for example after a clean-up).

## 2. Fail only on regressions

```bash
sentinel diff ./data --baseline .sentinel/baseline.json --fail-on new-error
```

`diff` marks every finding as **new** or **known** (finding ids are stable
hashes of detector, kind, involved samples and title). The run fails when:

* a **new** finding at or above `--fail-on` severity appears (`new-error`
  default; `new-warning`, `new-info`);
* the split integrity violation rate exceeds the baseline's rate (use
  `--max-violation-rate` to allow a higher absolute rate);
* a detector crashed (`fail_on.detector_errors`).

The console, Markdown, HTML and JSON outputs all carry the *changes since
baseline*: new / resolved / known findings, severity deltas, split-size deltas,
classes that disappeared from or appeared in a split, and new cross-split
groups.

## 3. Allowlist known, permitted findings

```yaml
# sentinel.yaml
allowlist:
  - id: 2cc9116ec30d
    reason: "duplicate kept on purpose for the calibration set"
  - kind: near_duplicate_within_split
    reason: "video frames inside train are expected to be similar"
  - samples: ["test/calib_*.jpg", "train/calib_*.jpg"]
    reason: "calibration images appear in every split by design"
    expires: 2027-01-01
```

Suppressed findings stay visible (tagged *suppressed*), drop to `info` and no
longer count toward the violation rate. Expired and unused entries are reported
so the allowlist does not rot.

## 4. GitHub Action

```yaml
name: Dataset integrity
on:
  pull_request:
    paths: ["data/**", "sentinel.yaml", ".sentinel/baseline.json"]

jobs:
  sentinel:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write        # only for the SARIF upload
    steps:
      - uses: actions/checkout@v4
      - uses: actions/cache@v4
        with:
          path: .sentinel-cache
          key: sentinel-${{ hashFiles('data/**') }}
          restore-keys: sentinel-
      - uses: N-Lampl/Sentinel@main       # pin to a tag once releases exist
        id: sentinel
        with:
          path: data
          baseline: .sentinel/baseline.json
          new-only: "true"
          fail-on: error
          sarif: "true"
          fix-plan: "true"
      - uses: github/codeql-action/upload-sarif@v3
        if: always() && steps.sentinel.outputs.sarif-report != ''
        with:
          sarif_file: ${{ steps.sentinel.outputs.sarif-report }}
```

What the action does: installs the package from its own checkout (set the
`version` input to a pip specifier such as `==0.1.0` once the package is on
PyPI), runs the scan, appends the
Markdown summary to the job summary, prints workflow annotations
(`::error` / `::warning`, with file paths when the dataset lives in the
repository), uploads the report directory as an artifact and fails the job
according to the policy. Outputs: `passed`, `violation-rate`, `errors`,
`new-findings`, `json-report`, `html-report`, `sarif-report`.

A scheduled scan for datasets outside pull requests is the same job with
`on: schedule: - cron: "0 6 * * 1"`.

## 5. SARIF

`--sarif report.sarif` writes SARIF 2.1.0. Levels: error -> `error`,
warning -> `warning`, info -> `note`. Every result has a
`partialFingerprints` entry equal to the finding id, so repeated uploads do
not create duplicate alerts, and one location per involved sample (relative
to the dataset root, `uriBaseId: DATASET`). Suppressed findings carry a SARIF
`suppressions` entry with the allowlist reason.

## 6. DVC metrics

```bash
sentinel scan ./data --dvc metrics/dataset-integrity.yaml
```

```yaml
integrity:
  passed: false
  split_integrity_violation_rate: 0.0018
  strict_violation_rate: 0.0018
  violating_samples: 29
  cross_split_clusters: 7
  exact_duplicate_groups: 4
  near_duplicate_groups: 3
  invalid_annotation_findings: 0
  errors: 9
  samples_train: 12840
  samples_val: 1600
  samples_test: 1600
```

Example `dvc.yaml` stage:

```yaml
stages:
  dataset-integrity:
    cmd: sentinel scan data --dvc metrics/dataset-integrity.yaml --json reports/integrity.json --fail-on none
    deps: [data, sentinel.yaml]
    metrics:
      - metrics/dataset-integrity.yaml: {cache: false}
```

`dvc metrics diff` then shows how the integrity numbers moved between dataset
versions.

## 7. Fix plan

`--fix-plan plan.json` (or `.csv`) exports one action per sample in a
violating group: `remove` from its split, with `move_to` naming the split the
rest of the group is kept in (the earliest split in `policy.split_order`,
normally `train`). The plan is a suggestion; Sentinel never modifies files.
