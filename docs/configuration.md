# Configuration reference

`sentinel scan` merges, in order: built-in defaults, `sentinel.yaml`
(`--config`, or auto-discovered as `sentinel.yaml` / `sentinel.yml` /
`.sentinel.yaml` / `sentinel.json` in the dataset root or the current
directory), then command-line flags and `--set key=value` overrides.
`sentinel init [--format coco|yolo|voc|image-folder]` writes a commented
template. Keys that are not part of the schema are reported as warnings with
a suggestion (`unknown config key 'policy.near_dup' (did you mean
'policy.near_duplicate'?)`).

Relative paths in the config are resolved against the config file's directory;
`dataset.root` is the base for dataset paths.

## dataset

| Key | Default | Meaning |
|-----|---------|---------|
| `name` | directory name | display name |
| `format` | auto | `coco`, `yolo`, `voc`, `image-folder` or a plugin adapter name |
| `root` | `.` | dataset root |
| `splits` | `{}` | COCO: `{split: {annotations: file, images: dir}}` or `{split: file}`; VOC: `{split: ImageSets/Main/x.txt}`; image-folder: `{split: dir}`; YOLO: optional overrides of `data.yaml` entries |
| `data` | auto | YOLO `data.yaml` path |
| `metadata.file` | none | CSV / TSV / JSON / JSONL / YAML / Parquet manifest with per-sample columns (a YAML file may wrap the records in a `samples:` section) |
| `metadata.key` | `file_name` | column that identifies the sample |
| `metadata.match` | `name` | how to join: `name` (file name), `stem`, `path` (relative uri), `id` (native id) |
| `groups.auto` | `true` | use well-known entity keys found in metadata |
| `groups.keys` | `[]` | extra metadata keys that identify an entity / source / batch |
| `groups.from_filename` | `{}` | `{group_name: regex}`; the named group `value` (or group 1) is the identifier |
| `groups.resolver` | none | `"package.module:function"`; called with each `Sample`, returns `{group_name: value}` or `None` (custom logic, e.g. a lookup service) |
| `lineage.keys` | `derived_from, parent, parent_id, parent_file, source_image, source_file, original, original_id, original_file, base_image` | metadata keys naming a parent sample |
| `lineage.from_filename` | `[]` | regexes with a named group `parent` (matched on the file stem, then name) |
| `time.keys` | `captured_at, timestamp, date_captured, datetime, capture_time, acquired_at, acquisition_time, time, date, created_at, recorded_at` | metadata keys holding a timestamp |
| `time.format` | auto | `strptime` format when auto-parsing is not enough |
| `text.fields` | auto | text datasets: record field(s) holding the text, concatenated (CLI `--text-field`, repeatable); auto picks `text`, `content`, `prompt`, `question`, `input`, `instruction`, `messages`, ... |
| `text.id_field` / `text.label_field` | auto | record fields for the sample id (`id`, `_id`, `idx`, `uid`, ...) and the label (`label`, `target`, `category`, `class`); CLI `--id-field`, `--label-field` |
| `text.lowercase` / `text.strip_punctuation` | `true` / `true` | normalisation before hashing |

## policy

Values are severities: `error` (alias `forbid`, `fail`), `warning`, `info`
(alias `allow`), `ignore` (alias `off`, `none`, `false`).

| Key | Default |
|-----|---------|
| `split_order` | `[train, val, test]` (roles; names are mapped with aliases such as `valid`, `holdout`, `train2017`) |
| `split_pair_overrides` | `{}`: severity per pair of split roles a leakage group spans, e.g. `{"train/val": warning}`. Applied when every spanned pair has an override; the most severe wins |
| `exact_duplicate.cross_split` / `within_split` | `error` / `warning` |
| `near_duplicate.cross_split` / `within_split` | `error` / `info` |
| `near_duplicate.threshold` | `6` (Hamming bits of 64) |
| `near_duplicate.min_correlation` | `0.8` (16x16 thumbnail correlation required to keep a candidate) |
| `near_duplicate.max_group_findings` | `500` |
| `near_duplicate.embedding.enabled` | `false`: re-rank hash candidates with embeddings (candidate pairs only, never all-pairs) |
| `near_duplicate.embedding.provider` | `builtin` (numpy colour + structure descriptor); `torchvision` (ResNet-18, `pip install "dataset-sentinel[embeddings]"`, experimental); or a plugin |
| `near_duplicate.embedding.threshold` | `10`: wider hash threshold used to generate candidates when enabled |
| `near_duplicate.embedding.min_cosine` | `0.9`: candidates below this cosine similarity are dropped |
| `derivative.cross_split` / `within_split` / `threshold` / `min_correlation` | `error` / `info` / `6` / `0.8` |
| `derivative.crops` | `true` (2x2 tiles, halves, 50% and 75% centre crops) |
| `derivative.crop_threshold` / `crop_min_correlation` | `8` / `0.9` (crop candidates use a slightly wider hash threshold and stricter verification; only cross-split pairs are searched) |
| `derivative.crops_max_images` | `100000`: skip the crop search above this many images and say so in the report (0 = no limit) |
| `contamination.cross_split` | `error` (text: evaluation items contained in training documents) |
| `contamination.ngram` / `min_containment` / `min_shared_ngrams` | `8` / `0.5` / `2` |
| `text_near_duplicate.min_jaccard` / `shingle` / `num_perm` / `bands` | `0.7` / `2` / `128` / `32` (severity comes from `near_duplicate.*`) |
| `group_overlap.cross_split` | `error` |
| `lineage.cross_split` | `error` |
| `temporal.enabled` | `auto` (`true` = chronological splits required, `false` = skip) |
| `temporal.cross_split` | `error` |
| `temporal.min_coverage` | `0.5` |
| `labels.missing_file` / `unreadable_image` / `empty_image` | `error` |
| `labels.blank_image` / `size_mismatch` / `missing_annotations` / `no_valid_annotations` / `missing_label_file` | `warning` |
| `labels.empty_split` / `invalid_bbox` / `out_of_bounds_bbox` / `degenerate_bbox` | `error` |
| `labels.min_bbox_size_px` | `1.0` |
| `labels.unknown_category` / `duplicate_annotation_id` / `orphan_annotation` / `malformed_label` | `error` |
| `labels.duplicate_annotation` / `invalid_segmentation` | `warning` |
| `labels.invalid_keypoints` / `keypoints_out_of_bounds` | `error` / `warning` (COCO keypoints and YOLO pose: count, visibility flags, `num_keypoints`, bounds) |
| `labels.category_mismatch` / `category_set_mismatch` | `error` / `warning` (COCO category lists that differ between split files) |
| `labels.empty_text` / `short_text` / `min_tokens` | `error` / `warning` / `3` (text records) |
| `labels.allowed_categories` | `null` (list of allowed class names) |
| `distribution.imbalance` / `imbalance_ratio` | `warning` / `20` |
| `distribution.shift` / `shift_threshold` | `warning` / `0.1` |
| `distribution.unseen_class` | `error` |
| `distribution.missing_class_in_eval` | `warning` |
| `distribution.image_size_shift` / `bbox_size_shift` / `cooccurrence` | `info` |
| `distribution.min_samples` | `20` |
| `distribution.rare_class` / `min_samples_per_class_per_split` | `warning` / `5` (classes present in a split with fewer images than this; 0 disables) |
| `distribution.conditioned` / `conditioned_min_samples` | `info` / `30` (class mix per metadata group, e.g. camera or session, compared with the rest of its split) |
| `distribution.truncated_boxes` / `class_box_size_shift` / `min_boxes_per_class` | `info` / `info` / `10` (per class: boxes cut off at the image border; median object size differing >= 2x from training) |
| `consistency.conflicting_labels` / `duplicate_registration` | `error` |

## allowlist

A list of entries. Each needs a `reason`; matching findings are kept in the
report as *suppressed*, drop to `info`, and their groups stop counting toward
the violation rate. Entries can match by `id` (finding id), `group` (group id),
`kind`, `detector`, or `samples` (list of globs on sample uris; **every**
sample of the finding must match). `expires: YYYY-MM-DD` retires an entry.
Unused and expired entries are reported.

## evaluation (metric impact)

| Key | Default | Meaning |
|-----|---------|---------|
| `evaluation.predictions` | `{}` | `{split: path}`: model predictions for an evaluation split (COCO results JSON, directory of YOLO `.txt` predictions, or classification CSV / JSON). CLI: `--predictions SPLIT=PATH` (repeatable) |
| `evaluation.task` | `auto` | `detection` (mAP@[.5:.95], AP50, AP75, per-class AP) or `classification` (accuracy, macro-F1); `auto` picks from the predictions and annotations |

The split is scored on all samples, on the clean samples (flagged leaked
samples removed), on the flagged samples only, and on the strict-clean
subset (deterministic and high-confidence flags only). The report's
`impact` section holds the numbers; the difference between *all* and *clean*
is the inflation caused by leakage.

## baseline, detectors, output, fail_on, performance

| Key | Default | Meaning |
|-----|---------|---------|
| `baseline.file` | none | JSON report of a previous run; findings present there are marked known, others `new` |
| `detectors.enabled` | all | list of detector names to run |
| `detectors.disabled` | `[]` | detectors to skip |
| `output.json` / `output.html` / `output.markdown` | `sentinel-report.json` / `.html` / none (CLI) | report paths |
| `output.sarif` / `output.dvc` / `output.fix_plan` | none | SARIF 2.1.0, DVC metrics (YAML/JSON), fix plan (JSON/CSV) |
| `output.thumbnails` | `true` | embed thumbnails for leakage findings in the HTML report |
| `output.max_thumbnails` | `200` | findings that get thumbnails |
| `output.thumbnail_size` | `96` | pixels |
| `output.max_findings_per_kind` | `1000` | cap per finding kind (overflow is summarised, metrics are unaffected) |
| `output.max_findings_html` | `2000` | findings rendered in the HTML report |
| `fail_on.severity` | `error` | lowest severity that fails the run (`none` to disable) |
| `fail_on.max_violation_rate` | `0.0` | fail when the violation rate exceeds this fraction |
| `fail_on.detector_errors` | `true` | a detector that crashes fails the run |
| `fail_on.new_only` | `false` | with a baseline: only new findings fail the severity gate and the rate only fails above the baseline's rate |
| `performance.mode` | `exact` | `fast` = reduced-resolution JPEG decode, no pixel-identity hash |
| `performance.workers` | `0` (auto) | fingerprinting threads |
| `performance.cache` | none | fingerprint cache directory (sqlite) |
| `performance.max_samples_per_split` | none | debugging limit |

## Command line

```
sentinel scan [PATH] [-c sentinel.yaml] [-f coco|yolo|voc|image-folder]
              [--split NAME=SPEC ...] [--data data.yaml]
              [--metadata|--manifest FILE] [--metadata-key COL] [--group-key|--group-by KEY ...]
              [--predictions SPLIT=PATH ...]
              [--baseline FILE] [--new-only]
              [--json FILE] [--html FILE] [--markdown FILE] [--sarif FILE] [--dvc FILE] [--fix-plan FILE]
              [--report-dir DIR] [--no-report] [--github-annotations]
              [--fail-on error|warning|info|none] [--max-violation-rate X]
              [--detectors a,b] [--disable c,d] [--fast]
              [--workers N] [--cache DIR] [--no-cache] [--no-thumbnails] [--max-samples N]
              [--set KEY=VALUE ...] [-q] [-v] [--no-color]
sentinel baseline create [PATH] -o .sentinel/baseline.json [same options]
sentinel diff [PATH] --baseline FILE [--fail-on new-error|new-warning|new-info|error|warning|info|none] [same options]
sentinel init [-f coco|yolo|voc|image-folder] [-o sentinel.yaml] [--force]
sentinel plugins
sentinel version
```

Exit codes: `0` pass, `1` fail condition hit, `2` usage / runtime error.
When `GITHUB_STEP_SUMMARY` is set and no `--markdown` path is given, the
Markdown summary is appended to the job summary automatically.
