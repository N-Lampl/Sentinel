# Detectors

All detectors are enabled by default. Select with `--detectors a,b` /
`--disable c` or `detectors.enabled` / `detectors.disabled` in the config.
Policy keys live under `policy.` in `sentinel.yaml`; their values are
severities (`error`, `warning`, `info`, `ignore`).

Confidence classes used below:

* **deterministic**: follows from data that cannot be wrong (identical bytes, explicit metadata, a negative box width);
* **high_confidence**: very strong signal with a low false-positive rate;
* **heuristic**: plausible, review recommended;
* **inconclusive**: the check could not be evaluated (missing metadata) and says so.

Only relationship groups from the leakage detectors (exact_duplicate,
near_duplicate, derivative, group_overlap, lineage, temporal) count toward the
split integrity violation rate. After all detectors ran, the engine applies
`policy.split_pair_overrides`, the `allowlist`, marks findings as new / known
against the baseline, and unifies all violating groups into **cross-split
clusters** (see `metric.md`).

---

## exact_duplicate

Byte-identical or pixel-identical samples, and the same file path registered in
several splits (typical YOLO mistake: `val: images/train`).

* Signals, strongest first: same path; SHA-256 of the file bytes; BLAKE2b of
  the decoded RGB pixels (re-saved / re-encoded lossless copies, PNG -> BMP).
  The pixel signal is skipped in `performance.mode: fast`.
* Evidence: `match_types` (`same_path`, `byte_identical`, `pixel_identical`), hashes, file sizes, paths.
* Confidence: deterministic.
* Policy: `policy.exact_duplicate.cross_split` (default error), `within_split` (default warning).
* Remediation: keep one copy in one split, or move the whole group to one split, then re-evaluate.

## near_duplicate

Perceptually near-identical images. Each image gets a 64-bit dHash of a 64x64
grayscale thumbnail; pairs within `threshold` Hamming bits (default 6) are
candidates, and every candidate is verified with the normalised correlation of
16x16 thumbnails (`min_correlation`, default 0.8). Exact duplicates are excluded
(reported by `exact_duplicate`). Blank / uniform images are skipped. Connected
pairs form clusters.

* Evidence: algorithm, threshold, closest cross-split pair (distance,
  correlation), up to 20 pairs, cluster size.
* Confidence: high_confidence when the best cross-split pair has distance <= 3
  and correlation >= 0.9; heuristic otherwise (and always for clusters > 200).
* Policy: `policy.near_duplicate.cross_split` / `within_split` / `threshold` / `min_correlation`.
* Limits: catches resizes, re-encodes, colour / brightness edits, light crops
  and small overlays. Does not catch heavy crops or semantic similarity
  (different photos of the same object).

## derivative

Two families of content-derived copies:

* **Dihedral transforms**: flips, rotations by 90 / 180 / 270 degrees,
  transposes. The dHash of all eight transforms of every image is stored;
  non-identity transforms are queried against the identity index; candidates
  are verified by correlating the transformed 16x16 thumbnail with the
  candidate's thumbnail. Statements are oriented from the earlier split to the
  later one ("test:x looks like flip horizontal of train:y").
* **Regular tiles and crops** (`policy.derivative.crops`, default on): 2x2
  tiles, left / right / top / bottom halves, 50% and 75% centre crops. The
  dHash of each region of the parent's thumbnail is indexed; a sample whose
  identity hash matches a region hash (within `crop_threshold`, default 12)
  is verified by correlating the parent's thumbnail region with the child's
  thumbnail (`crop_min_correlation`, default 0.9). Statements are always
  oriented parent -> crop.

* Evidence: transform names (`flip_horizontal`, `rotate_90`, `crop:tile_top_left`, ...), distances, correlations, pairs.
* Confidence: high_confidence for transforms with distance <= 3 and correlation >= 0.9, for crops with correlation >= 0.95; heuristic otherwise.
* Limits: arbitrary crops, other grid sizes, colour-space conversions of
  unrelated content and non-dihedral rotations are not detected by content.
  Use `lineage` metadata or file-name patterns for those.

## group_overlap

Samples sharing an entity / source / session / batch identifier that appear in
more than one split. Identifiers come from `sample.groups`, filled by
enrichment from:

* metadata keys listed in `dataset.groups.keys` (`--group-by KEY` on the CLI);
* well-known keys found automatically when `dataset.groups.auto` is true
  (`patient_id`, `subject_id`, `source`, `batch`, `session`, `device`, `camera`,
  `location`, `site`, `author`, `account`, `user_id`, `video_id`, `sequence_id`,
  `capture_id`, `scene`, `study_id`, `case_id`, `group_id`, ...);
* file-name regexes in `dataset.groups.from_filename` (`{name: pattern}` with a
  named group `value`).

Metadata itself comes from COCO image fields, VOC `<source>` fields, a manifest
(`dataset.metadata.file` / `--manifest`: CSV, TSV, JSON, JSONL or Parquet joined
on file name, stem, path or id) or adapters.

* Evidence: key, value, per-split counts, source (metadata or filename pattern).
* Confidence: deterministic (metadata) or high_confidence (file-name pattern).
* Policy: `policy.group_overlap.cross_split`; use `policy.split_pair_overrides`
  to allow specific pairs (e.g. train/val) while forbidding others.
* When no identifier exists at all, one *inconclusive* info finding says so.

## lineage

`derived_from` relationships that cross splits. Edges are resolved during
enrichment from metadata keys (`dataset.lineage.keys`, default `derived_from`,
`parent`, `parent_id`, `source_image`, `original`, ...) whose value is a file
name, stem, relative path or native id of another sample, and from file-name
regexes (`dataset.lineage.from_filename`, named group `parent`). Families are
connected components of the lineage graph.

* Confidence: deterministic (metadata) or heuristic (file-name pattern).
* Policy: `policy.lineage.cross_split`.
* Inconclusive info finding when no lineage information exists.

## temporal

Overlapping capture-time ranges between splits with known roles
(`policy.split_order`, default train -> val -> test). Timestamps are parsed from
`dataset.time.keys` (default `captured_at`, `timestamp`, `date_captured`, ...;
ISO-8601, epoch seconds / millis, `YYYYMMDD`, EXIF and common date formats, or
`dataset.time.format`).

* `policy.temporal.enabled: auto` (default): overlaps are reported as an
  *info / inconclusive* finding and do not count, because a random split
  legitimately overlaps in time.
* `enabled: true`: the splits must be chronological. Every later-split sample
  captured before the last earlier-split sample, plus the earlier-split samples
  after the first later-split sample, form a violating group
  (deterministic; heuristic when timestamp coverage is below `min_coverage`).
* `enabled: false`: skipped.

## label_validity

Deterministic structural checks (never count toward the rate):

| kind | meaning | policy key |
|------|---------|------------|
| `missing_file`, `unreadable_image`, `empty_image`, `blank_image` | file problems | same names |
| `size_mismatch` | declared width/height differs from the file | `size_mismatch` |
| `missing_annotations` (aggregated per split), `missing_label_file` (YOLO / VOC), `empty_split` | coverage | same names |
| `invalid_bbox`, `degenerate_bbox`, `out_of_bounds_bbox` | boxes (`min_bbox_size_px`); NaN / infinite values are `invalid_bbox` | same names |
| `unknown_category`, `unknown_category_summary` | undeclared or disallowed categories (`allowed_categories`) | `unknown_category` |
| `malformed_label` | unparsable YOLO line / COCO entry / VOC XML | `malformed_label` |
| `invalid_segmentation`, `duplicate_annotation` | polygons / RLE, repeated identical annotations | same names |

Adapter-level structural findings (`adapter:coco`, `adapter:voc`): orphan
annotations, duplicate annotation / image ids, duplicate image registrations,
and COCO category lists that differ between split files
(`category_definition_conflict`: same id, different name, error;
`category_set_mismatch`: different id sets, warning).

## distribution

Label and metadata distribution quality (never counts toward the rate):

* `class_imbalance`: max/min class count ratio in a split above `imbalance_ratio`.
* `unseen_class` (error, deterministic): classes in an evaluation split with no training examples.
* `missing_class_in_eval`: training classes absent from an evaluation split.
* `label_shift`: Jensen-Shannon divergence (base 2, 0..1) between the training
  split's class distribution and another split above `shift_threshold`.
* `image_size_shift`, `bbox_size_shift`: JS divergence over log-binned image
  area / relative box area; evidence also carries the Wasserstein distance
  (raw and normalised by the training split's standard deviation) for image
  area, image aspect ratio, objects per image, relative box area and box
  aspect ratio.
* `cooccurrence_anomaly`: class pairs that co-occur in >= 50% of the training
  images containing either class but never together in an evaluation split.
* `min_samples` guards all comparisons. Unknown categories (reported by
  `label_validity`) are excluded from the statistics.

The class distribution and the continuous-distribution summaries per split
are exposed in the report (`detectors[].stats`) and charted in the HTML report.
This is *split distribution divergence*, not bias detection: no deployment
distribution is assumed.

## consistency

A review assistant, not a ground-truth judge. Every finding is phrased as a
*potential* inconsistency with a review recommendation:

* `conflicting_labels_identical` (error, deterministic): byte- or
  pixel-identical images with different label sets (the message says whether
  the classes differ or only the box geometry).
* `conflicting_labels_near_duplicate` (warning, heuristic): verified
  near-duplicate pairs whose class sets differ.
* `object_count_mismatch_near_duplicate` (warning, heuristic): verified
  near-duplicate pairs with the same classes but a very different number of
  annotated objects (difference >= 2 and a factor >= 2).

Policy: `policy.consistency.conflicting_labels`. Adapter-level duplicate
registrations (same image id / file name twice in a COCO file) use
`policy.consistency.duplicate_registration`.

---

## Writing a detector

```python
from dataset_sentinel.detectors.base import Detector, DetectorContext, DetectorResult
from dataset_sentinel.model import Confidence, Severity
from dataset_sentinel.registry import detectors

@detectors.register("blurry_images")
class BlurryImages(Detector):
    name = "blurry_images"
    description = "Images with very low sharpness"
    modalities = ("image",)
    needs_fingerprints = True

    def run(self, ctx: DetectorContext) -> DetectorResult:
        res = DetectorResult()
        sev = ctx.severity("policy.blurry_images.severity", Severity.WARNING)
        for s in ctx.dataset.samples:
            fp = ctx.fingerprints.get(s.id)
            if fp and fp.ok and fp.std_luma is not None and fp.std_luma < 4:
                res.add(self.finding(
                    kind="blurry_image", title=f"Low contrast image {s.uri}",
                    severity=sev, confidence=Confidence.HEURISTIC,
                    policy=ctx.policy_label("policy.blurry_images.severity"),
                    policy_description="Images should have visible structure.",
                    message=f"luma std {fp.std_luma:.1f}", remediation="Review or drop the image.",
                    samples=[s.ref()], evidence={"std_luma": fp.std_luma},
                ))
        return res
```

Register third-party detectors through the `dataset_sentinel.detectors`
entry-point group; a `policy.<detector name>` section is then accepted by the
config validator. Emit `RelationshipGroup`s with `self.group(...)` and set
`counts=True` on the corresponding finding when the group violates a split
policy, so the metric, the clusters and the fix plan include it.
