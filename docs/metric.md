# Split integrity violation rate

## Definition

```
                    | { s : s belongs to at least one relationship group that violates split policy } |
violation_rate  =  ---------------------------------------------------------------------------------
                                              | evaluated samples |
```

* **Evaluated samples**: every sample the adapter loaded (all splits).
* **Relationship group**: a set of samples connected by one kind of
  relationship: exact identity, near-duplicate similarity, a transform
  (derivative), a shared entity / source / session / batch identifier, a
  lineage family, or a temporal overlap window.
* **Violates split policy**: the group spans more than one split *and* the
  policy for that relationship (`policy.<kind>.cross_split`) is `error` or
  `warning`. Groups reported under an `info` policy are listed but do not
  count; `ignore` disables the detector.
* A sample is counted **once** even when several detectors flag it or it sits
  in several groups.

## Breakdowns in the report

| Field | Meaning |
|-------|---------|
| `metrics.split_integrity_violation_rate` | the headline number |
| `metrics.violating_samples`, `violating_groups`, `evaluated_samples` | numerator components |
| `metrics.by_split[split].violation_rate` | violating samples in that split / samples in that split |
| `metrics.by_detector[name]` | groups and samples contributed by each detector (samples can overlap between detectors) |
| `metrics.by_confidence[level]` | samples in violating groups of that confidence |
| `metrics.by_confidence.strict.violation_rate` | **strict rate**: deterministic + high-confidence groups only |
| `metrics.severity_counts` | error / warning / info counts over all findings |

Use the headline rate for reporting and the strict rate when gating CI on a
dataset where heuristic matches are known to be noisy (simple graphics,
screenshots, synthetic renders).

## What does not count

Label validity, distribution and consistency findings describe data quality,
not split leakage, so they never enter the rate. They still fail the run
through `fail_on.severity` when their severity is high enough.

Temporal overlaps count only when `policy.temporal.enabled` is `true`; in
`auto` mode they are informational because random splits overlap in time by
design.

## Worked example

Dataset: 100 samples (train 70, val 15, test 15).

* exact_duplicate: 2 groups across train/test, 2 samples each -> 4 samples
* near_duplicate: 1 cluster of 3 samples across train/val (heuristic) -> 3 samples
* group_overlap on `patient_id`: 1 group of 5 samples, 4 in train and 1 in test -> 5 samples;
  one of them is also in an exact-duplicate group
* near_duplicate within train: 1 cluster of 4 -> does not cross splits, not counted

Violating samples = |{4} ∪ {3} ∪ {5}| = 4 + 3 + 5 - 1 = 11; rate = 11 / 100 = 11%.
Strict rate = (4 + 5 - 1) / 100 = 8% (the heuristic near-duplicate cluster is excluded).

## Recommended gates

* `fail_on.severity: error` and `fail_on.max_violation_rate: 0` (defaults): no
  leakage of any kind reaches training.
* For legacy datasets being cleaned up incrementally: `--max-violation-rate 0.02`
  plus `--fail-on warning`, then lower the rate as the dataset improves. The
  JSON report's stable finding ids let you diff runs.
