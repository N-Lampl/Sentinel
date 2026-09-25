# Text datasets and LLM evaluation contamination

The same question Sentinel asks about images applies to language models:
**is the evaluation data independent of the training data?** For text the
failure modes are benchmark items that leaked into a pretraining or
fine-tuning corpus, eval sets that overlap with SFT data, and paraphrased or
reformatted duplicates across splits. Sentinel treats the training corpus as
the `train` split and the benchmark as the `test` (or `val`) split and runs
the same policy, metric, cluster and CI machinery.

## Quick start

```bash
python examples/make_text_demo.py text-demo
sentinel scan text-demo --text-field question --text-field text --label-field answer
```

```
Split integrity violation rate: 1.24%  (... in 3 violating groups; 3 cross-split clusters)
  text_contamination  (3 error, 1 warning)
    [ERROR] 1 evaluation item from test appear in train (containment 100%)  [deterministic; test,train]
    [WARN ] test: 3 of 40 items (7.5%) appear in the training data  [deterministic; test]
```

## Input

`dataset.format: text` (auto-detected when the root holds `train.jsonl`,
`test.jsonl`, ... ). Each split is a file or a directory of files in JSONL,
JSON (a list, or an object with a `data` / `rows` / `examples` list), CSV,
TSV, Parquet (`pip install "dataset-sentinel[parquet]"`) or plain text (one
document per line).

```yaml
dataset:
  format: text
  splits:
    train: data/sft.jsonl            # or a directory of shards
    test: benchmarks/mmlu.jsonl
  text:
    fields: [question, choices]      # concatenated; null = auto (text, content, prompt, question, input, instruction, messages, ...)
    id_field: null                   # auto: id, _id, idx, uid; else the record position
    label_field: answer              # auto: label, target, category, class
    lowercase: true
    strip_punctuation: true
```

Chat-style `messages` lists are flattened to `role: content` lines. Every
other scalar field of a record becomes metadata, so `source`, `session_id`,
`captured_at` and similar fields feed the group-overlap, lineage and temporal
checks exactly as for images. Hugging Face datasets: export the split with
`dataset.to_json("split.jsonl")`.

## Checks

| Detector | What it finds | Method | Confidence |
|---|---|---|---|
| `text_contamination` | evaluation items that appear inside training documents | fraction of the item's word 8-grams found in a training record (containment); only `test`/`val` items against `train` records | 1.0 = deterministic, >= 0.8 high, >= `min_containment` (0.5) heuristic |
| `text_exact_duplicate` | identical texts across or within splits after normalisation (case, punctuation, whitespace) | SHA-256 of the normalised text | deterministic |
| `text_near_duplicate` | lightly edited or reformatted copies | MinHash / LSH candidates over word 2-gram shingles, verified with the exact Jaccard index (>= 0.7) | >= 0.85 high, else heuristic |
| `text_label_validity` | empty or very short texts, missing labels, labels outside `allowed_categories`, duplicate ids | structural | deterministic |
| `group_overlap`, `lineage`, `temporal`, `distribution` | shared sources / sessions, declared derivations, time overlap, label distribution | metadata, as for images | see `detectors.md` |

`text_contamination` also emits one **contamination rate** finding per
evaluation split (`contaminated / items`), which is the number to report
next to any benchmark score. Items shorter than the n-gram size are matched
as a whole and only count at containment 1.0.

Policies:

```yaml
policy:
  contamination:       {cross_split: error, ngram: 8, min_containment: 0.5, min_shared_ngrams: 2}
  text_near_duplicate: {min_jaccard: 0.7, shingle: 2, num_perm: 128, bands: 32}
  exact_duplicate:     {cross_split: error, within_split: warning}   # shared with images
  near_duplicate:      {cross_split: error, within_split: info}      # severity for text near-duplicates
  labels:
    empty_text: error
    short_text: warning
    min_tokens: 3
```

Everything else works unchanged: the split integrity violation rate,
clusters, the fix plan (`remove` the contaminated evaluation items), the
allowlist, baselines and `sentinel diff`, SARIF / DVC output and the GitHub
Action. The HTML report shows text snippets instead of thumbnails.

## Interpreting results

* Containment 100% on an 8-gram basis means the item's text occurs verbatim
  (after normalisation) in a training record: a memorisation test, not a
  capability test. Remove the item from the evaluation, or report scores on
  the clean subset alongside the full one.
* Containment between 0.5 and 0.8 is a partial overlap: a shared passage,
  a quoted premise, a templated question. Review it; templated benchmarks
  produce these legitimately, and raising `ngram` (e.g. 13) makes the check
  stricter.
* Near-duplicates at Jaccard 0.7 to 0.85 are paraphrase-level edits and
  need review; above 0.85 they are reformatting. Templated benchmarks are the
  known weak spot of any lexical measure: sibling items of a two-slot
  template score about 0.68 (just below the default), one-slot templates
  ("What is 2 + {n}?") score higher and will look like near-duplicates of
  each other, and a training copy of one such item is then near every
  sibling. For those benchmarks raise `min_jaccard` or allowlist
  `near_duplicate_within_split`, and rely on the contamination check, which
  is unaffected.

## Limits

* Matching is on normalised words, not model tokens; a tokenizer-level
  check would need the model's tokenizer and is not implemented.
* Real paraphrases (different words, same meaning) are not detected: the
  optional embedding re-ranking applies to images only in this release, and
  semantic contamination checks are deliberately out of scope until the
  n-gram results prove useful.
* All records are held in memory. Tens of thousands of SFT records and a
  benchmark are fine; scanning a multi-billion-token pretraining corpus is
  not what this release is for. Shard the corpus and scan a shard at a
  time, or keep only the benchmark in the `test` split and stream the
  corpus through the `train` split directory.
