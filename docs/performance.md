# Performance

Measured on a laptop (8 fingerprinting threads, Python 3.10, numpy + Pillow)
with synthetic YOLO datasets of 160x120 JPEGs (1% of the training images
copied into `test`). Small images make fingerprinting cheap per image, so the
detector costs below are the relevant part; for large photos the decode
dominates (see *Where the time goes*).

| Dataset | Cold scan | of which fingerprints | derivative | other detectors | Re-scan from cache | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| 5,050 images | 20 s | 17 s | 1.7 s | < 0.5 s | 3 s | ~160 MB |
| 30,300 images | 132 s | 92 s | 21 s | < 3 s | 41 s | 478 MB |

Before the crop-search rewrite the 30k scan took 365 s (246 s in the
derivative detector) and 850 MB, which is why the crop stage now only
searches cross-split pairs, skips flat regions, hashes regions from a 128px
intermediate thumbnail (so a tighter Hamming threshold suffices) and has a
size guard (`policy.derivative.crops_max_images`).

## Where the time goes

Per-image fingerprint cost (single thread):

| Image size | Total | JPEG decode | Draft decode (`--fast`) | Pixel hash | Grayscale + thumbnails | 18 hashes |
|---|---:|---:|---:|---:|---:|---:|
| 160x120 | 2.8 ms | 0.4 ms | 0.3 ms | 0.1 ms | 0.1 ms | 1.4 ms |
| 1920x1080 | 56 ms | 21 ms | 8 ms | 11 ms | 6.6 ms | 1.2 ms |
| 4000x3000 | 269 ms | 114 ms | 32 ms | 70 ms | 41 ms | 1.1 ms |

For photos, decoding and the full-resolution pixel hash dominate; the hashing
itself is about a millisecond. Consequences:

* **`--fast`** (`performance.mode: fast`): JPEGs are decoded at reduced
  resolution (DCT scaling) and the pixel-identity hash is skipped. 3-5x
  faster on large photos; byte-identical and perceptual detection are
  unaffected, pixel-identical detection (re-saved lossless copies) is not
  available. Cached `exact` fingerprints still serve `fast` runs.
* **Threads**: decoding releases the GIL; `performance.workers` defaults to
  min(8, CPU count). Raise it on machines with more cores and fast storage.
* **Cache** (`--cache DIR` / `performance.cache`): fingerprints are stored in
  sqlite keyed by path, size, mtime, schema version and mode. A re-scan of an
  unchanged dataset only pays for the detectors (41 s for 30k images above).
  In CI, persist the cache directory with `actions/cache`.

## Detector complexity

* Exact duplicates: hash-map grouping, linear.
* Near duplicates and dihedral transforms: multi-index Hamming search. A
  64-bit code is split into `threshold + 1` chunks; candidates share a chunk,
  so cost scales with `N x (N / buckets)`. With the default threshold of 6
  (7 chunks, 512 buckets) 30k images take about a second. Wide thresholds
  (>= 10) shrink the buckets and approach an all-pairs scan; the crop stage
  therefore uses hashes that agree more closely instead of a wider threshold.
* Crops / tiles: 10 region hashes per image queried against the other
  splits' identity hashes; the cost is roughly `N x regions x (N_other /
  buckets)`. Skipped above `crops_max_images` (default 100,000) with a note
  in the detector stats; force it with `--set policy.derivative.crops_max_images=0`.
* Group overlap, lineage, temporal, distribution: linear in samples and
  annotations.
* Embedding re-ranking (optional): computed for candidate samples only, never
  all-pairs; the `builtin` descriptor costs about 1 ms per image, torchvision
  features tens of milliseconds on CPU. Cached in `embeddings.sqlite`.

## Memory

Python objects dominate: a sample with one annotation and its fingerprint
(hashes, 256-byte thumbnail, paths) is roughly 5 KB, so 30k images need
about 150 MB of tracked memory plus interpreter, numpy and Pillow overhead
(478 MB peak RSS at 30k, most of it transient decode buffers). Expect
roughly 1.5 GB at 100k images with the default detectors. The HTML report
renders at most `output.max_findings_html` findings (default 2,000) with
thumbnails for the first `output.max_thumbnails` leakage findings.

## Reproducing

```bash
python examples/make_scale_dataset.py 30000 scale-dataset          # 160x120 images
python examples/make_scale_dataset.py 2000 scale-photos 1920x1080  # photo-sized
time sentinel scan scale-dataset --cache scale-dataset/.cache --no-thumbnails -q
time sentinel scan scale-dataset --cache scale-dataset/.cache --no-thumbnails -q   # cached
```

Run the scan twice to see the cached time. The JSON report's
`detectors[].duration_s`, `stats.fingerprints.seconds` and
`stats.duration_s` give the breakdown; `/usr/bin/time -l` (macOS) or
`-v` (Linux) reports peak memory.
