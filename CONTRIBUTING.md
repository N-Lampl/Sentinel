# Contributing

## Setup

```bash
git clone <repo> && cd dataset-sentinel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

Tests build small synthetic datasets in temporary directories (see
`tests/conftest.py`: `CocoBuilder`, `YoloBuilder`, `image_from_seed`) and plant
specific problems, then assert the exact findings, confidence classes and
metric values. Add a planted case for every new check.

## Principles

* **Every finding must state** the violated policy, the evidence, a confidence
  class, the samples and splits, and a remediation. `Detector.finding(...)`
  enforces the fields; keep the texts specific.
* **Deterministic before heuristic.** Prefer signals that cannot be wrong;
  when a heuristic is used, verify candidates (as the hash detectors verify
  with thumbnails) and label the result `heuristic`.
* **Do not hide gaps.** When a check cannot run because metadata is missing,
  emit an `inconclusive` info finding.
* **Do not claim unimplemented capabilities** in docs, CLI help or reports.
* Detectors never mutate the dataset. Adapters never run checks.
* Only leakage relationship groups count toward the violation rate.

## Adding a detector

1. Create `src/dataset_sentinel/detectors/<name>.py`, subclass `Detector`,
   register with `@detectors.register("<name>")`.
2. Add the module to `registry.detectors` built-ins and to
   `pyproject.toml` entry points.
3. Add policy defaults to `config.DEFAULT_CONFIG` and document them in
   `docs/configuration.md` and `docs/detectors.md`.
4. Add tests with planted problems.

## Adding an adapter

Subclass `DatasetAdapter` (`adapters/base.py`), implement `detect()` and
`load()`, register it, add tests with a tiny on-disk fixture. Put format
extras into `Annotation.attributes` / `Sample.metadata` rather than extending
the model. See `docs/adapters.md` for the modality checklist.

## Releasing

Bump `version` in `pyproject.toml` and `__version__` in
`src/dataset_sentinel/__init__.py`, update `CHANGELOG.md`, tag `vX.Y.Z`.
