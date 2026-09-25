"""Computes and caches fingerprints for all samples of a dataset."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..model import Dataset, Sample
from .image import REGION_NAMES, ImageFingerprint, fingerprint_image
from .index import to_uint64

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]


class _Cache:
    """Tiny sqlite cache keyed by (path, size, mtime_ns, version, mode).

    ``version`` is the fingerprint schema version; ``mode`` is ``exact`` or
    ``fast``. An ``exact`` entry satisfies a ``fast`` request (it carries a
    superset of the data) but not the other way round.
    """

    VERSION = 4

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "fingerprints.sqlite"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("DROP TABLE IF EXISTS fp")  # pre-0.1 layout
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS fingerprints (path TEXT, size INTEGER, mtime_ns INTEGER, version INTEGER, "
            "mode TEXT, data TEXT, PRIMARY KEY (path, size, mtime_ns, version, mode))"
        )
        self._conn.commit()

    def get(self, path: Path, mode: str) -> Optional[dict]:
        try:
            st = path.stat()
        except OSError:
            return None
        modes = ("exact",) if mode == "exact" else ("exact", "fast")
        with self._lock:
            for m in modes:
                row = self._conn.execute(
                    "SELECT data FROM fingerprints WHERE path=? AND size=? AND mtime_ns=? AND version=? AND mode=?",
                    (str(path), st.st_size, st.st_mtime_ns, self.VERSION, m),
                ).fetchone()
                if row:
                    return json.loads(row[0])
        return None

    def put_many(self, items: Iterable[Tuple[Path, dict]]) -> None:
        rows = []
        for path, data in items:
            try:
                st = path.stat()
            except OSError:
                continue
            rows.append((str(path), st.st_size, st.st_mtime_ns, self.VERSION, str(data.get("mode", "exact")), json.dumps(data)))
        if not rows:
            return
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO fingerprints VALUES (?,?,?,?,?,?)", rows)
            self._conn.commit()

    def prune_old_versions(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM fingerprints WHERE version <> ?", (self.VERSION,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class FingerprintStore:
    """Holds one :class:`ImageFingerprint` per sample."""

    def __init__(self, dataset: Dataset, workers: int = 0, cache_dir: Optional[Path] = None, mode: str = "exact"):
        self.dataset = dataset
        self.workers = workers if workers and workers > 0 else min(8, os.cpu_count() or 4)
        self.cache_dir = cache_dir
        self.mode = "fast" if str(mode).lower() == "fast" else "exact"
        self._fps: Dict[str, ImageFingerprint] = {}
        self._computed = False
        self.stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ compute
    def compute(self, progress: Optional[ProgressFn] = None) -> "FingerprintStore":
        if self._computed:
            return self
        t0 = time.time()
        samples = [s for s in self.dataset.samples if s.modality == "image"]
        cache = _Cache(self.cache_dir) if self.cache_dir else None
        if cache is not None:
            cache.prune_old_versions()
        todo: List[Sample] = []
        hits = 0
        for s in samples:
            if s.path is None:
                self._fps[s.id] = ImageFingerprint(sample_id=s.id, path=s.uri, ok=False, error="no file path")
                continue
            if cache is not None:
                cached = cache.get(s.path, self.mode)
                if cached is not None:
                    cached["sample_id"] = s.id
                    self._fps[s.id] = ImageFingerprint.from_dict(cached)
                    hits += 1
                    continue
            todo.append(s)

        done = hits
        total = len(samples)
        new_items: List[Tuple[Path, dict]] = []

        fast = self.mode == "fast"

        def work(sample: Sample) -> ImageFingerprint:
            assert sample.path is not None
            return fingerprint_image(sample.path, sample.id, fast=fast)

        if todo:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for fp in pool.map(work, todo, chunksize=8):
                    self._fps[fp.sample_id] = fp
                    if cache is not None and fp.ok:
                        new_items.append((Path(fp.path), fp.to_dict()))
                    done += 1
                    if progress and (done % 50 == 0 or done == total):
                        progress(done, total)
                    if cache is not None and len(new_items) >= 500:
                        cache.put_many(new_items)
                        new_items = []
        if cache is not None:
            cache.put_many(new_items)
            cache.close()
        self._computed = True
        self.stats = {
            "images": total,
            "mode": self.mode,
            "cache_hits": hits,
            "computed": len(todo),
            "failed": sum(1 for fp in self._fps.values() if not fp.ok),
            "seconds": round(time.time() - t0, 3),
        }
        log.info("fingerprinted %d images (%d cached) in %.1fs", total, hits, time.time() - t0)
        return self

    # ------------------------------------------------------------------ access
    def get(self, sample_id: str) -> Optional[ImageFingerprint]:
        return self._fps.get(sample_id)

    def all(self) -> Dict[str, ImageFingerprint]:
        return self._fps

    def ok_fingerprints(self) -> List[ImageFingerprint]:
        return [fp for fp in self._fps.values() if fp.ok and fp.dhash is not None]

    def dhash_arrays(self) -> Tuple[List[str], np.ndarray]:
        fps = self.ok_fingerprints()
        ids = [fp.sample_id for fp in fps]
        codes = to_uint64([fp.dhash for fp in fps]) if fps else np.zeros(0, dtype=np.uint64)
        return ids, codes

    def dihedral_arrays(self) -> Tuple[List[str], np.ndarray]:
        """(ids, array of shape (n, 8)) with the dihedral hashes."""
        fps = [fp for fp in self.ok_fingerprints() if fp.dhash_dihedral]
        ids = [fp.sample_id for fp in fps]
        if not fps:
            return ids, np.zeros((0, 8), dtype=np.uint64)
        arr = np.array([[int(h) & 0xFFFFFFFFFFFFFFFF for h in fp.dhash_dihedral] for fp in fps], dtype=np.uint64)
        return ids, arr

    def region_arrays(self) -> Tuple[List[str], np.ndarray]:
        """(ids, array of shape (n, len(REGION_NAMES))) with the sub-region hashes."""
        n_regions = len(REGION_NAMES)
        fps = [fp for fp in self.ok_fingerprints() if fp.dhash_regions and len(fp.dhash_regions) == n_regions]
        ids = [fp.sample_id for fp in fps]
        if not fps:
            return ids, np.zeros((0, n_regions), dtype=np.uint64)
        arr = np.array([[int(h) & 0xFFFFFFFFFFFFFFFF for h in fp.dhash_regions] for fp in fps], dtype=np.uint64)
        return ids, arr
