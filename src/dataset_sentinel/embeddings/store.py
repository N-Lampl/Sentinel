"""Computes embeddings for a subset of samples with an sqlite cache."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..model import Dataset
from .base import EmbeddingProvider

log = logging.getLogger(__name__)


class _EmbeddingCache:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "embeddings.sqlite"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (path TEXT, size INTEGER, mtime_ns INTEGER, provider TEXT, dim INTEGER, "
            "data BLOB, PRIMARY KEY (path, size, mtime_ns, provider))"
        )
        self._conn.commit()

    def get_many(self, paths: Sequence[Path], provider: str) -> Dict[Path, np.ndarray]:
        out: Dict[Path, np.ndarray] = {}
        with self._lock:
            for p in paths:
                try:
                    st = p.stat()
                except OSError:
                    continue
                row = self._conn.execute(
                    "SELECT dim, data FROM emb WHERE path=? AND size=? AND mtime_ns=? AND provider=?",
                    (str(p), st.st_size, st.st_mtime_ns, provider),
                ).fetchone()
                if row:
                    out[p] = np.frombuffer(row[1], dtype=np.float16).astype(np.float32)[: row[0]]
        return out

    def put_many(self, items: Iterable[Tuple[Path, np.ndarray]], provider: str) -> None:
        rows = []
        for p, vec in items:
            try:
                st = p.stat()
            except OSError:
                continue
            rows.append((str(p), st.st_size, st.st_mtime_ns, provider, int(vec.shape[0]), np.asarray(vec, dtype=np.float16).tobytes()))
        if rows:
            with self._lock:
                self._conn.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?,?,?,?)", rows)
                self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class EmbeddingStore:
    """Lazily embeds the samples that are asked for (candidate pairs only)."""

    def __init__(self, dataset: Dataset, provider: EmbeddingProvider, cache_dir: Optional[Path] = None):
        self.dataset = dataset
        self.provider = provider
        self.cache_dir = cache_dir
        self._vectors: Dict[str, np.ndarray] = {}
        self.stats: Dict[str, float] = {"computed": 0, "cache_hits": 0, "seconds": 0.0}

    def ensure(self, sample_ids: Iterable[str]) -> None:
        t0 = time.time()
        wanted: List[Tuple[str, Path]] = []
        for sid in sample_ids:
            if sid in self._vectors:
                continue
            s = self.dataset.get_or_none(sid)
            if s is None or s.path is None:
                self._vectors[sid] = np.zeros(self.provider.dimension, dtype=np.float32)
                continue
            wanted.append((sid, s.path))
        if not wanted:
            return
        cache = _EmbeddingCache(self.cache_dir) if self.cache_dir else None
        todo: List[Tuple[str, Path]] = []
        if cache is not None:
            cached = cache.get_many([p for _, p in wanted], self.provider.cache_key)
            for sid, p in wanted:
                if p in cached:
                    self._vectors[sid] = cached[p]
                    self.stats["cache_hits"] += 1
                else:
                    todo.append((sid, p))
        else:
            todo = wanted
        if todo:
            vectors = self.provider.embed([p for _, p in todo])
            for (sid, _p), vec in zip(todo, vectors):
                self._vectors[sid] = vec
            self.stats["computed"] += len(todo)
            if cache is not None:
                cache.put_many([(p, vec) for (_, p), vec in zip(todo, vectors)], self.provider.cache_key)
        if cache is not None:
            cache.close()
        self.stats["seconds"] += time.time() - t0

    def get(self, sample_id: str) -> Optional[np.ndarray]:
        return self._vectors.get(sample_id)
