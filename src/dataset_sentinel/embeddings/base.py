from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, List, Optional, Sequence

import numpy as np

from ..registry import Registry

log = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Turns image files into L2-normalised float32 vectors."""

    name: ClassVar[str] = "base"
    #: bump when the features change so cached vectors are recomputed
    version: ClassVar[str] = "1"
    dimension: ClassVar[int] = 0
    description: ClassVar[str] = ""

    @classmethod
    def available(cls) -> bool:
        """False when an optional dependency is missing."""
        return True

    @abstractmethod
    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        """Return an array of shape (len(paths), dimension). Rows for files that
        cannot be read are all zeros."""

    @property
    def cache_key(self) -> str:
        return f"{self.name}:{self.version}"


embeddings: Registry = Registry(
    "dataset_sentinel.embeddings",
    builtin_modules=["dataset_sentinel.embeddings.builtin", "dataset_sentinel.embeddings.torchvision_provider"],
)


def get_provider(name: str) -> Optional[EmbeddingProvider]:
    """Instantiate a registered provider, or ``None`` when its dependencies are missing."""
    cls = embeddings.get(name)
    if not cls.available():
        log.warning("embedding provider %s is not available (missing optional dependency)", name)
        return None
    return cls()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return x / norms


__all__ = ["EmbeddingProvider", "embeddings", "get_provider", "cosine_similarity", "normalize_rows", "List"]
