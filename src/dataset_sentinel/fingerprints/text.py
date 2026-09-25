"""Text fingerprints for exact / near-duplicate detection and contamination.

* ``normalize_text``: NFKC, lower-case, punctuation stripped, whitespace
  collapsed. Two texts that differ only in case, punctuation or spacing are
  *exact* duplicates after normalisation.
* word n-gram **shingles** (default 2 words) hashed to 64 bits for
  near-duplicate detection via **MinHash** (128 permutations) and **LSH**
  banding for candidate generation, verified with the exact Jaccard index on
  the shingle sets.
* longer word n-grams (default 8) for **contamination**: the fraction of an
  evaluation item's n-grams that appear inside a training document
  (containment), the standard check used for LLM benchmark contamination.

Only numpy is required.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_MERSENNE_31 = np.uint64((1 << 31) - 1)
_MASK_32 = np.uint64(0xFFFFFFFF)


def normalize_text(text: str, lowercase: bool = True, strip_punctuation: bool = True) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    if lowercase:
        t = t.lower()
    if strip_punctuation:
        t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def tokens(normalized: str) -> List[str]:
    return normalized.split(" ") if normalized else []


def hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def ngram_hashes(toks: Sequence[str], n: int) -> np.ndarray:
    """Sorted unique 64-bit hashes of the word n-grams (whole text when shorter than n)."""
    if not toks:
        return np.zeros(0, dtype=np.uint64)
    if len(toks) < n:
        grams: Iterable[str] = [" ".join(toks)]
        count = 1
    else:
        count = len(toks) - n + 1
        grams = (" ".join(toks[i : i + n]) for i in range(count))
    arr = np.fromiter((hash64(g) for g in grams), dtype=np.uint64, count=count)
    return np.unique(arr)


class MinHasher:
    """Vectorised MinHash over 64-bit shingle hashes (universal hashing mod 2^31-1)."""

    def __init__(self, num_perm: int = 128, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, int(_MERSENNE_31), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE_31), size=num_perm, dtype=np.uint64)

    def signature(self, shingles: np.ndarray, block: int = 4096) -> np.ndarray:
        if len(shingles) == 0:
            return np.full(self.num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)
        x = (shingles.astype(np.uint64) & _MASK_32)
        best = np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        for start in range(0, len(x), block):
            part = x[start : start + block]
            vals = (part[:, None] * self.a[None, :] + self.b[None, :]) % _MERSENNE_31
            best = np.minimum(best, vals.min(axis=0))
        return best.astype(np.uint32)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (len(a) + len(b) - inter)


def containment(needle: np.ndarray, haystack: np.ndarray) -> float:
    """Fraction of ``needle``'s n-grams present in ``haystack``."""
    if len(needle) == 0:
        return 0.0
    return np.intersect1d(needle, haystack, assume_unique=True).size / len(needle)


def lsh_candidate_pairs(signatures: np.ndarray, bands: int = 32) -> List[Tuple[int, int]]:
    """Pairs of row indices whose MinHash signatures collide in at least one band."""
    n, num_perm = signatures.shape
    if n < 2:
        return []
    rows = num_perm // bands
    pairs: set = set()
    for b in range(bands):
        chunk = np.ascontiguousarray(signatures[:, b * rows : (b + 1) * rows])
        keys = chunk.view(np.dtype((np.void, chunk.dtype.itemsize * rows))).ravel()
        buckets: Dict[bytes, List[int]] = defaultdict(list)
        for i, key in enumerate(keys):
            buckets[key.tobytes()].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
    return sorted(pairs)


@dataclass
class TextFingerprint:
    sample_id: str
    exact_hash: str
    n_tokens: int
    shingles: np.ndarray  # sorted unique uint64 (near-duplicate shingles)
    minhash: np.ndarray  # uint32[num_perm]


class TextFingerprintStore:
    """Per-sample text fingerprints for the samples of a dataset (modality ``text``)."""

    def __init__(self, dataset, shingle_n: int = 2, num_perm: int = 128, lowercase: bool = True, strip_punctuation: bool = True):
        self.dataset = dataset
        self.shingle_n = shingle_n
        self.num_perm = num_perm
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self._fps: Dict[str, TextFingerprint] = {}
        self._normalized: Dict[str, str] = {}
        self._computed = False
        self.hasher = MinHasher(num_perm)
        self.stats: Dict[str, float] = {}

    def normalized(self, sample_id: str) -> str:
        if sample_id not in self._normalized:
            s = self.dataset.get_or_none(sample_id)
            text = s.metadata.get("text", "") if s is not None else ""
            self._normalized[sample_id] = normalize_text(text, self.lowercase, self.strip_punctuation)
        return self._normalized[sample_id]

    def tokens(self, sample_id: str) -> List[str]:
        return tokens(self.normalized(sample_id))

    def compute(self) -> "TextFingerprintStore":
        if self._computed:
            return self
        import time

        t0 = time.time()
        n = 0
        for s in self.dataset.samples:
            if s.modality != "text":
                continue
            norm = self.normalized(s.id)
            toks = tokens(norm)
            shingles = ngram_hashes(toks, self.shingle_n)
            self._fps[s.id] = TextFingerprint(
                sample_id=s.id,
                exact_hash=hashlib.sha256(norm.encode("utf-8")).hexdigest(),
                n_tokens=len(toks),
                shingles=shingles,
                minhash=self.hasher.signature(shingles),
            )
            n += 1
        self._computed = True
        self.stats = {"samples": n, "seconds": round(time.time() - t0, 3), "shingle_n": self.shingle_n, "num_perm": self.num_perm}
        return self

    def get(self, sample_id: str) -> Optional[TextFingerprint]:
        return self._fps.get(sample_id)

    def all(self) -> Dict[str, TextFingerprint]:
        return self._fps

    def signature_matrix(self, sample_ids: Sequence[str]) -> np.ndarray:
        return np.stack([self._fps[sid].minhash for sid in sample_ids]) if sample_ids else np.zeros((0, self.num_perm), dtype=np.uint32)


def snippet(text: str, limit: int = 160) -> str:
    t = _WS_RE.sub(" ", str(text)).strip()
    return t if len(t) <= limit else t[: limit - 1] + "…"
