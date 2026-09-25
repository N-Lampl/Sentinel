"""Hamming-distance index for 64-bit perceptual hashes.

Uses multi-index hashing: a 64-bit code is split into ``threshold + 1``
disjoint chunks. Two codes within Hamming distance ``threshold`` must agree
exactly on at least one chunk (pigeonhole principle), so candidates are found
by bucketing on each chunk and verified with a vectorised popcount. This
gives exact recall for the configured threshold without an N^2 scan.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Popcount for an array of uint64 (any shape)."""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    bytes_view = x.view(np.uint8).reshape(x.shape + (8,))
    return _POP8[bytes_view].sum(axis=-1, dtype=np.int32)


def to_uint64(values: Sequence[int]) -> np.ndarray:
    return np.array([int(v) & 0xFFFFFFFFFFFFFFFF for v in values], dtype=np.uint64)


def _chunk_bounds(bits: int, n_chunks: int) -> List[Tuple[int, int]]:
    base, extra = divmod(bits, n_chunks)
    bounds = []
    start = 0
    for i in range(n_chunks):
        width = base + (1 if i < extra else 0)
        bounds.append((start, width))
        start += width
    return bounds


class HammingIndex:
    """Index over a fixed array of 64-bit codes."""

    def __init__(self, codes: np.ndarray, threshold: int, bits: int = 64):
        self.codes = np.ascontiguousarray(codes, dtype=np.uint64)
        self.threshold = int(max(0, threshold))
        self.bits = bits
        n_chunks = min(self.threshold + 1, bits)
        self.bounds = _chunk_bounds(bits, n_chunks)
        self._buckets: List[Dict[int, List[int]]] = []
        self._build()

    def _chunk_values(self, codes: np.ndarray, start: int, width: int) -> np.ndarray:
        mask = np.uint64((1 << width) - 1)
        return (codes >> np.uint64(start)) & mask

    def _build(self) -> None:
        """Bucket indices by chunk value with one argsort per chunk (no Python
        loop over every code), keeping only buckets with >= 2 members for
        pairs_within and all buckets for query."""
        for start, width in self.bounds:
            vals = self._chunk_values(self.codes, start, width)
            if len(vals) == 0:
                self._buckets.append({})
                continue
            order = np.argsort(vals, kind="stable")
            sorted_vals = vals[order]
            boundaries = np.flatnonzero(np.diff(sorted_vals)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(sorted_vals)]))
            bucket: Dict[int, List[int]] = {}
            uniq = sorted_vals[starts].tolist()
            for v, a, b in zip(uniq, starts.tolist(), ends.tolist()):
                bucket[v] = order[a:b].tolist()
            self._buckets.append(bucket)

    # ------------------------------------------------------------------ pairs
    def pairs_within(self, block: int = 2048) -> List[Tuple[int, int, int]]:
        """All (i, j, distance) with i < j and distance <= threshold."""
        seen: set = set()
        results: List[Tuple[int, int, int]] = []
        if len(self.codes) < 2:
            return results
        if self.threshold == 0:
            # exact matches: group by value directly
            order: Dict[int, List[int]] = defaultdict(list)
            for idx, v in enumerate(self.codes.tolist()):
                order[v].append(idx)
            for members in order.values():
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        results.append((members[a], members[b], 0))
            return results

        for bucket in self._buckets:
            for members in bucket.values():
                if len(members) < 2:
                    continue
                idx = np.array(members, dtype=np.int64)
                sub = self.codes[idx]
                for r0 in range(0, len(idx), block):
                    rows = sub[r0 : r0 + block]
                    d = popcount64(rows[:, None] ^ sub[None, :])
                    ii, jj = np.nonzero(d <= self.threshold)
                    for a, b in zip(ii.tolist(), jj.tolist()):
                        ga, gb = int(idx[r0 + a]), int(idx[b])
                        if ga >= gb:
                            continue
                        key = (ga, gb)
                        if key in seen:
                            continue
                        seen.add(key)
                        results.append((ga, gb, int(d[a, b])))
        results.sort()
        return results

    # ------------------------------------------------------------------ query
    def query(self, queries: np.ndarray, block: int = 2048) -> List[Tuple[int, int, int]]:
        """For each query code, all (query_idx, index_idx, distance) within threshold."""
        queries = np.ascontiguousarray(queries, dtype=np.uint64)
        results: List[Tuple[int, int, int]] = []
        if len(self.codes) == 0 or len(queries) == 0:
            return results
        seen: set = set()
        for (start, width), bucket in zip(self.bounds, self._buckets):
            qvals = self._chunk_values(queries, start, width).tolist()
            groups: Dict[int, List[int]] = defaultdict(list)
            for qi, v in enumerate(qvals):
                if v in bucket:
                    groups[v].append(qi)
            for v, q_members in groups.items():
                cand = np.array(bucket[v], dtype=np.int64)
                qidx = np.array(q_members, dtype=np.int64)
                csub = self.codes[cand]
                for r0 in range(0, len(qidx), block):
                    qrows = queries[qidx[r0 : r0 + block]]
                    d = popcount64(qrows[:, None] ^ csub[None, :])
                    ii, jj = np.nonzero(d <= self.threshold)
                    for a, b in zip(ii.tolist(), jj.tolist()):
                        key = (int(qidx[r0 + a]), int(cand[b]))
                        if key in seen:
                            continue
                        seen.add(key)
                        results.append((key[0], key[1], int(d[a, b])))
        results.sort()
        return results


def union_find_groups(n: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """Connected components over ``n`` nodes given undirected edges."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comps: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return [sorted(m) for m in comps.values() if len(m) > 1]
