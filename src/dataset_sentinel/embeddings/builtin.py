"""Dependency-free image descriptor (colour + coarse structure).

Not a learned embedding. It complements the grayscale dHash with colour
information and a coarse gradient-orientation layout, which is enough to
separate "same picture, re-encoded / recoloured" from "different picture with
similar luminance structure" when re-ranking hash candidates.

Layout (288 floats, L2-normalised):

* 8x8x3 mean-colour grid over a 32x32 RGB thumbnail (192 values, [0, 1]);
* 4x4 grid of 6-bin gradient-orientation histograms on the grayscale
  thumbnail (96 values, magnitude-weighted, normalised per cell).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from .base import EmbeddingProvider, embeddings, normalize_rows

_SIZE = 32


def _describe(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB").resize((_SIZE, _SIZE), Image.Resampling.BOX), dtype=np.float32) / 255.0
    colour = rgb.reshape(8, 4, 8, 4, 3).mean(axis=(1, 3)).ravel()  # 8x8x3
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)  # unsigned orientation
    bins = np.minimum((ang / np.pi * 6).astype(int), 5)
    hist = np.zeros((4, 4, 6), dtype=np.float32)
    cell = _SIZE // 4
    for cy in range(4):
        for cx in range(4):
            sl = (slice(cy * cell, (cy + 1) * cell), slice(cx * cell, (cx + 1) * cell))
            np.add.at(hist[cy, cx], bins[sl].ravel(), mag[sl].ravel())
            s = hist[cy, cx].sum()
            if s > 1e-6:
                hist[cy, cx] /= s
    return np.concatenate([colour, hist.ravel() * 0.5]).astype(np.float32)


@embeddings.register("builtin")
class BuiltinDescriptor(EmbeddingProvider):
    name = "builtin"
    version = "1"
    dimension = 288
    description = "Colour grid + gradient-orientation layout descriptor (numpy only, not a learned embedding)"

    def embed(self, paths: Sequence[Path]) -> np.ndarray:
        out = np.zeros((len(paths), self.dimension), dtype=np.float32)
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    im.draft("RGB", (256, 256))
                    out[i] = _describe(im)
            except Exception:
                continue
        return normalize_rows(out)
