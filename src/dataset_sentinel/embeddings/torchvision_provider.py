"""ImageNet ResNet-18 features via torchvision (optional, experimental).

Install with ``pip install "dataset-sentinel[embeddings]"``. Runs on CPU by
default (set ``SENTINEL_TORCH_DEVICE=cuda`` to use a GPU). Weights are
downloaded by torchvision on first use; the cache key includes the weights
name so cached vectors are recomputed when it changes.

This provider is exercised only through the shared provider interface in the
test-suite (the tests use a stub); treat it as best-effort.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .base import EmbeddingProvider, embeddings, normalize_rows

_WEIGHTS = "IMAGENET1K_V1"


@embeddings.register("torchvision")
class TorchvisionResNet18(EmbeddingProvider):
    name = "torchvision"
    version = f"resnet18-{_WEIGHTS}"
    dimension = 512
    description = "ImageNet ResNet-18 penultimate features (optional extra: torch + torchvision), experimental"

    @classmethod
    def available(cls) -> bool:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self) -> None:
        import torch
        from torchvision import models, transforms

        self._torch = torch
        self._device = os.environ.get("SENTINEL_TORCH_DEVICE", "cpu")
        weights = models.ResNet18_Weights.__members__[_WEIGHTS]
        model: Any = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(self._device)
        self._model = model
        self._tf = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def embed(self, paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        from PIL import Image, ImageOps

        torch = self._torch
        out = np.zeros((len(paths), self.dimension), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                chunk = paths[start : start + batch_size]
                tensors = []
                ok = []
                for j, p in enumerate(chunk):
                    try:
                        with Image.open(p) as im:
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            tensors.append(self._tf(im))
                            ok.append(start + j)
                    except Exception:
                        continue
                if not tensors:
                    continue
                batch = torch.stack(tensors).to(self._device)
                feats = self._model(batch).cpu().numpy().astype(np.float32)
                for row, idx in zip(feats, ok):
                    out[idx] = row
        return normalize_rows(out)
