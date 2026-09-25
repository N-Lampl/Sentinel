"""Adapter interface.

An adapter knows how to read one dataset format and produce a
:class:`~dataset_sentinel.model.Dataset`. It should:

* create one :class:`Sample` per unit of data with ``split`` set;
* attach :class:`Annotation` objects (labels/targets) to samples;
* put any per-sample metadata the format carries into ``sample.metadata``
  (image ``width``/``height`` when known, timestamps, licences, ...);
* report *load-time* structural problems (missing files, malformed
  annotation entries) as :class:`Finding` objects in ``LoadResult.findings``.
  Semantic checks belong in detectors.

Adapters must not compute fingerprints or run integrity checks; the engine
does that so every adapter benefits from every detector.

Future modality adapters (video, tabular, text, audio, time series,
multimodal) implement this same interface and set ``modality`` accordingly.
Detectors declare which modalities they support and are skipped otherwise.
See ``docs/adapters.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from ..config import SentinelConfig
from ..model import Dataset, Finding

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".jp2", ".dng", ".heic", ".heif"}


@dataclass
class LoadResult:
    dataset: Dataset
    findings: List[Finding] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Base class for dataset format adapters."""

    #: registry name, e.g. ``"coco"``
    name: ClassVar[str] = "base"
    #: modality produced, e.g. ``"image"``
    modality: ClassVar[str] = "image"
    description: ClassVar[str] = ""

    def __init__(self, config: SentinelConfig):
        self.config = config

    @classmethod
    def detect(cls, root: Path) -> Optional[Dict[str, Any]]:
        """Return a partial ``dataset`` config when ``root`` looks like this
        format, otherwise ``None``. Used by ``sentinel scan <path>`` when no
        format is given."""
        return None

    @abstractmethod
    def load(self) -> LoadResult:
        """Read the dataset described by ``self.config``."""

    # ------------------------------------------------------------------ helpers
    @property
    def root(self) -> Path:
        return self.config.root

    def resolve(self, value: Any, relative_to: Optional[Path] = None) -> Optional[Path]:
        return self.config.resolve_path(value, relative_to or self.root)

    def max_samples(self) -> Optional[int]:
        v = self.config.get("performance.max_samples_per_split")
        return int(v) if v else None

    @staticmethod
    def display_uri(path: Path, root: Path) -> str:
        """Portable sample uri: relative to the root when possible, always with
        forward slashes so finding ids and reports are identical on every OS."""
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS
