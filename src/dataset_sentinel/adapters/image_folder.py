"""Image-folder adapter (classification layout or plain image directories).

Configuration::

    dataset:
      format: image-folder
      root: ./data
      splits: {train: train, val: val, test: test}

If a split directory contains sub-directories with images, each
sub-directory name becomes the class label of the images inside
(``train/cat/1.jpg`` -> category ``cat``). Otherwise samples have no
annotations and only the leakage / quality checks apply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..model import Annotation, Dataset, Sample
from ..registry import adapters
from .base import DatasetAdapter, LoadResult, is_image_file

_SPLIT_DIR_NAMES = ("train", "training", "val", "valid", "validation", "test", "testing", "dev", "holdout")


@adapters.register("image-folder")
class ImageFolderAdapter(DatasetAdapter):
    name = "image-folder"
    modality = "image"
    description = "Directories per split, optional class sub-directories"

    @classmethod
    def detect(cls, root: Path) -> Optional[Dict[str, Any]]:
        splits = {}
        for name in _SPLIT_DIR_NAMES:
            d = root / name
            if d.is_dir():
                splits[name] = name
        if len(splits) >= 2:
            return {"format": "image-folder", "splits": splits}
        return None

    def load(self) -> LoadResult:
        splits_cfg = self.config.section("dataset.splits")
        if not splits_cfg:
            detected = self.detect(self.root)
            if not detected:
                raise ValueError(f"No split directories found under {self.root}. Set dataset.splits in the config.")
            splits_cfg = detected["splits"]

        samples: List[Sample] = []
        categories: Dict[Any, str] = {}
        stats: Dict[str, Any] = {"splits": {}}
        limit = self.max_samples()
        source: Dict[str, Any] = {"format": "image-folder", "root": str(self.root), "splits": {}}

        for split_name, spec in splits_cfg.items():
            d = self.resolve(spec if not isinstance(spec, dict) else spec.get("images") or spec.get("dir"))
            assert d is not None
            source["splits"][split_name] = str(d)
            if not d.is_dir():
                raise FileNotFoundError(f"Split directory for {split_name!r} not found: {d}")
            split_samples: List[Sample] = []
            for path in sorted(p for p in d.rglob("*") if p.is_file() and is_image_file(p)):
                rel_in_split = path.relative_to(d)
                label = rel_in_split.parts[0] if len(rel_in_split.parts) > 1 else None
                uri = self.display_uri(path, self.root)
                sample = Sample(id=f"{split_name}:{uri}", split=split_name, uri=uri, path=path, native_id=str(rel_in_split))
                if label is not None:
                    categories.setdefault(label, label)
                    sample.annotations.append(
                        Annotation(id=f"{sample.id}#label", sample_id=sample.id, category=label, category_id=label)
                    )
                split_samples.append(sample)
                if limit and len(split_samples) >= limit:
                    break
            samples.extend(split_samples)
            stats["splits"][split_name] = {"images": len(split_samples)}

        dataset = Dataset(
            name=self.config.get("dataset.name") or self.root.name or "dataset",
            samples=samples,
            splits=list(splits_cfg.keys()),
            modality="image",
            categories=categories,
            source=source,
        )
        return LoadResult(dataset=dataset, findings=[], stats=stats)
