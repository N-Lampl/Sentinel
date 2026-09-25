"""Plugin registry for adapters, detectors and reporters.

Built-in plugins are registered by importing their modules. Third-party
plugins are discovered through ``importlib.metadata`` entry points in the
groups ``dataset_sentinel.adapters``, ``dataset_sentinel.detectors`` and
``dataset_sentinel.reporters``.

Example third-party plugin (``pyproject.toml``)::

    [project.entry-points."dataset_sentinel.detectors"]
    my_check = "my_package.checks:MyDetector"
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Callable, Dict, Generic, Iterable, List, Optional, Type, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, group: str, builtin_modules: Iterable[str] = ()):
        self.group = group
        self._items: Dict[str, Type[T]] = {}
        self._order: List[str] = []
        self._builtin_modules = list(builtin_modules)
        self._builtins_loaded = False
        self._entry_points_loaded = False

    # --------------------------------------------------------------- registration
    def register(self, name: str, cls: Optional[Type[T]] = None) -> Callable[[Type[T]], Type[T]] | Type[T]:
        def _register(target: Type[T]) -> Type[T]:
            if name in self._items and self._items[name] is not target:
                log.debug("Overriding %s plugin %r", self.group, name)
            if name not in self._items:
                self._order.append(name)
            self._items[name] = target
            return target

        if cls is not None:
            return _register(cls)
        return _register

    # --------------------------------------------------------------- discovery
    def _load_builtins(self) -> None:
        if self._builtins_loaded:
            return
        self._builtins_loaded = True
        for module in self._builtin_modules:
            try:
                importlib.import_module(module)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Failed to import built-in plugin module %s: %s", module, exc)

    def _load_entry_points(self) -> None:
        if self._entry_points_loaded:
            return
        self._entry_points_loaded = True
        try:
            from importlib import metadata
        except ImportError:  # pragma: no cover
            return
        try:
            if sys.version_info >= (3, 10):
                eps = metadata.entry_points(group=self.group)
            else:  # pragma: no cover - py39
                eps = metadata.entry_points().get(self.group, [])
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("entry point discovery failed for %s: %s", self.group, exc)
            return
        for ep in eps:
            if ep.name in self._items:
                continue
            try:
                self.register(ep.name, ep.load())
            except Exception as exc:
                log.warning("Failed to load plugin %s from %s: %s", ep.name, self.group, exc)

    def load_all(self) -> None:
        self._load_builtins()
        self._load_entry_points()

    # --------------------------------------------------------------- lookup
    def get(self, name: str) -> Type[T]:
        self.load_all()
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"Unknown {self.group.split('.')[-1][:-1]} {name!r}. Available: {', '.join(self.names())}") from None

    def names(self) -> List[str]:
        self.load_all()
        return list(self._order)

    def items(self) -> List[tuple[str, Type[T]]]:
        self.load_all()
        return [(n, self._items[n]) for n in self._order]

    def __contains__(self, name: str) -> bool:
        self.load_all()
        return name in self._items


adapters: Registry = Registry(
    "dataset_sentinel.adapters",
    builtin_modules=[
        "dataset_sentinel.adapters.coco",
        "dataset_sentinel.adapters.yolo",
        "dataset_sentinel.adapters.voc",
        "dataset_sentinel.adapters.image_folder",
    ],
)

detectors: Registry = Registry(
    "dataset_sentinel.detectors",
    builtin_modules=[
        "dataset_sentinel.detectors.label_validity",
        "dataset_sentinel.detectors.exact_duplicate",
        "dataset_sentinel.detectors.near_duplicate",
        "dataset_sentinel.detectors.derivative",
        "dataset_sentinel.detectors.group_overlap",
        "dataset_sentinel.detectors.lineage",
        "dataset_sentinel.detectors.temporal",
        "dataset_sentinel.detectors.consistency",
        "dataset_sentinel.detectors.distribution",
    ],
)

reporters: Registry = Registry(
    "dataset_sentinel.reporters",
    builtin_modules=[
        "dataset_sentinel.reporters.json_reporter",
        "dataset_sentinel.reporters.html_reporter",
        "dataset_sentinel.reporters.console",
        "dataset_sentinel.reporters.markdown_reporter",
        "dataset_sentinel.reporters.sarif_reporter",
        "dataset_sentinel.reporters.dvc_reporter",
    ],
)
