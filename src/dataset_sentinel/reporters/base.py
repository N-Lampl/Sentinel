from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Optional

from ..config import SentinelConfig
from ..model import Report


class Reporter(ABC):
    name: ClassVar[str] = "base"
    extension: ClassVar[str] = ".txt"

    def __init__(self, config: Optional[SentinelConfig] = None):
        self.config = config or SentinelConfig()

    @abstractmethod
    def render(self, report: Report) -> str:
        ...

    def write(self, report: Report, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(report), encoding="utf-8")
        return p
