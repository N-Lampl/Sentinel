from __future__ import annotations

import json

from ..model import Report
from ..registry import reporters
from .base import Reporter


@reporters.register("json")
class JsonReporter(Reporter):
    name = "json"
    extension = ".json"

    def render(self, report: Report) -> str:
        return json.dumps(report.to_dict(), indent=2, sort_keys=False, default=str) + "\n"
