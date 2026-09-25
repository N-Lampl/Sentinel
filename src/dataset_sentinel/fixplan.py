"""Suggested remediation plan for split-integrity violations.

For every policy-violating relationship group the plan keeps the members that
live in the *earliest* split of ``policy.split_order`` (normally ``train``,
so no training data is lost) and proposes to **remove** the members in the
other splits. Removing from the evaluation side is the conservative choice:
it never leaks training content and only shrinks the evaluation sets. Each
action also lists the alternative (``move_to`` the kept split), so that a
group can be relocated instead if evaluation coverage matters more.

The plan is a suggestion. It does not modify any file.
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SentinelConfig, split_role
from .model import Dataset, Report


def build_fix_plan(report: Report, dataset: Dataset, config: SentinelConfig) -> Dict[str, Any]:
    order = {role: i for i, role in enumerate(config.split_order)}

    def rank(split: str) -> tuple:
        return (order.get(split_role(split) or "", 99), split)

    actions: Dict[str, Dict[str, Any]] = {}
    groups_handled = 0
    for g in report.groups:
        if not g.violates_policy:
            continue
        members = [dataset.get_or_none(m) for m in g.member_ids]
        members = [m for m in members if m is not None]
        splits = {m.split for m in members}
        if len(splits) < 2:
            continue
        groups_handled += 1
        keep_split = min(splits, key=rank)
        for m in members:
            if m.split == keep_split:
                continue
            entry = actions.setdefault(
                m.id,
                {
                    "sample_id": m.id,
                    "split": m.split,
                    "uri": m.uri,
                    "path": str(m.path) if m.path else None,
                    "action": "remove",
                    "move_to": keep_split,
                    "reasons": [],
                    "groups": [],
                    "kept_samples": [],
                },
            )
            if g.kind not in entry["reasons"]:
                entry["reasons"].append(g.kind)
            entry["groups"].append(g.id)
            for k in members:
                if k.split == keep_split and k.id not in entry["kept_samples"]:
                    entry["kept_samples"].append(k.id)
            if rank(keep_split) < rank(entry["move_to"]):
                entry["move_to"] = keep_split

    by_split: Dict[str, int] = defaultdict(int)
    for a in actions.values():
        by_split[a["split"]] += 1
    sizes = dataset.split_sizes()
    resulting = {s: n - by_split.get(s, 0) for s, n in sizes.items()}
    summary = {
        "violating_groups": groups_handled,
        "samples_to_remove": len(actions),
        "by_split": dict(sorted(by_split.items())),
        "current_split_sizes": sizes,
        "resulting_split_sizes": resulting,
        "strategy": "keep the members in the earliest split of policy.split_order, remove the others (or move them there)",
    }
    return {
        "schema_version": 1,
        "dataset": report.dataset.get("name"),
        "generated_at": report.generated_at,
        "summary": summary,
        "actions": sorted(actions.values(), key=lambda a: (a["split"], a["uri"])),
    }


def render_fix_plan(plan: Dict[str, Any], fmt: str = "json") -> str:
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["sample_id", "split", "uri", "path", "action", "move_to", "reasons", "groups", "kept_samples"])
        for a in plan["actions"]:
            w.writerow([a["sample_id"], a["split"], a["uri"], a["path"] or "", a["action"], a["move_to"], "|".join(a["reasons"]), "|".join(a["groups"]), "|".join(a["kept_samples"])])
        return buf.getvalue()
    return json.dumps(plan, indent=2) + "\n"


def write_fix_plan(plan: Dict[str, Any], path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = "csv" if p.suffix.lower() == ".csv" else "json"
    p.write_text(render_fix_plan(plan, fmt), encoding="utf-8")
    return p


__all__ = ["build_fix_plan", "render_fix_plan", "write_fix_plan"]
_ = (List, Optional)
