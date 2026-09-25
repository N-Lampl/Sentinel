"""Post-processing applied after all detectors ran:

* split-pair severity overrides (``policy.split_pair_overrides``);
* allowlist suppressions (``allowlist``);
* unified cross-split clusters over every violating relationship group.
"""

from __future__ import annotations

import fnmatch
import hashlib
from collections import Counter, defaultdict
from datetime import date, datetime
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .config import SentinelConfig, split_role
from .fingerprints.index import union_find_groups
from .model import SEVERITY_ORDER, Confidence, Dataset, Finding, RelationshipGroup, Severity

_CONF_RANK = {Confidence.DETERMINISTIC: 0, Confidence.HIGH_CONFIDENCE: 1, Confidence.HEURISTIC: 2, Confidence.INCONCLUSIVE: 3}


# --------------------------------------------------------------------------- split pair overrides
def _pair_key(a: str, b: str) -> str:
    return "/".join(sorted((a, b)))


def apply_split_pair_overrides(findings: List[Finding], groups: List[RelationshipGroup], config: SentinelConfig) -> Dict[str, Any]:
    raw = config.get("policy.split_pair_overrides") or {}
    if not raw:
        return {"applied": False}
    overrides: Dict[str, Optional[Severity]] = {}
    for key, value in raw.items():
        parts = [p.strip() for p in str(key).replace(",", "/").split("/") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"policy.split_pair_overrides key {key!r} must name two split roles, e.g. 'train/val'")
        overrides[_pair_key(*parts)] = Severity.parse(value)
    by_group = {g.id: g for g in groups}
    changed = 0
    for f in findings:
        g = by_group.get(f.group_id or "")
        if g is None or len(g.splits) < 2:
            continue
        roles = sorted({split_role(s) or s for s in g.splits})
        pair_keys = [_pair_key(a, b) for a, b in combinations(roles, 2)] if len(roles) > 1 else []
        if not pair_keys or any(k not in overrides for k in pair_keys):
            continue  # some spanned pair keeps the detector's default policy
        sev_values = [overrides[k] for k in pair_keys]
        if all(v is None for v in sev_values):
            new_sev: Optional[Severity] = None
        else:
            new_sev = min((v for v in sev_values if v is not None), key=lambda s: SEVERITY_ORDER[s])
        if new_sev is None:
            new_sev = Severity.INFO
        if new_sev is not f.severity:
            changed += 1
        f.severity = new_sev
        f.policy = f"policy.split_pair_overrides[{'|'.join(pair_keys)}] = {new_sev.value}"
        violates = new_sev in (Severity.ERROR, Severity.WARNING)
        f.counts_toward_violation_rate = violates
        g.violates_policy = violates
    return {"applied": True, "changed_findings": changed, "overrides": {k: (v.value if v else "ignore") for k, v in overrides.items()}}


# --------------------------------------------------------------------------- allowlist
def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise ValueError(f"allowlist 'expires' must be YYYY-MM-DD, got {value!r}") from None


def _entry_matches(entry: Dict[str, Any], f: Finding) -> bool:
    if entry.get("id") and str(entry["id"]) != f.id:
        return False
    if entry.get("group") and str(entry["group"]) != (f.group_id or ""):
        return False
    if entry.get("kind") and str(entry["kind"]) != f.kind:
        return False
    if entry.get("detector") and str(entry["detector"]) != f.detector:
        return False
    patterns = entry.get("samples")
    if patterns:
        pats = [str(p) for p in (patterns if isinstance(patterns, list) else [patterns])]
        if not f.samples:
            return False
        for s in f.samples:
            if not any(fnmatch.fnmatch(s.uri, p) or fnmatch.fnmatch(s.id, p) for p in pats):
                return False
    keys = {"id", "group", "kind", "detector", "samples"}
    return any(entry.get(k) for k in keys)


def apply_allowlist(findings: List[Finding], groups: List[RelationshipGroup], config: SentinelConfig, today: Optional[date] = None) -> Dict[str, Any]:
    entries = config.get("allowlist") or []
    if not entries:
        return {"applied": False, "suppressed": 0}
    today = today or date.today()
    by_group = {g.id: g for g in groups}
    active: List[Tuple[int, Dict[str, Any]]] = []
    expired: List[int] = []
    warnings: List[str] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"allowlist[{i}] must be a mapping")
        if not entry.get("reason"):
            warnings.append(f"allowlist[{i}] has no 'reason'; add one so reviewers know why the finding is permitted")
        exp = _parse_date(entry.get("expires"))
        if exp is not None and exp < today:
            expired.append(i)
            continue
        active.append((i, entry))
    suppressed = 0
    used: Counter = Counter()
    for f in findings:
        for i, entry in active:
            if _entry_matches(entry, f):
                f.suppressed = {"reason": entry.get("reason", ""), "entry": i, "expires": str(entry.get("expires")) if entry.get("expires") else None}
                f.severity = Severity.INFO
                f.counts_toward_violation_rate = False
                g = by_group.get(f.group_id or "")
                if g is not None:
                    g.violates_policy = False
                suppressed += 1
                used[i] += 1
                break
    unused = [i for i, _ in active if used[i] == 0]
    return {
        "applied": True,
        "entries": len(entries),
        "suppressed": suppressed,
        "expired_entries": expired,
        "unused_entries": unused,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- clusters
def build_clusters(dataset: Dataset, groups: Sequence[RelationshipGroup], config: SentinelConfig, max_samples: int = 24) -> List[Dict[str, Any]]:
    """Connected components over the members of all policy-violating groups."""
    violating = [g for g in groups if g.violates_policy]
    if not violating:
        return []
    ids: List[str] = []
    pos: Dict[str, int] = {}
    for g in violating:
        for m in g.member_ids:
            if m not in pos:
                pos[m] = len(ids)
                ids.append(m)
    edges = []
    for g in violating:
        members = [pos[m] for m in g.member_ids]
        edges.extend(zip(members, members[1:]))
    comps = union_find_groups(len(ids), edges)
    order = {role: i for i, role in enumerate(config.split_order)}

    def rank(split: str) -> tuple:
        return (order.get(split_role(split) or "", 99), split)

    clusters: List[Dict[str, Any]] = []
    for comp in comps:
        member_ids = [ids[i] for i in comp]
        member_set = set(member_ids)
        samples = [dataset.get_or_none(m) for m in member_ids]
        samples = [s for s in samples if s is not None]
        per_split: Counter = Counter(s.split for s in samples)
        if len(per_split) < 2:
            continue
        cluster_groups = [g for g in violating if member_set.intersection(g.member_ids)]
        by_kind: Counter = Counter(g.kind for g in cluster_groups)
        by_detector: Counter = Counter(g.detector for g in cluster_groups)
        group_keys: Dict[str, Set[str]] = defaultdict(set)
        for g in cluster_groups:
            if g.kind.startswith("group_overlap:"):
                group_keys[g.kind.split(":", 1)[1]].add(str(g.evidence.get("group_value")))
        best_conf = min((g.confidence for g in cluster_groups), key=lambda c: _CONF_RANK[c])
        keep = min(per_split, key=rank)
        to_move = sum(n for s, n in per_split.items() if s != keep)
        evidence_lines = []
        for kind, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
            if kind.startswith("group_overlap:"):
                continue  # covered by the "shared <key>" lines below
            evidence_lines.append(f"{n} {kind.replace('_', ' ')} group{'s' if n != 1 else ''}")
        for key, values in group_keys.items():
            vals = sorted(values)
            evidence_lines.append(f"shared {key}: {', '.join(vals[:5])}{' ...' if len(vals) > 5 else ''}")
        cid = hashlib.sha1(",".join(sorted(member_ids)).encode()).hexdigest()[:12]

        def plural(n: int) -> str:
            return "sample" if n == 1 else "samples"

        rec = (
            f"Keep the {per_split[keep]} {plural(per_split[keep])} in {keep} and remove (or move to {keep}) the {to_move} {plural(to_move)} in "
            + ", ".join(f"{s} ({n})" for s, n in sorted(per_split.items()) if s != keep)
        )
        if group_keys:
            key, values = next(iter(group_keys.items()))
            vals = sorted(values)
            rec = f"Move every sample with {key} in {{{', '.join(vals[:5])}{', ...' if len(vals) > 5 else ''}}} into one split, then rescan. " + rec
        clusters.append(
            {
                "id": cid,
                "size": len(samples),
                "splits": dict(sorted(per_split.items())),
                "groups": len(cluster_groups),
                "group_ids": [g.id for g in cluster_groups][:50],
                "by_kind": dict(by_kind),
                "by_detector": dict(by_detector),
                "group_keys": {k: sorted(v)[:20] for k, v in group_keys.items()},
                "confidence": best_conf.value,
                "evidence": evidence_lines,
                "recommended_split": keep,
                "recommendation": rec,
                "samples": [s.ref().to_dict() for s in samples[:max_samples]],
            }
        )
    clusters.sort(key=lambda c: (-c["size"], c["id"]))
    return clusters


__all__ = ["apply_split_pair_overrides", "apply_allowlist", "build_clusters"]
