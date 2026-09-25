"""Config, enrichment helpers, metrics, reporters and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CocoBuilder, run_scan
from dataset_sentinel import __version__
from dataset_sentinel.cli import main
from dataset_sentinel.config import SentinelConfig, config_template, deep_merge, load_config, split_role
from dataset_sentinel.enrich import parse_timestamp
from dataset_sentinel.metrics import compute_metrics, evaluate_fail_conditions
from dataset_sentinel.model import (
    Confidence,
    Dataset,
    Finding,
    RelationshipGroup,
    Sample,
    SampleRef,
    Severity,
)
from dataset_sentinel.registry import adapters, detectors, reporters
from dataset_sentinel.reporters.html_reporter import HtmlReporter
from dataset_sentinel.reporters.json_reporter import JsonReporter
from dataset_sentinel.reporters.markdown_reporter import MarkdownReporter


# ----------------------------------------------------------------------------- config
def test_deep_merge_and_accessors():
    cfg = SentinelConfig({"policy": {"near_duplicate": {"threshold": 3}}})
    assert cfg.get("policy.near_duplicate.threshold") == 3
    assert cfg.get("policy.near_duplicate.cross_split") == "error"  # default kept
    assert cfg.severity("policy.near_duplicate.cross_split") is Severity.ERROR
    cfg.set("policy.near_duplicate.cross_split", "allow")
    assert cfg.severity("policy.near_duplicate.cross_split") is Severity.INFO
    cfg.set("policy.near_duplicate.cross_split", "ignore")
    assert cfg.severity("policy.near_duplicate.cross_split") is None
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}


def test_severity_parse_rejects_garbage():
    with pytest.raises(ValueError):
        Severity.parse("loud")


def test_load_config_yaml_and_json(tmp_path):
    y = tmp_path / "sentinel.yaml"
    y.write_text("dataset:\n  format: yolo\n  root: data\npolicy:\n  exact_duplicate:\n    cross_split: warning\n")
    cfg = load_config(y)
    assert cfg.get("dataset.format") == "yolo" and cfg.root == (tmp_path / "data")
    assert cfg.severity("policy.exact_duplicate.cross_split") is Severity.WARNING
    j = tmp_path / "sentinel.json"
    j.write_text(json.dumps({"fail_on": {"severity": "warning"}}))
    assert load_config(j).get("fail_on.severity") == "warning"
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_split_role():
    assert split_role("train2017") == "train"
    assert split_role("valid") == "val"
    assert split_role("holdout") == "test"
    assert split_role("weird") is None


def test_config_template_is_valid_yaml():
    import yaml

    for fmt in ("coco", "yolo", "image-folder"):
        data = yaml.safe_load(config_template(fmt))
        assert data["dataset"]["format"] == fmt and "policy" in data


# ----------------------------------------------------------------------------- enrichment
@pytest.mark.parametrize(
    "value,expected",
    [
        ("2024-01-02T03:04:05Z", 1704164645.0),
        ("2024-01-02 03:04:05", 1704164645.0),
        ("2024:01:02 03:04:05", 1704164645.0),
        (1704164645, 1704164645.0),
        (1704164645000, 1704164645.0),
        ("1704164645", 1704164645.0),
        ("20240102", 1704153600.0),
        ("not a date", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_timestamp(value, expected):
    assert parse_timestamp(value) == expected


# ----------------------------------------------------------------------------- metrics
def _dataset(n_train=6, n_test=4) -> Dataset:
    samples = [Sample(id=f"train:{i}", split="train", uri=f"t{i}.jpg") for i in range(n_train)]
    samples += [Sample(id=f"test:{i}", split="test", uri=f"e{i}.jpg") for i in range(n_test)]
    return Dataset(name="x", samples=samples, splits=["train", "test"])


def test_compute_metrics_counts_unique_samples_and_confidence():
    ds = _dataset()
    groups = [
        RelationshipGroup("g1", "exact_duplicate", ["train:0", "test:0"], ["train", "test"], Confidence.DETERMINISTIC, "exact_duplicate", True),
        RelationshipGroup("g2", "near_duplicate", ["train:0", "test:1"], ["train", "test"], Confidence.HEURISTIC, "near_duplicate", True),
        RelationshipGroup("g3", "near_duplicate", ["train:1", "train:2"], ["train"], Confidence.HEURISTIC, "near_duplicate", False),
    ]
    m = compute_metrics(ds, groups, [])
    assert m.evaluated_samples == 10 and m.violating_samples == 3 and m.violating_groups == 2
    assert m.violation_rate == pytest.approx(0.3)
    assert m.by_confidence["strict"]["violating_samples"] == 2
    assert m.by_split["test"]["violating_samples"] == 2 and m.by_split["train"]["violating_samples"] == 1
    assert m.by_detector["near_duplicate"]["violating_samples"] == 2


def test_evaluate_fail_conditions():
    ds = _dataset()
    m = compute_metrics(ds, [], [])
    f = Finding("d", "k", "t", Severity.WARNING, Confidence.HEURISTIC, "p", "pd", "m", "r", samples=[SampleRef("train:0", "train", "x")])
    assert evaluate_fail_conditions({"severity": "error", "max_violation_rate": 0}, m, [f]) == []
    assert evaluate_fail_conditions({"severity": "warning", "max_violation_rate": 0}, m, [f])
    m.violation_rate = 0.05
    reasons = evaluate_fail_conditions({"severity": "none", "max_violation_rate": 0.01}, m, [f])
    assert len(reasons) == 1 and "exceeds" in reasons[0]


def test_finding_id_is_stable():
    a = Finding("d", "k", "t", Severity.INFO, Confidence.HEURISTIC, "p", "pd", "m", "r", samples=[SampleRef("a", "train", "a.jpg")])
    b = Finding("d", "k", "t", Severity.INFO, Confidence.HEURISTIC, "p", "pd", "m", "r", samples=[SampleRef("a", "train", "a.jpg")])
    assert a.id == b.id and a.splits == ["train"]


# ----------------------------------------------------------------------------- registry
def test_registries_list_builtins():
    assert {"coco", "yolo", "image-folder"} <= set(adapters.names())
    assert {"exact_duplicate", "near_duplicate", "derivative", "group_overlap", "lineage", "temporal", "label_validity", "distribution", "consistency"} <= set(detectors.names())
    assert {"json", "html", "console", "markdown"} <= set(reporters.names())
    with pytest.raises(KeyError):
        detectors.get("does_not_exist")


# ----------------------------------------------------------------------------- reporters
@pytest.fixture
def leaky_report(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(3)
    src = b.path("train", b.images["train"][0]["file_name"])
    (b.root / "test" / "copy.jpg").write_bytes(src.read_bytes())
    b.images["test"].append({"id": 900, "file_name": "copy.jpg", "width": 160, "height": 120, "patient_id": "Q"})
    b.images["train"][1]["patient_id"] = "Q"
    b.write()
    return run_scan(b.root, output__thumbnails=True)


def test_json_report_round_trip(leaky_report):
    text = JsonReporter().render(leaky_report)
    data = json.loads(text)
    assert data["schema_version"] == 1 and data["tool"]["version"] == __version__
    assert data["metrics"]["split_integrity_violation_rate"] == pytest.approx(leaky_report.metrics.violation_rate)
    assert data["findings"] and all({"policy", "evidence", "confidence", "samples", "splits", "remediation"} <= set(f) for f in data["findings"])
    assert data["passed"] is False and data["failure_reasons"]


def test_html_report_contains_findings_and_thumbnails(leaky_report):
    html = HtmlReporter(SentinelConfig({"output": {"thumbnails": True}})).render(leaky_report)
    assert "<!DOCTYPE html>" in html and "Split integrity violation rate" in html
    assert "Exact duplicate across splits" in html and "patient_id=Q" in html
    assert "data:image/jpeg;base64," in html
    assert "prefers-color-scheme: dark" in html
    assert "<script" in html and "Class distribution" in html
    # escaping: no raw '<' from evidence leaks into markup
    assert "&lt;no path&gt;" not in html


def test_markdown_report(leaky_report):
    md = MarkdownReporter().render(leaky_report)
    assert md.startswith("## Dataset Sentinel: ❌ FAIL")
    assert "| Split integrity violation rate |" in md and "exact_duplicate" in md


# ----------------------------------------------------------------------------- CLI
def test_cli_scan_exit_codes_and_outputs(tmp_path, capsys):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(3)
    src = b.path("train", b.images["train"][0]["file_name"])
    (b.root / "test" / "copy.jpg").write_bytes(src.read_bytes())
    b.images["test"].append({"id": 900, "file_name": "copy.jpg", "width": 160, "height": 120})
    b.add_annotation("test", 900, 1, [10, 10, 50, 40])  # same labels as the source image
    b.write()
    out_json = tmp_path / "r.json"
    out_html = tmp_path / "r.html"
    out_md = tmp_path / "r.md"
    code = main(["scan", str(b.root), "--json", str(out_json), "--html", str(out_html), "--markdown", str(out_md), "--no-color", "--no-cache"])
    assert code == 1
    captured = capsys.readouterr()
    assert "RESULT: FAIL" in captured.out and "exact_duplicate" in captured.out
    assert out_json.exists() and out_html.exists() and out_md.exists()
    data = json.loads(out_json.read_text())
    assert data["metrics"]["violating_samples"] == 2

    code = main(["scan", str(b.root), "--no-report", "--fail-on", "none", "--max-violation-rate", "0.5", "-q"])
    assert code == 0
    assert capsys.readouterr().out.startswith("PASS")

    code = main(["scan", str(b.root), "--no-report", "-q", "--set", "policy.exact_duplicate.cross_split=info", "--set", "fail_on.max_violation_rate=1"])
    assert code == 0


def test_cli_scan_errors_return_2(tmp_path, capsys):
    assert main(["scan", str(tmp_path), "--no-report"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_config_discovery_and_split_args(tmp_path, capsys):
    b = CocoBuilder(tmp_path / "d", splits=("train", "val")).fill(2)
    b.write()
    (b.root / "sentinel.yaml").write_text("dataset:\n  name: named\nfail_on:\n  severity: none\n")
    code = main(["scan", str(b.root), "--no-report", "--no-color", "--split", "train=annotations/instances_train.json:train", "--split", "val=annotations/instances_val.json"])
    assert code == 0
    out = capsys.readouterr()
    assert "named" in out.out and "using config" in out.err


def test_cli_init_and_plugins(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--format", "yolo"]) == 0
    assert (tmp_path / "sentinel.yaml").exists()
    assert main(["init"]) == 2  # exists
    assert main(["init", "--force"]) == 0
    assert main(["plugins"]) == 0
    out = capsys.readouterr().out
    assert "near_duplicate" in out and "coco" in out
    assert main(["version"]) == 0


def test_engine_rejects_unknown_format(tmp_path):
    from dataset_sentinel.engine import scan

    with pytest.raises(KeyError):
        scan(SentinelConfig({"dataset": {"root": str(tmp_path), "format": "parquet"}}))


def test_dataset_reindex_orders_splits():
    ds = Dataset(name="x", samples=[Sample(id="b:1", split="b", uri="1"), Sample(id="a:1", split="a", uri="2")], splits=["a"])
    assert ds.splits == ["a", "b"] and ds.get("b:1").split == "b" and ds.split_sizes() == {"a": 1, "b": 1}
    assert Path(ds.get("a:1").uri).name == "2"
