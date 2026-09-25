"""Tests for the second iteration: crops/tiles, VOC, baselines & diff, allowlist,
split-pair overrides, clusters, fix plan, SARIF / DVC output, config validation,
detector failures, GitHub annotations and the baseline/diff CLI commands."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from conftest import CocoBuilder, findings_of, image_from_seed, make_config, run_scan
from dataset_sentinel.cli import emit_github_annotations, main
from dataset_sentinel.config import SentinelConfig, find_unknown_keys
from dataset_sentinel.engine import scan
from dataset_sentinel.fixplan import render_fix_plan
from dataset_sentinel.model import Confidence, Severity
from dataset_sentinel.registry import detectors
from dataset_sentinel.reporters.dvc_reporter import DvcReporter
from dataset_sentinel.reporters.sarif_reporter import SarifReporter


def _leaky(tmp_path: Path, name: str = "d") -> CocoBuilder:
    """train/test dataset with one byte-identical copy and one shared patient."""
    b = CocoBuilder(tmp_path / name, splits=("train", "test")).fill(4)
    src = b.path("train", b.images["train"][0]["file_name"])
    (b.root / "test" / "copy.jpg").write_bytes(src.read_bytes())
    b.images["test"].append({"id": 900, "file_name": "copy.jpg", "width": 160, "height": 120, "patient_id": "Q"})
    b.add_annotation("test", 900, 1, [10, 10, 50, 40])
    b.images["train"][1]["patient_id"] = "Q"
    b.write()
    return b


# ----------------------------------------------------------------------------- crops / tiles
def test_tiles_and_center_crops_detected(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(4)
    img = Image.open(b.path("train", b.images["train"][0]["file_name"])).convert("RGB")
    w, h = img.size
    b.add("test", name="tile_tr.jpg", image=img.crop((w // 2, 0, w, h // 2)))
    img2 = Image.open(b.path("train", b.images["train"][1]["file_name"])).convert("RGB")
    b.add("test", name="center.jpg", image=img2.crop((w // 4, h // 4, w - w // 4, h - h // 4)))
    img3 = Image.open(b.path("train", b.images["train"][2]["file_name"])).convert("RGB")
    b.add("test", name="left.jpg", image=img3.crop((0, 0, w // 2, h)))
    b.write()
    report = run_scan(b.root)
    deriv = findings_of(report, "derivative", "derivative_cross_split")
    transforms = {d.evidence["transforms"][0] for d in deriv}
    assert transforms == {"crop:tile_top_right", "crop:center_crop_50", "crop:left_half"}, transforms
    for d in deriv:
        p = d.evidence["pairs"][0]
        assert p["source"].startswith("train:") and p["derived"].startswith("test:")
        assert p["correlation"] >= 0.9
        assert d.confidence is Confidence.HIGH_CONFIDENCE
    assert "tile" in deriv[0].message or "crop" in deriv[0].message or "half" in deriv[0].message


def test_crop_detection_can_be_disabled(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(3)
    img = Image.open(b.path("train", b.images["train"][0]["file_name"])).convert("RGB")
    b.add("test", name="tile.jpg", image=img.crop((0, 0, 80, 60)))
    b.write()
    assert findings_of(run_scan(b.root), "derivative")
    assert not findings_of(run_scan(b.root, policy__derivative__crops=False), "derivative")


# ----------------------------------------------------------------------------- VOC
def _voc(root: Path, ids_by_split) -> Path:
    (root / "Annotations").mkdir(parents=True)
    (root / "JPEGImages").mkdir()
    (root / "ImageSets" / "Main").mkdir(parents=True)
    seed = 0
    for split, ids in ids_by_split.items():
        (root / "ImageSets" / "Main" / f"{split}.txt").write_text("\n".join(ids) + "\n")
        for i in ids:
            seed += 1
            image_from_seed(seed).save(root / "JPEGImages" / f"{i}.jpg", quality=90)
            (root / "Annotations" / f"{i}.xml").write_text(
                f"<annotation><folder>VOC</folder><filename>{i}.jpg</filename><source><database>Demo</database></source>"
                "<size><width>160</width><height>120</height><depth>3</depth></size><segmented>0</segmented>"
                "<object><name>cat</name><pose>Left</pose><truncated>0</truncated><difficult>0</difficult>"
                "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>50</ymax></bndbox></object></annotation>"
            )
    return root


def test_voc_adapter_autodetect_and_boxes(tmp_path):
    root = _voc(tmp_path / "voc", {"train": ["000001", "000002", "000003"], "val": ["000004", "000005"]})
    # class-specific list files must be ignored, missing xml tolerated
    (root / "ImageSets" / "Main" / "cat_train.txt").write_text("000001 1\n000002 -1\n")
    (root / "ImageSets" / "Main" / "val.txt").write_text("000004\n000005\n000006\n")
    image_from_seed(99).save(root / "JPEGImages" / "000006.jpg")
    report = run_scan(root)
    assert report.dataset["format"] == "voc"
    assert report.dataset["splits"]["train"]["samples"] == 3 and report.dataset["splits"]["val"]["samples"] == 3
    ds = report.dataset_ref
    s = ds.get("train:000001")
    assert s.annotations[0].category == "cat" and s.annotations[0].bbox.as_list() == [10, 10, 50, 40]
    assert s.metadata["source_database"] == "Demo" and s.width == 160
    assert findings_of(report, "label_validity", "missing_label_file")
    assert not [f for f in report.findings if f.severity is Severity.ERROR]


def test_voc_cross_split_duplicate(tmp_path):
    root = _voc(tmp_path / "voc", {"train": ["a1", "a2"], "test": ["b1"]})
    (root / "JPEGImages" / "b1.jpg").write_bytes((root / "JPEGImages" / "a1.jpg").read_bytes())
    report = run_scan(root)
    assert findings_of(report, "exact_duplicate", "exact_duplicate_cross_split")


# ----------------------------------------------------------------------------- baseline / diff
def test_baseline_marks_new_and_resolved(tmp_path):
    b = _leaky(tmp_path)
    first = run_scan(b.root)
    baseline = tmp_path / "baseline.json"
    from dataset_sentinel.reporters.json_reporter import JsonReporter

    JsonReporter().write(first, baseline)
    # add a new leak: flipped copy in test
    img = Image.open(b.path("train", b.images["train"][2]["file_name"])).convert("RGB")
    b.add("test", name="flip.jpg", image=img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    b.write()
    second = run_scan(b.root, baseline__file=str(baseline))
    new = [f for f in second.findings if f.new]
    assert new and all(f.detector == "derivative" for f in new)
    assert all(f.new is False for f in second.findings if f.detector in ("exact_duplicate", "group_overlap"))
    stats = second.stats["baseline"]
    assert stats["applied"] and stats["new"] == len(new) and stats["known"] >= 2
    diff = second.stats["diff"]
    assert diff["findings"]["new_cross_split_groups"] == 1
    assert diff["violation_rate"]["delta"] > 0
    assert diff["split_sizes"]["test"]["delta"] == 1
    assert any("new cross-split" in h for h in diff["headline"])


def test_new_only_fails_only_on_regressions(tmp_path):
    b = _leaky(tmp_path)
    first = run_scan(b.root)
    assert not first.passed
    baseline = tmp_path / "baseline.json"
    from dataset_sentinel.reporters.json_reporter import JsonReporter

    JsonReporter().write(first, baseline)
    same = run_scan(b.root, baseline__file=str(baseline), fail_on__new_only=True)
    assert same.passed, same.failure_reasons
    img = Image.open(b.path("train", b.images["train"][2]["file_name"])).convert("RGB")
    b.add("test", name="flip.jpg", image=img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    b.write()
    worse = run_scan(b.root, baseline__file=str(baseline), fail_on__new_only=True)
    assert not worse.passed and any("new finding" in r for r in worse.failure_reasons)
    assert any("baseline rate" in r for r in worse.failure_reasons)


def test_cli_baseline_create_and_diff(tmp_path, capsys, monkeypatch):
    b = _leaky(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["baseline", "create", str(b.root), "-o", "base/baseline.json", "--no-color", "--no-cache"]) == 0
    assert (tmp_path / "base" / "baseline.json").exists()
    out = capsys.readouterr().out
    assert "baseline written" in out
    assert main(["diff", str(b.root), "--baseline", "base/baseline.json", "--no-report", "--no-color", "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert "Since baseline" in out and "no material change" in out
    img = Image.open(b.path("train", b.images["train"][2]["file_name"])).convert("RGB")
    b.add("test", name="flip.jpg", image=img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    b.write()
    assert main(["diff", str(b.root), "--baseline", "base/baseline.json", "--no-report", "--no-color", "--no-cache"]) == 1
    out = capsys.readouterr().out
    assert "NEW" in out and "RESULT: FAIL" in out
    # --fail-on none disables the severity gate; the rate regression gate is separate
    assert main(["diff", str(b.root), "--baseline", "base/baseline.json", "--no-report", "--fail-on", "none", "-q", "--no-cache"]) == 1
    assert main(["diff", str(b.root), "--baseline", "base/baseline.json", "--no-report", "--fail-on", "none", "--max-violation-rate", "1", "-q", "--no-cache"]) == 0


# ----------------------------------------------------------------------------- allowlist
def test_allowlist_suppresses_by_id_samples_and_kind(tmp_path):
    b = _leaky(tmp_path)
    report = run_scan(b.root)
    dup = findings_of(report, "exact_duplicate", "exact_duplicate_cross_split")[0]
    allow = [
        {"id": dup.id, "reason": "kept on purpose"},
        {"kind": "group_overlap", "reason": "patients may span splits in this study", "expires": "2999-01-01"},
        {"samples": ["nothing/*"], "reason": "stale entry"},
        {"kind": "lineage_not_evaluated", "reason": "expired", "expires": "2000-01-01"},
    ]
    report2 = run_scan(b.root, allowlist=allow)
    suppressed = [f for f in report2.findings if f.suppressed]
    assert {f.kind for f in suppressed} == {"exact_duplicate_cross_split", "group_overlap"}
    assert all(f.severity is Severity.INFO and not f.counts_toward_violation_rate for f in suppressed)
    assert report2.metrics.violation_rate == 0.0 and report2.passed
    st = report2.stats["allowlist"]
    assert st["suppressed"] == 2 and st["unused_entries"] == [2] and st["expired_entries"] == [3]
    # sample-glob based suppression
    # every sample of the finding must match a pattern: the duplicate pair is
    # (train_0001, copy) while the patient group is (train_0002, copy)
    report3 = run_scan(b.root, allowlist=[{"samples": ["test/copy.jpg", "train/train_0001.jpg"], "reason": "calibration"}])
    assert findings_of(report3, "exact_duplicate")[0].suppressed
    assert not findings_of(report3, "group_overlap")[0].suppressed


def test_allowlist_requires_mapping(tmp_path):
    b = _leaky(tmp_path)
    with pytest.raises(ValueError):
        run_scan(b.root, allowlist=["oops"])


# ----------------------------------------------------------------------------- split pair overrides
def test_split_pair_overrides(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "val", "test")).fill(3)
    for split in ("val", "test"):
        (b.root / split / f"copy_{split}.jpg").write_bytes(b.path("train", b.images["train"][0]["file_name"]).read_bytes())
        b.images[split].append({"id": 900 + len(split), "file_name": f"copy_{split}.jpg", "width": 160, "height": 120})
        b.add_annotation(split, 900 + len(split), 1, [10, 10, 50, 40])
    b.write()
    report = run_scan(b.root, policy__split_pair_overrides={"train/val": "warning", "val/test": "warning", "train/test": "error"})
    f = findings_of(report, "exact_duplicate", "exact_duplicate_cross_split")
    assert len(f) == 1 and f[0].severity is Severity.ERROR  # spans train/val/test -> most severe pair wins
    report = run_scan(b.root, policy__split_pair_overrides={"train/val": "warning", "val/test": "info", "train/test": "info"})
    f = findings_of(report, "exact_duplicate", "exact_duplicate_cross_split")[0]
    assert f.severity is Severity.ERROR or f.severity is Severity.WARNING
    report = run_scan(b.root, policy__split_pair_overrides={"train/val": "info", "val/test": "info", "train/test": "info"})
    f = findings_of(report, "exact_duplicate", "exact_duplicate_cross_split")[0]
    assert f.severity is Severity.INFO and not f.counts_toward_violation_rate and report.metrics.violation_rate == 0.0
    with pytest.raises(ValueError):
        run_scan(b.root, policy__split_pair_overrides={"train": "info"})


# ----------------------------------------------------------------------------- clusters + fix plan
def test_clusters_unify_evidence_and_fix_plan(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(4)
    # exact copy + same patient on two different images -> one cluster of 4 samples
    src = b.path("train", b.images["train"][0]["file_name"])
    (b.root / "test" / "copy.jpg").write_bytes(src.read_bytes())
    b.images["test"].append({"id": 900, "file_name": "copy.jpg", "width": 160, "height": 120, "patient_id": "Q"})
    b.add_annotation("test", 900, 1, [10, 10, 50, 40])
    b.images["train"][1]["patient_id"] = "Q"
    b.images["test"][0]["patient_id"] = "Q"
    b.write()
    report = run_scan(b.root)
    assert len(report.clusters) == 1
    c = report.clusters[0]
    assert c["size"] == 4 and c["splits"] == {"test": 2, "train": 2}
    assert set(c["by_detector"]) == {"exact_duplicate", "group_overlap"}
    assert c["group_keys"] == {"patient_id": ["Q"]}
    assert c["recommended_split"] == "train" and "patient_id" in c["recommendation"]
    plan = report.fix_plan
    assert plan["summary"]["samples_to_remove"] == 2 and plan["summary"]["by_split"] == {"test": 2}
    assert plan["summary"]["resulting_split_sizes"]["test"] == 3
    actions = {a["sample_id"]: a for a in plan["actions"]}
    assert all(a["split"] == "test" and a["move_to"] == "train" for a in actions.values())
    assert "exact_duplicate" in actions["test:900"]["reasons"]
    csv_text = render_fix_plan(plan, "csv")
    assert csv_text.splitlines()[0].startswith("sample_id,split,uri") and "test/copy.jpg" in csv_text
    data = json.loads(render_fix_plan(plan, "json"))
    assert data["summary"]["samples_to_remove"] == 2


# ----------------------------------------------------------------------------- reporters
def test_sarif_and_dvc_outputs(tmp_path):
    b = _leaky(tmp_path)
    report = run_scan(b.root, allowlist=[{"kind": "group_overlap", "reason": "ok"}])
    sarif = json.loads(SarifReporter().render(report))
    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "dataset-sentinel"
    results = run["results"]
    dup = [r for r in results if r["ruleId"] == "exact_duplicate_cross_split"][0]
    assert dup["level"] == "error" and dup["partialFingerprints"]["datasetSentinel/findingId/v1"]
    assert dup["locations"][0]["physicalLocation"]["artifactLocation"]["uri"].endswith(".jpg")
    sup = [r for r in results if r["ruleId"] == "group_overlap"][0]
    assert sup["suppressions"][0]["justification"] == "ok" and sup["level"] == "note"
    rule_ids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    assert "exact_duplicate_cross_split" in rule_ids

    yaml_text = DvcReporter().render(report)
    assert "integrity:" in yaml_text and "exact_duplicate_groups: 1" in yaml_text
    out = DvcReporter().write(report, tmp_path / "m.json")
    data = json.loads(out.read_text())
    assert data["integrity"]["samples_train"] == 4 and data["integrity"]["passed"] is False


def test_html_shows_new_suppressed_clusters(tmp_path):
    b = _leaky(tmp_path)
    from dataset_sentinel.reporters.html_reporter import HtmlReporter
    from dataset_sentinel.reporters.json_reporter import JsonReporter

    first = run_scan(b.root)
    JsonReporter().write(first, tmp_path / "base.json")
    report = run_scan(b.root, baseline__file=str(tmp_path / "base.json"), allowlist=[{"kind": "group_overlap", "reason": "ok"}], output__thumbnails=True)
    html = HtmlReporter(SentinelConfig({"output": {"thumbnails": True}})).render(report)
    assert "Cross-split clusters" in html and "Changes since baseline" in html
    assert "Hide suppressed" in html and "New only" in html and "suppressed" in html
    assert "data:image/jpeg;base64," in html


# ----------------------------------------------------------------------------- config validation / detector errors / annotations
def test_unknown_config_keys_are_reported(tmp_path):
    warnings = find_unknown_keys({"policy": {"near_dup": {"threshold": 3}}, "outpt": {}, "dataset": {"splits": {"x": 1}}})
    assert any("policy.near_dup" in w and "near_duplicate" in w for w in warnings)
    assert any("'outpt'" in w for w in warnings)
    assert not any("dataset.splits" in w for w in warnings)
    b = _leaky(tmp_path)
    cfg = make_config(b.root)
    cfg.set("policy.exact_dup.cross_split", "info")
    report = scan(cfg)
    assert any("policy.exact_dup" in w for w in report.stats["config_warnings"])
    # plugin policy sections are allowed
    assert not find_unknown_keys({"policy": {"my_plugin": {}}}, extra_policy_keys=["my_plugin"])


def test_detector_failure_fails_the_run(tmp_path, monkeypatch):
    b = CocoBuilder(tmp_path / "d", splits=("train",)).fill(2)
    b.write()
    cls = detectors.get("distribution")

    def boom(self, ctx):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cls, "run", boom)
    report = run_scan(b.root)
    run = [d for d in report.detectors if d.name == "distribution"][0]
    assert run.status == "error" and "kaboom" in run.message
    assert not report.passed and any("detector(s) failed" in r for r in report.failure_reasons)
    assert run_scan(b.root, fail_on__detector_errors=False).passed


def test_github_annotations(tmp_path, monkeypatch):
    b = _leaky(tmp_path)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    report = run_scan(b.root)
    buf = io.StringIO()
    n = emit_github_annotations(report, stream=buf)
    lines = buf.getvalue().splitlines()
    assert n == len(lines) >= 2
    err = [ln for ln in lines if ln.startswith("::error ")][0]
    assert "file=d/" in err and "title=" in err and "::" in err[8:]
    assert all(not ln.startswith("::notice") for ln in lines)


def test_cli_scan_new_flags(tmp_path, capsys):
    b = _leaky(tmp_path)
    out_dir = tmp_path / "out"
    code = main([
        "scan", str(b.root), "--json", str(out_dir / "r.json"), "--html", str(out_dir / "r.html"),
        "--sarif", str(out_dir / "r.sarif"), "--dvc", str(out_dir / "m.yaml"), "--fix-plan", str(out_dir / "plan.csv"),
        "--no-color", "--no-cache", "--github-annotations", "--group-by", "patient_id",
    ])
    assert code == 1
    for name in ("r.json", "r.html", "r.sarif", "m.yaml", "plan.csv"):
        assert (out_dir / name).exists(), name
    out = capsys.readouterr().out
    assert "::error " in out and "Suggested fix" in out and "Cross-split clusters" in out


def test_thumb16_cache_roundtrip(tmp_path):
    from dataset_sentinel.fingerprints.image import ImageFingerprint, fingerprint_image, thumb_array

    p = tmp_path / "a.jpg"
    image_from_seed(3).save(p)
    fp = fingerprint_image(p, "a")
    back = ImageFingerprint.from_dict(json.loads(json.dumps(fp.to_dict())))
    assert back.thumb16 == fp.thumb16 and isinstance(back.thumb16, bytes)
    assert thumb_array(back).shape == (16, 16)
    assert back.dhash_regions == fp.dhash_regions and len(fp.dhash_regions) == 10


def test_coco_category_definition_mismatch(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "val")).fill(2)
    b.write()
    val = json.loads((b.root / "annotations" / "instances_val.json").read_text())
    val["categories"] = [{"id": 1, "name": "dog"}, {"id": 2, "name": "cat"}, {"id": 3, "name": "bird"}]
    (b.root / "annotations" / "instances_val.json").write_text(json.dumps(val))
    report = run_scan(b.root)
    kinds = {f.kind for f in findings_of(report, "adapter:coco")}
    assert "category_definition_conflict" in kinds and "category_set_mismatch" in kinds
    conflict = findings_of(report, "adapter:coco", "category_definition_conflict")
    assert conflict[0].severity is Severity.ERROR and conflict[0].confidence is Confidence.DETERMINISTIC


def test_consistency_object_count_mismatch(tmp_path):
    import numpy as np

    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(3)
    img = Image.open(b.path("train", b.images["train"][0]["file_name"])).convert("RGB")
    shifted = Image.fromarray(np.clip(np.asarray(img).astype(np.int16) + 6, 0, 255).astype(np.uint8))
    b.add("test", name="near.jpg", image=shifted, boxes=[(1, [10, 10, 50, 40]), (1, [70, 10, 30, 30]), (1, [5, 60, 40, 40]), (1, [100, 60, 30, 30])])
    b.write()
    report = run_scan(b.root)
    f = findings_of(report, "consistency", "object_count_mismatch_near_duplicate")
    assert len(f) == 1 and "Potential inconsistency" in f[0].title and f[0].confidence is Confidence.HEURISTIC


# ----------------------------------------------------------------------------- fast mode / large images
def test_fast_mode_skips_pixel_hash_but_keeps_perceptual_hash(tmp_path):
    from dataset_sentinel.fingerprints.image import fingerprint_image, hamming

    p = tmp_path / "big.jpg"
    image_from_seed(11, size=(1920, 1080)).save(p, quality=88)
    exact = fingerprint_image(p, "a")
    fast = fingerprint_image(p, "a", fast=True)
    assert exact.mode == "exact" and exact.pixel_hash and (exact.width, exact.height) == (1920, 1080)
    assert fast.mode == "fast" and fast.pixel_hash is None and (fast.width, fast.height) == (1920, 1080)
    assert hamming(exact.dhash, fast.dhash) <= 4
    assert all(hamming(a, b) <= 6 for a, b in zip(exact.dhash_dihedral, fast.dhash_dihedral))


@pytest.mark.parametrize("t", [1, 3, 4, 6])
def test_reduced_large_images_keep_dihedral_symmetry(tmp_path, t):
    from dataset_sentinel.fingerprints.image import _TRANSPOSE_OPS, fingerprint_image, hamming

    img = image_from_seed(21 + t, size=(1200, 800))
    img.save(tmp_path / "o.png")
    img.transpose(_TRANSPOSE_OPS[t]).save(tmp_path / "t.png")
    o = fingerprint_image(tmp_path / "o.png", "o")
    tr = fingerprint_image(tmp_path / "t.png", "t")
    assert hamming(o.dhash_dihedral[t], tr.dhash) <= 2


def test_fast_mode_end_to_end_and_cache_mode(tmp_path):
    b = _leaky(tmp_path)
    cache = tmp_path / "cache"
    fast = run_scan(b.root, performance__mode="fast", performance__cache=str(cache))
    assert fast.stats["fingerprints"]["mode"] == "fast"
    assert findings_of(fast, "exact_duplicate", "exact_duplicate_cross_split")  # byte-identical still found
    exact = run_scan(b.root, performance__mode="exact", performance__cache=str(cache))
    assert exact.stats["fingerprints"]["cache_hits"] == 0  # fast entries do not satisfy exact requests
    fast2 = run_scan(b.root, performance__mode="fast", performance__cache=str(cache))
    assert fast2.stats["fingerprints"]["cache_hits"] == fast2.stats["fingerprints"]["images"]  # exact entries satisfy fast requests


def test_cli_fast_flag(tmp_path, capsys):
    b = _leaky(tmp_path)
    assert main(["scan", str(b.root), "--no-report", "--fast", "-q", "--no-cache"]) == 1
    assert capsys.readouterr().out.startswith("FAIL")
