"""End-to-end detector tests on synthetic datasets with planted problems."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PIL import Image

from conftest import CocoBuilder, YoloBuilder, findings_of, image_from_seed, run_scan
from dataset_sentinel.model import Confidence, Severity


def _shift(img: Image.Image, delta: int = 8) -> Image.Image:
    arr = np.clip(np.asarray(img).astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------------------------------------------------------- clean dataset
def test_clean_dataset_passes(coco_root):
    report = run_scan(coco_root)
    errors = [f for f in report.findings if f.severity is Severity.ERROR]
    assert errors == [], [f.title for f in errors]
    assert report.metrics.violation_rate == 0.0
    assert report.passed
    # inconclusive checks are surfaced, not hidden
    kinds = {f.kind for f in report.findings}
    assert "group_overlap_not_evaluated" in kinds and "lineage_not_evaluated" in kinds
    assert all(f.confidence is Confidence.INCONCLUSIVE for f in report.findings if f.kind.endswith("_not_evaluated"))


def test_every_finding_states_required_fields(coco_root):
    b = CocoBuilder(coco_root / "x", splits=("train", "test"))
    a = b.add("train")
    b.add("test", image=Image.open(b.path("train", b.images["train"][0]["file_name"])).convert("RGB"), name="dup.png", fmt="PNG")
    b.write()
    report = run_scan(b.root)
    assert report.findings
    for f in report.findings:
        assert f.policy and f.policy_description and f.remediation and f.message and f.title
        assert f.confidence in Confidence and f.severity in Severity
        assert isinstance(f.evidence, dict)
    _ = a


# ----------------------------------------------------------------------------- exact duplicates
def test_exact_duplicate_cross_split(tmp_path):
    b = CocoBuilder(tmp_path / "d").fill(4)
    src = b.path("train", b.images["train"][0]["file_name"])
    (b.root / "test" / "copy.jpg").write_bytes(src.read_bytes())
    b.images["test"].append({"id": 900, "file_name": "copy.jpg", "width": 160, "height": 120})
    b.add_annotation("test", 900, 1, [10, 10, 50, 40])
    img = Image.open(b.path("train", b.images["train"][1]["file_name"])).convert("RGB")
    b.add("val", name="resaved.png", image=img, fmt="PNG")  # pixel-identical, different bytes
    b.write()
    report = run_scan(b.root)
    f = findings_of(report, "exact_duplicate", "exact_duplicate_cross_split")
    assert len(f) == 2
    hows = {tuple(x.evidence["match_types"]) for x in f}
    assert ("byte_identical", "pixel_identical") in hows and ("pixel_identical",) in hows
    assert all(x.confidence is Confidence.DETERMINISTIC and x.severity is Severity.ERROR and x.counts_toward_violation_rate for x in f)
    assert report.metrics.violating_samples == 4
    assert report.metrics.violation_rate == pytest.approx(4 / 14)
    assert report.metrics.by_detector["exact_duplicate"]["groups"] == 2
    assert not report.passed


def test_same_path_in_two_splits_yolo(tmp_path):
    b = YoloBuilder(tmp_path / "y", splits=("train", "val")).fill(3)
    (b.root / "data.yaml").write_text("train: images/train\nval: images/train\nnames: [cat, dog]\n")
    report = run_scan(b.root, "yolo")
    f = findings_of(report, "exact_duplicate", "exact_duplicate_cross_split")
    assert len(f) == 3 and all("same_path" in x.evidence["match_types"] for x in f)
    assert report.metrics.violation_rate == 1.0


def test_within_split_duplicate_is_warning_only(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train",)).fill(3)
    src = b.path("train", b.images["train"][0]["file_name"])
    (b.root / "train" / "copy.jpg").write_bytes(src.read_bytes())
    b.images["train"].append({"id": 900, "file_name": "copy.jpg", "width": 160, "height": 120})
    b.write()
    report = run_scan(b.root)
    f = findings_of(report, "exact_duplicate")
    assert len(f) == 1 and f[0].kind == "exact_duplicate_within_split" and f[0].severity is Severity.WARNING
    assert report.metrics.violation_rate == 0.0


# ----------------------------------------------------------------------------- near duplicates / derivatives
def test_near_duplicate_and_flip(tmp_path):
    b = CocoBuilder(tmp_path / "d").fill(5)
    train0 = Image.open(b.path("train", b.images["train"][0]["file_name"])).convert("RGB")
    train1 = Image.open(b.path("train", b.images["train"][1]["file_name"])).convert("RGB")
    b.add("test", name="near.jpg", image=_shift(train0, 8))
    b.add("val", name="flip.jpg", image=train1.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    b.add("val", name="rot.jpg", image=Image.open(b.path("train", b.images["train"][2]["file_name"])).convert("RGB").transpose(Image.Transpose.ROTATE_90))
    b.write()
    report = run_scan(b.root)
    near = findings_of(report, "near_duplicate", "near_duplicate_cross_split")
    assert len(near) == 1 and near[0].confidence is Confidence.HIGH_CONFIDENCE
    assert near[0].evidence["min_cross_split_distance"] <= 3
    deriv = findings_of(report, "derivative", "derivative_cross_split")
    assert len(deriv) == 2
    transforms = {d.evidence["transforms"][0] for d in deriv}
    assert transforms == {"flip_horizontal", "rotate_90"}
    assert all(d.confidence is Confidence.HIGH_CONFIDENCE for d in deriv)
    # the statement is oriented from the training image to its transformed copy
    for d in deriv:
        p = d.evidence["pairs"][0]
        assert p["source"].startswith("train:") and not p["derived"].startswith("train:")
    assert report.metrics.by_detector["near_duplicate"]["violating_samples"] == 2
    assert report.metrics.by_detector["derivative"]["violating_samples"] == 4


def test_near_duplicate_policy_can_be_relaxed(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(3)
    train0 = Image.open(b.path("train", b.images["train"][0]["file_name"])).convert("RGB")
    b.add("test", name="near.jpg", image=_shift(train0, 5))
    b.write()
    report = run_scan(b.root, policy__near_duplicate__cross_split="info")
    f = findings_of(report, "near_duplicate")
    assert f and f[0].severity is Severity.INFO and not f[0].counts_toward_violation_rate
    assert report.metrics.violation_rate == 0.0
    report = run_scan(b.root, policy__near_duplicate__cross_split="ignore", policy__near_duplicate__within_split="ignore")
    assert not findings_of(report, "near_duplicate")


# ----------------------------------------------------------------------------- groups / lineage / temporal
def test_group_overlap_from_metadata_and_sidecar(tmp_path):
    b = CocoBuilder(tmp_path / "d").fill(3)
    b.add("train", extra={"patient_id": "P7"})
    b.add("test", extra={"patient_id": "P7"})
    b.add("val", extra={"patient_id": "P8"})
    b.write()
    report = run_scan(b.root)
    f = findings_of(report, "group_overlap", "group_overlap")
    assert len(f) == 1 and f[0].evidence["group_value"] == "P7" and f[0].confidence is Confidence.DETERMINISTIC
    assert set(f[0].splits) == {"train", "test"}
    assert report.metrics.by_detector["group_overlap"]["violating_samples"] == 2

    # sidecar CSV adds a 'source' column joined on file name
    sidecar = b.root / "meta.csv"
    with sidecar.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_name", "source"])
        w.writerow([b.images["train"][0]["file_name"], "camA"])
        w.writerow([b.images["val"][0]["file_name"], "camA"])
    report = run_scan(b.root, dataset__metadata__file=str(sidecar))
    f = findings_of(report, "group_overlap", "group_overlap")
    assert {x.evidence["group_key"] for x in f} == {"patient_id", "source"}
    assert report.stats["enrich"]["metadata"]["matched_samples"] == 2


def test_group_from_filename_pattern(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"))
    b.add("train", name="S01_a.jpg")
    b.add("train", name="S02_a.jpg")
    b.add("test", name="S01_b.jpg")
    b.write()
    report = run_scan(b.root, dataset__groups__from_filename={"subject": r"^(?P<value>S\d+)_"})
    f = findings_of(report, "group_overlap", "group_overlap")
    assert len(f) == 1 and f[0].evidence["group_value"] == "S01" and f[0].confidence is Confidence.HIGH_CONFIDENCE


def test_lineage_from_metadata_and_filename(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"))
    b.add("train", name="img_001.jpg")
    b.add("train", name="img_002.jpg")
    b.add("test", name="crop.jpg", image=image_from_seed(77), extra={"derived_from": "img_001.jpg"})
    b.add("test", name="img_002_tile3.jpg", image=image_from_seed(78))
    b.write()
    report = run_scan(b.root, dataset__lineage__from_filename=[r"^(?P<parent>.+?)_tile\d+$"])
    f = findings_of(report, "lineage", "lineage_cross_split")
    assert len(f) == 2
    confs = {x.confidence for x in f}
    assert confs == {Confidence.DETERMINISTIC, Confidence.HEURISTIC}
    assert report.metrics.by_detector["lineage"]["violating_samples"] == 4
    assert report.dataset["lineage_edges"] == 2


def test_temporal_auto_vs_strict(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"))
    for i in range(4):
        b.add("train", extra={"date_captured": f"2024-01-{10 + i:02d} 10:00:00"})
    b.add("test", extra={"date_captured": "2024-01-05 10:00:00"})  # before all training samples
    b.add("test", extra={"date_captured": "2024-02-01 10:00:00"})
    b.write()
    auto = run_scan(b.root)
    f = findings_of(auto, "temporal")
    assert len(f) == 1 and f[0].kind == "temporal_overlap_info" and f[0].severity is Severity.INFO
    assert f[0].confidence is Confidence.INCONCLUSIVE and auto.metrics.violation_rate == 0.0

    strict = run_scan(b.root, policy__temporal__enabled=True)
    f = findings_of(strict, "temporal")
    assert len(f) == 1 and f[0].kind == "temporal_overlap" and f[0].severity is Severity.ERROR
    assert f[0].confidence is Confidence.DETERMINISTIC
    assert f[0].evidence["later_samples_before_earlier_end"] == 1
    assert strict.metrics.by_detector["temporal"]["violating_samples"] == 5  # 1 test + 4 train after it

    off = run_scan(b.root, policy__temporal__enabled=False)
    assert not findings_of(off, "temporal")


# ----------------------------------------------------------------------------- labels
def test_label_validity_coco(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train",)).fill(2)
    img = b.add("train", boxes=[])
    b.add_annotation("train", img, 1, [10, 10, -5, 20])  # degenerate
    b.add_annotation("train", img, 2, [150, 100, 50, 50])  # out of bounds
    b.add_annotation("train", img, 99, [1, 1, 5, 5])  # unknown category
    b.add_annotation("train", img, 1, [1, 1, 5, 5], segmentation=[[1, 1, 2]])  # invalid polygon
    b.add_annotation("train", img, 1, [1, 1, 5, 5])  # duplicate of previous box/class
    b.add("train", boxes=[])  # no annotations
    b.images["train"].append({"id": 777, "file_name": "missing.jpg", "width": 10, "height": 10})
    b.add("train", declared_size=(999, 999))  # size mismatch
    b.write()
    report = run_scan(b.root)
    kinds = {f.kind for f in findings_of(report, "label_validity")}
    assert {"degenerate_bbox", "out_of_bounds_bbox", "unknown_category", "invalid_segmentation", "duplicate_annotation",
            "missing_annotations", "missing_file", "size_mismatch"} <= kinds
    assert all(f.confidence is Confidence.DETERMINISTIC for f in findings_of(report, "label_validity"))
    # label problems never count toward the leakage metric
    assert report.metrics.violation_rate == 0.0


def test_label_validity_yolo(tmp_path):
    b = YoloBuilder(tmp_path / "y", splits=("train", "val")).fill(2)
    b.add("train", "oob.jpg", labels=["0 0.9 0.9 0.5 0.5"])
    b.add("train", "bad.jpg", labels=["0 0.5 abc 0.3 0.4"])
    b.add("train", "unk.jpg", labels=["7 0.5 0.5 0.3 0.4"])
    b.add("train", "zero.jpg", labels=["0 0.5 0.5 0 0.4"])
    b.add("val", "nolabel.jpg", write_label=False)
    b.write()
    report = run_scan(b.root)
    kinds = {f.kind for f in findings_of(report, "label_validity")}
    assert {"out_of_bounds_bbox", "malformed_label", "unknown_category", "degenerate_bbox", "missing_label_file"} <= kinds
    ml = findings_of(report, "label_validity", "missing_label_file")[0]
    assert ml.severity is Severity.WARNING and ml.splits == ["val"]


def test_allowed_categories_policy(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train",)).fill(2)
    b.write()
    report = run_scan(b.root, policy__labels__allowed_categories=["cat"])
    f = findings_of(report, "label_validity", "unknown_category")
    assert f and all("dog" in x.title for x in f)


def test_empty_split_reported(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "val"))
    b.add("train")
    b.write()
    report = run_scan(b.root)
    f = findings_of(report, "label_validity", "empty_split")
    assert len(f) == 1 and f[0].splits == ["val"]


# ----------------------------------------------------------------------------- distribution
def test_distribution_unseen_missing_and_imbalance(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"), categories=("cat", "dog", "bird"))
    for i in range(30):
        b.add("train", boxes=[(1, [10, 10, 50, 40])] + ([(2, [5, 5, 20, 20])] if i == 0 else []))
    for _ in range(25):
        b.add("test", boxes=[(1, [10, 10, 50, 40]), (3, [5, 5, 20, 20])])
    b.write()
    report = run_scan(b.root)
    unseen = findings_of(report, "distribution", "unseen_class")
    assert len(unseen) == 1 and unseen[0].evidence["classes"] == ["bird"] and unseen[0].severity is Severity.ERROR
    missing = findings_of(report, "distribution", "missing_class_in_eval")
    assert len(missing) == 1 and missing[0].evidence["classes"] == ["dog"]
    imbalance = findings_of(report, "distribution", "class_imbalance")
    assert imbalance and imbalance[0].splits == ["train"] and imbalance[0].evidence["ratio"] == 30.0
    shift = findings_of(report, "distribution", "label_shift")
    assert shift and shift[0].confidence is Confidence.HEURISTIC
    assert report.metrics.violation_rate == 0.0


def test_distribution_stats_exposed_for_report(coco_root):
    report = run_scan(coco_root)
    run = [d for d in report.detectors if d.name == "distribution"][0]
    assert set(run.stats["class_distribution"]) == {"train", "val", "test"}


# ----------------------------------------------------------------------------- consistency
def test_conflicting_labels_on_identical_images(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "val")).fill(2)
    img = Image.open(b.path("train", b.images["train"][0]["file_name"])).convert("RGB")
    b.add("val", name="same.png", image=img, fmt="PNG", boxes=[(2, [1, 1, 20, 20])])
    b.write()
    report = run_scan(b.root)
    f = findings_of(report, "consistency", "conflicting_labels_identical")
    assert len(f) == 1 and f[0].confidence is Confidence.DETERMINISTIC and len(f[0].samples) == 2


def test_fingerprint_cache_reused(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train",)).fill(3)
    b.write()
    cache = tmp_path / "cache"
    first = run_scan(b.root, performance__cache=str(cache))
    second = run_scan(b.root, performance__cache=str(cache))
    assert first.stats["fingerprints"]["cache_hits"] == 0
    assert second.stats["fingerprints"]["cache_hits"] == 3
    assert (cache / "fingerprints.sqlite").exists()


def test_detectors_can_be_disabled(coco_root):
    report = run_scan(coco_root, detectors__enabled=["exact_duplicate"])
    assert [d.name for d in report.detectors] == ["exact_duplicate"]
    report = run_scan(coco_root, detectors__disabled=["distribution", "temporal"])
    assert "distribution" not in {d.name for d in report.detectors}
