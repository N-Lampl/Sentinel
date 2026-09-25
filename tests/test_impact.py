"""Metric impact: scoring an evaluation split with and without flagged samples."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CocoBuilder, YoloBuilder, image_from_seed, make_config, run_scan
from dataset_sentinel.cli import main
from dataset_sentinel.engine import load_dataset
from dataset_sentinel.impact import (
    Prediction,
    PredictionSet,
    load_predictions,
    score_classification,
    score_detection,
)
from dataset_sentinel.model import BBox
from dataset_sentinel.reporters.html_reporter import HtmlReporter
from dataset_sentinel.reporters.json_reporter import JsonReporter
from dataset_sentinel.reporters.markdown_reporter import MarkdownReporter


def _preds(items):
    ps = PredictionSet(task="detection")
    for sid, cat, score, box in items:
        ps.by_sample.setdefault(sid, []).append(Prediction(sid, cat, score, BBox(*box)))
    return ps


# ----------------------------------------------------------------------------- scorer
def test_map_perfect_and_empty(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("test",), categories=("cat", "dog"))
    for _ in range(4):
        b.add("test", boxes=[(1, [10, 10, 50, 40]), (2, [70, 20, 30, 30])])
    b.write()
    ds = load_dataset(make_config(b.root)).dataset
    perfect = _preds([(s.id, a.category, 0.9, a.bbox.as_list()) for s in ds.samples for a in s.annotations])
    res = score_detection(ds.samples, perfect, ["cat", "dog"])
    assert res["mAP"] == pytest.approx(1.0) and res["AP50"] == pytest.approx(1.0) and res["classes"] == 2
    empty = score_detection(ds.samples, PredictionSet(task="detection"), ["cat", "dog"])
    assert empty["mAP"] == 0.0 and empty["per_class"]["cat"]["AP"] == 0.0


def test_map_half_recall_and_localisation(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("test",), categories=("cat",))
    ids = [b.add("test", boxes=[(1, [10, 10, 50, 40])]) for _ in range(2)]
    b.write()
    ds = load_dataset(make_config(b.root)).dataset
    first = f"test:{ids[0]}"
    half = _preds([(first, "cat", 0.9, [10, 10, 50, 40])])
    res = score_detection(ds.samples, half, ["cat"])
    assert res["AP50"] == pytest.approx(0.5, abs=0.02)
    # a box with IoU ~0.6 counts at 0.5 but not at 0.75
    shifted = _preds([(f"test:{i}", "cat", 0.9, [20, 10, 50, 40]) for i in ids])
    res = score_detection(ds.samples, shifted, ["cat"])
    assert res["AP50"] == pytest.approx(1.0) and res["AP75"] == 0.0 and 0 < res["mAP"] < 1
    # a false positive with a higher score than the true positives lowers precision
    fp = _preds([(f"test:{i}", "cat", 0.9, [10, 10, 50, 40]) for i in ids] + [(first, "cat", 0.99, [100, 90, 20, 20])])
    res = score_detection(ds.samples, fp, ["cat"])
    assert 0.5 < res["AP50"] < 1.0


def test_crowd_boxes_are_ignored(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("test",), categories=("cat",))
    img = b.add("test", boxes=[(1, [10, 10, 50, 40])])
    b.add_annotation("test", img, 1, [100, 80, 40, 30], iscrowd=1)
    b.write()
    ds = load_dataset(make_config(b.root)).dataset
    # a prediction on the crowd region is neither TP nor FP
    preds = _preds([(f"test:{img}", "cat", 0.9, [10, 10, 50, 40]), (f"test:{img}", "cat", 0.95, [100, 80, 40, 30])])
    res = score_detection(ds.samples, preds, ["cat"])
    assert res["AP50"] == pytest.approx(1.0) and res["per_class"]["cat"]["gt"] == 1


def test_classification_scoring(tmp_path):
    root = tmp_path / "f"
    for split in ("train", "test"):
        for cls in ("cat", "dog"):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(3):
                image_from_seed(hash((split, cls, i)) % 10_000).save(d / f"{i}.jpg")
    ds = load_dataset(make_config(root)).dataset
    test = [s for s in ds.samples if s.split == "test"]
    ps = PredictionSet(task="classification")
    for k, s in enumerate(test):
        ps.by_sample[s.id] = [Prediction(s.id, "cat" if k < 4 else s.annotations[0].category, 0.8)]
    res = score_classification(test, ps)
    assert res["scored"] == 6 and 0 < res["accuracy"] < 1 and set(res["per_class"]) == {"cat", "dog"}
    perfect = PredictionSet(task="classification")
    for s in test:
        perfect.by_sample[s.id] = [Prediction(s.id, s.annotations[0].category, 1.0)]
    assert score_classification(test, perfect)["accuracy"] == 1.0 and score_classification(test, perfect)["macro_f1"] == 1.0


# ----------------------------------------------------------------------------- loaders
def test_load_coco_results_and_yolo_dir(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("test",), categories=("cat", "dog")).fill(3)
    b.write()
    ds = load_dataset(make_config(b.root)).dataset
    rows = [{"image_id": s.native_id, "category_id": 2, "bbox": [1, 2, 3, 4], "score": 0.5} for s in ds.samples]
    rows.append({"image_id": 424242, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.5})
    p = tmp_path / "preds.json"
    p.write_text(json.dumps(rows))
    ps = load_predictions(p, ds, "test")
    assert ps.task == "detection" and len(ps.by_sample) == 3 and ps.unmatched == 1
    assert ps.by_sample[ds.samples[0].id][0].category == "dog"

    y = YoloBuilder(tmp_path / "y", names=("a", "b"), splits=("val",)).fill(2)
    y.write()
    yds = load_dataset(make_config(y.root, "yolo")).dataset
    out = tmp_path / "labels"
    out.mkdir()
    for s in yds.samples:
        (out / (Path(s.uri).stem + ".txt")).write_text("1 0.5 0.5 0.3 0.4 0.77\n")
    yps = load_predictions(out, yds, "val")
    pred = yps.by_sample[yds.samples[0].id][0]
    # no fingerprint pass here, so the sample has no size yet and the box stays normalised
    assert pred.category == "b" and pred.score == pytest.approx(0.77) and pred.bbox.w == pytest.approx(0.3)


# ----------------------------------------------------------------------------- end to end
def _leaky_with_predictions(tmp_path: Path):
    """test split: 3 byte-identical copies of training images (perfectly predicted)
    plus 6 clean images (badly predicted)."""
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"), categories=("cat", "dog")).fill(6)
    for k in range(3):
        src = b.path("train", b.images["train"][k]["file_name"])
        name = f"copy{k}.jpg"
        (b.root / "test" / name).write_bytes(src.read_bytes())
        b.images["test"].append({"id": 900 + k, "file_name": name, "width": 160, "height": 120})
        b.add_annotation("test", 900 + k, 1 + (k % 2), [10, 10, 50, 40])
    b.write()
    ds = load_dataset(make_config(b.root)).dataset
    rows = []
    for s in ds.samples:
        if s.split != "test":
            continue
        for a in s.annotations:
            good = Path(s.uri).name.startswith("copy")
            box = a.bbox.as_list() if good else [a.bbox.x + 45, a.bbox.y + 35, a.bbox.w, a.bbox.h]
            rows.append({"image_id": s.native_id, "category_id": a.category_id, "bbox": box, "score": 0.9})
    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps(rows))
    return b, preds


def test_impact_end_to_end(tmp_path):
    b, preds = _leaky_with_predictions(tmp_path)
    report = run_scan(b.root, evaluation__predictions={"test": str(preds)})
    imp = report.impact
    assert imp and "test" in imp["splits"]
    e = imp["splits"]["test"]
    assert e["task"] == "detection" and e["metric"] == "mAP"
    assert e["leaked_samples"] == 3 and e["samples"] == 9
    assert e["leaked"] == pytest.approx(1.0)
    assert e["clean"] < 0.2 and e["all"] > e["clean"]
    assert e["inflation"] == pytest.approx(e["all"] - e["clean"])
    assert e["strict_inflation"] == pytest.approx(e["inflation"])  # exact duplicates are deterministic
    assert e["per_class_delta"] and e["per_class_delta"][0]["delta"] > 0
    data = json.loads(JsonReporter().render(report))
    assert data["impact"]["splits"]["test"]["inflation"] > 0
    html = HtmlReporter().render(report)
    assert "Metric impact of the flagged samples" in html and "inflation" in html
    md = MarkdownReporter().render(report)
    assert "Metric impact" in md and "test mAP" in md


def test_impact_without_leaks_notes_identity(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"), categories=("cat", "dog")).fill(3)
    b.write()
    ds = load_dataset(make_config(b.root)).dataset
    rows = [{"image_id": s.native_id, "category_id": a.category_id, "bbox": a.bbox.as_list(), "score": 0.9} for s in ds.samples if s.split == "test" for a in s.annotations]
    p = tmp_path / "p.json"
    p.write_text(json.dumps(rows))
    report = run_scan(b.root, evaluation__predictions={"test": str(p)})
    e = report.impact["splits"]["test"]
    assert e["all"] == pytest.approx(1.0) and e["clean"] == pytest.approx(1.0) and e["inflation"] == pytest.approx(0.0)
    assert any("no leaked samples" in n for n in report.impact["notes"])


def test_cli_predictions_flag(tmp_path, capsys):
    b, preds = _leaky_with_predictions(tmp_path)
    code = main(["scan", str(b.root), "--no-report", "--no-cache", "--no-color", "--predictions", f"test={preds}"])
    assert code == 1
    out = capsys.readouterr().out
    assert "Metric impact of the flagged samples" in out and "test mAP: all" in out
    with pytest.raises(SystemExit):
        main(["scan", str(b.root), "--no-report", "--no-cache", "-q", "--predictions", "nonsense"])


def test_predictions_for_unknown_split_and_missing_file(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(2)
    b.write()
    report = run_scan(b.root, evaluation__predictions={"holdout": str(tmp_path / "x.json")})
    assert any("unknown split" in n for n in report.impact["notes"]) and not report.impact["splits"]
    with pytest.raises(FileNotFoundError):
        run_scan(b.root, evaluation__predictions={"test": str(tmp_path / "missing.json")})
