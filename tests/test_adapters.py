from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import CocoBuilder, YoloBuilder, image_from_seed, make_config
from dataset_sentinel.adapters.coco import CocoAdapter
from dataset_sentinel.adapters.image_folder import ImageFolderAdapter
from dataset_sentinel.adapters.yolo import YoloAdapter, img2label_path
from dataset_sentinel.engine import detect_format, load_dataset


# ----------------------------------------------------------------------------- COCO
def test_coco_autodetect_and_load(coco_root):
    cfg = make_config(coco_root)
    detected = detect_format(cfg)
    assert detected and detected["format"] == "coco"
    assert set(detected["splits"]) == {"train", "val", "test"}
    result = load_dataset(cfg)
    ds = result.dataset
    assert len(ds) == 30 and ds.splits == ["train", "val", "test"] or set(ds.splits) == {"train", "val", "test"}
    assert ds.categories == {1: "cat", 2: "dog"}
    s = ds.samples[0]
    assert s.path is not None and s.path.exists()
    assert s.width == 160 and s.height == 120
    assert s.annotations and s.annotations[0].bbox is not None and s.annotations[0].category in {"cat", "dog"}


def test_coco_explicit_splits_and_metadata_fields(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"))
    b.add("train", extra={"date_captured": "2024-01-02 10:00:00", "patient_id": "P1", "license": 3})
    b.add("test")
    b.write()
    cfg = make_config(b.root, "coco")
    cfg.set("dataset.splits", {"train": {"annotations": "annotations/instances_train.json", "images": "train"}, "test": "annotations/instances_test.json"})
    ds = load_dataset(cfg).dataset
    train = [s for s in ds.samples if s.split == "train"][0]
    assert train.metadata["patient_id"] == "P1" and train.metadata["date_captured"].startswith("2024")
    assert train.metadata["license"] == 3
    assert [s for s in ds.samples if s.split == "test"][0].path.exists()


def test_coco_structural_problems_reported(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train",))
    img_id = b.add("train")
    b.add("train")
    b.add_annotation("train", 999, 1, [1, 1, 2, 2])  # orphan
    b.add_annotation("train", img_id, 1, "oops")  # invalid bbox type
    b.annotations["train"].append(dict(b.annotations["train"][0]))  # duplicate annotation id
    b.images["train"].append(dict(b.images["train"][0]))  # duplicate image id
    b.write()
    result = load_dataset(make_config(b.root, "coco"))
    kinds = {f.kind for f in result.findings}
    assert {"orphan_annotation", "duplicate_annotation_id", "duplicate_image_registration"} <= kinds
    ann = [a for s in result.dataset.samples for a in s.annotations if a.attributes.get("bbox_error")]
    assert ann and ann[0].bbox is None


def test_coco_missing_annotation_file_raises(tmp_path):
    cfg = make_config(tmp_path, "coco")
    cfg.set("dataset.splits", {"train": "annotations/nope.json"})
    with pytest.raises(FileNotFoundError):
        load_dataset(cfg)


def test_coco_image_dir_guessing(tmp_path):
    # annotations/instances_train2017.json + images/train2017/
    root = tmp_path / "d"
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "train2017").mkdir(parents=True)
    image_from_seed(1).save(root / "images" / "train2017" / "1.jpg")
    (root / "annotations" / "instances_train2017.json").write_text(
        json.dumps({"images": [{"id": 1, "file_name": "1.jpg", "width": 160, "height": 120}], "annotations": [], "categories": [{"id": 1, "name": "x"}]})
    )
    ds = load_dataset(make_config(root)).dataset
    assert ds.samples[0].path.exists()
    assert ds.splits == ["train2017"]


# ----------------------------------------------------------------------------- YOLO
def test_img2label_path():
    assert img2label_path(Path("/d/images/train/a.jpg")) == Path("/d/labels/train/a.txt")
    assert img2label_path(Path("/d/x/a.png")) == Path("/d/x/a.txt")


def test_yolo_autodetect_and_load(yolo_root):
    cfg = make_config(yolo_root)
    assert detect_format(cfg)["format"] == "yolo"
    ds = load_dataset(cfg).dataset
    assert len(ds) == 30 and ds.categories == {0: "cat", 1: "dog"}
    s = ds.samples[0]
    assert s.annotations and s.annotations[0].attributes["bbox_norm"] == [0.5, 0.5, 0.3, 0.4]
    assert s.annotations[0].category in {"cat", "dog"}
    assert s.metadata["label_file_exists"] is True


def test_yolo_label_parsing_cases(tmp_path):
    b = YoloBuilder(tmp_path / "y", names=("a", "b"), splits=("train",))
    b.add("train", "poly.jpg", labels=["1 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5"])
    b.add("train", "bad.jpg", labels=["0 0.5 abc 0.3 0.4", "0 0.5 0.5 0.3"])
    b.add("train", "nolabel.jpg", write_label=False)
    b.add("train", "empty.jpg", labels=[])
    b.write()
    ds = load_dataset(make_config(b.root, "yolo")).dataset
    by_name = {Path(s.uri).name: s for s in ds.samples}
    poly = by_name["poly.jpg"].annotations[0]
    assert "polygon_norm" in poly.attributes and poly.attributes["bbox_norm"] == pytest.approx([0.3, 0.3, 0.4, 0.4])
    bad = by_name["bad.jpg"].annotations
    assert bad[0].attributes["malformed"] == "non-numeric token"
    assert "expected 4 box values" in bad[1].attributes["malformed"]
    assert by_name["nolabel.jpg"].metadata["label_file_exists"] is False
    assert by_name["empty.jpg"].annotations == [] and by_name["empty.jpg"].metadata.get("label_file_empty")


def test_yolo_txt_listing_and_names_list(tmp_path):
    b = YoloBuilder(tmp_path / "y", names=("a", "b"), splits=("train", "val"))
    b.fill(3)
    listing = b.root / "train.txt"
    listing.write_text("\n".join(f"images/train/{p.name}" for p in (b.root / "images" / "train").glob("*.jpg")))
    (b.root / "data.yaml").write_text("train: train.txt\nval: images/val\nnames: [a, b]\n")
    ds = load_dataset(make_config(b.root, "yolo")).dataset
    assert ds.split_sizes() == {"train": 3, "val": 3}
    assert ds.categories == {0: "a", 1: "b"}


# ----------------------------------------------------------------------------- image folder
def test_image_folder_classification(tmp_path):
    root = tmp_path / "f"
    for split in ("train", "val"):
        for cls in ("cat", "dog"):
            d = root / split / cls
            d.mkdir(parents=True)
            image_from_seed(hash((split, cls)) % 1000).save(d / "1.jpg")
    cfg = make_config(root)
    assert detect_format(cfg)["format"] == "image-folder"
    ds = load_dataset(cfg).dataset
    assert len(ds) == 4 and set(ds.categories) == {"cat", "dog"}
    assert all(s.annotations[0].category in {"cat", "dog"} for s in ds.samples)


def test_adapter_descriptions_present():
    for cls in (CocoAdapter, YoloAdapter, ImageFolderAdapter):
        assert cls.name and cls.description and cls.modality == "image"
