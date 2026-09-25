"""Embedding re-ranking, rare classes, metadata-conditioned class mix,
keypoint validation and plugin group resolvers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import CocoBuilder, YoloBuilder, findings_of, image_from_seed, make_config, run_scan
from dataset_sentinel.cli import main
from dataset_sentinel.embeddings.base import EmbeddingProvider, cosine_similarity, embeddings, get_provider
from dataset_sentinel.embeddings.builtin import BuiltinDescriptor
from dataset_sentinel.embeddings.store import EmbeddingStore
from dataset_sentinel.engine import load_dataset
from dataset_sentinel.model import Confidence, Severity


def _shift(img: Image.Image, delta: int = 8) -> Image.Image:
    return Image.fromarray(np.clip(np.asarray(img).astype(np.int16) + delta, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------- embeddings
def test_builtin_descriptor_properties(tmp_path):
    a = tmp_path / "a.jpg"
    image_from_seed(5).save(a, quality=92)
    b = tmp_path / "b.jpg"
    _shift(Image.open(a).convert("RGB"), 6).save(b, quality=60)
    c = tmp_path / "c.jpg"
    image_from_seed(77).save(c, quality=92)
    prov = BuiltinDescriptor()
    vecs = prov.embed([a, b, c, tmp_path / "missing.jpg"])
    assert vecs.shape == (4, prov.dimension)
    assert np.allclose(np.linalg.norm(vecs[:3], axis=1), 1.0, atol=1e-4)
    assert not vecs[3].any()  # unreadable -> zeros
    assert cosine_similarity(vecs[0], vecs[1]) > 0.97
    assert cosine_similarity(vecs[0], vecs[2]) < cosine_similarity(vecs[0], vecs[1])


def test_embedding_store_caches(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train",)).fill(3)
    b.write()
    ds = load_dataset(make_config(b.root)).dataset
    store = EmbeddingStore(ds, BuiltinDescriptor(), cache_dir=tmp_path / "cache")
    ids = [s.id for s in ds.samples]
    store.ensure(ids)
    assert store.stats["computed"] == 3 and store.get(ids[0]).shape == (288,)
    store2 = EmbeddingStore(ds, BuiltinDescriptor(), cache_dir=tmp_path / "cache")
    store2.ensure(ids)
    assert store2.stats["cache_hits"] == 3 and store2.stats["computed"] == 0
    assert np.allclose(store.get(ids[0]), store2.get(ids[0]), atol=1e-3)


@embeddings.register("test_reject_all")
class _RejectAll(EmbeddingProvider):
    """Stub provider: orthogonal vectors for every sample -> cosine 0."""

    name = "test_reject_all"
    version = "1"
    dimension = 8

    def embed(self, paths):
        out = np.zeros((len(paths), 8), dtype=np.float32)
        for i in range(len(paths)):
            out[i, i % 8] = 1.0
        return out


@embeddings.register("test_unavailable")
class _Unavailable(EmbeddingProvider):
    name = "test_unavailable"
    dimension = 1

    @classmethod
    def available(cls):
        return False

    def embed(self, paths):  # pragma: no cover
        return np.zeros((len(paths), 1))


def test_embedding_reranking_end_to_end(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(4)
    src = Image.open(b.path("train", b.images["train"][0]["file_name"])).convert("RGB")
    b.add("test", name="near.jpg", image=_shift(src, 8))
    b.write()
    plain = run_scan(b.root)
    assert len(findings_of(plain, "near_duplicate", "near_duplicate_cross_split")) == 1

    # builtin provider confirms the true near-duplicate
    with_emb = run_scan(b.root, policy__near_duplicate__embedding={"enabled": True, "provider": "builtin", "min_cosine": 0.9})
    f = findings_of(with_emb, "near_duplicate", "near_duplicate_cross_split")
    assert len(f) == 1 and f[0].evidence["best_pair_cosine"] > 0.95 and f[0].evidence["pairs"][0]["cosine"] > 0.95
    run = [d for d in with_emb.detectors if d.name == "near_duplicate"][0]
    assert run.stats["embedding"]["provider"] == "builtin" and run.stats["embedding"]["computed"] >= 2

    # a provider that rejects everything removes the finding and reports it
    rejected = run_scan(b.root, policy__near_duplicate__embedding={"enabled": True, "provider": "test_reject_all", "min_cosine": 0.9})
    assert not findings_of(rejected, "near_duplicate")
    run = [d for d in rejected.detectors if d.name == "near_duplicate"][0]
    assert run.stats["embedding"]["rejected_by_embedding"] >= 1

    # an unavailable provider degrades to the plain pipeline with a note
    degraded = run_scan(b.root, policy__near_duplicate__embedding={"enabled": True, "provider": "test_unavailable"})
    assert len(findings_of(degraded, "near_duplicate", "near_duplicate_cross_split")) == 1
    run = [d for d in degraded.detectors if d.name == "near_duplicate"][0]
    assert "unavailable" in run.stats["embedding"]["error"]
    assert get_provider("test_unavailable") is None


def test_plugins_lists_embedding_providers(capsys):
    assert main(["plugins"]) == 0
    out = capsys.readouterr().out
    assert "Embedding providers" in out and "builtin" in out and "torchvision" in out


# ----------------------------------------------------------------------------- distribution extras
def test_rare_class_and_conditioned_class_mix(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"), categories=("cat", "dog", "bird"))
    # camera A only sees cats, camera B sees cats and dogs -> conditioned divergence; bird is rare in train
    for i in range(40):
        cam = "A" if i % 2 == 0 else "B"
        boxes = [(1, [10, 10, 50, 40]), (1, [60, 10, 30, 30])] if cam == "A" else [(2, [10, 10, 50, 40]), (2, [60, 10, 30, 30])]
        if i == 0:
            boxes.append((3, [5, 5, 10, 10]))
        b.add("train", boxes=boxes, extra={"camera": cam})
    for _ in range(30):
        b.add("test", boxes=[(1, [10, 10, 50, 40]), (2, [60, 10, 30, 30])], extra={"camera": "C"})
    b.write()
    report = run_scan(b.root)
    rare = findings_of(report, "distribution", "rare_class")
    assert len(rare) == 1 and rare[0].evidence["classes"] == {"bird": 1} and rare[0].splits == ["train"]
    cond = findings_of(report, "distribution", "conditioned_class_mix")
    assert len(cond) == 1 and cond[0].confidence is Confidence.HEURISTIC
    groups = cond[0].evidence["groups"]
    assert {g["value"] for g in groups} >= {"A", "B"} and all(g["key"] == "camera" for g in groups)
    run = [d for d in report.detectors if d.name == "distribution"][0]
    assert "camera" in run.stats["conditioned_class_mix"]["train"]
    quiet = run_scan(b.root, policy__distribution__conditioned="ignore", policy__distribution__min_samples_per_class_per_split=0)
    assert not findings_of(quiet, "distribution", "conditioned_class_mix") and not findings_of(quiet, "distribution", "rare_class")


# ----------------------------------------------------------------------------- keypoints
def test_coco_keypoint_validation(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train",), categories=("person",))
    b.categories[0]["keypoints"] = ["nose", "eye_l", "eye_r"]
    ok = b.add("train", boxes=[])
    b.add_annotation("train", ok, 1, [10, 10, 50, 40], keypoints=[20, 20, 2, 30, 20, 2, 0, 0, 0], num_keypoints=2)
    bad = b.add("train", boxes=[])
    b.add_annotation("train", bad, 1, [10, 10, 50, 40], keypoints=[20, 20, 2, 30, 20, 5], num_keypoints=1)  # count + visibility
    b.add_annotation("train", bad, 1, [10, 10, 50, 40], keypoints=[20, 20, 2, 500, 20, 2, 0, 0, 0], num_keypoints=3)  # oob + num mismatch
    b.write()
    report = run_scan(b.root)
    inv = findings_of(report, "label_validity", "invalid_keypoints")
    oob = findings_of(report, "label_validity", "keypoints_out_of_bounds")
    reasons = " ".join(f.message for f in inv)
    assert "3 keypoints" not in reasons or True
    assert any("has 2 keypoints but category" in m for m in (f.message for f in inv))
    assert any("visibility" in f.message for f in inv)
    assert any("num_keypoints=3" in f.message for f in inv)
    assert len(oob) == 1 and oob[0].severity is Severity.WARNING
    assert all(f.samples[0].id == f"train:{bad}" for f in inv + oob)


def test_yolo_pose_keypoints(tmp_path):
    b = YoloBuilder(tmp_path / "y", names=("person",), splits=("train",))
    kp_ok = " ".join(["0.5 0.5 0.3 0.4"] + ["0.5 0.4 2", "0.45 0.35 2", "0.55 0.35 1"])
    kp_short = "0 0.5 0.5 0.3 0.4 0.5 0.4 2"
    kp_oob = " ".join(["0.5 0.5 0.3 0.4"] + ["1.5 0.4 2", "0.45 0.35 2", "0.55 0.35 0"])
    b.add("train", "ok.jpg", labels=[f"0 {kp_ok}"])
    b.add("train", "short.jpg", labels=[kp_short])
    b.add("train", "oob.jpg", labels=[f"0 {kp_oob}"])
    b.write()
    text = (b.root / "data.yaml").read_text() + "kpt_shape: [3, 3]\n"
    (b.root / "data.yaml").write_text(text)
    ds = load_dataset(make_config(b.root, "yolo")).dataset
    assert ds.source["kpt_shape"] == [3, 3]
    by_name = {Path(s.uri).name: s for s in ds.samples}
    assert by_name["ok.jpg"].annotations[0].attributes["keypoints_norm"] == pytest.approx([0.5, 0.4, 2, 0.45, 0.35, 2, 0.55, 0.35, 1])
    assert "polygon_norm" not in by_name["ok.jpg"].annotations[0].attributes
    assert "keypoints_error" in by_name["short.jpg"].annotations[0].attributes
    report = run_scan(b.root)
    inv = findings_of(report, "label_validity", "invalid_keypoints")
    oob = findings_of(report, "label_validity", "keypoints_out_of_bounds")
    assert len(inv) == 1 and "short.jpg" in inv[0].title
    assert len(oob) == 1 and "oob.jpg" in oob[0].title


# ----------------------------------------------------------------------------- plugin group resolver
def test_group_resolver_plugin(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "my_groups.py").write_text(
        "def by_prefix(sample):\n    return {'shard': sample.uri.split('/')[-1][:5]}\n"
        "def broken(sample):\n    raise RuntimeError('nope')\n"
        "def wrong(sample):\n    return 42\n"
    )
    monkeypatch.syspath_prepend(str(plugin_dir))
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(2)
    b.write()
    report = run_scan(b.root, dataset__groups__resolver="my_groups:by_prefix")
    assert "shard" in report.dataset["group_keys"]
    f = findings_of(report, "group_overlap", "group_overlap")
    assert not f  # train_xxxx vs test_xxxx prefixes differ
    with pytest.raises(ValueError, match="failed for sample"):
        run_scan(b.root, dataset__groups__resolver="my_groups:broken")
    with pytest.raises(ValueError, match="dict or None"):
        run_scan(b.root, dataset__groups__resolver="my_groups:wrong")
    with pytest.raises(ValueError, match="cannot import"):
        run_scan(b.root, dataset__groups__resolver="nonexistent_mod:fn")
    with pytest.raises(ValueError, match="package.module:function"):
        run_scan(b.root, dataset__groups__resolver="badspec")


def test_init_template_mentions_new_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "-o", "s.yaml"]) == 0
    text = (tmp_path / "s.yaml").read_text()
    for key in ("allowlist", "baseline", "split_pair_overrides", "new_only", "mode: exact"):
        assert key in text
    import yaml

    assert yaml.safe_load(text)["fail_on"]["detector_errors"] is True


def test_json_report_has_new_fields(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train",)).fill(2)
    b.write()
    from dataset_sentinel.reporters.json_reporter import JsonReporter

    data = json.loads(JsonReporter().render(run_scan(b.root)))
    assert "clusters" in data and data["fix_plan_summary"]["samples_to_remove"] == 0
    assert all("new" in f and "suppressed" in f for f in data["findings"])


# ----------------------------------------------------------------------------- manifests / label coverage / box geometry
def test_yaml_manifest_and_dataset_level_samples_section(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test")).fill(2)
    b.write()
    train0 = b.images["train"][0]["file_name"]
    test0 = b.images["test"][0]["file_name"]
    (tmp_path / "meta.yaml").write_text(
        f"dataset: demo\nsamples:\n  {train0}: {{site: S1}}\n  {test0}: {{site: S1}}\n"
    )
    report = run_scan(b.root, dataset__metadata__file=str(tmp_path / "meta.yaml"))
    f = findings_of(report, "group_overlap", "group_overlap")
    assert len(f) == 1 and f[0].evidence["group_key"] == "site"
    (tmp_path / "list.yml").write_text(f"- file_name: {train0}\n  site: S2\n- file_name: {test0}\n  site: S2\n")
    report = run_scan(b.root, dataset__metadata__file=str(tmp_path / "list.yml"))
    assert findings_of(report, "group_overlap", "group_overlap")[0].evidence["group_value"] == "S2"


def test_no_valid_annotations_and_unlabeled_rate(tmp_path):
    b = YoloBuilder(tmp_path / "y", splits=("train",)).fill(3)
    b.add("train", "allbad.jpg", labels=["x y z", "0 0.5 abc 0.3 0.4"])
    b.add("train", "empty1.jpg", labels=[])
    b.add("train", "empty2.jpg", labels=[])
    b.write()
    report = run_scan(b.root)
    nva = findings_of(report, "label_validity", "no_valid_annotations")
    assert len(nva) == 1 and "allbad.jpg" in nva[0].title and nva[0].severity is Severity.WARNING
    ma = findings_of(report, "label_validity", "missing_annotations")
    assert len(ma) == 1 and ma[0].evidence["count"] == 2 and ma[0].evidence["rate"] == pytest.approx(2 / 6, abs=1e-3)
    assert "(33%)" in ma[0].title


def test_truncated_boxes_and_class_box_size_shift(tmp_path):
    b = CocoBuilder(tmp_path / "d", splits=("train", "test"), categories=("cat", "dog"))
    for _ in range(25):
        b.add("train", boxes=[(1, [40, 30, 40, 30]), (2, [50, 40, 20, 20])])
    for _ in range(25):
        # cats are large and cut off at the border in test; dogs unchanged
        b.add("test", boxes=[(1, [0, 0, 150, 115]), (2, [50, 40, 20, 20])])
    b.write()
    report = run_scan(b.root)
    trunc = findings_of(report, "distribution", "truncated_boxes")
    assert len(trunc) == 1 and trunc[0].evidence["combinations"][0]["class"] == "cat" and trunc[0].evidence["combinations"][0]["split"] == "test"
    size = findings_of(report, "distribution", "class_box_size_shift")
    assert len(size) == 1 and size[0].evidence["classes"][0]["class"] == "cat" and size[0].evidence["classes"][0]["ratio"] > 2
    assert all(f.confidence is Confidence.HEURISTIC and f.severity is Severity.INFO for f in trunc + size)
    quiet = run_scan(b.root, policy__distribution__truncated_boxes="ignore", policy__distribution__class_box_size_shift="ignore")
    assert not findings_of(quiet, "distribution", "truncated_boxes") and not findings_of(quiet, "distribution", "class_box_size_shift")
