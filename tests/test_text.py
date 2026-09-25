"""Text modality: adapter, fingerprints, duplicates, contamination, labels, CLI."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from conftest import findings_of, make_config, run_scan
from dataset_sentinel.cli import main
from dataset_sentinel.engine import detect_format, load_dataset
from dataset_sentinel.fingerprints.text import (
    MinHasher,
    containment,
    jaccard,
    lsh_candidate_pairs,
    ngram_hashes,
    normalize_text,
    tokens,
)
from dataset_sentinel.model import Confidence, Severity
from dataset_sentinel.reporters.html_reporter import HtmlReporter

WORDS = "the quick brown fox jumps over lazy dog while seven wizards quietly juggle glass boxes near river bank under pale moonlight".split()


def _doc(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _text_dataset(tmp_path: Path):
    """train: 60 random docs + planted leaks; test: 20 benchmark questions."""
    rng = random.Random(7)
    evals = [{"id": f"q{i}", "question": f"What is the capital of country number {i} and why does it matter for trade route {i * 7}?", "answer": "ABCD"[i % 4]} for i in range(20)]
    train = [{"id": f"d{i}", "text": _doc(rng, rng.randint(40, 120)), "source": f"site{i % 5}"} for i in range(60)]
    train.append({"id": "embedded", "text": _doc(rng, 30) + " " + evals[3]["question"] + " " + _doc(rng, 30), "source": "site9"})
    train.append({"id": "exact", "text": evals[5]["question"].upper() + "!!", "source": "site9"})
    train.append({"id": "near", "text": evals[4]["question"].replace("does it matter", "does it count"), "source": "site9"})
    root = tmp_path / "txt"
    root.mkdir()
    _write_jsonl(root / "train.jsonl", train)
    _write_jsonl(root / "test.jsonl", evals)
    return root, evals, train


# ----------------------------------------------------------------------------- fingerprints
def test_normalize_and_ngrams():
    assert normalize_text("  Hello, WORLD!  It's   fine. ") == "hello world it s fine"
    toks = tokens(normalize_text("a b c d e"))
    assert len(ngram_hashes(toks, 3)) == 3 and len(ngram_hashes(toks, 8)) == 1 and len(ngram_hashes([], 3)) == 0
    a = ngram_hashes(tokens("one two three four five six"), 2)
    b = ngram_hashes(tokens("one two three four"), 2)
    assert containment(b, a) == 1.0 and 0.5 < jaccard(a, b) < 1.0


def test_minhash_estimates_jaccard_and_lsh_finds_neighbours():
    rng = random.Random(3)
    base = tokens(_doc(rng, 300))
    variant = list(base)
    for i in range(0, 300, 10):
        variant[i] = "zzz"
    other = tokens(_doc(rng, 300))
    sa, sv, so = (ngram_hashes(t, 3) for t in (base, variant, other))
    hasher = MinHasher(128)
    est = float(np.mean(hasher.signature(sa) == hasher.signature(sv)))
    true = jaccard(sa, sv)
    assert abs(est - true) < 0.15 and true > 0.5
    sigs = np.stack([hasher.signature(x) for x in (sa, sv, so)])
    pairs = lsh_candidate_pairs(sigs, bands=32)
    assert (0, 1) in pairs and (0, 2) not in pairs


# ----------------------------------------------------------------------------- adapter
def test_text_adapter_autodetect_fields_and_metadata(tmp_path):
    root, evals, train = _text_dataset(tmp_path)
    cfg = make_config(root)
    assert detect_format(cfg)["format"] == "text"
    cfg.set("dataset.text.fields", ["question", "text"])
    cfg.set("dataset.text.label_field", "answer")
    ds = load_dataset(cfg).dataset
    assert ds.modality == "text" and ds.split_sizes() == {"train": 63, "test": 20}
    q = ds.get("test:q0")
    assert q.metadata["text"].startswith("What is the capital") and q.annotations[0].category == "A"
    d = ds.get("train:d0")
    assert d.metadata["source"] == "site0" and d.path is None and d.annotations == []
    assert ds.source["text_fields"]["test"]["label"] == "answer"


def test_text_adapter_formats_and_messages(tmp_path):
    root = tmp_path / "t"
    root.mkdir()
    (root / "train.csv").write_text("id,text,label\n1,hello world one,pos\n2,another line here,neg\n")
    (root / "test.json").write_text(json.dumps([{"messages": [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "hello"}]}, {"prompt": "just a prompt"}]))
    (root / "val.txt").write_text("plain line one\nplain line two\n")
    ds = load_dataset(make_config(root)).dataset
    assert ds.split_sizes() == {"train": 2, "test": 2, "val": 2}
    assert ds.get("train:1").annotations[0].category == "pos"
    msgs = [s for s in ds.samples if s.split == "test"][0]
    assert "user: hi there" in msgs.metadata["text"] and "assistant: hello" in msgs.metadata["text"]
    assert [s for s in ds.samples if s.split == "val"][0].metadata["text"] == "plain line one"


# ----------------------------------------------------------------------------- detectors
def test_contamination_exact_and_near_duplicates(tmp_path):
    root, evals, train = _text_dataset(tmp_path)
    report = run_scan(root, dataset__text__fields=["question", "text"], dataset__text__label_field="answer")
    names = {d.name for d in report.detectors if d.status == "ok"}
    assert {"text_contamination", "text_exact_duplicate", "text_near_duplicate", "text_label_validity", "group_overlap", "distribution"} <= names
    assert all(d.status == "skipped" for d in report.detectors if d.name in ("exact_duplicate", "near_duplicate", "derivative", "label_validity"))

    cont = findings_of(report, "text_contamination", "contamination")
    by_eval = {p["eval"]: (p["containment"], f.confidence) for f in cont for p in f.evidence["pairs"]}
    assert by_eval["test:q3"][0] == pytest.approx(1.0) and by_eval["test:q3"][1] is Confidence.DETERMINISTIC
    assert "test:q5" in by_eval and by_eval["test:q5"][0] == pytest.approx(1.0)
    rate = findings_of(report, "text_contamination", "contamination_rate")[0]
    assert rate.evidence["contaminated"] >= 2 and rate.evidence["items"] == 20 and "test" in rate.title

    exact = findings_of(report, "text_exact_duplicate", "exact_duplicate_cross_split")
    assert len(exact) == 1 and {s.id for s in exact[0].samples} == {"test:q5", "train:exact"}
    assert exact[0].confidence is Confidence.DETERMINISTIC and exact[0].evidence["snippets"]

    near = findings_of(report, "text_near_duplicate", "near_duplicate_cross_split")
    assert len(near) == 1 and {s.id for s in near[0].samples} == {"test:q4", "train:near"}
    assert 0.5 < near[0].evidence["best_cross_split_jaccard"] < 1.0

    assert report.metrics.violating_samples >= 5 and report.clusters
    assert report.fix_plan["summary"]["samples_to_remove"] >= 2
    html = HtmlReporter().render(report)
    assert "snippet" in html and "capital of country number 3" in html


def test_contamination_respects_policy_and_ngram(tmp_path):
    root, *_ = _text_dataset(tmp_path)
    off = run_scan(root, dataset__text__fields=["question", "text"], policy__contamination__cross_split="ignore")
    assert not findings_of(off, "text_contamination")
    strict = run_scan(root, dataset__text__fields=["question", "text"], policy__contamination__ngram=4, policy__contamination__min_containment=0.9)
    cont = findings_of(strict, "text_contamination", "contamination")
    assert cont and all(p["containment"] >= 0.9 for f in cont for p in f.evidence["pairs"])
    info = run_scan(root, dataset__text__fields=["question", "text"], policy__contamination__cross_split="info")
    assert info.metrics.by_detector.get("text_contamination", {}).get("violating_samples", 0) == 0


def test_text_label_validity(tmp_path):
    root = tmp_path / "t"
    root.mkdir()
    _write_jsonl(root / "train.jsonl", [{"id": "a", "text": "a proper sentence with words", "label": "x"}, {"id": "a", "text": "   ", "label": "y"}, {"id": "b", "text": "hi", "label": "zz"}, {"id": "c", "text": "no label at all here"}])
    _write_jsonl(root / "test.jsonl", [{"id": "t", "text": "another proper sentence here", "label": "x"}])
    report = run_scan(root, policy__labels__allowed_categories=["x", "y"])
    kinds = {f.kind for f in findings_of(report, "text_label_validity")}
    assert {"empty_text", "short_text", "unknown_category", "duplicate_registration", "missing_annotations"} <= kinds
    assert findings_of(report, "text_label_validity", "empty_text")[0].severity is Severity.ERROR


def test_text_group_overlap_and_cli(tmp_path, capsys):
    root, *_ = _text_dataset(tmp_path)
    # same source on both sides -> group overlap via metadata
    rows = [json.loads(line) for line in (root / "test.jsonl").read_text().splitlines()]
    rows[0]["source"] = "site1"
    _write_jsonl(root / "test.jsonl", rows)
    code = main(["scan", str(root), "--no-report", "--no-cache", "--no-color", "--text-field", "question", "--text-field", "text", "--label-field", "answer"])
    assert code == 1
    out = capsys.readouterr().out
    assert "text_contamination" in out and "group_overlap" in out and "RESULT: FAIL" in out
    assert main(["init", "-f", "text", "-o", str(tmp_path / "s.yaml")]) == 0
    assert "format: text" in (tmp_path / "s.yaml").read_text()


def test_shared_file_metadata_does_not_explode_lineage(tmp_path):
    """Every record of a split shares the same file; that must not become 7000^2 lineage edges."""
    root = tmp_path / "t"
    root.mkdir()
    _write_jsonl(root / "train.jsonl", [{"id": i, "text": f"record number {i} says something different {i * 3}", "source_file": "shared/train.jsonl"} for i in range(400)])
    _write_jsonl(root / "test.jsonl", [{"id": "t", "text": "an evaluation record that is unique"}])
    import time

    t = time.time()
    report = run_scan(root)
    assert time.time() - t < 20
    assert report.dataset["lineage_edges"] == 0
    lin = report.stats["enrich"]["lineage"]
    assert lin["unresolved"] + lin["ambiguous"] >= 1  # the shared file is not a parent sample
    ds = report.dataset_ref
    assert "_file" in ds.get("train:0").metadata and "source_file" in ds.get("train:0").metadata
