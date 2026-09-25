#!/usr/bin/env python3
"""Generate a small LLM-style text dataset with planted benchmark contamination.

    python examples/make_text_demo.py text-demo
    sentinel scan text-demo --text-field question --text-field text --label-field answer

train.jsonl: 300 synthetic "documents" (an SFT / pretraining stand-in) into
which a few evaluation items were leaked: one embedded verbatim inside a long
document, one copied exactly (different casing / punctuation), one lightly
paraphrased, plus one record sharing a ``source`` with an evaluation item.
test.jsonl: 40 benchmark-style questions with an ``answer`` label.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

WORDS = (
    "the quick brown fox jumps over lazy dog while seven wizards quietly juggle glass boxes near river bank under pale "
    "moonlight and engineers measure latency across distributed systems using careful experiments with reproducible seeds"
).split()


def doc(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def main(root: Path) -> None:
    rng = random.Random(42)
    root.mkdir(parents=True, exist_ok=True)
    evals = [
        {"id": f"q{i:03d}", "question": f"Question {i}: which of the following best explains phenomenon {i * 13 % 97} in context {i * 7 % 31}?", "choices": ["A", "B", "C", "D"], "answer": "ABCD"[i % 4], "source": f"bench{i}"}
        for i in range(40)
    ]
    train = [{"id": f"doc{i:04d}", "text": doc(rng, rng.randint(40, 160)), "source": f"crawl{i % 7}"} for i in range(300)]
    train.append({"id": "leak_embedded", "text": doc(rng, 50) + " " + evals[7]["question"] + " " + doc(rng, 40), "source": "crawl1"})
    train.append({"id": "leak_exact", "text": evals[12]["question"].upper().replace("?", "!"), "source": "crawl2"})
    train.append({"id": "leak_paraphrase", "text": evals[20]["question"].replace("best explains", "best describes"), "source": "crawl3"})
    train.append({"id": "same_source", "text": doc(rng, 60), "source": "bench3"})  # shares a source with one eval item
    (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (root / "test.jsonl").write_text("\n".join(json.dumps(r) for r in evals) + "\n")
    print(f"wrote {root}/train.jsonl ({len(train)} records) and test.jsonl ({len(evals)} items)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "text-demo"))
