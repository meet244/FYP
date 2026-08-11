"""Freeze the evaluation subset (seed 1337) plus a 200-utt dev slice for tuning."""
import json
import random
from pathlib import Path

FULL = Path("data/manifests/test.jsonl")
rows = [json.loads(l) for l in open(FULL, encoding="utf-8")]
random.Random(1337).shuffle(rows)
subset = rows[:800]


def dump(rs, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in sorted(rs, key=lambda x: x["utt_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mins = sum(r["duration"] or 0 for r in rs) / 60
    print(f"{len(rs)} utts, {mins:.1f} min -> {path}")


# Held-out slice for threshold / k selection; the other 600 are the ablation set.
dump(subset, "data/manifests/subset.jsonl")
dump(subset[:200], "data/manifests/dev200.jsonl")
dump(subset[200:], "data/manifests/eval600.jsonl")
