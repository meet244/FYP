"""Export the original leakage-free FYRP S5 LM without decoding/training.

Run from backend: .venv/bin/python scripts/export_s5_lm.py
Requires the research manifest; the exported app artifact does not.
"""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.asr.normalize import normalize
from app.asr.rescore import DEFAULT_ASSET


def export():
    from transformers import AutoTokenizer
    root = BACKEND.parent / "research" / "sgcd"
    rows = [json.loads(line) for line in (root / "out/manifest.jsonl").read_text().splitlines() if line.strip()]
    tune = [r for r in rows if r["split"] == "dev" and r.get("in_eval", True)][::2][:30]
    tune_ids = {r["utt_id"] for r in tune}
    dev = [r for r in rows if r["split"] == "dev" and r["utt_id"] not in tune_ids]
    test = [r for r in rows if r["split"] == "test"]
    assert not ({r["lecture_id"] for r in dev} & {r["lecture_id"] for r in test})
    corpus = []
    for path in sorted((root / "syllabi").glob("*.json")):
        if path.stem in ("lecture_map", "lecture_titles"):
            continue
        for unit in json.loads(path.read_text())["units"]:
            corpus.append(normalize(f"{unit['title']} {unit['prose']} {' '.join(unit['keywords'])}"))
    lectures = defaultdict(list)
    for row in dev:
        lectures[row["lecture_id"]].append(row)
    for utterances in lectures.values():
        utterances.sort(key=lambda row: row["start"])
        for i in range(0, len(utterances), 4):
            corpus.append(normalize(" ".join(r["ref"] for r in utterances[i:i+4])))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B-hf")
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    uni, bi, tri = Counter(), defaultdict(Counter), defaultdict(Counter)
    for text in corpus:
        tokens = [-1, -1] + tokenizer.encode(normalize(text), add_special_tokens=False) + [eos]
        for u, v, token in zip(tokens, tokens[1:], tokens[2:]):
            uni[token] += 1
            bi[(v,)][token] += 1
            tri[(u, v)][token] += 1
    data = dict(format="fyrp-token-trigram-v1", vocab_size=151936, eos_id=eos,
                discount=0.75, alpha=0.01, unigram=uni,
                bigram=sorted(bi.items()), trigram=sorted(tri.items()),
                tokenizer_sha256=hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True,
                                                          separators=(",", ":")).encode()).hexdigest(),
                provenance=dict(method="S5", dataset="OpenSLR SLR104 Hindi-English",
                                license="CC BY-SA 4.0", source_url="https://www.openslr.org/104/",
                                corpus_segments=len(corpus), dev_utterances=len(dev), tuning_excluded=30,
                                corpus_sha256=hashlib.sha256(json.dumps(corpus, ensure_ascii=False).encode()).hexdigest(),
                                dev_utterance_ids=sorted(r["utt_id"] for r in dev),
                                dev_lecture_ids=sorted(lectures), test_lecture_ids=sorted({r["lecture_id"] for r in test}),
                                beams=5, lm_weight=0.2, length_bonus=0.5))
    DEFAULT_ASSET.parent.mkdir(parents=True, exist_ok=True)
    # Fixed gzip timestamp makes repeated exports byte reproducible.
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    DEFAULT_ASSET.write_bytes(gzip.compress(encoded, mtime=0))
    print(f"Exported {len(corpus)} segments to {DEFAULT_ASSET} ({DEFAULT_ASSET.stat().st_size} bytes)")


if __name__ == "__main__":
    export()
