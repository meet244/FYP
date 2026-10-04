"""FYRP S5: frozen token-trigram LM and independent N-best rescoring.

Scalar lookups preserve fyrp_lm.py's absolute-discount interpolation without
caching a 151,936-entry distribution for every context on an 8 GB laptop.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

from app.asr.normalize import normalize

DEFAULT_ASSET = Path(__file__).resolve().parents[2] / "assets" / "fyrp_s5_lm.json.gz"


def asset_info(path=DEFAULT_ASSET) -> dict:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("S5 domain LM is missing; run scripts/export_s5_lm.py or select Baseline.")
    return {"lm_path": str(path.resolve()), "lm_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class DomainLM:
    def __init__(self, tokenizer, path=DEFAULT_ASSET, expected_sha256=None):
        info = asset_info(path)
        if expected_sha256 and info["lm_sha256"] != expected_sha256:
            raise RuntimeError("S5 LM changed since this job was created; submit a new transcription run.")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            data = json.load(source)
        if data["format"] != "fyrp-token-trigram-v1":
            raise ValueError("Unsupported S5 LM format")
        vocab_hash = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
        if vocab_hash != data["tokenizer_sha256"]:
            raise RuntimeError("S5 LM tokenizer does not match the selected checkpoint.")
        self.tokenizer = tokenizer
        self.metadata = data["provenance"]
        self.eos = data["eos_id"]
        self.discount, self.alpha = data["discount"], data["alpha"]
        self.uni = {int(k): v for k, v in data["unigram"].items()}
        self.denominator = sum(self.uni.values()) + self.alpha * data["vocab_size"]
        self.bi = self._tables(data["bigram"])
        self.tri = self._tables(data["trigram"])

    @staticmethod
    def _tables(rows):
        result = {}
        for context, counts in rows:
            counts = {int(k): v for k, v in counts.items()}
            result[tuple(context)] = (counts, sum(counts.values()))
        return result

    def _mix(self, table, context, token, lower):
        entry = table.get(context)
        if entry is None:
            return lower
        counts, total = entry
        return (lower * (self.discount * len(counts) / total)
                + max(counts.get(token, 0) - self.discount, 0) / total)

    def seq_logprob(self, text):
        tokens = [-1, -1] + self.tokenizer.encode(normalize(text), add_special_tokens=False) + [self.eos]
        score = 0.0
        for u, v, token in zip(tokens, tokens[1:], tokens[2:]):
            lower = (self.uni.get(token, 0) + self.alpha) / self.denominator
            lower = self._mix(self.bi, (v,), token, lower)
            score += math.log(self._mix(self.tri, (u, v), token, lower))
        return score


def select_candidate(candidates, lm, weight=0.2, length_bonus=0.5):
    if not candidates:
        raise RuntimeError("S5 decoder returned no hypotheses")
    return max(candidates, key=lambda h: h["am"] + weight * lm.seq_logprob(h["text"])
               + length_bonus * len(normalize(h["text"]).split()))
