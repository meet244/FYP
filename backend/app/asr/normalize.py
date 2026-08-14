"""Text normalisation, ported verbatim from research/sgcd/src/normalize.py.

Kept byte-identical in behaviour so transcripts produced here are comparable
with the published numbers. Do not "improve" this file.
"""
from __future__ import annotations

import re
import unicodedata

_PUNCT = r"""!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥“”‘’—–…"""
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

DEV_RE = re.compile(r"[ऀ-ॿ]")
LAT_RE = re.compile(r"[a-z]")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.translate(_DEV_DIGITS)
    s = s.lower()  # affects Latin only
    s = re.sub(f"[{_PUNCT}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def script_of(w: str) -> str:
    if DEV_RE.search(w):
        return "dev"
    if LAT_RE.search(w):
        return "lat"
    return "other"


def script_mix(s: str) -> dict[str, float]:
    """Fraction of tokens in each script — the observable the paper's script-fidelity
    metric is built on. Surfaced per lecture so a deployment can watch for the
    Devanagari-transliteration failure mode without reference transcripts."""
    toks = normalize(s).split()
    if not toks:
        return {"dev": 0.0, "lat": 0.0, "other": 0.0}
    counts = {"dev": 0, "lat": 0, "other": 0}
    for t in toks:
        counts[script_of(t)] += 1
    return {k: v / len(toks) for k, v in counts.items()}
