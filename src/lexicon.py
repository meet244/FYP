"""The frozen syllabus term lexicon (§5.3b).

The lexicon has two roles and they must not be confused:

  1. it supplies the bias list for the grounding mechanisms, and
  2. it *defines the metric* — B-WER counts errors on reference words that are lexicon
     members, U-WER counts the rest.

Because of role 2 the lexicon is frozen before any grounded condition runs, and its
size and content hash are recorded (§5.3 "critical constraint"). `Lexicon.stamp()`
returns that record; every metrics.json embeds it, so a metric can always be traced
back to the exact term list that produced it.

Membership is tested at two normalisation levels. The corpus writes English technical
terms sometimes in Latin and sometimes in Devanagari script ('slide' vs 'स्लाइड'), so
level-1 membership alone would silently exclude every Devanagari-written term from
B-WER. `in_bias` therefore also accepts a level-2 (romanised, phonetically folded)
match, and both counts are reported.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from common import ROOT, file_hash
from normalize import level1, romanise_token


class Lexicon:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = [t.strip() for t in self.path.read_text(encoding="utf-8").splitlines()]
        self.terms = sorted({t.lower() for t in raw if t.strip()})
        # Surface form after level-1 normalisation (what a reference token looks like).
        self.l1 = {level1(t) for t in self.terms}
        self.l1.discard("")
        # Romanised/folded form, for terms written in Devanagari in the reference.
        self.l2 = {romanise_token(t) for t in self.terms}
        self.l2.discard("")

    def __len__(self) -> int:
        return len(self.terms)

    def in_bias(self, token: str) -> bool:
        """Is this (level-1 normalised) reference/hypothesis token a bias term?"""
        if token in self.l1:
            return True
        return romanise_token(token) in self.l2

    def bias_flags(self, tokens: list[str]) -> list[bool]:
        return [_cached_member(self, t) for t in tokens]

    def stamp(self) -> dict:
        """Provenance record embedded in every metrics.json (§13)."""
        try:
            shown = str(self.path.relative_to(ROOT))
        except ValueError:
            shown = str(self.path)
        return {
            "lexicon_path": shown,
            "lexicon_size": len(self.terms),
            "lexicon_sha256_12": file_hash(self.path),
        }

    def coverage(self, refs: list[str]) -> dict:
        """§5.4: what fraction of reference word tokens are lexicon terms?

        This is the ceiling on achievable gain from terminology biasing and belongs in
        the results section.
        """
        tot = hit = hit_l1_only = 0
        for r in refs:
            for tok in level1(r).split():
                tot += 1
                if tok in self.l1:
                    hit += 1
                    hit_l1_only += 1
                elif romanise_token(tok) in self.l2:
                    hit += 1
        return {
            "ref_tokens": tot,
            "bias_tokens": hit,
            "bias_token_rate": hit / tot if tot else 0.0,
            "bias_tokens_same_script": hit_l1_only,
            "bias_token_rate_same_script": hit_l1_only / tot if tot else 0.0,
        }


_MEMO: dict[tuple[int, str], bool] = {}


def _cached_member(lex: Lexicon, token: str) -> bool:
    """Memoised membership: romanise_token is the hot path in scoring."""
    key = (id(lex), token)
    v = _MEMO.get(key)
    if v is None:
        v = lex.in_bias(token)
        _MEMO[key] = v
    return v


@lru_cache(maxsize=8)
def load_lexicon(path: str | Path = "syllabus/index/terms.txt") -> Lexicon:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return Lexicon(p)
