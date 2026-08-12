"""M3a — lexical/phonetic output-level correction (§7.3).

For each output token absent from the term lexicon, find the closest retrieved term
under a string-similarity measure and replace it if the similarity exceeds a threshold.
Deterministic, fast and easily explained. The threshold is swept on Tier 1 only.

Two matching passes, both symmetric in script:

  1. single token  — "matplotlip" -> "matplotlib"
  2. token n-gram  — "mat plot lib" -> "matplotlib", which is the characteristic failure
                     mode this study is about: a compound technical term split into
                     several ordinary words. A single-token matcher cannot repair it.

Comparison happens on the level-2 (romanised, phonetically folded) form, so a term the
model emitted in Devanagari can still be matched against a Latin-script lexicon entry.
Replacement is written in the lexicon's own script, which is what the reference uses for
technical terms.

When `spans` is supplied (confidence gating, §7.4) only tokens inside a flagged span are
considered for replacement; everything else is left untouched.
"""
from __future__ import annotations

from rapidfuzz import fuzz, process

from normalize import level1, romanise_token


def _folded(term: str) -> str:
    return " ".join(romanise_token(t) for t in level1(term).split()).strip()


class TermMatcher:
    """Fuzzy matcher over one utterance's candidate term list."""

    def __init__(self, terms: list[str], max_ngram: int = 3):
        self.terms = [t for t in dict.fromkeys(terms)]
        self.max_ngram = max_ngram
        self.folded: dict[str, str] = {}
        self.by_folded: dict[str, str] = {}
        for t in self.terms:
            f = _folded(t)
            if not f:
                continue
            self.folded[t] = f
            self.by_folded.setdefault(f, t)
        self.choices = list(self.by_folded)
        # Terms whose own folded form is multi-word are matched against token n-grams.
        self.single = [c for c in self.choices if " " not in c]

    def in_lexicon(self, token: str) -> bool:
        return _folded(token) in self.by_folded

    def canonical(self, token: str) -> str | None:
        """The lexicon's own surface form for a token that folds onto a term."""
        return self.by_folded.get(_folded(token))

    def best(self, query: str, threshold: int, choices=None) -> tuple[str, float] | None:
        f = _folded(query)
        if not f:
            return None
        pool = choices if choices is not None else self.single
        if not pool:
            return None
        m = process.extractOne(f, pool, scorer=fuzz.ratio, score_cutoff=threshold)
        if not m:
            return None
        return self.by_folded[m[0]], m[1]


def correct_tokens(tokens: list[str], matcher: TermMatcher, threshold: int = 88,
                   min_len: int = 4, allowed: set[int] | None = None) -> tuple[list[str], list[dict]]:
    """Return (corrected tokens, list of edits). `allowed` restricts editable indices."""
    out: list[str] = []
    edits: list[dict] = []
    i = 0
    n = len(tokens)
    while i < n:
        if allowed is not None and i not in allowed:
            out.append(tokens[i])
            i += 1
            continue

        # --- multi-token merge: does a window of 2..max_ngram tokens spell a term? ---
        merged = None
        for w in range(matcher.max_ngram, 1, -1):
            if i + w > n:
                continue
            if allowed is not None and any(j not in allowed for j in range(i, i + w)):
                continue
            window = tokens[i:i + w]
            if any(matcher.in_lexicon(t) for t in window):
                continue          # do not merge tokens that are already terms
            joined = "".join(_folded(t) for t in window)
            if len(joined) < min_len:
                continue
            m = process.extractOne(joined, matcher.single, scorer=fuzz.ratio,
                                   score_cutoff=threshold)
            if m:
                merged = (matcher.by_folded[m[0]], m[1], w, " ".join(window))
                break
        if merged:
            term, score, w, src = merged
            out.append(term)
            edits.append({"type": "merge", "from": src, "to": term,
                          "score": round(score, 1)})
            i += w
            continue

        # --- spelling canonicalisation ---------------------------------------
        # A token can fold onto a lexicon term while its surface spelling differs
        # ("printff" folds to "printf"). Scoring compares level-1 surfaces, so the
        # spelling is still an error and rewriting it to the lexicon's own form is a
        # correction, not a no-op. Restricted to Latin-script tokens: when the model
        # wrote the term in Devanagari the script choice is the reference's business
        # and level-2 scoring already treats the two as equal.
        tok = tokens[i]
        canon = matcher.canonical(tok)
        if canon is not None:
            if tok.isascii() and canon != tok and len(_folded(tok)) >= min_len:
                out.append(canon)
                edits.append({"type": "canon", "from": tok, "to": canon,
                              "score": 100.0})
            else:
                out.append(tok)
            i += 1
            continue

        # --- single-token replacement ---
        if len(_folded(tok)) >= min_len and not matcher.in_lexicon(tok):
            m = matcher.best(tok, threshold)
            if m and m[0] != tok:
                out.append(m[0])
                edits.append({"type": "sub", "from": tok, "to": m[0],
                              "score": round(m[1], 1)})
                i += 1
                continue
        out.append(tok)
        i += 1
    return out, edits


def correct_utterance(hyp: str, candidate_terms: list[str], threshold: int = 88,
                      min_len: int = 4, spans: list[int] | None = None) -> tuple[str, list[dict]]:
    """Correct one hypothesis string. `spans` = indices of low-confidence tokens (§7.4)."""
    if not candidate_terms or not hyp.strip():
        return hyp, []
    tokens = level1(hyp).split()
    matcher = TermMatcher(candidate_terms)
    allowed = set(spans) if spans is not None else None
    fixed, edits = correct_tokens(tokens, matcher, threshold, min_len, allowed)
    return " ".join(fixed), edits
