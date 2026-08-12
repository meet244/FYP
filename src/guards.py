"""Guards and instrumentation (§7.5).

Guards are not defensive engineering to be hidden; they are measurements to be
reported. Each function returns the decision *and* the evidence, and the caller stores
both on the hypothesis row so that `score.py` can aggregate firing rates into
metrics.json.

  context-echo guard  On short or near-silent utterances a context-conditioned decode
                      may continue the injected context instead of transcribing the
                      audio (§7.1's expected failure mode). If the output overlaps the
                      injected context above a threshold, fall back to the unbiased
                      hypothesis and count the firing.

  rewrite guard       For M3b, discard any correction that alters more than a fixed
                      fraction of tokens. An unconstrained rewrite destroys WER;
                      detecting and preventing it is a reportable result.
"""
from __future__ import annotations

import jiwer

from normalize import level1


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def context_echo_score(hyp: str, context: str | None, n: int = 3) -> float:
    """Fraction of the hypothesis's n-grams that also occur in the injected context.

    0.0 means the output is independent of the context; values near 1.0 mean the model
    transcribed the context rather than the audio.
    """
    if not context or not hyp.strip():
        return 0.0
    h = _ngrams(level1(hyp).split(), n)
    if not h:
        return 0.0
    c = _ngrams(level1(context).split(), n)
    return len(h & c) / len(h)


def apply_context_echo_guard(row: dict, fallback_hyp: str, n: int = 3,
                             threshold: float = 0.5,
                             injected: str | None = None) -> dict:
    """Mutates and returns `row`, recording the guard decision and its evidence.

    `injected` is the text that was actually put in front of the decoder. It is passed
    explicitly rather than read from `row["context"]` because M2 injects hint terms via
    `row["hotwords"]` while `row["context"]` is present and None — reading the row would
    silently guard against nothing.
    """
    score = context_echo_score(row.get("hyp", ""),
                               injected if injected is not None else row.get("context"),
                               n)
    row["context_echo_score"] = round(score, 4)
    fired = score >= threshold
    row["guard_context_echo"] = bool(fired)
    if fired:
        row["hyp_before_guard"] = row["hyp"]
        row["hyp"] = fallback_hyp
    return row


def token_change_ratio(before: str, after: str) -> float:
    """Fraction of tokens altered, relative to the pre-correction hypothesis length."""
    b, a = level1(before), level1(after)
    if not b.strip():
        return 0.0 if not a.strip() else 1.0
    o = jiwer.process_words([b], [a if a.strip() else ""])
    return (o.substitutions + o.insertions + o.deletions) / max(1, len(b.split()))


def apply_rewrite_guard(row: dict, corrected: str, max_change: float = 0.20) -> dict:
    """Accept a correction only if it changes at most `max_change` of the tokens."""
    ratio = token_change_ratio(row["hyp"], corrected)
    row["rewrite_change_ratio"] = round(ratio, 4)
    if ratio > max_change:
        row["guard_rewrite_discarded"] = True
        row["corrected"] = False
        return row
    row["guard_rewrite_discarded"] = False
    row["corrected"] = corrected != row["hyp"]
    if row["corrected"]:
        row["hyp_before_correction"] = row["hyp"]
        row["hyp"] = corrected
    return row
