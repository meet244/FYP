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


def echo_guard_fires(hyp: str, injected: str | None, n: int = 3,
                     threshold: float = 0.5) -> tuple[bool, float]:
    """Decision + evidence, without touching the hypothesis."""
    score = context_echo_score(hyp, injected, n)
    return score >= threshold and bool(injected), round(score, 4)


def runaway_guard_fires(hyp: str, fallback_hyp: str, max_ratio: float = 1.75,
                        pad: int = 3) -> tuple[bool, float | None]:
    """Decision + evidence, without touching the hypothesis.

    Kept separate from the substitution so that several guards can be evaluated against
    the *same* original hypothesis: applying one guard first would rewrite the output the
    next guard is supposed to judge, and the second would then never fire.
    """
    n_ref = len((fallback_hyp or "").split())
    n_hyp = len((hyp or "").split())
    ratio = round(n_hyp / n_ref, 3) if n_ref else None
    return bool(n_ref and n_hyp > max_ratio * n_ref + pad), ratio


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


def apply_runaway_guard(row: dict, fallback_hyp: str, max_ratio: float = 1.75,
                        pad: int = 3) -> dict:
    """Fall back when a grounded decode produces far more words than the unbiased one.

    The context-echo guard looks for the injected text appearing verbatim in the output.
    On this corpus that is the wrong thing to measure: conditioning on a syllabus passage
    did not make the model repeat the passage, it made the model keep *generating* in the
    passage's register — new terminology that was never spoken. Insertions doubled or
    tripled while n-gram echo stayed near zero, so the echo guard fired on 2% of
    utterances while the damage was happening everywhere.

    Output length relative to the unbiased hypothesis catches it directly, and it is a
    fair test: the two decodes saw identical audio, so a hypothesis far longer than the
    unbiased one is generating rather than transcribing. Measured on Tier 2, this fires on
    3-4% of utterances for the context conditions and 0-1 utterances for the baseline,
    token-biasing and correction conditions — it touches only pathological output.
    """
    n_ref = len((fallback_hyp or "").split())
    n_hyp = len((row.get("hyp") or "").split())
    row["runaway_ratio"] = round(n_hyp / n_ref, 3) if n_ref else None
    fired = bool(n_ref and n_hyp > max_ratio * n_ref + pad)
    row["guard_runaway"] = fired
    if fired:
        row.setdefault("hyp_before_guard", row["hyp"])
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
