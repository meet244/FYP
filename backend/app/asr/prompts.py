"""Render syllabus units into a decoder `initial_prompt`.

Three constraints from Section III-B, all load-bearing:

  1. Register/orthography — the prompt is fluent code-mixed *narration*, never a
     terminology list. A comma-separated list reproduces the Sun et al. failure
     mode: terminology improves, everything else degrades, aggregate WER +19.39.
  2. Terminal weighting — later context tokens dominate, so units are emitted in
     ASCENDING relevance and truncation drops from the LEFT.
  3. Budget — code-mixed prose costs ~3 tokens/word; the channel is capped at 200.
"""
from __future__ import annotations

import functools

MAX_PROMPT_TOKENS = 200

# One generic code-mixed sentence with no course content (condition G). It lifts
# script fidelity 32.6% -> 41.6% on its own, but on the only set where it was
# measured it also regressed aggregate WER (+16.63, CI spanning zero) — so it is
# used only as a degenerate filler when a retrieved unit set turns out empty,
# never as a blanket substitute for having no syllabus. See sgcd.transcribe().
GENERIC_PROMPT = (
    "यह एक technical tutorial है जिसमें Hindi और English दोनों "
    "का प्रयोग होता है। आइए अब आगे बढ़ते हैं।"
)


@functools.lru_cache(maxsize=1)
def _encoder():
    """Whisper's own multilingual BPE, so the budget is exact rather than a proxy."""
    try:
        from mlx_whisper.tokenizer import get_tokenizer

        return get_tokenizer(multilingual=True, language="hi", task="transcribe").encoding
    except Exception:  # pragma: no cover - non-Apple-silicon hosts
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str | None) -> int:
    if not text:
        return 0
    return len(_encoder().encode(text))


def truncate_keep_end(text: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    enc = _encoder()
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[-max_tokens:])  # keep the END: highest-influence tokens


def prose_prompt(units, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """`units` is ordered LEAST -> MOST relevant, so the best match lands last.

    Each element needs a `.prose` attribute (a SyllabusUnit) or a "prose" key.
    """
    parts = []
    for u in units:
        prose = getattr(u, "prose", None) if not isinstance(u, dict) else u.get("prose")
        if prose and prose.strip():
            parts.append(prose.strip())
    if not parts:
        return GENERIC_PROMPT
    return truncate_keep_end(" ".join(parts), max_tokens)


def build_prompt(units, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """The only prompt builder the production path uses.

    The research code also implements keyword-list and whole-syllabus conditions;
    they exist to be measured, not deployed. Enumeration degraded aggregate WER on
    both evaluation sets, so it is deliberately absent here.
    """
    if not units:
        return GENERIC_PROMPT
    return prose_prompt(units, max_tokens)
