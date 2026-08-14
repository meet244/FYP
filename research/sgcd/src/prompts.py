"""Render syllabus content into Whisper initial_prompt strings.

Design constraints (grounded in how Whisper's prompt slot works):
  - only the final ~224 tokens are consumed  -> we cap at MAX_PROMPT_TOKENS and LEFT-truncate
  - later tokens carry more influence        -> most distinctive content goes LAST
  - the slot expects previous-segment transcript, not a word list
                                             -> C3/C4 use fluent prose (the hypothesis)
"""
MAX_PROMPT_TOKENS = 200  # DEV-tunable: {120, 200}; Whisper consumes the last ~224

try:
    # Whisper's own multilingual BPE — exact budget accounting, no proxy error.
    from mlx_whisper.tokenizer import get_tokenizer

    _ENC = get_tokenizer(multilingual=True, language="hi", task="transcribe").encoding
except Exception:  # pragma: no cover - fallback when MLX is unavailable
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

# Measured: Hindi-English code-mixed prose costs ~3 Whisper tokens per word
# (Devanagari is byte-encoded), so a 20-30 word unit is ~60-90 tokens and k=2
# retrieved units fit inside the cap without truncation.


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))


def _truncate_keep_end(text: str, max_tokens: int = None) -> str:
    max_tokens = MAX_PROMPT_TOKENS if max_tokens is None else max_tokens
    ids = _ENC.encode(text)
    if len(ids) <= max_tokens:
        return text
    return _ENC.decode(ids[-max_tokens:])  # keep the END: highest-influence tokens


# ---- C1: generic style control (NO course content) ----
GENERIC_PROMPT = (
    "यह एक technical tutorial है जिसमें Hindi और English दोनों "
    "का प्रयोग होता है। आइए अब आगे बढ़ते हैं।"
)


# ---- C2: naive keyword list (the known-weak baseline we replicate) ----
def keyword_prompt(units, max_tokens=None):
    kws = []
    for u in units:
        for k in u["keywords"]:
            if k not in kws:
                kws.append(k)
    return _truncate_keep_end(", ".join(kws), max_tokens)


# ---- C3 / C4: prose rendering (the proposed rendering) ----
def prose_prompt(units, max_tokens=None):
    """units is ordered LEAST -> MOST relevant; most relevant lands at the end."""
    return _truncate_keep_end(" ".join(u["prose"].strip() for u in units), max_tokens)


def build(condition, course, retrieved_units=None, max_tokens=None):
    if condition == "C0":
        return None
    if condition == "C1":
        return GENERIC_PROMPT
    if condition == "C2":
        return keyword_prompt(course["units"], max_tokens)
    if condition == "C3":
        return prose_prompt(course["units"], max_tokens)  # whole syllabus
    if condition in ("C4", "C5", "C6", "C7"):
        if not retrieved_units:
            raise ValueError(f"{condition} requires retrieved_units")
        return prose_prompt(retrieved_units, max_tokens)  # retrieved subset
    raise ValueError(condition)
