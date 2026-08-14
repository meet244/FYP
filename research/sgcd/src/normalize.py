"""Text normalisation for scoring. Fixed once, applied identically everywhere.

Normalisation choices move WER by several points, so this file is frozen before
the TEST run and reported verbatim in the paper.
"""
import re
import unicodedata

_PUNCT = r"""!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥“”‘’—–…"""
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = s.translate(_DEV_DIGITS)
    s = s.lower()  # affects Latin only
    s = re.sub(f"[{_PUNCT}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


DEV_RE = re.compile(r"[ऀ-ॿ]")
LAT_RE = re.compile(r"[a-z]")


def script_of(w: str) -> str:
    if DEV_RE.search(w):
        return "dev"
    if LAT_RE.search(w):
        return "lat"
    return "other"


# ---------------------------------------------------------------------------
# Script-agnostic (transliteration-tolerant) scoring — a SECONDARY metric.
#
# Whisper often writes an English technical term in Devanagari ("स्क्रीन") where
# the reference writes it in Latin ("screen"). Strict WER counts that as an
# error, which conflates "recognised the wrong word" with "recognised the right
# word in the other script". The skeleton below romanises Devanagari and reduces
# both scripts to a coarse consonant skeleton, so the two forms collide:
#   screen / स्क्रीन -> skrn      print / प्रिंट -> prnt      paste / पेस्ट -> pst
# It is deliberately lossy and is reported ALONGSIDE strict WER, never instead
# of it: it is a lower bound on error, as strict WER is an upper bound.
# ---------------------------------------------------------------------------
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    _HAVE_TRANSLIT = True
except ImportError:  # pragma: no cover
    _HAVE_TRANSLIT = False

_PRE_DEV = str.maketrans({"ॉ": "ो", "ॅ": "े", "़": ""})
_DIGRAPHS = [
    ("ph", "f"), ("gh", "g"), ("kh", "k"), ("chh", "c"), ("ch", "c"), ("sh", "s"),
    ("th", "t"), ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"),
    ("ce", "se"), ("ci", "si"), ("cy", "si"),  # soft c, before c -> k below
    ("ee", "i"), ("oo", "u"), ("aa", "a"), ("ai", "e"), ("au", "o"),
]
_SINGLES = str.maketrans({"c": "k", "q": "k", "z": "j", "w": "v", "m": "n", "y": ""})
_VOWELS = str.maketrans({c: "" for c in "aeiou"})


def skeleton(w: str) -> str:
    """Coarse cross-script phonetic skeleton of one word."""
    if DEV_RE.search(w):
        if not _HAVE_TRANSLIT:
            return w
        w = transliterate(w.translate(_PRE_DEV), sanscript.DEVANAGARI, sanscript.ITRANS)
    w = re.sub(r"[^a-zA-Z]", "", w.lower())
    if not w:
        return ""
    first = w[0]  # fallback for words that reduce to nothing (e.g. "a", "y")
    for a, b in _DIGRAPHS:
        w = w.replace(a, b)
    w = w.translate(_SINGLES)
    w = w.translate(_VOWELS)
    w = re.sub(r"(.)\1+", r"\1", w)
    return w or first


def normalize_sa(s: str) -> str:
    """Script-agnostic normalisation: normalize() then per-word skeleton."""
    return " ".join(filter(None, (skeleton(w) for w in normalize(s).split())))


def script_mix(s: str):
    """Fraction of tokens in each script — used to describe the corpus."""
    toks = normalize(s).split()
    if not toks:
        return {"dev": 0.0, "lat": 0.0, "other": 0.0}
    c = {"dev": 0, "lat": 0, "other": 0}
    for t in toks:
        c[script_of(t)] += 1
    return {k: v / len(toks) for k, v in c.items()}
