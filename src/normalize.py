"""Text normalisation for scoring code-switched HI-EN ASR output.

Two levels:
  basic_norm            script-preserving. THE headline metric.
  script_invariant_norm romanise + light phonetic folding, so a word written in
                        Devanagari and the same word written in Latin compare equal.
                        SECONDARY, lenient, always labelled as such in the report.
"""
import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

PUNCT = re.compile(r"[।॥,.\?!;:\"'`\(\)\[\]\{\}—–\-_/\\|@#\$%\^&\*\+=<>~]")
WS = re.compile(r"\s+")
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# The candra vowels used to write English loanwords (ऑफिस, फॉन्ट, कॉपी) have no ITRANS
# equivalent and survive transliteration unconverted, so fold them onto ओ / ए first.
CANDRA = str.maketrans({"ॉ": "ो", "ऑ": "ओ", "ॅ": "े", "ऍ": "ए", "ॲ": "अ"})


def basic_norm(s: str) -> str:
    """Standard, script-preserving normalisation. Use for the headline WER."""
    s = unicodedata.normalize("NFC", s)
    s = s.translate(DEVA_DIGITS)
    s = PUNCT.sub(" ", s)
    s = s.lower()
    s = WS.sub(" ", s).strip()
    return s


def _fold(tok: str, latin: bool = False) -> str:
    """Light phonetic folding applied to BOTH sides, so the comparison stays fair.

    Handles the systematic Devanagari<->Latin spelling differences for English
    loanwords: फ़/फ = ph = f, क = c = k, व = v = w, and the inherent final schwa
    that ITRANS renders as a trailing 'a' (फॉन्ट -> phonta -> font).
    """
    t = re.sub(r"[^a-z0-9]", "", tok.lower())
    if not t:
        return ""
    if latin:
        # English orthographic rules that have no counterpart in the Devanagari
        # spelling: soft c/g before a front vowel, and -tion.
        t = re.sub(r"tion", "shan", t)
        t = re.sub(r"c(?=[eiy])", "s", t)
        t = re.sub(r"g(?=[eiy])", "j", t)
    t = t.replace("ph", "f")
    t = t.replace("ck", "k").replace("ch", "c")
    t = re.sub(r"c(?=[^h]|$)", "k", t)
    t = t.replace("q", "k").replace("x", "ks").replace("z", "j").replace("w", "v")
    t = re.sub(r"(.)\1+", r"\1", t)          # collapse doubled letters
    t = re.sub(r"([bcdgjkt])h", r"\1", t)    # aspirated stops -> plain
    t = re.sub(r"m(?=[bcdfgjklnprstv])", "n", t)   # anusvara -> homorganic nasal
    if len(t) > 3:
        t = re.sub(r"[ae]$", "", t)          # inherent schwa / silent final e
    t = re.sub(r"y$", "i", t)
    t = re.sub(r"m$", "n", t)
    return t


VOWELS = re.compile(r"[aeiou]")


def consonant_skeleton_norm(s: str) -> str:
    """Vowel-free consonant skeleton of the romanised text.

    English and Devanagari spellings of the same loanword agree on consonants but
    rarely on vowels (स्लाइड 'slaid' vs 'slide', सिलेक्ट 'silekt' vs 'select'), so this
    strips vowels entirely. Deliberately lenient: report it as an orthography-agnostic
    LOWER BOUND on WER, never as the WER.
    """
    out = []
    for tok in script_invariant_norm(s).split():
        c = VOWELS.sub("", tok)
        c = re.sub(r"[vy]$", "", c) or tok[:1]   # final semivowel is spelling noise
        out.append(c)
    return " ".join(t for t in out if t)


def script_invariant_norm(s: str) -> str:
    """Romanise Devanagari, then phonetically fold every token."""
    s = basic_norm(s)
    out = []
    for tok in s.split():
        deva = bool(DEVANAGARI.search(tok))
        if deva:
            tok = transliterate(tok.translate(CANDRA), sanscript.DEVANAGARI,
                                sanscript.ITRANS).lower()
        tok = _fold(tok, latin=not deva)
        if tok:
            out.append(tok)
    return " ".join(out)
