"""Normalisation for scoring code-switched Hindi-English ASR output (§8.1).

Two levels, exactly as specified:

  level1  Standard, script-preserving: Unicode NFC, numeral unification, punctuation
          removal, case folding, whitespace collapse. THIS PRODUCES THE HEADLINE WER
          and is comparable with published baselines on this corpus.

  level2  Script-invariant: additionally romanises Devanagari and applies a light,
          symmetric phonetic folding so that a word written in either script compares
          equal. A clearly labelled SECONDARY metric, never presented as the WER.

`skeleton` (vowel-free consonant skeleton) is an additional, even more lenient
diagnostic used only in the error analysis as an orthography-agnostic lower bound.
Both folding functions are applied to reference and hypothesis alike, so no level can
flatter the hypothesis.
"""
from __future__ import annotations

import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

PUNCT = re.compile(r"[।॥,.\?!;:\"'`\(\)\[\]\{\}—–\-_/\\|@#\$%\^&\*\+=<>~]")
WS = re.compile(r"\s+")
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# The candra vowels used when writing English loanwords in Devanagari (ऑफिस, फॉन्ट,
# कॉपी) have no ITRANS equivalent and would survive transliteration unconverted, so
# fold them onto ओ / ए before romanising.
CANDRA = str.maketrans({"ॉ": "ो", "ऑ": "ओ", "ॅ": "े", "ऍ": "ए", "ॲ": "अ"})

# Spelled-out numbers, unified with digits at level 1 (§8.1 "numeral form").
NUKTA = "़"
# Precomposed nukta consonants, mapped to their base letters. Hindi orthography uses
# the two interchangeably (फ़/फ, ज़/ज, ड़/ड) and the corpus mixes them, so folding them
# is standard Devanagari normalisation for ASR scoring, not leniency. Applied to
# reference and hypothesis alike.
NUKTA_MAP = str.maketrans({
    "ऩ": "न",  # ऩ -> न
    "ऱ": "र",  # ऱ -> र
    "ऴ": "ळ",  # ऴ -> ळ
    "क़": "क",  # क़ -> क
    "ख़": "ख",  # ख़ -> ख
    "ग़": "ग",  # ग़ -> ग
    "ज़": "ज",  # ज़ -> ज
    "ड़": "ड",  # ड़ -> ड
    "ढ़": "ढ",  # ढ़ -> ढ
    "फ़": "फ",  # फ़ -> फ
    "य़": "य",  # य़ -> य
    NUKTA: "",           # standalone combining nukta
    # Chandrabindu and anusvara are interchangeable in modern Hindi typing
    # (यहाँ/यहां, जाएँ/जाएं); fold the former onto the latter.
    "ँ": "ं",
})

NUMWORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5",
    "पांच": "5", "छह": "6", "छः": "6", "सात": "7", "आठ": "8", "नौ": "9", "दस": "10",
}


def level1(s: str) -> str:
    """Standard, script-preserving normalisation. Use for the headline WER."""
    s = unicodedata.normalize("NFC", s)
    s = s.translate(NUKTA_MAP)
    s = s.translate(DEVA_DIGITS)
    s = PUNCT.sub(" ", s)
    s = s.lower()
    s = WS.sub(" ", s).strip()
    return " ".join(NUMWORDS.get(t, t) for t in s.split())


VOWELS = re.compile(r"[aeiou]")


def _fold(tok: str, latin: bool = False) -> str:
    """Symmetric phonetic folding of one romanised token.

    Handles the systematic Devanagari<->Latin spelling differences for English
    loanwords: फ़/फ = ph = f, क = c = k, व = v = w, and the inherent final schwa that
    ITRANS renders as a trailing 'a' (फॉन्ट -> phonta -> font).
    """
    t = re.sub(r"[^a-z0-9]", "", tok.lower())
    if not t:
        return ""
    if latin:
        # English orthographic rules with no counterpart in a Devanagari spelling:
        # soft c/g before a front vowel, and -tion.
        t = re.sub(r"tion", "shan", t)
        t = re.sub(r"c(?=[eiy])", "s", t)
        t = re.sub(r"g(?=[eiy])", "j", t)
    t = t.replace("ph", "f")
    t = t.replace("ck", "k").replace("ch", "c")
    t = re.sub(r"c(?=[^h]|$)", "k", t)
    t = t.replace("q", "k").replace("x", "ks").replace("z", "j").replace("w", "v")
    t = re.sub(r"(.)\1+", r"\1", t)                 # collapse doubled letters
    t = re.sub(r"([bcdgjkt])h", r"\1", t)           # aspirated stops -> plain
    t = re.sub(r"m(?=[bcdfgjklnprstv])", "n", t)    # anusvara -> homorganic nasal
    # A vowel cluster is reduced to its final vowel. Devanagari spellings of English
    # loanwords use vowel digraphs where the Latin spelling uses one letter and vice
    # versa (स्लाइड -> 'slaid' vs 'slide' -> 'slid'; साइज़ -> 'saij' vs 'size' -> 'sij').
    # Applied to both sides, so it cannot favour the hypothesis.
    t = re.sub(r"[aeiou]{2,}", lambda mo: mo.group(0)[-1], t)
    if len(t) > 3:
        t = re.sub(r"[ae]$", "", t)                 # inherent schwa / silent final e
    t = re.sub(r"y$", "i", t)
    t = re.sub(r"m$", "n", t)
    return t


def romanise_token(tok: str) -> str:
    """Romanise one token if it is Devanagari, then fold it phonetically."""
    deva = bool(DEVANAGARI.search(tok))
    if deva:
        tok = transliterate(tok.translate(CANDRA), sanscript.DEVANAGARI,
                            sanscript.ITRANS).lower()
    return _fold(tok, latin=not deva)


def level2(s: str) -> str:
    """Script-invariant normalisation. SECONDARY metric (§8.1)."""
    out = []
    for tok in level1(s).split():
        t = romanise_token(tok)
        if t:
            out.append(t)
    return " ".join(out)


def skeleton(s: str) -> str:
    """Vowel-free consonant skeleton of the romanised text.

    English and Devanagari spellings of the same loanword agree on consonants but
    rarely on vowels (स्लाइड 'slaid' vs 'slide', सिलेक्ट 'silekt' vs 'select'). Very
    lenient: used only as an orthography-agnostic lower bound in the error analysis.
    """
    out = []
    for tok in level2(s).split():
        c = VOWELS.sub("", tok)
        c = re.sub(r"[vy]$", "", c) or tok[:1]
        if c:
            out.append(c)
    return " ".join(out)


LEVELS = {"level1": level1, "level2": level2, "skeleton": skeleton}

# Backwards-compatible aliases used by earlier scripts.
basic_norm = level1
script_invariant_norm = level2
consonant_skeleton_norm = skeleton


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        line = line.rstrip("\n")
        print(f"raw    : {line}\nlevel1 : {level1(line)}\nlevel2 : {level2(line)}\n"
              f"skel   : {skeleton(line)}\n")
