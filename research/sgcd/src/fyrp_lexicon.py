"""FYRP post-hoc text methods: glossary snapping (S1), phonetic snapping (S2),
BM25-constrained span correction (S4). Pure text, no model — they rewrite cached
baseline hypotheses.

Glossary / corpus sources are leakage-free: syllabus JSON + SLR104 *dev*-split
reference transcripts (dev lectures are disjoint from test lectures).
"""
import collections
import json
import math
import re

from rapidfuzz import fuzz, process

from config import SYL
from normalize import normalize, script_of, DEV_RE

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ImportError:  # pragma: no cover
    transliterate = None


# ------------------------------------------------------------- resources ---
def syllabus_units():
    units = []
    for p in sorted(SYL.glob("*.json")):
        if p.stem in ("lecture_map", "lecture_titles"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for u in d["units"]:
            units.append(dict(u, course_id=d["course_id"]))
    return units


def build_glossary(dev_refs, min_count=2):
    """Flat term -> frequency glossary.

    syllabus keywords (always in) + in-domain tokens from dev transcripts seen
    >= min_count times (both scripts: English terms in Latin *and* their
    Devanagari transliterations, e.g. 'font' and 'फॉन्ट')."""
    cnt = collections.Counter(w for r in dev_refs for w in normalize(r).split())
    gl = {w: c for w, c in cnt.items() if c >= min_count and not w.isdigit()}
    for u in syllabus_units():
        for k in u["keywords"]:
            for w in normalize(k).split():
                gl[w] = gl.get(w, 0) + 5  # syllabus prior: outranks one-off dev tokens
    return gl


def top_terms(glossary, n=80):
    """Syllabus keywords ranked by in-domain frequency — the S3 hotword list."""
    kw = {w for u in syllabus_units() for k in u["keywords"] for w in normalize(k).split()
          if len(w) > 2}
    return [w for w, _ in sorted(((w, glossary.get(w, 0)) for w in kw), key=lambda x: -x[1])][:n]


# --------------------------------------------------------- phonetic key ---
_PRE_DEV = str.maketrans({"ॉ": "ो", "ॅ": "े", "ऑ": "ओ", "ऍ": "ए", "़": ""})
_PHON_RULES = [
    ("chh", "c"), ("ch", "c"), ("sh", "s"), ("ph", "f"), ("th", "t"), ("dh", "d"),
    ("kh", "k"), ("gh", "g"), ("bh", "b"), ("jh", "j"), ("ck", "k"), ("x", "ks"),
    ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"),   # vowel length
    ("ai", "e"), ("au", "o"), ("ou", "o"),
]
_PHON_SINGLE = str.maketrans({"w": "v", "q": "k", "c": "k", "z": "j", "y": "i"})


def phon(w):
    """Custom phonetic key for code-switched tokens. Unlike the scorer's
    consonant skeleton it KEEPS vowels (collapsed for length) so short words
    stay distinguishable; handles w/v, sh/s, ph/f, aspiration, Devanagari via ITRANS."""
    if DEV_RE.search(w):
        if transliterate is None:
            return w
        w = transliterate(w.translate(_PRE_DEV), sanscript.DEVANAGARI, sanscript.ITRANS)
        # आ+इ spells English /aI/ (स्लाइड = slide); anusvara is a nasal
        w = w.replace("Ai", "i").replace("M", "n")
    else:
        w = re.sub(r"c(?=[eiy])", "s", w.lower())                # soft c: office ~ ऑफिस
        w = re.sub(r"(?<=[^aeiou])e$", "", w)  # English silent e: slide -> slid
        w = re.sub(r"[ts]ion", "shan", w)                # option ~ ऑप्शन
    w = re.sub(r"[^a-z]", "", w.lower())
    for a, b in _PHON_RULES:
        w = w.replace(a, b)
    w = w.translate(_PHON_SINGLE)
    w = re.sub(r"h", "", w) or w           # residual aspiration / schwa-h
    w = re.sub(r"(.)\1+", r"\1", w)
    w = re.sub(r"a$", "", w) or w          # inherent-schwa deletion (कर -> kar/kara)
    return w


# ------------------------------------------------------ S1 / S2 snappers ---
class Snapper:
    """Token-level snapper. `mode='fuzzy'`: raw edit similarity, same script
    only. `mode='phonetic'`: similarity on phonetic keys, cross-script allowed,
    ties broken by corpus frequency so the dominant written form wins."""

    def __init__(self, glossary, mode="fuzzy", min_len=4):
        self.gl, self.mode, self.min_len = glossary, mode, min_len
        if mode == "fuzzy":
            self.by_script = collections.defaultdict(list)
            for w in glossary:
                self.by_script[script_of(w)].append(w)
        else:
            self.key2word = {}
            for w, c in glossary.items():
                k = phon(w)
                if len(k) >= 2 and (k not in self.key2word or c > glossary[self.key2word[k]]):
                    self.key2word[k] = w
            self.keys = list(self.key2word)
        self._cache = {}

    def snap_token(self, t, thr):
        if t in self.gl or len(t) < self.min_len or t.isdigit():
            return t
        ck = (t, thr)
        if ck in self._cache:
            return self._cache[ck]
        out = t
        if self.mode == "fuzzy":
            cands = self.by_script.get(script_of(t), [])
            m = process.extractOne(t, cands, scorer=fuzz.ratio, score_cutoff=thr)
            if m:
                out = m[0]
        else:
            k = phon(t)
            if k in self.key2word:
                out = self.key2word[k]
            elif thr <= 100:  # thr > 100 = exact phonetic-key match only
                m = process.extractOne(k, self.keys, scorer=fuzz.ratio, score_cutoff=thr)
                if m:
                    out = self.key2word[m[0]]
        self._cache[ck] = out
        return out

    def correct(self, hyp, thr):
        return " ".join(self.snap_token(t, thr) for t in normalize(hyp).split())


# ------------------------------------------------------------- S4 BM25 ---
def build_corpus(dev_rows, win=4):
    """Syllabus corpus: every syllabus unit + dev transcripts chunked into
    windows of `win` consecutive utterances per lecture. Also the LM corpus (S5/S6)."""
    segs = [normalize(f"{u['title']} {u['prose']} {' '.join(u['keywords'])}")
            for u in syllabus_units()]
    by_lec = collections.defaultdict(list)
    for r in dev_rows:
        by_lec[r["lecture_id"]].append(r)
    for rs in by_lec.values():
        rs.sort(key=lambda r: r["start"])
        for i in range(0, len(rs), win):
            segs.append(normalize(" ".join(r["ref"] for r in rs[i:i + win])))
    return segs


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [d.split() for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        df = collections.Counter(w for d in self.docs for w in set(d))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [collections.Counter(d) for d in self.docs]

    def top(self, query, k=3):
        q = query.split()
        sc = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avg)
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm)
            sc.append((s, i))
        sc.sort(reverse=True)
        return [i for _, i in sc[:k]]


class RetrievalCorrector:
    """S4: retrieve top-k segments per hypothesis; only snap tokens to vocabulary
    that occurs in those segments (phonetic keys, cross-script)."""

    def __init__(self, segs, glossary, k=3):
        self.segs, self.gl, self.k = segs, glossary, k
        self.bm = BM25(segs)

    def correct(self, hyp, thr):
        h = normalize(hyp)
        vocab = {}
        for i in self.bm.top(h, self.k):
            for w in self.segs[i].split():
                vocab[w] = self.gl.get(w, 1)
        local = Snapper(vocab, mode="phonetic")
        out = []
        for t in h.split():
            # a token already valid in the global lexicon is left alone
            out.append(t if t in self.gl else local.snap_token(t, thr))
        return " ".join(out)
