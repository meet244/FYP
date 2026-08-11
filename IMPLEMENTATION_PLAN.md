# Syllabus-Grounded ASR for Code-Switched Classroom Lectures
## Implementation Plan — Whisper baseline → syllabus grounding → WER reduction

**Scope of this document:** everything from environment setup to a final results table with
statistical significance. Note generation (the downstream LLM summarisation) is explicitly
**out of scope** and is your part.

**Target system:** MacBook Air (Apple Silicon), Python 3.11.
**Dataset:** OpenSLR SLR104, Hindi-English code-switched subset (MUCS 2021 subtask-2).
**Model:** Whisper large-v3, run locally via `faster-whisper` (CTranslate2), with the Groq
hosted endpoint as an optional fast path.

---

## 0. The research claim you are testing

State it precisely before writing any code, because it determines the whole harness:

> **H1.** Injecting retrieved syllabus context into the decoding process of a general-purpose
> ASR model reduces word error rate on code-switched technical lecture speech, and the
> reduction is concentrated in domain-specific terminology rather than distributed uniformly.

Two things follow from that phrasing:

1. You need a **term-level metric**, not just corpus WER. If Whisper mistranscribes
   *"matplotlib"* as *"mat plot lib"*, that is 2 insertions out of a ~15-word utterance — a
   real failure that barely moves corpus WER. Reporting only WER will make a working method
   look like it did nothing.
2. You must be honest about the **direction of the effect**. Prompt biasing can *increase*
   WER by hallucinating syllabus terms into utterances that never contained them. Measure
   both gains and regressions. A finding of "term recall +14 points, corpus WER −0.9 points,
   with 3% of utterances regressed" is a much better final-year result than an unexamined
   "WER went down".

---

## 1. Environment setup

```bash
# Homebrew deps
brew install ffmpeg wget

# Python via uv (fast, reproducible)
curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir -p ~/projects/lecture-asr && cd ~/projects/lecture-asr
uv venv --python 3.11
source .venv/bin/activate
```

`requirements.txt`:

```
faster-whisper==1.1.1
ctranslate2>=4.5.0
huggingface-hub>=0.26
soundfile
librosa
numpy
pandas
pyyaml
tqdm
jiwer==3.0.4
rapidfuzz
sentence-transformers
indic-transliteration
groq
python-dotenv
```

```bash
uv pip install -r requirements.txt
```

Repository layout — create this now, it keeps the ablations manageable:

```
lecture-asr/
├── configs/config.yaml
├── data/
│   ├── raw/slr104/            # tarballs + extracted Kaldi-style dirs
│   └── manifests/             # test.jsonl, subset.jsonl
├── syllabus/
│   ├── raw/                   # one .md per course/topic
│   └── index/                 # embeddings + term lexicon
├── cache/asr/                 # one JSON per (utt_id, backend, config_hash)
├── runs/                      # one dir per experiment: hyps.jsonl + metrics.json
├── src/
│   ├── prepare_slr104.py
│   ├── backends.py
│   ├── transcribe.py
│   ├── normalize.py
│   ├── score.py
│   ├── build_syllabus.py
│   ├── retrieve.py
│   ├── correct_lexical.py
│   ├── correct_llm.py
│   └── run_experiment.py
└── report/
```

```bash
mkdir -p configs data/raw/slr104 data/manifests syllabus/raw syllabus/index \
         cache/asr runs src report
git init && printf 'data/\ncache/\n.venv/\n.env\nruns/*/hyps.jsonl\n' > .gitignore
```

---

## 2. Data acquisition

**Download only the test set.** The Hindi-English train tarball is 7.3 GB and you are not
fine-tuning; the test tarball is 443 MB and contains the 5.18 hours you will evaluate on.
Pull the train tarball later *only* if you decide to mine a term lexicon from its transcripts
(§7 offers a cheaper alternative).

```bash
cd data/raw/slr104

# Hindi-English test set (443 MB) — the evaluation data
wget -c https://openslr.trmal.net/resources/104/Hindi-English_test.tar.gz

# Optional, only if you want in-domain term mining from train transcripts (7.3 GB)
# wget -c https://openslr.trmal.net/resources/104/Hindi-English_train.tar.gz

tar -xzf Hindi-English_test.tar.gz
find . -maxdepth 3 -type d | head -30
find . -name 'text' -o -name 'segments' -o -name 'wav.scp' | head
```

Facts worth putting in your report's dataset section, all from the OpenSLR record:
the Hindi-English and Bengali-English sets are extracted from spoken tutorials covering
technical topics, and the code-switching arises predominantly from that technical content;
the Hindi-English train and test portions are 89.86 and 5.18 hours; audio is 16 kHz,
16-bit; the Hindi-English vocabulary is 17,877 types. The distribution is Kaldi-style —
`wav.scp`, `text`, `utt2spk`, and a `segments` file carrying sentence timestamps used to cut
utterances out of the long tutorial recordings.

The 16 kHz sample rate matters: no resampling step is needed, which removes a whole class of
silent bugs.

### `src/prepare_slr104.py`

Converts the Kaldi dir into a flat JSONL manifest and, where a `segments` file exists, cuts
per-utterance WAVs once so every later run reads the same audio.

```python
"""Build data/manifests/test.jsonl from the SLR104 Kaldi-style test directory."""
import argparse, json, os, subprocess
from pathlib import Path


def read_kaldi_map(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def read_segments(path):
    segs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            utt, rec, start, end = line.strip().split()
            segs[utt] = (rec, float(start), float(end))
    return segs


def main(kaldi_dir: Path, audio_out: Path, manifest: Path):
    audio_out.mkdir(parents=True, exist_ok=True)
    text = read_kaldi_map(kaldi_dir / "text")
    wavscp = read_kaldi_map(kaldi_dir / "wav.scp")
    seg_path = kaldi_dir / "segments"
    segments = read_segments(seg_path) if seg_path.exists() else None

    rows = []
    for utt, ref in text.items():
        out_wav = audio_out / f"{utt}.wav"
        if not out_wav.exists():
            if segments:
                rec, start, end = segments[utt]
                src = wavscp[rec]
                # wav.scp may hold a pipe command; handle the plain-path case
                assert not src.endswith("|"), f"pipe entry needs manual handling: {src}"
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                     "-i", src, "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                     "-ac", "1", "-ar", "16000", str(out_wav)],
                    check=True,
                )
            else:
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                     "-i", wavscp[utt], "-ac", "1", "-ar", "16000", str(out_wav)],
                    check=True,
                )
        dur = None
        if segments and utt in segments:
            dur = round(segments[utt][2] - segments[utt][1], 3)
        rows.append({"utt_id": utt, "audio": str(out_wav), "ref": ref, "duration": dur})

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["utt_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(r["duration"] or 0 for r in rows)
    print(f"{len(rows)} utterances, {total/3600:.2f} h -> {manifest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kaldi-dir", type=Path, required=True)
    p.add_argument("--audio-out", type=Path, default=Path("data/audio/test"))
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/test.jsonl"))
    a = p.parse_args()
    main(a.kaldi_dir, a.audio_out, a.manifest)
```

```bash
python src/prepare_slr104.py --kaldi-dir data/raw/slr104/Hindi-English/test
```

### Freeze an evaluation subset

Every ablation re-runs the whole set. Do not evaluate on 5.18 hours for each of ~8 conditions
on a laptop — you will spend the semester waiting. Sample once, log the seed, use it
everywhere, and report the full test set only for the final two systems.

```python
# src/make_subset.py
import json, random, sys
rows = [json.loads(l) for l in open("data/manifests/test.jsonl", encoding="utf-8")]
random.Random(1337).shuffle(rows)
subset = rows[:800]
with open("data/manifests/subset.jsonl", "w", encoding="utf-8") as f:
    for r in sorted(subset, key=lambda x: x["utt_id"]):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(subset), "utts,", sum(r["duration"] or 0 for r in subset)/60, "min")
```

Write in the report: *"All development and ablation experiments use a fixed random subset of
800 utterances (seed 1337); final systems are additionally evaluated on the complete
Hindi-English test set."* That is defensible and honest.

---

## 3. Download the model

`faster-whisper` pulls a CTranslate2 build of large-v3 from the Hugging Face Hub on first
use. Do it explicitly so the download is a setup step, not a surprise mid-experiment.

```bash
# ~3 GB, cached in ~/.cache/huggingface
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Systran/faster-whisper-large-v3")
print("model at:", p)
PY
```

On an 8 GB Air, run large-v3 with `compute_type="int8"`. On 16 GB you can use `int8_float32`
or `float32`. Apple Silicon GPU is not used by CTranslate2 — this runs on CPU, roughly
1–3× real time for large-v3 int8. For 800 utterances of ~5 s each (~70 min of audio) budget
30–90 minutes per full decode pass. That is exactly why the cache in §4 is mandatory.

### `src/backends.py`

One interface, two implementations, so every experiment script is backend-agnostic.

```python
"""ASR backends. Both return {'text': str, 'segments': [...]}"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DecodeConfig:
    language: str = "hi"          # code-switched HI-EN; 'hi' is the better prior than 'en'
    temperature: float = 0.0
    beam_size: int = 5
    prompt: Optional[str] = None  # syllabus context goes here
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        import hashlib, json
        blob = json.dumps(self.__dict__, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


class LocalWhisper:
    name = "local-large-v3"

    def __init__(self, model_size="large-v3", compute_type="int8", device="cpu"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, cfg: DecodeConfig):
        segments, info = self.model.transcribe(
            audio_path,
            language=cfg.language,
            temperature=cfg.temperature,
            beam_size=cfg.beam_size,
            initial_prompt=cfg.prompt,
            condition_on_previous_text=False,   # utterances are independent
            vad_filter=False,                   # already sentence-segmented
        )
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        return {"text": "".join(s["text"] for s in segs).strip(),
                "segments": segs,
                "language_prob": getattr(info, "language_probability", None)}


class GroqWhisper:
    name = "groq-large-v3"

    def __init__(self, model="whisper-large-v3"):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def transcribe(self, audio_path: str, cfg: DecodeConfig):
        with open(audio_path, "rb") as fh:
            r = self.client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), fh.read()),
                model=self.model,
                language=cfg.language,
                temperature=cfg.temperature,
                prompt=cfg.prompt or "",
                response_format="verbose_json",
            )
        d = r if isinstance(r, dict) else r.model_dump()
        return {"text": d.get("text", "").strip(),
                "segments": d.get("segments", []),
                "language_prob": None}


def get_backend(name: str):
    return {"local": LocalWhisper, "groq": GroqWhisper}[name]()
```

**Note the `prompt` asymmetry.** Locally it is Whisper's `initial_prompt`, which conditions
the decoder's text context. On Groq it is the OpenAI-compatible `prompt` field, capped at
224 tokens. Keep your syllabus context under ~200 tokens so results are comparable across
backends, and record the truncation policy in the report.

---

## 4. Transcription with caching

Never call a model twice for the same (utterance, config). This is the single most important
engineering decision in the project — it turns a re-run from 45 minutes into 2 seconds and
protects you if you switch to the API's daily quota.

### `src/transcribe.py`

```python
import json, hashlib
from pathlib import Path
from tqdm import tqdm
from backends import DecodeConfig

CACHE = Path("cache/asr")


def _cache_path(backend_name, utt_id, cfg: DecodeConfig) -> Path:
    d = CACHE / backend_name / cfg.key()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{utt_id}.json"


def transcribe_manifest(backend, manifest_path, cfg_fn, out_path):
    """cfg_fn(row) -> DecodeConfig, so per-utterance prompts are supported."""
    rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    out = []
    for row in tqdm(rows, desc=backend.name):
        cfg = cfg_fn(row)
        cp = _cache_path(backend.name, row["utt_id"], cfg)
        if cp.exists():
            res = json.loads(cp.read_text(encoding="utf-8"))
        else:
            res = backend.transcribe(row["audio"], cfg)
            res["_cfg"] = cfg.__dict__
            cp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        out.append({"utt_id": row["utt_id"], "ref": row["ref"], "hyp": res["text"]})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out
```

---

## 5. Normalisation and scoring — build this **before** any improvement

This is where most student ASR projects quietly go wrong. On code-switched data, a large
fraction of apparent errors are formatting mismatches, not recognition failures. If you skip
this, your baseline WER will be inflated by 15–25 points and every later "improvement" will
be measuring your normaliser, not your method.

Concretely, on SLR104 Hindi-English you will hit:

- **Script choice for English words.** The reference writes technical terms in Latin
  (`function`, `Scilab`); Whisper sometimes emits them in Devanagari (`फंक्शन`). Same word,
  scored as an error.
- **Devanagari numerals vs. Latin digits vs. spelled-out numbers.**
- **Punctuation and casing.** Whisper adds full punctuation and capitalisation; Kaldi-style
  references have neither.
- **Nukta and other Unicode normalisation** (`क़` as one codepoint vs. `क` + U+093C).
- **Compound splitting** of technical terms (`matplotlib` → `mat plot lib`).

### `src/normalize.py`

```python
import re, unicodedata
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

PUNCT = re.compile(r"[।॥,.\?!;:\"'`\(\)\[\]\{\}—–\-_/\\|@#\$%\^&\*\+=<>~]")
WS = re.compile(r"\s+")
DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def basic_norm(s: str) -> str:
    """Standard, script-preserving normalisation. Use for the headline WER."""
    s = unicodedata.normalize("NFC", s)
    s = s.translate(DEVA_DIGITS)
    s = PUNCT.sub(" ", s)
    s = s.lower()
    s = WS.sub(" ", s).strip()
    return s


def script_invariant_norm(s: str) -> str:
    """Romanise Devanagari so HI written in either script compares equal.
    Report this as a SECONDARY metric, clearly labelled — never as the WER."""
    s = basic_norm(s)
    out = []
    for tok in s.split():
        if any("\u0900" <= ch <= "\u097F" for ch in tok):
            tok = transliterate(tok, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
            tok = re.sub(r"[^a-z0-9]", "", tok)
        out.append(tok)
    return " ".join(t for t in out if t)
```

Report **both**. The primary number is `basic_norm` WER — that is what is comparable to the
MUCS baselines and to published Whisper numbers. The script-invariant number tells you (and
your examiner) how much of the gap is orthographic rather than acoustic. That decomposition
is a genuine contribution on its own and belongs in your results chapter.

### `src/score.py`

```python
import json, re
import jiwer
from normalize import basic_norm, script_invariant_norm


def load_terms(path="syllabus/index/terms.txt"):
    return {t.strip().lower() for t in open(path, encoding="utf-8") if t.strip()}


def term_metrics(refs, hyps, terms):
    """Recall/precision restricted to syllabus terminology."""
    tp = fp = fn = 0
    for r, h in zip(refs, hyps):
        rt = [w for w in r.split() if w in terms]
        ht = [w for w in h.split() if w in terms]
        from collections import Counter
        rc, hc = Counter(rt), Counter(ht)
        for t in set(rc) | set(hc):
            tp += min(rc[t], hc[t])
            fn += max(0, rc[t] - hc[t])
            fp += max(0, hc[t] - rc[t])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"term_precision": prec, "term_recall": rec, "term_f1": f1}


def score_file(hyp_jsonl, terms_path=None):
    rows = [json.loads(l) for l in open(hyp_jsonl, encoding="utf-8")]
    refs = [basic_norm(r["ref"]) for r in rows]
    hyps = [basic_norm(r["hyp"]) for r in rows]
    m = jiwer.compute_measures(refs, hyps)
    out = {
        "n_utts": len(rows),
        "wer": m["wer"],
        "sub": m["substitutions"], "ins": m["insertions"], "del": m["deletions"],
        "cer": jiwer.cer(refs, hyps),
        "wer_script_invariant": jiwer.wer(
            [script_invariant_norm(r["ref"]) for r in rows],
            [script_invariant_norm(r["hyp"]) for r in rows]),
    }
    if terms_path:
        out.update(term_metrics(refs, hyps, load_terms(terms_path)))
    return out
```

**Also compute per-utterance WER and store it.** You need it for the regression analysis and
for the paired significance test in §10.

---

## 6. Baseline and error analysis

```bash
python src/run_experiment.py --name baseline --backend local --manifest data/manifests/subset.jsonl
python -c "import sys; sys.path.insert(0,'src'); from score import score_file; print(score_file('runs/baseline/hyps.jsonl'))"
```

Then **spend a week here**, not on code. Produce:

1. Overall WER, CER, and the sub/ins/del split.
2. WER vs. utterance duration (short utterances usually dominate the error mass).
3. The top-100 substitution pairs, ranked by frequency. Eyeball them and classify each into:
   orthographic/script, technical-term error, function-word error, hallucination.
4. The fraction of total word errors falling on syllabus terminology.

That last number is your **headroom estimate**. If only 8% of errors are on technical terms,
then even perfect term recognition caps your WER gain at 8% relative — and you should say so
in the report *before* presenting the method, which turns a modest result into a
well-predicted one.

```python
# src/analyze_errors.py — top substitution pairs
import json, collections, jiwer, sys
sys.path.insert(0, "src")
from normalize import basic_norm

rows = [json.loads(l) for l in open("runs/baseline/hyps.jsonl", encoding="utf-8")]
pairs = collections.Counter()
for r in rows:
    out = jiwer.process_words([basic_norm(r["ref"])], [basic_norm(r["hyp"])])
    ref_w, hyp_w = out.references[0], out.hypotheses[0]
    for ch in out.alignments[0]:
        if ch.type == "substitute":
            for i, j in zip(range(ch.ref_start_idx, ch.ref_end_idx),
                            range(ch.hyp_start_idx, ch.hyp_end_idx)):
                pairs[(ref_w[i], hyp_w[j])] += 1
for (a, b), c in pairs.most_common(100):
    print(f"{c:4d}  {a}  ->  {b}")
```

---

## 7. Building the syllabus resource

The SLR104 audio comes from Spoken Tutorial content on technical topics — Linux, C/C++,
Python, Scilab, LaTeX, PHP, Java, R and similar. Your "syllabus" is a set of topic documents
covering that vocabulary.

Two sources, use both:

**(a) Hand-built syllabus documents.** Write 8–15 short markdown files, one per course topic,
each listing the concepts, commands, and jargon a lecture on that topic would contain. This
mirrors what a real deployment has (a university uploads its syllabus) and is the honest
framing for your report. 200–400 words each is plenty.

```
syllabus/raw/python_basics.md
syllabus/raw/linux_shell.md
syllabus/raw/c_programming.md
syllabus/raw/latex.md
syllabus/raw/scilab.md
...
```

**(b) A term lexicon.** Extract candidate technical terms from your syllabus documents,
plus general programming vocabulary. **Do not mine the test transcripts** — that is test-set
leakage and an examiner will catch it. If you want in-domain mining, download the *train*
tarball and mine `train/text` only, and state clearly that train and test are disjoint.

### `src/build_syllabus.py`

```python
import json, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

RAW = Path("syllabus/raw")
IDX = Path("syllabus/index")
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def chunk(text, size=120, overlap=30):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i+size]) for i in range(0, max(1, len(words)-overlap), step)]


def main():
    IDX.mkdir(parents=True, exist_ok=True)
    docs = []
    for p in sorted(RAW.glob("*.md")):
        for i, c in enumerate(chunk(p.read_text(encoding="utf-8"))):
            docs.append({"doc_id": f"{p.stem}#{i}", "topic": p.stem, "text": c})

    model = SentenceTransformer(MODEL)
    emb = model.encode([d["text"] for d in docs], normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(IDX / "emb.npy", emb)
    (IDX / "docs.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")

    # term lexicon: multi-case tokens, code-ish tokens, and curated words
    terms = set()
    for p in RAW.glob("*.md"):
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_\.\-]{2,}", p.read_text(encoding="utf-8")):
            terms.add(tok.lower())
    (IDX / "terms.txt").write_text("\n".join(sorted(terms)), encoding="utf-8")
    print(len(docs), "chunks,", len(terms), "terms")


if __name__ == "__main__":
    main()
```

### `src/retrieve.py`

Chicken-and-egg problem: you need to know the topic to bias the decoding, but you need a
transcript to know the topic. Solve it with a **two-pass design** — this is the core
architecture of your system and deserves a figure in the report.

```
Pass 1: baseline decode (no prompt)
   ↓
Retrieve top-k syllabus chunks using the pass-1 transcript as the query
   ↓
Build a ≤200-token context string (topic name + key terms)
   ↓
Pass 2: re-decode with that context as initial_prompt
   ↓
Post-hoc lexical correction against retrieved terms
```

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

IDX = Path("syllabus/index")


class SyllabusRetriever:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.docs = [json.loads(l) for l in open(IDX / "docs.jsonl", encoding="utf-8")]
        self.emb = np.load(IDX / "emb.npy")
        self.model = SentenceTransformer(model_name)

    def topk(self, query: str, k: int = 3):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        idx = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in idx]

    def prompt_for(self, query: str, k: int = 3, max_words: int = 120) -> str:
        hits = self.topk(query, k)
        topics = ", ".join(sorted({h["topic"].replace("_", " ") for h, _ in hits}))
        terms, seen = [], set()
        for h, _ in hits:
            for w in h["text"].split():
                wl = w.strip(".,:;()").lower()
                if wl.isascii() and len(wl) > 3 and wl not in seen:
                    seen.add(wl); terms.append(w.strip(".,:;()"))
        body = " ".join(terms[:max_words])
        return f"विषय: {topics}. Technical terms: {body}"
```

The Hindi framing word in the prompt is deliberate — it signals the code-switched register to
the decoder rather than pushing it toward English-only output. Test the alternative
(English-only prompt) as an ablation; it is a cheap extra row in your results table.

---

## 8. Method A — decode-time prompt biasing

```python
# inside src/run_experiment.py
from backends import get_backend, DecodeConfig
from transcribe import transcribe_manifest
from retrieve import SyllabusRetriever

# Pass 1
backend = get_backend("local")
base_cfg = DecodeConfig()
pass1 = transcribe_manifest(backend, "data/manifests/subset.jsonl",
                            lambda row: base_cfg, "runs/baseline/hyps.jsonl")

# Pass 2
retr = SyllabusRetriever()
prompts = {r["utt_id"]: retr.prompt_for(r["hyp"]) for r in pass1}

def cfg_fn(row):
    return DecodeConfig(prompt=prompts[row["utt_id"]])

transcribe_manifest(backend, "data/manifests/subset.jsonl", cfg_fn,
                    "runs/prompt_bias/hyps.jsonl")
```

**Watch for prompt-induced hallucination.** Whisper will occasionally continue the prompt
instead of transcribing, especially on near-silent utterances. Guard:

```python
def guard(hyp: str, prompt: str, ref_free_baseline: str) -> str:
    """Fall back to the unbiased hypothesis if the output looks like prompt echo."""
    hw = set(hyp.lower().split()); pw = set(prompt.lower().split())
    if hw and len(hw & pw) / len(hw) > 0.6:
        return ref_free_baseline
    return hyp
```

Report how often the guard fires. It is a real finding, not an embarrassment.

---

## 9. Method B — post-hoc correction

Two variants. Build B1 first; it is deterministic, fast, and often captures most of the gain.

**B1: phonetic/lexical correction.** For each hypothesis token that is *not* in the term
lexicon, find the closest retrieved syllabus term above a similarity threshold and replace it.

```python
# src/correct_lexical.py
from rapidfuzz import process, fuzz


def correct(hyp: str, candidate_terms: list[str], threshold: int = 88) -> str:
    out = []
    for tok in hyp.split():
        if tok.lower() in {t.lower() for t in candidate_terms}:
            out.append(tok); continue
        if not tok.isascii() or len(tok) < 4:
            out.append(tok); continue
        m = process.extractOne(tok.lower(), candidate_terms, scorer=fuzz.ratio)
        out.append(m[0] if m and m[1] >= threshold else tok)
    return " ".join(out)
```

Sweep the threshold on a held-out slice of the subset (not the whole subset — otherwise you
are tuning on your test data). Report the chosen value and the sweep curve.

**B2: LLM correction with retrieved context.** Constrain it hard; an unconstrained LLM will
paraphrase and destroy your WER.

```python
# src/correct_llm.py
SYSTEM = (
    "You correct ASR transcripts of Hindi-English code-switched technical lectures. "
    "Fix ONLY misrecognised technical terms, using the reference term list. "
    "Rules: preserve every other word exactly, including disfluencies and grammar. "
    "Preserve the original script of each word. Do not translate, punctuate, or reorder. "
    "Do not add or remove words. Output the corrected transcript and nothing else."
)

def build_user(hyp: str, terms: list[str]) -> str:
    return f"Reference terms: {', '.join(terms[:40])}\n\nTranscript: {hyp}"
```

Add a hard post-check: if the corrected output differs from the input by more than ~20% of
tokens, discard the correction and keep the original. Log the discard rate.

---

## 10. Experiment matrix and significance

| ID | System | Purpose |
|----|--------|---------|
| S0 | large-v3, no prompt | baseline |
| S1 | S0 + generic prompt ("technical lecture") | controls for *any* prompt vs. *retrieved* prompt |
| S2 | S0 + **random** syllabus doc as prompt | controls for retrieval quality — critical |
| S3 | S0 + retrieved prompt (k=3) | Method A |
| S4 | S3 + lexical correction | A + B1 |
| S5 | S3 + LLM correction | A + B2 |
| S6 | S0 + oracle syllabus doc | upper bound on retrieval |

S1 and S2 are what separate a research project from a demo. Without them you cannot claim the
gain comes from *syllabus grounding* rather than from prompting at all.

**Paired bootstrap significance test** (the standard in ASR, and easy):

```python
# src/bootstrap.py
import numpy as np

def paired_bootstrap(err_a, len_a, err_b, len_b, n=10000, seed=0):
    """err_*: per-utterance edit counts; len_*: per-utterance ref lengths."""
    rng = np.random.default_rng(seed)
    ea, la, eb, lb = map(np.asarray, (err_a, len_a, err_b, len_b))
    n_utt = len(ea)
    wins = 0
    for _ in range(n):
        idx = rng.integers(0, n_utt, n_utt)
        wer_a = ea[idx].sum() / la[idx].sum()
        wer_b = eb[idx].sum() / lb[idx].sum()
        wins += wer_b < wer_a
    return wins / n   # p ≈ 1 - this, for "B better than A"
```

Report: WER, CER, script-invariant WER, term-F1, % utterances improved, % regressed, and the
bootstrap p-value for each system against S0.

---

## 11. Timeline

| Week | Deliverable |
|------|-------------|
| 1 | Env, data downloaded, manifest built, subset frozen |
| 2 | Model running locally, cache working, baseline decoded |
| 3 | Normalisation + scoring harness, baseline WER/CER locked |
| 4 | Error analysis, headroom estimate, top-100 substitution table |
| 5 | Syllabus documents written, index + term lexicon built |
| 6 | Method A (two-pass prompt biasing) + guard |
| 7 | Methods B1 and B2, threshold sweep |
| 8 | Full ablation matrix S0–S6 on subset |
| 9 | Final two systems on full 5.18 h test set, bootstrap tests |
| 10 | Figures, results chapter, hand-off of transcripts to your note-generation stage |

---

## 12. Things that will go wrong (pre-empt them)

- **`wav.scp` contains pipe commands** in some Kaldi distributions (`sox ... |`). The prep
  script asserts on this; if it fires, strip the pipe and call the tool directly.
- **Baseline WER looks catastrophic (>60%).** Almost always normalisation, not the model.
  Print 20 ref/hyp pairs side by side before believing any number.
- **`language="hi"` vs `language="en"` vs auto-detect** changes WER substantially on
  code-switched audio. Run all three on 100 utterances in week 2 and fix the choice; report it.
- **Prompt biasing makes WER worse.** Common and publishable. Check whether term-F1 rose
  while WER fell — that dissociation is the interesting result.
- **Tuning on the test set.** Hold out 200 of the 800 subset utterances for threshold and
  k-value selection. Say so in the report.
- **Comparing across backends.** Never mix local and Groq hypotheses inside one results
  table; Groq's inference stack is not numerically identical to the reference model.
