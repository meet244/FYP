# Syllabus-Grounded Decoding for Code-Switched Lecture ASR
## Complete implementation plan — laptop-scale, reproducible, paper-ready

**Author:** (you)
**Target artifact:** one conference/journal-style short paper + a reproducible repo
**Compute budget:** single MacBook (Apple Silicon), main experiment ≤ ~6 minutes of wall-clock decoding
**Status of this document:** every factual claim below is marked either **[verified]** (checked against a primary source) or **[assumption]** (must be confirmed in Step 0 before you trust it).

---

# PART 0 — CONTEXT: WHAT WE ARE ACTUALLY DOING AND WHY

## 0.1 The one-sentence version

We test whether giving Whisper the **course syllabus** as decoder context reduces word error rate on Hindi–English code-mixed technical lectures — and, more importantly, we build and evaluate a method that makes that context *actually help overall WER* rather than only helping rare words while hurting everything else.

## 0.2 The naive hypothesis, and why it is not enough

Your original hypothesis:

> Adding the syllabus gives the model context about what will be taught, so it produces better output.

This is intuitive and partly right, but as stated it is not yet a publishable claim, for three reasons.

**Reason 1 — the mechanism is narrow.** Whisper does not "understand" a syllabus. It exposes an `initial_prompt` parameter that is prepended to the decoder's input as *previous-context tokens*. It biases decoding toward preferred vocabulary, spellings and styles. Only roughly the last 224 tokens are consumed, and tokens near the end of the prompt exert more influence than those at the start. So a full syllabus cannot even fit; something must select and compress it. **[verified]**

**Reason 2 — the naive version is known to backfire.** Published work on rare-word biasing in Whisper reports that passing a biasing list via the prompt reduced rare-word WER from 23.7% to 18.0% and out-of-vocabulary WER from 60% to 37.1%, but *increased* WER on non-biased words on every dataset tested, so overall WER got **worse on 6 of 11 datasets**. The stated cause: Whisper's prompt slot expects a transcript of previous speech, not a keyword list, so a list is a distribution mismatch. **[verified]**

**Reason 3 — without controls, a positive result is uninterpretable.** If you only run "no prompt" vs "syllabus prompt" and WER drops, a reviewer immediately asks: did it drop because of *the syllabus content*, or because *any* text in the prompt nudged Whisper into the right script and register? You cannot answer that without a mismatched-syllabus control.

## 0.3 The reframed contribution (this is what makes it a paper)

**Claim:** Syllabus context helps code-switched lecture ASR **only if it is rendered and selected correctly.** We show:

1. Naive keyword-list prompting reproduces the known failure — it improves technical-term recognition but degrades overall WER. *(negative result, replicated in a new language setting)*
2. Rendering the same syllabus content as **fluent code-mixed prose in the format Whisper's prompt slot expects** removes most of that degradation.
3. **Retrieval-selecting** the syllabus unit relevant to *this* utterance (via a cheap first-pass transcript) beats using the whole syllabus, because it spends the 224-token budget on the right content.
4. A **confidence guard** that falls back to the unprompted hypothesis when prompting destabilises decoding converts the remaining regressions into a net overall WER reduction.
5. A **mismatched-syllabus control** shows the gain is content-specific, not style-priming — this is the load-bearing evidence for your hypothesis.

Working name for the method: **SGCD — Syllabus-Grounded Contextual Decoding.**

Note the honest framing: contributions 1–5 are a *method + rigorous evaluation* paper at laptop scale, not a new model. That is entirely appropriate for an undergraduate final-year research output, and it is far more defensible than claiming a new architecture.

## 0.4 What "success" means (pre-register this before you run anything)

Write these down and commit them to git **before** the first main run. Pre-registration is what separates a result from p-hacking.

| ID | Hypothesis | Pass criterion |
|----|-----------|----------------|
| H1 | SGCD reduces overall WER vs no-prompt baseline | ΔWER < 0 with 95% bootstrap CI excluding 0 |
| H2 | Syllabus context improves technical-term recognition | Keyword-WER (K-WER) drops ≥ 15% relative |
| H3 | The gain is content-specific, not style-priming | WER(matched syllabus) < WER(mismatched syllabus), CI excludes 0 |
| H4 | Prose rendering beats keyword-list rendering | WER(C3) < WER(C2) |
| H5 | Retrieval beats whole-syllabus prompting | WER(C4) < WER(C3) |

**If H1 fails but H2 and H3 hold, you still have a paper.** The result "syllabus context reliably improves domain-term recognition in Hindi–English lecture ASR but does not improve overall WER, and here is exactly why" is a genuine, reportable finding. Decide now that you will report it either way. Do not let the plan depend on a positive result.

---

# PART 1 — DATA

## 1.1 Dataset choice: MUCS 2021 Subtask-2, Hindi–English (OpenSLR SLR104)

**Why this dataset and not something else.** It is the only public, properly transcribed Hindi–English code-mixed corpus whose audio is *literally technical lectures*, which means a syllabus is a natural, non-contrived context source.

Verified facts about it:

- It is the Subtask-2 dataset of MUCS 2021 (Multilingual and Code-Switching ASR Challenge for Low Resource Indian Languages), hosted at OpenSLR as SLR104 under **CC BY-SA 4.0**. **[verified]**
- The Hindi–English and Bengali–English portions are **extracted from spoken tutorials** covering a range of technical topics, and the code-switching arises predominantly from the technical content of the lectures. **[verified]**
- Audio is **16 kHz, 16-bit**. **[verified]**
- Hindi–English: ~89.85 h train, ~5.18 h test, ~6.24 h blind test. **[verified]**
- Download sizes: `Hindi-English_train.tar.gz` ≈ 7.3 GB, **`Hindi-English_test.tar.gz` ≈ 443 MB**. Mirror links on OpenSLR are password-free. **[verified]** → *We only need the test tarball.*
- The release ships a **`segments` file giving sentence-level timestamps**, used to cut segments from the long audio files and align them with the `text` file. This is Kaldi-style data preparation. **[verified]**
- Speaker IDs were not available, but **the organisers do retain information about which underlying tutorial each sentence came from**, and treated each tutorial as a distinct speaker. **[verified — and this is the single most important fact for us]**
- Transcriptions include mathematical symbols and technical content; the transcripts were the narrator scripts for the tutorial videos. **[verified]**
- Reference text is dual-script: Hindi in Devanagari, English in Latin. **[verified from challenge description]**

**Why the tutorial-grouping fact matters:** it means each test utterance can be traced to a *lecture*, and each lecture belongs to a *course/tutorial series*. That gives us exactly the unit a syllabus attaches to. Without it, "syllabus" would be meaningless.

**Zero-shot only.** We never train. The reported 33.9% test/train overlap in this corpus is irrelevant to us because we never touch the training split. State this explicitly in the paper — it pre-empts a reviewer question.

## 1.2 Download

```bash
mkdir -p ~/sgcd/data && cd ~/sgcd/data
# Get the exact current mirror URL from https://www.openslr.org/104/ (use a Mirror1 link, password-free)
curl -L -O "<PASTE_MIRROR1_URL_FOR_Hindi-English_test.tar.gz>"
tar -xzf Hindi-English_test.tar.gz
```

Only the **test** tarball (~443 MB). Do not download the 7.3 GB train tarball; you have no use for it.

## 1.3 Step 0 — DISCOVERY (do this before writing anything else)

**[assumption]** I have not personally opened this tarball, so I will not tell you its exact directory layout. Kaldi-style releases of this kind normally contain `transcripts/text`, `transcripts/wav.scp`, `transcripts/segments`, `transcripts/utt2spk` and an `Audio/` directory, but **verify, don't assume.** Run this first:

```bash
cd ~/sgcd/data
find . -maxdepth 4 -type d | head -50
find . -maxdepth 4 -type f ! -name "*.wav" | head -50
echo "--- text ---";      find . -name "text"     | head -3 | xargs -I{} sh -c 'head -5 "{}"'
echo "--- segments ---";  find . -name "segments" | head -3 | xargs -I{} sh -c 'head -5 "{}"'
echo "--- wav.scp ---";   find . -name "wav.scp"  | head -3 | xargs -I{} sh -c 'head -5 "{}"'
echo "--- wav count ---"; find . -name "*.wav" | wc -l
```

**What you are looking for, and what to do with it:**

| Question | Why it matters | If the answer is "no" |
|---|---|---|
| Is there a `segments` file? | Tells you whether audio is long recordings needing slicing, or already-cut clips | If absent, each `.wav` is one utterance; skip slicing |
| Do utterance IDs share a prefix per recording? | This is your lecture grouping | Fall back to grouping by source `.wav` filename |
| Do recording IDs encode a topic/series name? | Lets you map lecture → course for free | Use §2.2 Tier-B syllabus generation instead |
| Is reference text dual-script (Devanagari + Latin)? | Determines normalisation and the script-fidelity metric | Adjust normaliser accordingly |
| How many utterances, and what duration distribution? | Sets your sampling budget | — |

**Do not proceed to Part 3 until you have pasted the real head of `text` and `segments` into your notes.** Every path in the code below is written to be adjusted from what you actually find.

## 1.4 Evaluation subset construction

Fixed rules, applied once, then frozen:

1. **Duration filter: keep utterances of 2.0 s ≤ dur ≤ 28.0 s.** Rationale: Whisper processes a 30-second window; keeping every clip inside one window means `initial_prompt` applies to the entire utterance with identical semantics across all conditions. This removes a confound rather than creating one. State it in the paper's Limitations.
2. **Reference filter: drop utterances with fewer than 4 reference words** — per-utterance WER on 1–3 word references is extremely high-variance and will distort your plots.
3. **Split by lecture, not by utterance:**
   - **DEV** = utterances from ~30% of lectures. Used *only* for choosing decoding config, prompt phrasing, and guard thresholds.
   - **TEST** = the remaining ~70% of lectures. Touched **once**, at the end, for the numbers that go in the paper.
   This is the discipline that makes H1 credible. Every knob you turn must be turned on DEV.
4. **Stratified sample:** from TEST, sample N utterances *proportionally per lecture*, so no single lecture dominates. Use a fixed seed (`SEED = 1337`).
5. **Sizes:** DEV N=60, TEST N=150 for the main run. See the runtime table in §3.8 to trade N against wall clock.

---

# PART 2 — THE SYLLABUS, AND THE LEAKAGE PROBLEM

## 2.1 The failure mode that would kill your paper

If your "syllabus" is built from the reference transcripts of the utterances you evaluate on, you have leaked the answer into the prompt. WER will drop dramatically, the result will be meaningless, and any competent reviewer or examiner will spot it in thirty seconds.

**Hard rule, enforced in code:** no syllabus unit may contain text derived from the reference transcript of any utterance in the evaluation set. Every syllabus unit carries a `provenance` field, and the runner **asserts** that provenance is not `"oracle"` for any scored condition except the explicitly-labelled topline.

## 2.2 Three tiers of syllabus, in order of preference

**Tier A — real published syllabus (best; use if available).**
The corpus is drawn from spoken tutorials on technical topics **[verified]**. Tutorial-series projects of this kind publish per-series outlines, learning objectives and keyword lists on their own websites. If Step 0 reveals recognisable series names in the recording IDs, go to the corresponding public course page and copy the outline. This is fully external to the audio, so leakage is structurally impossible. Record the URL in the provenance field. **[assumption — depends on what Step 0 reveals; verify before relying on it.]**

**Tier B — topic-conditioned generated syllabus (reliable fallback).**
Give an LLM **only the series/lecture title** (e.g. "Linux — file permissions") and ask it to produce a realistic course syllabus: 6–12 units, each with a title, a 40–60 word description written in Hindi–English code-mixed register, and 6–10 technical keywords. It never sees any audio or any reference transcript. Provenance: `"generated-from-title"`.

This is defensible and worth saying out loud in the paper: **a real deployment has the syllabus before the lecture is recorded, so generating it from the course outline is exactly the realistic condition, not a shortcut.**

**Tier C — oracle (topline only, never a headline number).**
Build the unit prose from the reference transcripts of *other* lectures in the same series, never the evaluated one. This estimates the ceiling if your syllabus were perfectly matched to the delivery. Label it **Topline (oracle)** in every table. Provenance: `"oracle"`.

## 2.3 Syllabus file format

One JSON per course, in `~/sgcd/syllabi/`:

```json
{
  "course_id": "linux",
  "title": "Linux Operating System",
  "provenance": "generated-from-title",
  "source_url": null,
  "units": [
    {
      "unit_id": "linux-u03",
      "title": "File permissions and ownership",
      "prose": "इस tutorial में हम Linux के file permissions समझेंगे। हर file के लिए read, write और execute permission होती है, और ये owner, group तथा others के लिए अलग-अलग set की जाती हैं। chmod command से हम permission बदल सकते हैं और ls -l से current permission देख सकते हैं।",
      "keywords": ["chmod", "permission", "execute", "owner", "group", "directory", "ls", "read", "write"]
    }
  ]
}
```

**Critical detail — how `prose` must be written.** This is the core of the method and the reason C3 beats C2.

Whisper's prompt slot is trained to hold *the transcript of the preceding speech*. So the prose must look like preceding speech from the same lecture:

- Written as **flowing narration**, not bullets, not a comma-separated word list.
- In the **same dual-script convention as the references** — Hindi function words in Devanagari, English technical terms in Latin. This is what teaches Whisper the output *format* as well as the vocabulary.
- Same register as a tutorial narrator (second person, present/future tense, "हम … देखेंगे").
- **Densest, most distinctive technical terms placed at the END of the prose**, because later prompt tokens carry more influence. **[verified mechanism]**
- Target 45–70 tokens per unit so that 2–3 units fit inside the 224-token budget.

Write one unit by hand first and eyeball it. Then generate the rest to match.

## 2.4 Mismatched-syllabus control (H3)

Build a `mismatched` pool by pairing each course with a **different** course's syllabus (e.g. Linux lectures get the LaTeX syllabus). Same length, same prose style, same rendering pipeline — only the content is wrong. If C4 (matched) beats C5 (mismatched) with a CI excluding zero, you have demonstrated the effect is **semantic**, not stylistic. Without this condition, your central claim is unfalsifiable.

---

# PART 3 — METHOD AND IMPLEMENTATION

## 3.1 Environment

```bash
cd ~/sgcd
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install mlx-whisper jiwer soundfile numpy pandas scikit-learn scipy regex tiktoken
```

- **`mlx-whisper`** — Apple-Silicon-native Whisper (MLX framework). This is what makes the "few minutes on a MacBook" budget realistic. If it gives you trouble, `faster-whisper` (CPU, int8) is the fallback; both expose `initial_prompt`.
- **`jiwer`** — WER/CER plus word-level alignments (needed for K-WER).
- **`scikit-learn`** — TF-IDF retrieval. Deliberately *not* a neural embedding model: with only 6–12 syllabus units per course, character-n-gram TF-IDF is instant, needs no download, and handles mixed Devanagari/Latin natively. Using a sentence-transformer here would add a model download and minutes of runtime for no measurable benefit — and you should say so in the paper.

Models to pull (cached on first use):

| Role | HF repo | Notes |
|---|---|---|
| Primary | `mlx-community/whisper-large-v3-turbo` | Best accuracy per second on Apple Silicon |
| Secondary | `mlx-community/whisper-small-mlx` | Second model size, for the generalisation row |
| Smoke test | `mlx-community/whisper-tiny-mlx` | Pipeline debugging only, never reported |

Reporting two model sizes is cheap and materially strengthens the paper: it shows the effect is not an artifact of one checkpoint.

## 3.2 Repository layout

```
~/sgcd/
  data/                     # extracted SLR104 test set
  syllabi/                  # *.json, one per course
  out/
    manifest.jsonl          # frozen eval set
    hyps/<model>__<cond>.jsonl   # cached decodes
    scores.csv
    tables/
  src/
    build_manifest.py
    prompts.py
    retrieve.py
    decode.py
    normalize.py
    score.py
    stats.py
  RUNLOG.md                 # append-only: date, git hash, command, headline numbers
```

`RUNLOG.md` is not optional. When you write the paper three weeks from now you will not remember which run produced which number.

## 3.3 `src/build_manifest.py` — freeze the evaluation set

```python
"""Build the frozen evaluation manifest from the SLR104 Kaldi-style test dir.
ADJUST THE PATHS IN load_kaldi() TO MATCH WHAT STEP 0 REVEALED."""
import json, random, re, pathlib, collections
import soundfile as sf

DATA = pathlib.Path("~/sgcd/data").expanduser()
OUT  = pathlib.Path("~/sgcd/out").expanduser(); OUT.mkdir(parents=True, exist_ok=True)
SEED = 1337
MIN_DUR, MAX_DUR, MIN_WORDS = 2.0, 28.0, 4
DEV_LECTURE_FRAC = 0.30

def find(name):
    hits = list(DATA.rglob(name))
    if not hits: raise FileNotFoundError(f"{name} not found under {DATA} — re-check Step 0")
    return hits[0]

def load_kaldi():
    text = {}
    for line in find("text").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        utt, _, tr = line.partition(" ")
        text[utt] = tr.strip()

    wavscp = {}
    for line in find("wav.scp").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        rec, _, rest = line.partition(" ")
        m = re.search(r"\S+\.wav", rest)          # handles both plain paths and pipe commands
        if m: wavscp[rec] = m.group(0)

    segs = {}
    segf = list(DATA.rglob("segments"))
    if segf:
        for line in segf[0].read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            utt, rec, s, e = line.split()
            segs[utt] = (rec, float(s), float(e))
    else:
        # No segments file: every wav IS one utterance
        for utt in text:
            if utt in wavscp: segs[utt] = (utt, 0.0, None)
    return text, wavscp, segs

def resolve_wav(p):
    p = pathlib.Path(p)
    if p.exists(): return str(p)
    hits = list(DATA.rglob(p.name))
    if not hits: raise FileNotFoundError(f"wav {p.name} not found")
    return str(hits[0])

def main():
    text, wavscp, segs = load_kaldi()
    rows = []
    for utt, tr in text.items():
        if utt not in segs: continue
        rec, s, e = segs[utt]
        if rec not in wavscp: continue
        wav = resolve_wav(wavscp[rec])
        if e is None:
            info = sf.info(wav); e = info.frames / info.samplerate
        dur = e - s
        if not (MIN_DUR <= dur <= MAX_DUR): continue
        if len(tr.split()) < MIN_WORDS: continue
        rows.append(dict(utt_id=utt, lecture_id=rec, wav=wav,
                         start=s, end=e, dur=round(dur, 2), ref=tr))

    lectures = sorted({r["lecture_id"] for r in rows})
    rng = random.Random(SEED); rng.shuffle(lectures)
    n_dev = max(1, int(len(lectures) * DEV_LECTURE_FRAC))
    dev = set(lectures[:n_dev])
    for r in rows: r["split"] = "dev" if r["lecture_id"] in dev else "test"

    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = collections.Counter(r["split"] for r in rows)
    print(f"utterances={len(rows)}  lectures={len(lectures)}  {dict(by)}")
    print(f"total audio = {sum(r['dur'] for r in rows)/3600:.2f} h")
    print("\nSample lecture IDs (use these to name your courses):")
    for l in lectures[:15]: print("  ", l)

if __name__ == "__main__":
    main()
```

Run it, then **read the printed lecture IDs** — that list is what you use to decide how many courses you need syllabi for.

## 3.4 `src/prompts.py` — rendering the syllabus into the prompt slot

This file *is* the method. Everything else is plumbing.

```python
"""Render syllabus content into Whisper initial_prompt strings.

Design constraints (grounded in how Whisper's prompt slot works):
  - only the final ~224 tokens are consumed  -> we cap at 200 and LEFT-truncate
  - later tokens carry more influence        -> most distinctive content goes LAST
  - the slot expects previous-segment transcript, not a word list
                                             -> C3/C4 use fluent prose (the hypothesis)
"""
import tiktoken

MAX_PROMPT_TOKENS = 200
_ENC = tiktoken.get_encoding("cl100k_base")   # proxy tokenizer; ±10% vs Whisper's BPE, fine as a cap

def _truncate_keep_end(text: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    ids = _ENC.encode(text)
    if len(ids) <= max_tokens: return text
    return _ENC.decode(ids[-max_tokens:])      # keep the END: highest-influence tokens

# ---- C1: generic style control (NO course content) ----
GENERIC_PROMPT = ("यह एक technical tutorial है जिसमें Hindi और English दोनों "
                  "का प्रयोग होता है। आइए अब आगे बढ़ते हैं।")

# ---- C2: naive keyword list (the known-weak baseline we replicate) ----
def keyword_prompt(units):
    kws = []
    for u in units:
        for k in u["keywords"]:
            if k not in kws: kws.append(k)
    return _truncate_keep_end(", ".join(kws))

# ---- C3 / C4: prose rendering (the proposed rendering) ----
def prose_prompt(units):
    """units is ordered LEAST -> MOST relevant; most relevant lands at the end."""
    return _truncate_keep_end(" ".join(u["prose"].strip() for u in units))

def build(condition, course, retrieved_units=None):
    if condition == "C0": return None
    if condition == "C1": return GENERIC_PROMPT
    if condition == "C2": return keyword_prompt(course["units"])
    if condition == "C3": return prose_prompt(course["units"])          # whole syllabus
    if condition in ("C4", "C5", "C6", "C7"):
        return prose_prompt(retrieved_units)                            # retrieved subset
    raise ValueError(condition)
```

Two things a reviewer will like here: the left-truncation is *derived from* the documented behaviour of the prompt slot rather than arbitrary, and C2 vs C3 isolates rendering format while holding content constant.

## 3.5 `src/retrieve.py` — picking the right syllabus unit

```python
"""Two-pass retrieval: first-pass hypothesis -> most relevant syllabus units.
Character n-gram TF-IDF: no model download, handles Devanagari+Latin, ~1 ms/query."""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SyllabusIndex:
    def __init__(self, units, k=2):
        self.units, self.k = units, k
        docs = [f"{u['title']} {u['prose']} {' '.join(u['keywords'])}" for u in units]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.M = self.vec.fit_transform(docs)

    def query(self, text):
        if not text or not text.strip():
            return self.units[: self.k]
        sims = (self.M @ self.vec.transform([text]).T).toarray().ravel()
        top = np.argsort(-sims)[: self.k]
        # return ASCENDING relevance so the best unit is rendered LAST in the prompt
        return [self.units[i] for i in reversed(top)]
```

Why `char_wb` 3–5 grams rather than word tokens: code-mixed text has inconsistent spellings and transliteration, and first-pass hypotheses are noisy. Character n-grams degrade gracefully where word matching fails.

## 3.6 `src/decode.py` — the runner

```python
"""Decode every condition, cache every hypothesis. Re-runs are near-free."""
import json, pathlib, time, hashlib
import numpy as np, soundfile as sf
import mlx_whisper
from prompts import build
from retrieve import SyllabusIndex

OUT = pathlib.Path("~/sgcd/out").expanduser()
HYP = OUT / "hyps"; HYP.mkdir(parents=True, exist_ok=True)
SYL = pathlib.Path("~/sgcd/syllabi").expanduser()

MODELS = {"turbo": "mlx-community/whisper-large-v3-turbo",
          "small": "mlx-community/whisper-small-mlx",
          "tiny":  "mlx-community/whisper-tiny-mlx"}

DECODE = dict(language="hi",                  # tune on DEV; see §3.7
              task="transcribe",
              temperature=0.0,                # scalar -> disables fallback -> deterministic + fast
              condition_on_previous_text=False,  # prompt is the ONLY context: no cross-utt drift
              word_timestamps=False)

def load_manifest(split):
    return [json.loads(l) for l in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(l)["split"] == split]

def load_audio(row):
    info = sf.info(row["wav"])
    sr = info.samplerate
    a, _ = sf.read(row["wav"], start=int(row["start"] * sr), stop=int(row["end"] * sr), dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    assert sr == 16000, f"expected 16 kHz, got {sr}"   # corpus is 16 kHz [verified]
    return a

# ---- lecture -> course mapping. EDIT after Step 0 tells you the real ID format. ----
def course_of(lecture_id):
    return lecture_id.split("_")[0].lower()

_cache = {}
def get_course(cid, mismatched=False):
    key = (cid, mismatched)
    if key in _cache: return _cache[key]
    path = SYL / f"{cid}.json"
    course = json.loads(path.read_text(encoding="utf-8"))
    if mismatched:
        others = sorted(p for p in SYL.glob("*.json") if p.stem != cid)
        course = json.loads(others[hash(cid) % len(others)].read_text(encoding="utf-8"))
    _cache[key] = course
    return course

def run(model_key, condition, split="test", first_pass=None, oracle=False):
    tag = f"{model_key}__{condition}__{split}"
    path = HYP / f"{tag}.jsonl"
    if path.exists():
        print(f"[cached] {tag}"); return [json.loads(l) for l in path.read_text().splitlines()]

    rows, res, t0 = load_manifest(split), [], time.time()
    for r in rows:
        cid = course_of(r["lecture_id"])
        course = get_course(cid, mismatched=(condition == "C5"))
        units = None
        if condition in ("C4", "C5", "C6", "C7"):
            idx = SyllabusIndex(course["units"], k=2)
            # C6 = topline: retrieve with the REFERENCE (upper bound on retrieval quality)
            q = r["ref"] if (condition == "C6") else first_pass[r["utt_id"]]
            units = idx.query(q)
        prompt = build(condition if condition != "C7" else "C4", course, units)

        o = mlx_whisper.transcribe(load_audio(r), path_or_hf_repo=MODELS[model_key],
                                   initial_prompt=prompt, **DECODE)
        seg = o.get("segments") or [{}]
        res.append(dict(utt_id=r["utt_id"], ref=r["ref"], hyp=o["text"].strip(),
                        avg_logprob=seg[0].get("avg_logprob"),
                        compression_ratio=seg[0].get("compression_ratio"),
                        prompt_tokens=len(prompt.split()) if prompt else 0))
    el = time.time() - t0
    audio_s = sum(r["dur"] for r in rows)
    with path.open("w", encoding="utf-8") as f:
        for x in res: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[done] {tag}  {el:.1f}s  RTF={el/audio_s:.3f}  ({audio_s/el:.1f}x realtime)")
    return res

# ---- C7: confidence guard. Thresholds MUST be fitted on DEV, then frozen. ----
def apply_guard(base, prompted, d_logprob=0.15, max_cr=2.4, len_ratio=2.0):
    b = {x["utt_id"]: x for x in base}
    out, n_fb = [], 0
    for p in prompted:
        bb = p["utt_id"] in b and b[p["utt_id"]]
        bad = False
        if bb and p["avg_logprob"] is not None and bb["avg_logprob"] is not None:
            bad |= (p["avg_logprob"] < bb["avg_logprob"] - d_logprob)
        if p["compression_ratio"] is not None:
            bad |= (p["compression_ratio"] > max_cr)
        if bb and len(bb["hyp"].split()):
            bad |= (len(p["hyp"].split()) > len_ratio * len(bb["hyp"].split()))
        if bad and bb: out.append({**p, "hyp": bb["hyp"], "fallback": True}); n_fb += 1
        else:          out.append({**p, "fallback": False})
    print(f"[guard] fell back on {n_fb}/{len(prompted)} ({100*n_fb/len(prompted):.1f}%)")
    return out

if __name__ == "__main__":
    import sys
    m, split = (sys.argv[1] if len(sys.argv) > 1 else "turbo"), (sys.argv[2] if len(sys.argv) > 2 else "test")
    c0 = run(m, "C0", split)
    fp = {x["utt_id"]: x["hyp"] for x in c0}          # first pass feeds retrieval
    for c in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        run(m, c, split, first_pass=fp)
    c4 = [json.loads(l) for l in (HYP / f"{m}__C4__{split}.jsonl").read_text().splitlines()]
    guarded = apply_guard(c0, c4)
    with (HYP / f"{m}__C7__{split}.jsonl").open("w", encoding="utf-8") as f:
        for x in guarded: f.write(json.dumps(x, ensure_ascii=False) + "\n")
```

Three design decisions worth defending in the paper:

- `condition_on_previous_text=False` — otherwise Whisper's own rolling context contaminates the prompt slot and conditions become non-comparable.
- `temperature=0.0` as a scalar — disables Whisper's temperature-fallback loop, giving deterministic output and stable timing. It also means repetition loops are *not* silently repaired, which is what lets you honestly measure prompt-induced hallucination.
- Caching by `(model, condition, split)` — you will re-run scoring dozens of times while writing; decoding should happen once.

## 3.7 Conditions table

| ID | Prompt | Purpose |
|---|---|---|
| **C0** | none | Baseline |
| **C1** | generic code-mixed sentence, no course content | **Style control** — separates format priming from content |
| **C2** | syllabus keywords, comma-separated | Naive method; replicates the known failure mode |
| **C3** | whole-syllabus prose | Tests the *rendering* hypothesis (H4) |
| **C4** | retrieved 2 units, prose (**SGCD**) | The proposed method (H5) |
| **C5** | retrieved from a **different course** | **Content-specificity control** (H3) |
| **C6** | retrieved using the reference | Topline (oracle retrieval) — clearly labelled |
| **C7** | C4 + confidence guard | Full system (H1) |

Cost note: C4, C5, C6, C7 all reuse C0's hypotheses for the first pass, so the two-pass method costs **one** extra decode, not two.

**DEV-only tuning list** (freeze before touching TEST): `language="hi"` vs `language=None`; `k` ∈ {1, 2, 3} retrieved units; prompt token cap ∈ {120, 200}; guard thresholds. Record every DEV number in `RUNLOG.md` so you can state in the paper exactly what was tuned and where.

## 3.8 Runtime budget

Estimates assume Apple Silicon with `mlx-whisper`; measure your own RTF in the smoke test and adjust.

| Stage | Model | N utts | Audio | Est. wall clock |
|---|---|---|---|---|
| Smoke test | tiny | 10 | ~1.3 min | < 20 s |
| DEV tuning sweep | small | 60 | ~8 min | ~2–3 min total |
| **Main run, 7 conditions** | **turbo** | **150** | **~20 min** | **~5–7 min** |
| Generalisation row, 7 conditions | small | 150 | ~20 min | ~2–3 min |

To stay strictly inside "a few minutes": run the main table at **N=120 with turbo**, and use `small` for any exploratory sweep. Everything is cached, so an interrupted run resumes free.

---

# PART 4 — SCORING

## 4.1 `src/normalize.py`

Normalisation choices change WER by several points, so they must be fixed once, documented in the paper, and applied identically to every condition.

```python
import re, unicodedata

_PUNCT = r"""!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥“”‘’—–…"""
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_DEV_DIGITS)
    s = s.lower()                       # affects Latin only
    s = re.sub(f"[{_PUNCT}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

DEV_RE = re.compile(r"[\u0900-\u097F]")
LAT_RE = re.compile(r"[a-z]")
def script_of(w):
    if DEV_RE.search(w): return "dev"
    if LAT_RE.search(w): return "lat"
    return "other"
```

**Manual check, non-negotiable:** print 20 (ref, normalised-ref, hyp, normalised-hyp) quadruples and read them. You are looking for numerals ("5" vs "पाँच"), spelled-out English words appearing in Devanagari, and stray tokens. If numbers are a large error source, add an explicit number-word mapping and say so; do not silently drop utterances.

## 4.2 `src/score.py`

```python
import json, pathlib, csv
import jiwer
from normalize import normalize, script_of

OUT = pathlib.Path("~/sgcd/out").expanduser(); HYP = OUT / "hyps"

def keyword_set(course_dir="~/sgcd/syllabi"):
    ks = set()
    for p in pathlib.Path(course_dir).expanduser().glob("*.json"):
        for u in json.loads(p.read_text(encoding="utf-8"))["units"]:
            ks |= {normalize(k) for k in u["keywords"]}
    return ks

def score(path, kws):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines()]
    refs = [normalize(r["ref"]) for r in rows]
    hyps = [normalize(r["hyp"]) for r in rows]

    o = jiwer.process_words(refs, hyps)
    wer = o.wer
    cer = jiwer.process_characters(refs, hyps).cer

    # K-WER / U-WER split, and script fidelity, from the word alignments
    k_err = k_tot = u_err = u_tot = lat_ok = lat_tot = 0
    for ri, (rw, hw, chunks) in enumerate(zip(o.references, o.hypotheses, o.alignments)):
        for c in chunks:
            for j in range(c.ref_end_idx - c.ref_start_idx) if c.type != "insert" else []:
                w = rw[c.ref_start_idx + j]
                err = (c.type != "equal")
                if w in kws: k_tot += 1; k_err += err
                else:        u_tot += 1; u_err += err
                if script_of(w) == "lat":
                    lat_tot += 1
                    if c.type == "equal": lat_ok += 1

    per_utt = [jiwer.wer(a, b) if a.strip() else None for a, b in zip(refs, hyps)]
    return dict(n=len(rows), wer=wer, cer=cer,
                k_wer=(k_err / k_tot if k_tot else None), k_n=k_tot,
                u_wer=(u_err / u_tot if u_tot else None),
                script_fidelity=(lat_ok / lat_tot if lat_tot else None)), per_utt

if __name__ == "__main__":
    kws = keyword_set()
    rows = []
    for p in sorted(HYP.glob("*.jsonl")):
        m, cond, split = p.stem.split("__")
        s, per = score(p, kws)
        rows.append(dict(model=m, cond=cond, split=split, **s))
        json.dump(per, (OUT / f"perutt__{p.stem}.json").open("w"))
        print(f"{m:6} {cond:3} {split:5} WER={s['wer']*100:6.2f}  "
              f"K-WER={100*(s['k_wer'] or 0):6.2f}  U-WER={100*(s['u_wer'] or 0):6.2f}  "
              f"CER={s['cer']*100:6.2f}  script={100*(s['script_fidelity'] or 0):5.1f}%")
    with (OUT / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
```

**Why the K-WER / U-WER split is the most important metric here.** It is exactly the decomposition that exposed the naive-prompting failure in prior work: prompting cut rare-word error substantially while raising error on non-biased words, so overall WER got worse on most datasets. If you report only aggregate WER, you cannot tell whether your method is working or which half is failing. Reporting K-WER and U-WER separately is what lets you diagnose, and it is what makes your negative results interpretable rather than embarrassing.

## 4.3 `src/stats.py` — paired bootstrap

Overall WER differences from prompting are typically small. Without confidence intervals you cannot distinguish a real 0.8-point gain from noise, and a reviewer will not accept a bare point estimate.

```python
import json, pathlib, numpy as np
OUT = pathlib.Path("~/sgcd/out").expanduser()

def load(model, cond, split="test"):
    return np.array([x if x is not None else np.nan
                     for x in json.load((OUT / f"perutt__{model}__{cond}__{split}.json").open())])

def paired_bootstrap(a, b, n=10000, seed=1337):
    """a = baseline per-utt WER, b = system per-utt WER. Returns mean delta + 95% CI."""
    rng = np.random.default_rng(seed)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), (boot < 0).mean()

if __name__ == "__main__":
    base = load("turbo", "C0")
    for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
        mean, lo, hi, p = paired_bootstrap(base, load("turbo", c))
        sig = "*" if (lo < 0 < hi) is False else " "
        print(f"C0 -> {c}: ΔWER = {mean*100:+.2f}  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]  "
              f"P(improve)={p:.3f} {sig}")
    # H3: content-specificity — matched vs mismatched syllabus
    mean, lo, hi, p = paired_bootstrap(load("turbo", "C5"), load("turbo", "C4"))
    print(f"\nH3  C5(mismatched) -> C4(matched): ΔWER = {mean*100:+.2f}  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]")
```

Report **ΔWER with 95% CI** in every table. Report per-utterance **degradation rate** (% of utterances where the system is worse than C0) alongside it — a method that improves the mean while wrecking 30% of utterances is not deployable, and saying so is a strength.

---

# PART 5 — EXECUTION ORDER

| # | Task | Output | Time |
|---|---|---|---|
| 1 | Download test tarball, run Step 0 discovery | Confirmed file layout pasted into notes | 20–40 min |
| 2 | `build_manifest.py`, inspect lecture IDs | `manifest.jsonl` + lecture list | 30 min |
| 3 | Write **one** syllabus JSON by hand (Tier A or B) | `syllabi/<course>.json` | 45 min |
| 4 | Smoke test: tiny model, 10 utts, C0 + C4 | Pipeline runs end-to-end | 20 min |
| 5 | Sanity-read 20 normalised ref/hyp pairs | Normaliser fixed and frozen | 30 min |
| 6 | Generate remaining syllabi | `syllabi/*.json` | 1–2 h |
| 7 | Build mismatched pool; **assert no oracle provenance** | Leakage guard in code | 30 min |
| 8 | DEV sweep (small model): language, k, cap, guard thresholds | Frozen config in `RUNLOG.md` | 1 h |
| 9 | **Freeze everything. Commit. Then run TEST once.** | `hyps/*.jsonl` | ~6 min |
| 10 | `score.py` + `stats.py` | `scores.csv`, CI table | 30 min |
| 11 | Second model size (`small`) for generalisation row | Extra table rows | 15 min |
| 12 | Error analysis: 30 utterances where C4 beat C0, 30 where it lost | Qualitative section | 2 h |
| 13 | Write up | Paper | — |

Realistically two focused days for steps 1–11, plus writing. Step 12 is what turns a results table into a paper; do not skip it.

---

# PART 6 — PAPER STRUCTURE AND CLAIM MAPPING

**Title:** *Syllabus-Grounded Contextual Decoding for Hindi–English Code-Switched Lecture Transcription*

1. **Introduction** — classroom ASR for Indian higher education; code-switching is the hard part; syllabi are free, structured, available *before* the lecture, and universally unused.
2. **Related work** — Whisper; contextual biasing via prompting; keyword-spotting-driven biasing (CB-Whisper, KG-Whisper, KWS-Whisper); code-switched Indic ASR and the MUCS challenge. **Position your work as: no training, no auxiliary model, uses a document that already exists.** That is a real gap.
3. **Method** — prompt-slot mechanics and the 224-token limit; prose rendering vs keyword rendering; TF-IDF retrieval over syllabus units; the confidence guard. One figure: audio → first pass → retrieve → render → second pass → guard.
4. **Experimental setup** — SLR104 Hindi–English test set; DEV/TEST lecture-disjoint split; syllabus tiers and the leakage policy (give this its own subsection — it demonstrates rigour); metrics; decoding config.
5. **Results** — main table (C0–C7 × {WER, K-WER, U-WER, CER, script fidelity, degradation rate, ΔWER with CI}); second model size; the C4-vs-C5 matched/mismatched result as the headline evidence.
6. **Analysis** — where it wins (technical terms, script-correct English), where it loses (short utterances, off-syllabus digressions, prompt-induced repetition); guard fallback rate.
7. **Limitations** — one corpus; scripted narration is not spontaneous classroom speech; ≤28 s utterances; syllabus quality is a confound; small N.
8. **Conclusion + future work** — the natural bridge to your larger note-generation system.

**Claim → evidence map (fill this in as you go, and refuse to write any claim without a cell):**

| Claim | Evidence |
|---|---|
| Naive keyword prompting hurts overall WER | C2 vs C0, with K-WER/U-WER split |
| Prose rendering fixes most of it | C3 vs C2 |
| Retrieval beats whole-syllabus | C4 vs C3 |
| The gain is content-specific | **C4 vs C5** |
| Guard turns it into a net win | C7 vs C0, + fallback rate |
| Not a single-checkpoint artifact | turbo and small rows |
| Headroom remains | C6 topline |

---

# PART 7 — FAILURE MODES AND WHAT TO DO

| Symptom | Likely cause | Response |
|---|---|---|
| WER > 60% in every condition | Script mismatch — Whisper writing English in Devanagari while refs use Latin | Try `language=None`; add a transliteration-tolerant scoring variant and report both |
| Prompted output repeats or drifts | Prompt too long / list-like; the known instability | Lower cap to 120 tokens; this is exactly what C7's guard exists to catch — measure it, don't hide it |
| C4 ≈ C5 (matched no better than mismatched) | Syllabus content is not actually informative, or retrieval is failing | Check retrieval precision against C6. If C6 also ≈ C5, the honest conclusion is that this corpus's utterances are too short for content context to matter — **report that** |
| No WER change anywhere | Utterances too short for context to bite | Concatenate consecutive same-lecture segments into 25 s pseudo-utterances; re-run |
| Timing far worse than budget | Model loading per call, or fallback decoding | Confirm `temperature=0.0` scalar; keep the process alive across conditions |
| Runtime fine but results noisy | N too small | Raise N to 300 with `small` — more utterances beats a bigger model for CI width |

---

# PART 8 — VERIFIED VS ASSUMED (read before you build)

**Verified against primary sources:**
- SLR104 identity, licence (CC BY-SA 4.0), test tarball ≈ 443 MB, password-free Mirror1 links
- Hindi–English CS data is drawn from spoken tutorials on technical topics; transcripts were narrator scripts; content includes mathematical symbols
- 16 kHz / 16-bit audio; ~5.18 h test split
- A `segments` file with sentence timestamps exists in the baseline recipe
- Per-sentence source-tutorial information is retained by the organisers
- Whisper's `initial_prompt` consumes only the last <224 tokens, later tokens dominate, and it biases vocabulary/spelling/style
- Naive prompt biasing improves rare-word and OOV WER but raises unbiased-word WER and overall WER on most datasets tested

**Assumed — verify in Step 0 before relying on:**
- Exact directory layout and filenames inside the tarball
- Whether utterance/recording IDs encode a recognisable topic or series name
- Whether public per-series syllabi exist for the specific tutorials in this corpus (Tier A). **If not, Tier B is the plan — it costs nothing and is arguably more realistic.**
- Actual decoding throughput on your machine

Anywhere the code says "ADJUST AFTER STEP 0", that is a real instruction, not boilerplate.
