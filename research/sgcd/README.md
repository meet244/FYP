# SGCD — Syllabus-Grounded Contextual Decoding

Does giving Whisper the **course syllabus** as decoder context reduce WER on
Hindi–English code-mixed technical lectures? This repo tests that properly:
naive keyword prompting (the known failure mode), prose rendering, retrieval over
syllabus units, a confidence guard, and — the load-bearing control — a
**mismatched syllabus** condition that separates content from style priming.

Zero training. Whisper is used off the shelf; the only lever is the
`initial_prompt` slot, whose documented behaviour (last ~224 tokens, later tokens
dominate, expects previous-segment transcript) drives every design choice.

## Data

MUCS 2021 Subtask-2 Hindi–English, [OpenSLR SLR104](https://www.openslr.org/104/),
CC BY-SA 4.0. **Test tarball only** (~443 MB) — the training split is never used,
so the corpus's reported 33.9% train/test overlap does not apply here.

```bash
mkdir -p data && cd data
curl -L -O https://openslr.elda.org/resources/104/Hindi-English_test.tar.gz
tar -xzf Hindi-English_test.tar.gz
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx-whisper jiwer soundfile numpy pandas scikit-learn scipy regex tiktoken
```

Apple Silicon via `mlx-whisper`. On other hardware swap `mlx_whisper.transcribe`
in `src/decode.py` for `faster-whisper`; both expose `initial_prompt`.

## Pipeline

```bash
cd src                                  # every script is run from src/
python inspect_data.py                  # Step 0: verify the real corpus layout
python build_manifest.py                # freeze the eval set (manifest.jsonl)
python map_lectures.py --write          # lecture -> course mapping
python make_syllabi.py --list           # which syllabi are still missing
#   ... write syllabi/<course>.json (Tier A real, or Tier B generated-from-title)
python sanity_check.py                  # read 20 normalised references
python decode.py --model tiny --split dev --limit 10 --conditions C0 C4   # smoke test
python sweep_dev.py --model small       # DEV tuning; freeze the printed config
#   ... commit PREREGISTRATION.md + frozen config, THEN:
python decode.py --model turbo --split test         # the one main run
python score.py
python stats.py --model turbo --split test
python make_tables.py --model turbo --split test
python error_analysis.py --model turbo --cond C4
```

Hypotheses are cached per `(model, condition, split)`, so re-scoring is free and an
interrupted run resumes without redecoding.

## What we found

Full numbers and the honest reading are in [RUNLOG.md](RUNLOG.md); tables in
`out/tables/`. In brief, on 150 lecture-disjoint TEST utterances (turbo, zero-shot):

- **Naive keyword prompting reproduces the known failure mode in a new language
  setting.** C2 cuts keyword error 65.5% → 31.6% while pushing non-keyword error
  43.5% → 53.2% and overall WER +50.5 (CI excludes 0).
- **Prose rendering repairs most of it** (C2→C3 = −36.9, CI excludes 0), returning
  non-keyword error to *below* baseline.
- **The gain is not content-specific.** A syllabus from the *wrong course* does as
  well as the matched one (C4 vs C5 = +3.09, CI spans 0), and oracle retrieval is
  no better. What prompting actually supplies is the corpus's output convention —
  script fidelity rises 32.6% → 41.6% with a *contentless* prompt and → ~60% with
  any syllabus, matched or not.
- **On 26 s pseudo-utterances the method works without the guard** (−4.83, CI
  excludes 0; keyword error 48.0% → 21.2%), and baseline WER halves (85.7 → 43.5),
  showing the main set's absolute numbers were inflated by short spans. Even here
  the mismatched control captures **81%** of the gain.

Headline: *syllabus prompting reliably improves technical-term recognition and
script fidelity in Hindi–English lecture ASR; the benefit is predominantly a
format/register effect rather than semantic grounding in the specific syllabus.*

## Conditions

| ID | Prompt | Purpose |
|---|---|---|
| C0 | none | Baseline |
| C1 | generic code-mixed sentence | Style control (format priming without content) |
| C2 | syllabus keywords, comma-separated | Naive method — replicates the known failure |
| C3 | whole-syllabus prose | Rendering hypothesis |
| C4 | retrieved k units, prose (**SGCD**) | Proposed method |
| C5 | retrieved from a **different** course | Content-specificity control |
| C6 | retrieved using the reference | Topline (oracle retrieval), always labelled |
| C7 | C4 + confidence guard | Full system |

C4–C7 reuse C0's hypotheses as the retrieval first pass, so the two-pass method
costs **one** extra decode, not two.

## Metrics

WER, CER, **K-WER / U-WER** (keyword vs non-keyword reference words — the split
that exposes whether prompting helps terms while hurting everything else), script
fidelity, per-utterance degradation rate, guard fallback rate, and ΔWER with a 95%
paired bootstrap CI.

## Discipline

- `PREREGISTRATION.md` — hypotheses and pass criteria, committed before the TEST run.
- `RUNLOG.md` — append-only: date, git hash, command, headline numbers.
- DEV (30% of lectures) is the only split any knob is tuned on. TEST is decoded once.
- Leakage guard: `courses.assert_leakage_free` raises if an `oracle`-provenance
  syllabus reaches a scored condition, and `decode.run` asserts the reference is
  never used as a retrieval query outside the C6 topline.

## Layout

```
data/        extracted SLR104 test set (gitignored)
syllabi/     <course>.json + lecture_map.json
out/         manifest.jsonl, hyps/, scores.csv, tables/
src/         config, build_manifest, prompts, retrieve, decode, score, stats, ...
```
