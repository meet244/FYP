# SGCD — Syllabus-Grounded Contextual Decoding

This directory contains the original Whisper context study, the five-model
baseline benchmark, and the ten-method Qwen FYRP comparison. The application
defaults to Qwen 0.6B with S5; SGCD is a separate decoding option. See the
[research overview](../README.md) and [consolidated results](../../results.md).

Does giving Whisper the **course syllabus** as decoder context reduce WER on
Hindi–English code-mixed technical lectures? This repo tests that properly:
naive keyword prompting (the known failure mode), prose rendering, retrieval over
syllabus units, a confidence guard, and — the load-bearing control — a
**mismatched syllabus** condition that separates content from style priming.

The original SGCD study uses zero training. Whisper is used off the shelf;
the only lever is the
`initial_prompt` slot, whose documented behaviour (last ~224 tokens, later tokens
dominate, expects previous-segment transcript) drives every design choice.

## Data

MUCS 2021 Subtask-2 Hindi–English, [OpenSLR SLR104](https://www.openslr.org/104/),
CC BY-SA 4.0. **Test tarball only** (~443 MB) — the official training tarball is never used,
so the corpus's reported 33.9% train/test overlap does not apply here.

The tarball is internally divided into lecture-disjoint DEV and TEST lectures.
FYRP training uses internal DEV data, not the official training tarball or
internal TEST references.

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

This setup covers the original Whisper study. Qwen, FYRP fine-tuning, and
Parakeet need additional runtimes; see the [replication guide](../REPLICATION.md).

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

## Later experiments

| Experiment | Sample | Finding |
| --- | --- | --- |
| [Five-model benchmark](out/tables/bench__test.md) | 150 utterances, 862 seconds | Qwen3-ASR-0.6B has the lowest strict WER, 59.74%. Qwen 1.7B has better script-tolerant WER but a forced-Hindi script mismatch. |
| [Ten FYRP methods](out/fyrp/results.md) | 50 TEST utterances; 30 DEV tuning utterances | S5 reduces WER from 58.86% to 55.69%, delta -3.18 points, reported 95% CI [-6.3, -0.2]. S8's interval includes zero; S9/S10 have repetition runaways. |

These are separate experiments. The later benchmark's turbo C0 (80.23% WER)
differs from original SGCD C0 (85.69%); the reviewed material does not establish
the exact cause. Use each experiment's own baseline. The 50-utterance S0 and
150-utterance Qwen baseline are also not interchangeable.

Serving S5 uses a frozen, versioned domain LM, five candidates, roughly
25-second spans, and a 512-token cap; research FYRP used a 200-token cap.
See the [LM contract](../../backend/assets/README.md). S5 is independent of SGCD
and shallow fusion; the experiment does not measure serving WER. S7-S10 write
training metadata and hypotheses, not reusable fine-tuned weights. This checkout
contains their metrics and training metadata but lacks the FYRP hypothesis
directory. Qwen/Parakeet benchmark hypotheses are also absent; original Whisper
hypotheses are available. Independently rescoring the later results requires
recreating those outputs.

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

K-WER/U-WER count errors aligned to reference words; insertions are separate,
so those buckets do not account for every WER error. WER-sa is a secondary,
lossy transliteration-tolerant metric, not semantic accuracy. The field called
script fidelity counts exact matches on Latin-script reference words, so it
also depends on recognition. RTF is decode time divided by audio duration;
process RSS does not measure all accelerator memory. Original SGCD uses 10,000
bootstrap resamples (seed 1337); FYRP uses 2,000 (seed 0). Neither resamples
independent classroom deployments.

## Discipline

- `PREREGISTRATION.md` — hypotheses and pass criteria, committed before the TEST run.
- `RUNLOG.md` — append-only: date, git hash, command, headline numbers.
- DEV (30% of lectures) is the only split any knob is tuned on. TEST is decoded once.
- These preregistration rules describe original SGCD. The later benchmark and
  FYRP have their own saved settings; they are not covered by the original
  preregistration. Historical log entries are preserved as observations at the
  time, including older statements about the application's default model.
- Leakage guard: `courses.assert_leakage_free` raises if an `oracle`-provenance
  syllabus reaches a scored condition, and `decode.run` asserts the reference is
  never used as a retrieval query outside the C6 topline.

## Layout

```
data/        extracted SLR104 test set (gitignored)
syllabi/     <course>.json + lecture_map.json
out/         manifest.jsonl, hyps/, scores.csv, tables/
out/fyrp/    Qwen method hypotheses, training metadata, results.json, results.md
src/         config, build_manifest, prompts, retrieve, decode, score, stats, ...
```
