# Pre-registration — SGCD (Syllabus-Grounded Contextual Decoding)

Committed **before** the first TEST run. Nothing below may be changed after the
TEST split has been decoded; if something must change, it is recorded as a
post-hoc deviation in `RUNLOG.md` and reported as such in the paper.

## Hypotheses and pass criteria

| ID | Hypothesis | Pass criterion |
|----|-----------|----------------|
| H1 | SGCD (C7) reduces overall WER vs the no-prompt baseline (C0) | ΔWER < 0 with 95% bootstrap CI excluding 0 |
| H2 | Syllabus context improves technical-term recognition | K-WER drops ≥ 15% relative (C4 or C7 vs C0) |
| H3 | The gain is content-specific, not style-priming | WER(C4, matched) < WER(C5, mismatched), CI excludes 0 |
| H4 | Prose rendering beats keyword-list rendering | WER(C3) < WER(C2) |
| H5 | Retrieval beats whole-syllabus prompting | WER(C4) < WER(C3) |

**Reporting commitment:** the result is reported either way. If H1 fails while H2
and H3 hold, the finding reported is "syllabus context reliably improves
domain-term recognition in Hindi–English lecture ASR but does not improve overall
WER, and here is why" — that is a genuine result, not a failed experiment.

## Conditions

| ID | Prompt | Purpose |
|---|---|---|
| C0 | none | Baseline |
| C1 | generic code-mixed sentence, no course content | Style control |
| C2 | syllabus keywords, comma-separated | Naive method (known failure mode) |
| C3 | whole-syllabus prose | Rendering hypothesis (H4) |
| C4 | retrieved k units, prose (**SGCD**) | Proposed method (H5) |
| C5 | retrieved from a **different** course | Content-specificity control (H3) |
| C6 | retrieved using the reference | Topline (oracle retrieval) — always labelled |
| C7 | C4 + confidence guard | Full system (H1) |

## Frozen data rules (`src/build_manifest.py`)

- Duration filter: 2.0 s ≤ dur ≤ 28.0 s (whole utterance inside one 30 s Whisper window).
- Reference filter: ≥ 4 reference words.
- Lecture-disjoint split: 30% of lectures → DEV, 70% → TEST.
- Stratified sample, proportional per lecture, seed 1337: N_DEV = 60, N_TEST = 150.
- Zero-shot only. The training split is never downloaded or used, so the corpus's
  reported 33.9% train/test overlap does not apply.

## Frozen decoding config (`src/decode.py`)

```
task="transcribe", temperature=0.0 (scalar), condition_on_previous_text=False,
word_timestamps=False, language=<set by DEV sweep>
```

## Leakage policy

- No syllabus unit may contain text derived from the reference transcript of any
  evaluated utterance. Enforced by `courses.assert_leakage_free`, which raises if a
  syllabus with `provenance == "oracle"` is used by any condition outside
  `ORACLE_CONDITIONS`.
- Retrieval queries come from the first-pass C0 hypothesis for all conditions
  except C6, which is explicitly the oracle-retrieval topline. `decode.run`
  asserts the reference is never used as a query outside C6.

## Tuned on DEV only, then frozen

`language`, `k` ∈ {1,2,3}, prompt token cap ∈ {120, 200}, guard thresholds
(`d_logprob`, `max_cr`, `len_ratio`). Every DEV number is recorded in `RUNLOG.md`.
TEST is decoded once, after this file and the frozen config are committed.

## Metrics

WER (corpus-level), CER, K-WER / U-WER (keyword vs non-keyword reference words),
script fidelity (fraction of Latin-script reference words recovered exactly),
per-utterance degradation rate, guard fallback rate, and ΔWER with a 95% paired
bootstrap CI (10,000 resamples, seed 1337).
