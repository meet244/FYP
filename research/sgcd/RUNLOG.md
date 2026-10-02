# RUNLOG — append-only

Date, command, and headline numbers for every run that produced anything I might
cite. Written as I go, because in three weeks I will not remember which run
produced which number.

---

## 2026-08-14 — Step 0: discovery (SLR104 Hindi–English test)

Downloaded `Hindi-English_test.tar.gz` (443,929,204 bytes) from the ELDA mirror
(`https://openslr.elda.org/resources/104/`), extracted to `data/`.

**Real layout** (the plan's assumed layout was correct):

```
data/test/transcripts/{text, segments, wav.scp, utt2spk, spk2utt, spkr_list}
data/test/*.wav          # 30 files
```

- `text`: 3136 lines, `<utt_id> <transcript>`; dual-script (Devanagari + Latin) as expected.
- `segments`: 3136 lines, `<utt_id> <rec_id> <start> <end>`.
- `wav.scp`: 30 lines, `<rec_id> <rec_id>.wav`.
- Utterance IDs are `<speaker>_<recording>_<seq>`, e.g. `103085_w5Jyq3XMbb3WwiKQ_0000`.
  Recording ID = lecture. 30 lectures, one speaker each — matches the organisers'
  "each tutorial treated as a distinct speaker".
- Audio verified 16 kHz; wav durations match the max segment end exactly, so the
  slicing in `decode.load_audio` is correct.

**Answers to the Step-0 questions:**

| Question | Answer |
|---|---|
| `segments` file? | Yes — long recordings, sliced per utterance |
| Utterance IDs share a per-recording prefix? | Yes — recording hash is the lecture ID |
| Do IDs encode a topic/series name? | **No** — opaque hashes. Tier A via IDs is impossible; see below |
| Reference text dual-script? | Yes — Hindi in Devanagari, English technical terms in Latin |
| How many utterances? | 3136 raw; 3073 after filtering; median 5.0 s |

**Finding that changes the plan:** 99.6% of segment boundaries are rounded to whole
seconds (6250/6272), so adjacent-utterance words bleed into each clip. This inflates
absolute WER for *every* condition equally; relative comparisons are unaffected.
Goes in Limitations.

## 2026-08-14 — Manifest

```
python src/build_manifest.py
utterances=3073  lectures=30  split={'dev': 967, 'test': 2106}  skipped={'duration': 17, 'too_few_words': 46}
eval subset: {'dev': 60, 'test': 150}  (dev lectures=9, test lectures=21)
total audio = 5.06 h   median dur = 5.0 s
  dev: n=60  audio=5.5 min  mean=5.5 s
  test: n=150 audio=14.4 min mean=5.7 s
```

## 2026-08-14 — Syllabi (Tier B) and the title-leakage rule

IDs carry no topic, so each lecture's topic was read from its **opening
utterances**, which announce it ("… के इस spoken tutorial में आपका स्वागत है").
Consequence, enforced in `build_manifest.py`: the first 2 surviving utterances of
every lecture are marked `title_source` and **excluded from the evaluation set**,
so no scored utterance's reference contributed anything to a syllabus. The
title→course audit trail is `syllabi/lecture_titles.json`.

30 lectures → **7 courses**: libreoffice-impress (8 lectures), libreoffice-writer (5),
thunderbird (4), gedit (4), jchempaint (3), xfig (3), c-programming (3).
All Tier B, `provenance: "generated-from-title"`, 8–10 units each.

**Deviation from the plan (§2.3):** the plan targets 45–70 prompt tokens per unit
using cl100k as a proxy. Measured against Whisper's *actual* multilingual BPE
(`mlx_whisper.tokenizer`, now used directly in `prompts.py`), Hindi–English prose
costs ~3 tokens/word, so units are 74–105 tokens (~20–30 words) and **k=2 units ≈
185 tokens ≈ the 200-token cap** — which is the budget the plan intended.

## 2026-08-14 — Smoke test

```
python src/decode.py --model tiny  --split dev --limit 8 --conditions C0 C4 C5 C6 --suffix=-smoke
python src/decode.py --model turbo --split dev --limit 8 --conditions C0 C2 C4    --suffix=-smoke
```

Throughput after the model is cached: **tiny ~12x realtime, turbo ~9x realtime**
(first call per model includes the HF download: turbo 539 s, tiny 30 s).
Projected main run: 14.4 min of TEST audio ÷ 9 ≈ 96 s per condition, ~11 min for
7 conditions — inside budget.

Two failure modes from the plan's Part 7 confirmed on real output:

1. **Script mismatch.** turbo writes English terms in Devanagari ("स्क्रीन",
   "प्रिंट") where references use Latin ("screen", "print"). Strict script
   fidelity at C0 is only 42.9%. Response, as the plan prescribes: added a
   **transliteration-tolerant secondary metric** (`normalize.normalize_sa`,
   reported as `WER-sa`) that romanises Devanagari and reduces both scripts to a
   consonant skeleton, so `screen`/`स्क्रीन` → `skrn`. Reported *alongside*
   strict WER, never instead of it. Also to be tested on DEV: `language=None`.
2. **Repetition loops.** `temperature=0.0` as a scalar disables Whisper's
   fallback, so loops are not silently repaired (deliberate — it is what lets the
   guard's job be measured honestly). One looping utterance pushes small-sample
   WER over 100%.

Smoke numbers (n=8, tiny is debug-only and never reported):

```
turbo C0 dev-smoke  WER=115.25  WER-sa= 98.31  K-WER=40.00  U-WER=34.26  script=42.9%
turbo C2 dev-smoke  WER=170.34  WER-sa=145.76  K-WER=10.00  U-WER=58.33  script=64.3%
turbo C4 dev-smoke  WER= 66.95  WER-sa= 50.00  K-WER=20.00  U-WER=40.74  script=64.3%
```

Directionally the expected pattern is already visible at n=8: C2 (keyword list)
cuts K-WER hard while *raising* U-WER and overall WER — the known failure mode —
and prompting raises script fidelity 42.9% → 64.3%. n=8 proves nothing; noted
only as evidence the pipeline measures what it is supposed to.

## 2026-08-14 — DEV sweep (turbo), and the frozen config

```
python src/sweep_dev.py --model turbo
```

**Deviation from the plan (§3.8):** the plan sweeps on `small`. The
`whisper-small-mlx` download stalled repeatedly, and tuning on the same
checkpoint the main run uses removes a config-transfer assumption, so the sweep
was run on `turbo`. `small` is still used for the generalisation row.

**1. Language** — decisive, not a close call:

| `language` | C0 WER (DEV, n=60) |
|---|---|
| `"hi"` | **54.83** |
| `None` (auto-detect) | 74.01 |

**2/3. Retrieved units k, prompt cap** (C4 WER on DEV):

| | cap=120 | cap=200 |
|---|---|---|
| k=1 | 60.94 | 60.94 |
| k=2 | 58.10 | 64.35 |
| k=3 | 58.10 | **54.83** |

Chosen **k=3, cap=200**. Note what this actually means: 3 units × ~90 tokens = 270
tokens, left-truncated to 200, so mean prompt length is exactly 200.0 tokens. The
selected config is "fill the whole budget, most relevant unit last" rather than
"three whole units".

**4. Guard thresholds** — grid over d_logprob × max_cr × len_ratio, minimising DEV WER:

```
d_logprob=0.25  max_cr=2.0  len_ratio=1.5  ->  DEV C7 WER=37.50  (C4=54.83, C0=54.83)  fallback=8.3%
```

**FROZEN CONFIG** (now in `decode.py` / `prompts.py`):

```json
{"language": "hi", "k": 3, "max_prompt_tokens": 200,
 "guard": {"d_logprob": 0.25, "max_cr": 2.0, "len_ratio": 1.5}}
```

### Two things checked before trusting these numbers

**C4(k=3,cap=200) WER equals C0 WER to the decimal (54.83).** Not a bug: 59 of 60
hypotheses differ, and the per-utterance error profiles are completely different
(worst C0 utterance 71 errors/22 ref words; worst C4 utterance 65/19). Total
errors coincide at 386/704 in both. A coincidence at the aggregate, verified by
inspection rather than assumed.

**The guard's 17-point DEV gain comes from ~5 utterances.** Corpus WER here is
dominated by a few repetition-loop utterances (one contributes 65 errors against
a 19-word reference). Falling back on 8.3% of utterances removes 122 errors:
386 → 264, and 264/704 = 37.50%. This must be reported honestly — the guard
rescues catastrophic utterances, it does not broadly improve transcription.
Expect a smaller effect on TEST (36 threshold combinations were compared on
n=60, so some overfitting to DEV is certain). `stats.py` reports the macro
(per-utterance mean) delta and the degradation rate alongside the corpus delta
precisely so this distinction is visible.

## 2026-08-14 — TEST run (turbo), the one main run

```
python src/decode.py --model turbo --split test     # 8 conditions, N=150, frozen config
```

7 decodes × 150 utterances, 5.5–5.6x realtime, ~18 min total. C7 derived from C4
with no extra decode. Guard fell back on **19/150 (12.7%)**, every fallback
verified identical to the C0 hypothesis.

### Main results (turbo, TEST, N=150)

| Cond | WER | WER-sa | K-WER | U-WER | CER | Script fid. | ΔWER vs C0 [95% CI] | Worse % |
|---|---|---|---|---|---|---|---|---|
| C0 baseline | 85.69 | 71.87 | 65.53 | 43.50 | 67.92 | 32.6 | ref | — |
| C1 generic prompt | 102.32 | 90.48 | 55.83 | 44.38 | 82.29 | 41.6 | +16.63 [−3.61, +36.66] | 31 |
| C2 keyword list | 136.22 | 118.43 | 31.55 | 53.18 | 114.16 | 60.9 | **+50.53 [+16.55, +90.94]** | 46 |
| C3 whole-syllabus prose | 99.35 | 88.81 | 33.50 | 41.68 | 82.89 | 60.0 | +13.66 [−7.70, +35.35] | 32 |
| C4 SGCD | 90.80 | 81.27 | 35.44 | 41.95 | 75.97 | 58.8 | +5.11 [−13.24, +23.28] | 34 |
| C5 mismatched | 87.71 | 77.62 | 35.92 | 40.19 | 74.33 | 58.6 | +2.02 [−13.98, +18.98] | 29 |
| C6 oracle retrieval (topline) | 95.61 | 85.64 | 29.61 | 40.53 | 78.73 | 62.1 | +9.92 [−10.49, +30.23] | 33 |
| C7 SGCD + guard | **71.50** | 61.28 | 35.44 | 41.14 | 56.22 | 58.1 | **−14.19 [−26.55, −4.79]** | 24 |

### Pre-registered hypotheses — outcome

| ID | Claim | Result | Verdict |
|---|---|---|---|
| H1 | C7 reduces WER vs C0 | ΔWER −14.19, CI [−26.55, −4.79] | **PASS** |
| H2 | K-WER drops ≥15% relative | 65.53 → 35.44 = **46% relative** | **PASS** |
| H3 | Gain is content-specific (C4 < C5) | **+3.09**, CI [−9.82, +16.16] | **FAIL** |
| H4 | Prose beats keyword list (C3 < C2) | −36.88, CI [−75.64, −5.18] | **PASS** |
| H5 | Retrieval beats whole syllabus (C4 < C3) | −8.55, CI [−22.89, +5.71] | directional, n.s. |

### Honest reading of this

**The central hypothesis is not supported.** H3 is the load-bearing control, and
it fails: a syllabus from the *wrong course* (C5, 87.71) does as well as the
matched syllabus (C4, 90.80) — numerically slightly better, with a CI comfortably
spanning zero. The oracle-retrieval topline (C6, 95.61) is no better either. Per
the decision rule written in the plan's Part 7 *before* these numbers existed:
when C4 ≈ C5 ≈ C6, the honest conclusion is that syllabus **content** is not what
is helping.

**What is actually happening** is visible in the script-fidelity column: 32.6%
(no prompt) → 41.6% (generic prompt, zero course content) → ~59–62% (any syllabus
prompt, matched or not). The prompt is teaching Whisper the corpus's *output
convention* — write English technical terms in Latin script, Hindi in Devanagari —
not telling it which words to expect. That is a format/register effect, and it is
exactly what the C1 and C5 controls were built to detect. Without them this would
have been written up as a content result.

**The K-WER/U-WER split reproduces the known failure mode, in a new language
setting.** C2 (keyword list) cuts K-WER 65.53 → 31.55 while pushing U-WER
43.50 → 53.18 and overall WER +50.53 (CI excludes 0). Prose rendering repairs most
of it (H4, C2→C3 = −36.88): U-WER returns to 41.68, *below* baseline.

**H1 passes, but on the guard's strength, not the prompt's.** C4 alone is +5.11
(worse than baseline, n.s.). The entire H1 win is C4→C7 = −19.30, and the guard
wins by reverting 19 utterances to the unprompted hypothesis. Error analysis shows
why prompting both helps and hurts:

- *Wins:* C4 repairs C0's repetition loops and writes English in Latin
  (`record` / `operating system` / `tutorial` where C0 gives
  `डिकॉर्ड` / `ओपरेटिंग सिस्टम` / `टूटोरियल`).
- *Losses:* C4 **induces its own loops**, and the looped text echoes the prompt's
  narration register — `सीखेंगे`, `देखेंगे अपना`, `लिए और` repeated to the token
  limit. The model continues the syllabus prose instead of stopping.

So the guard is not cosmetic: prompting causes and cures loops roughly
symmetrically (C4 better on 59 utterances, worse on 51, unchanged on 40), and the
guard's job is to keep the cures. Degradation rate drops 34% → 24%.

**Caveat on the guard, stated plainly:** its thresholds were chosen by comparing
36 combinations on 60 DEV utterances, and its gain is concentrated in a few
catastrophic utterances rather than spread across the corpus. The TEST fallback
rate (12.7%) ran higher than DEV (8.3%), and the TEST gain (−14.19) is smaller
than DEV (−17.33), consistent with mild DEV overfitting.

**Absolute WER is very high (85.69% baseline) and is not comparable to published
MUCS numbers.** Three causes, all documented above: 1 s-rounded segment boundaries
bleed adjacent words into every clip; `temperature=0.0` leaves repetition loops
unrepaired by design; and the strict metric penalises script mismatch, which is
why WER-sa is reported alongside (71.87% at baseline).

## 2026-08-14 — Generalisation row (small, TEST, N=150)

```
python src/decode.py --model small --split test      # ~6-9x realtime, 15 min
```

| Cond | WER | K-WER | U-WER | Script fid. | ΔWER vs C0 [95% CI] |
|---|---|---|---|---|---|
| C0 | 171.50 | 99.51 | 75.17 | **1.6** | ref |
| C1 | 247.57 | 99.51 | 77.33 | 1.6 | +76.07 |
| C2 | 330.76 | 40.29 | 93.30 | 46.7 | +159.26 |
| C3 | 148.81 | 80.58 | 70.70 | 20.5 | −22.68 [−55.99, +10.58] |
| C4 | **143.23** | 74.27 | 69.15 | 23.7 | −28.27 [−60.77, +3.10] |
| C5 | 160.75 | 88.35 | 69.08 | 16.0 | −10.75 [−47.43, +26.30] |
| C6 | 168.88 | 67.48 | 69.15 | 25.6 | −2.61 [−42.13, +37.63] |
| C7 | 157.78 | 79.13 | 73.14 | 17.9 | −13.72 [−23.47, −5.73] |

This row does more than show the effect is not a single-checkpoint artifact —
it **changes the interpretation**, in three ways:

1. **`small` is nearly script-blind: 1.6% script fidelity at C0**, i.e. it writes
   essentially every English term in Devanagari, and gets 99.5% of keyword tokens
   wrong. Prompting is the only thing that recovers any Latin-script output
   (1.6% → 23.7%), and here prompting *does* cut overall WER (C4 −28.27).
2. **H3 reverses direction on the weaker model.** C5→C4 = **−17.52** (matched
   better than mismatched, P(improve)=0.888) versus **+3.09** on turbo. The
   natural reading: when a model is weak enough that it cannot produce the
   technical terms at all, syllabus *content* has something to contribute; once a
   model is strong enough to get the words (turbo), only the output *convention*
   is still missing, and any fluent code-mixed prose supplies that. Neither CI
   excludes zero, so this is a hypothesis for future work, not a finding.
3. **The guard does not transfer across checkpoints.** Thresholds fitted on
   turbo/DEV fall back on **46%** of `small` utterances and make things *worse*
   (C4→C7 = +14.55). It still beats C0 (−13.72) only because `small`'s baseline
   is so poor. Guard thresholds are model-specific and must be refitted per
   checkpoint — worth stating explicitly, since it is a real limitation of the
   method as specified.

## 2026-08-14 — Secondary experiment: 26 s pseudo-utterances (post-hoc)

Prompted by the H3 null, following the response the plan prescribes in Part 7 for
the C4 ≈ C5 symptom: "concatenate consecutive same-lecture segments into 25 s
pseudo-utterances; re-run". **Clearly post-hoc — not part of the pre-registration,
and reported as a separate secondary experiment.**

```
python src/build_concat_manifest.py    # 400 pseudo-utts; testcat N=100
python src/decode.py --model turbo --split testcat --conditions C0 C4 C5
```

Construction: only strictly adjacent segments of one lecture are merged, spans
capped at 28 s so each still fits a single Whisper window, `title_source`
utterances still excluded, same lecture-disjoint split and seed.
Result: mean 26.2 s and **55.1 reference words** per pseudo-utterance, against
~12 words in the main set — 4.7x more context for syllabus content to act on.

| Cond | WER | WER-sa | K-WER | U-WER | CER | Script fid. | ΔWER vs C0 [95% CI] | Worse % |
|---|---|---|---|---|---|---|---|---|
| C0 | 43.46 | 28.05 | 48.04 | 34.64 | 33.73 | 48.0 | ref | — |
| C4 | **38.64** | 25.53 | **21.24** | 33.94 | 28.72 | 70.0 | **−4.83 [−9.21, −0.53]** | 32 |
| C5 | 39.56 | 26.40 | 25.44 | 33.91 | 29.96 | 67.3 | −3.91 [−8.53, +0.61] | 37 |

H3 at longer context: C5→C4 = **−0.92**, 95% CI [−3.53, +1.34], P(improve)=0.768.

**This is the clearest result in the project, for three reasons.**

1. **Absolute WER halves — 85.69 → 43.46 at baseline** (WER-sa 71.87 → 28.05).
   That is strong evidence the headline numbers on the main set were inflated by
   short spans and 1 s-rounded boundary bleed, not by anything the method does.
   43.5% WER / 28.1% script-agnostic for zero-shot turbo on code-switched lecture
   audio is a believable figure; 85.7% was not.

2. **SGCD significantly improves WER here, with no guard at all.** C4 = −4.83 with
   a CI excluding zero, K-WER 48.04 → 21.24 (56% relative), script fidelity
   48% → 70%, and it wins on 55% of utterances against losing on 32%. On the main
   (short-utterance) set C4 alone was +5.11 and needed the guard to reach a net
   win. Context length, not the guard, is what the method actually needed.

3. **The content-specificity question gets a quantitative answer at last.** The
   direction finally favours matched syllabi, but the split is lopsided: of C4's
   4.83-point gain, the mismatched syllabus already delivers 3.91 points. Content
   contributes **0.92 points — roughly 19% of the effect — and its CI still spans
   zero.** So even under conditions maximally favourable to the hypothesis, the
   honest statement is: *the benefit of syllabus prompting for code-switched
   lecture ASR is predominantly a format/register effect, with at most a small
   content contribution that a 100-utterance evaluation cannot resolve.*

Estimating the sample needed to resolve a 0.92-point effect with this variance
(CI half-width ≈ 2.4 at n=100) puts it at roughly n≈700 pseudo-utterances — about
5 hours of audio, i.e. the entire SLR104 test split. That is the concrete
follow-up, and it is cheap: 5 h of audio at 15x realtime is ~20 minutes per
condition.

---

## Summary of what this project found

**Supported:** naive keyword-list prompting reproduces the known failure mode in a
new language setting (C2: K-WER −52% relative, U-WER +22% relative, overall WER
+50.53 with CI excluding zero); prose rendering repairs most of it (H4, −36.88);
prompting substantially improves technical-term recognition (H2, K-WER −46%
relative) and script fidelity (32.6% → ~60%).

**Not supported:** that the benefit comes from *this course's* syllabus content.
Across three settings the mismatched-syllabus control captures most or all of the
gain — turbo/short: C5 better than C4 by 3.09; turbo/long: C5 captures 81% of C4's
gain; only small/short favours matched (−17.52, n.s.). The mechanism is
principally teaching Whisper the corpus's output convention.

**Method-level:** the confidence guard converts prompting's regressions into a net
win on short utterances (−14.19, CI excludes zero) but is model-specific and does
not transfer across checkpoints. At realistic lecture-span lengths the guard is
not needed for a significant gain.




---

## 2026-09-01 — Cross-model ASR benchmark (5 checkpoints, zero-prompt)

**Git:** `6dad8ea` · **Hardware:** MacBook Air M3, 8 GB unified memory, macOS 26.6
**Data:** SLR104 Hindi–English TEST, frozen `manifest.jsonl`, n=150 lecture-disjoint
utterances, 862.0 s audio (mean 5.75 s). Same seed/filters as the SGCD study.

```bash
./run_bench_all.sh                       # fetch + decode + score, one process per model
python src/bench_report.py --split test
```

New code: `src/bench_asr.py` (multi-backend decode → study-compatible hyp cache),
`src/bench_report.py` (accuracy + speed + memory table), `run_bench_all.sh`.

All models decoded **zero-prompt** — the only condition comparable across
Whisper's `initial_prompt`, Qwen's `prompt=` hotwords, and Parakeet's no-context
transducer. Whisper kept the study's frozen C0 settings (`language="hi"`,
`temperature=0.0`, `condition_on_previous_text=False`); Qwen used
`language="Hindi"`; Parakeet has no language control.

| Model | WER % | WER-sa % | CER % | K-WER % | U-WER % | script fid % | RTF | x RT | peak RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR-0.6B (torch/MPS) | **59.74** | 50.39 | **50.49** | 44.66 | 39.92 | 47.2 | 0.339 | 2.95 | 1.41 |
| Qwen3-ASR-1.7B (torch/MPS) | 63.24 | **47.94** | 57.26 | 92.23 | **39.45** | 8.4 | 1.209 | 0.83 | 1.26 |
| Whisper large-v3 (fp16 MLX) | 74.17 | 61.94 | 62.39 | 63.59 | 42.63 | 34.4 | 0.761 | 1.31 | 0.61 |
| Whisper large-v3-turbo (MLX) | 80.23 | 66.61 | 65.62 | 65.53 | 43.71 | 32.8 | 0.229 | 4.36 | 0.58 |
| Parakeet RNNT 1.1B (MLX) | 97.45 | 81.93 | 81.53 | **37.86** | 91.54 | **58.8** | 0.076 | 13.18 | 0.99 |

**Harness validation.** turbo C0 K-WER = 65.53% here vs 65.5% logged for the
2026-08 TEST run — the rebuilt manifest reproduces the frozen eval set exactly.

**Findings.**

1. **Qwen3-ASR-0.6B wins on strict WER by 14.4 points over the best Whisper.**
   The gap is concentrated in keywords (K-WER 44.7 vs 63.6) and script fidelity
   (47.2% vs 34.4%), not in ordinary words (U-WER 39.9 vs 42.6). This is a
   training-data effect, not capacity: scaling turbo→large-v3 buys 6.1 points,
   switching model family buys 14.4.

2. **The 1.7B's strict WER is a script artifact, not a recognition failure.**
   Forced to `language="Hindi"` it transliterates every English technical term
   into Devanagari ("वन इंडेक्स वन" for "one index one"), collapsing script
   fidelity to 8.4% and K-WER to 92.2%. Its **WER-sa of 47.94 is the best of all
   five**, and its U-WER (39.45) is also the best — it hears the most and writes
   it in the wrong alphabet for this corpus. Untested: whether auto-detect
   (`--language none`) preserves Latin script. `run_lang_variants.sh` is written
   and ready; the run was interrupted before producing output.

3. **Parakeet's K-WER (37.86%, best in table) and script fidelity (58.8%, best)
   are metric artifacts and must not be read as quality.** It is English-only
   (`nvidia/parakeet-rnnt-1.1b` is tagged `en`; no multilingual 1.1B RNNT exists,
   and no Parakeet of any size supports Hindi). It transcribes English spans
   accurately and renders Devanagari spans as Latin phonetic noise — U-WER 91.54%
   is the honest figure. It wins K-WER because syllabus keywords are ~all English,
   and wins script fidelity because it cannot emit Devanagari at all.

4. **Speed ranking is inverse to nothing useful.** Parakeet 13.2x RT, turbo 4.4x,
   Qwen-0.6B 3.0x, large-v3 1.3x, Qwen-1.7B 0.83x — the 1.7B is *slower than
   realtime* on this machine and unusable for batch lecture transcription here.

**Implication for ClassScribe.** The pipeline's ASR backend (currently Whisper
turbo) should move to Qwen3-ASR-0.6B: ~20 WER points on this corpus at 3x
realtime in 1.4 GB. Qwen exposes a `prompt=` context slot, so the SGCD
syllabus-grounding method transfers — but every SGCD number would need re-running,
since those results are Whisper-prompt-specific.

**Caveat.** Absolute WERs are inflated by the 2–28 s utterance segmentation
(see the `testcat` 26 s pseudo-utterance run, where baseline WER roughly halves).
Rankings, not levels, are what this table supports.


## FYRP — 10 independent improvement methods on Qwen3-ASR-0.6B (2026-10-01)

Code: `src/fyrp_{exp,asr,lm,lexicon}.py`, driver `run_fyrp.sh`. Test = 50 utts (every 3rd of frozen 150-utt TEST eval set); tune = 30 DEV utts; training/LM/glossary text = DEV-split lectures only (0 lecture overlap with TEST). Each method applied alone to the baseline. Full table + settings: `out/fyrp/results.md`.

| Rank | Step | Method | WER % | Word Acc % | ΔWER vs base (95% CI) | CER % | WER-sa % | K-WER % | U-WER % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S5 | N-best rescoring with domain n-gram LM | 55.69 | 44.31 | -3.18 [-6.3, -0.2] | 43.84 | 47.14 | 42.17 | 34.95 | 1.36 |
| 2 | S8 | TTS synthetic data (partial FT on TTS only) | 55.85 | 44.15 | -3.01 [-6.8, +0.8] | 43.78 | 50.34 | 20.48 | 39.03 | 0.61 |
| 3 | S3 | Context biasing via system prompt | 57.36 | 42.64 | -1.51 [-4.8, +1.3] | 46.73 | 48.99 | 49.40 | 34.95 | 0.66 |
| 4 | S4 | BM25 retrieval-constrained span correction | 58.03 | 41.97 | -0.84 [-2.0, +0.2] | 46.29 | 49.83 | 48.19 | 35.73 | 0.29 |
| 5 | S0 | Baseline (Qwen3-ASR-0.6B, greedy, no context) | 58.86 | 41.14 | — | 46.70 | 50.00 | 50.60 | 36.31 | 0.29 |
| 6 | S7 | Embedding-only tuning (tokenizer expansion) | 58.86 | 41.14 | +0.00 [+0.0, +0.0] | 46.70 | 50.00 | 50.60 | 36.31 | 0.63 |
| 7 | S1 | Fuzzy / edit-distance lexicon correction | 58.86 | 41.14 | +0.00 [-0.8, +0.8] | 46.79 | 49.83 | 49.40 | 36.50 | 0.29 |
| 8 | S6 | Shallow fusion (n-gram LM in decoding) | 60.03 | 39.97 | +1.17 [-1.8, +4.4] | 49.65 | 50.67 | 62.65 | 36.12 | 0.29 |
| 9 | S2 | Phonetic matching for code-switched OOV | 60.37 | 39.63 | +1.51 [-0.5, +3.5] | 46.79 | 51.01 | 42.17 | 39.42 | 0.29 |
| 10 | S10 | LoRA r=8 enc+dec attention (real+TTS) | 62.21 | 37.79 | +3.34 [-9.0, +22.3] | 48.67 | 56.06 | 39.76 | 38.83 | 0.65 |
| 11 | S9 | Partial FT, last-4 decoder layers (real+TTS) | 67.22 | 32.78 | +8.36 [-0.6, +18.9] | 55.56 | 58.59 | 31.33 | 40.39 | 0.65 |

Notes: S7 output identical to baseline on 50/50 (14 new tokens never emitted). S9/S10 best K-WER but 3/2 repetition-loop runaways inflate WER — next run should add `repetition_penalty`/length guard. S2 helped DEV (-1.9) but hurt TEST (+1.5): 30-utt tune set too small. Only S5 CI excludes 0.
