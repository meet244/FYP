# ClassScribe research results

Documentation review: 2026-10-10. This report consolidates saved experiments and
the supporting material in [dump/](dump/). No new decoding, training, or
classroom-quality experiment was performed for this update.

## Experiments must be read separately

| Experiment | Model / condition | Evaluation sample | Primary evidence |
| --- | --- | --- | --- |
| Cross-model benchmark | Five checkpoints, zero prompt | 150 Hindi-English utterances; 862 seconds of audio | [Benchmark table](research/sgcd/out/tables/bench__test.md) and [JSON](research/sgcd/out/bench_scores__test.json) |
| FYRP improvement comparison | Qwen3-ASR-0.6B, S0 plus ten independent methods | 50 utterances from the frozen 150-utterance TEST subset | [Detailed report](research/sgcd/out/fyrp/results.md) and [JSON](research/sgcd/out/fyrp/results.json) |
| Original SGCD study | Whisper turbo, alternative decoder contexts | 150 short utterances and a separate 100-span concatenated evaluation | [Comparison](research/sgcd/COMPARISON.md) and [run log](research/sgcd/RUNLOG.md) |

The SLR104 Hindi-English test tarball is internally split into lecture-disjoint
DEV and TEST sets; the official training tarball is not used. FYRP training
uses the internal DEV lectures, not TEST references.

Saved metrics are present for all three experiments. Original Whisper
hypotheses are available, but Qwen/Parakeet benchmark hypotheses and the FYRP
hypothesis directory are absent in this checkout. The later numbers can be
checked against saved JSON; independent rescoring requires recreating outputs.

## 1. Five-model baseline benchmark

Recorded hardware: MacBook Air M3 with 8 GB unified memory. Each checkpoint runs
in its own process on the same frozen audio. Whisper uses Hindi transcription;
Qwen uses the forced Hindi language hint; Parakeet is an English-only checkpoint.

| Model | WER % | WER-sa % | CER % | K-WER % | U-WER % | Speed | Peak process RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 59.74 | 50.39 | 50.49 | 44.66 | 39.92 | 2.95x | 1.41 |
| Qwen3-ASR-1.7B | 63.24 | 47.94 | 57.26 | 92.23 | 39.45 | 0.83x | 1.26 |
| Whisper large-v3 | 74.17 | 61.94 | 62.39 | 63.59 | 42.63 | 1.31x | 0.61 |
| Whisper large-v3-turbo | 80.23 | 66.61 | 65.62 | 65.53 | 43.71 | 4.36x | 0.58 |
| Parakeet RNNT 1.1B | 97.45 | 81.93 | 81.53 | 37.86 | 91.54 | 13.18x | 0.99 |

Speed is audio duration divided by decode time. Process RSS is not a complete
measurement of GPU/unified-memory use. These measurements describe this run.

Qwen3-ASR-0.6B has the lowest strict WER: **59.74%**. Its gap to Whisper large-v3
is **14.43 percentage points**, and to turbo is **20.49 points**. These are
benchmark differences, not measured gains from changing the application.

Qwen 1.7B has the lowest transliteration-tolerant WER (47.94%) and non-keyword
error (39.45%), but often writes English terms in Devanagari with forced Hindi.
Auto-language control still needs a completed held-out evaluation. Parakeet's
low keyword error must be read alongside its **91.54% non-keyword error**:
English terminology alone does not establish bilingual recognition quality.

The original SGCD short-utterance table records turbo C0 at **85.69%**, while
this later benchmark records **80.23%**. These are separate saved runs; the
reviewed material does not establish the exact cause of the difference.
Use each experiment's own baseline rather than combining the figures.

## 2. Ten independent Qwen improvement methods

Settings were selected on 30 DEV tuning utterances and frozen. Each method is
applied independently to S0; methods are not stacked. There are **11 rows:
the baseline plus ten alternatives**. The glossary and language-model text
exclude tuning references and use DEV lecture text plus research syllabuses.

| Rank | Method | WER % | Delta vs S0 (points) | Reported 95% CI | K-WER % |
| ---: | --- | ---: | ---: | --- | ---: |
| 1 | S5: N-best rescoring with domain n-gram LM | 55.69 | -3.18 | [-6.3, -0.2] | 42.17 |
| 2 | S8: TTS synthetic data (partial FT on TTS only) | 55.85 | -3.01 | [-6.8, +0.8] | 20.48 |
| 3 | S3: Context biasing via system prompt | 57.36 | -1.51 | [-4.8, +1.3] | 49.40 |
| 4 | S4: BM25 retrieval-constrained span correction | 58.03 | -0.84 | [-2.0, +0.2] | 48.19 |
| 5 | S0: Baseline (Qwen3-ASR-0.6B, greedy, no context) | 58.86 | - | - | 50.60 |
| 6 | S7: Embedding-only tuning (tokenizer expansion) | 58.86 | +0.00 | [+0.0, +0.0] | 50.60 |
| 7 | S1: Fuzzy / edit-distance lexicon correction | 58.86 | +0.00 | [-0.8, +0.8] | 49.40 |
| 8 | S6: Shallow fusion (n-gram LM in decoding) | 60.03 | +1.17 | [-1.8, +4.4] | 62.65 |
| 9 | S2: Phonetic matching for code-switched OOV | 60.37 | +1.51 | [-0.5, +3.5] | 42.17 |
| 10 | S10: LoRA r=8 enc+dec attention (real+TTS) | 62.21 | +3.34 | [-9.0, +22.3] | 39.76 |
| 11 | S9: Partial FT, last-4 decoder layers (real+TTS) | 67.22 | +8.36 | [-0.6, +18.9] | 31.33 |

**S5 is the only method whose reported paired 95% interval lies entirely below
zero.** WER falls from 58.86% to 55.69%, a recorded delta of -3.18 percentage
points, approximately **5.4% relative error reduction**. These are
utterance-level paired bootstrap estimates (2,000 resamples, seed 0), not
evidence of robustness across classrooms or new subjects.

S8 has the lowest keyword error, 20.48%, but its overall-WER interval includes
zero. S9 and S10 reduce keyword error but have worse overall WER, with repetition
runaways reported in three and two outputs respectively. S7 emits the same test
hypotheses as S0. S2 improves tuning WER but worsens TEST WER; S6 also worsens the
recorded point estimate. These outcomes do not support deploying every method.

S5's reported research RTF is **1.36**, versus **0.29** for S0; lower is faster.
The accuracy improvement has a decoding-cost tradeoff. The dump's informal
wall-clock account and saved per-stage training times measure different
intervals. See the [detailed report](research/sgcd/out/fyrp/results.md).

## 3. Original Whisper syllabus-context experiment

This table is for **100 concatenated spans averaging 26.2 seconds**, not the
short-utterance benchmark.

| Condition | WER % | K-WER % | U-WER % |
| --- | ---: | ---: | ---: |
| C0: no context | 43.46 | 48.04 | 34.64 |
| C2: terminology list | 62.86 | 23.00 | 46.42 |
| C3: whole-syllabus narration | 37.23 | 21.38 | 32.42 |

Whole-syllabus narration reduces WER by **6.23 percentage points**, reported
paired 95% CI **[-10.59, -1.97]**. Retrieved narration (C4) records **38.64% WER**,
a different condition from the whole-syllabus result.

A terminology list improves keyword matching but harms the rest of the
transcription. Narration avoids that tradeoff in this experiment. A mismatched
syllabus reproduces roughly **81% of the retrieved-narration gain**, supporting
a substantial format/register effect rather than proving semantic correction
from the correct syllabus.

On short utterances, C3 records 99.35% WER against C0's 85.69%. The long-span
finding is not a universal prompting gain or a measured Qwen result. See the
[original comparison](research/sgcd/COMPARISON.md) for settings and controls.

## 4. Current application

The backend defaults to **Qwen3-ASR-0.6B with S5**: five beam candidates, a frozen
token-trigram LM, weight 0.2, and word-length bonus 0.5. S5, greedy baseline, and
syllabus-conditioned SGCD are separate selections. Serving uses roughly
25-second spans and a 512-token cap; FYRP used short utterances and a 200-token cap.

The LM artifact records provenance and tokenizer identity; jobs retain its
SHA-256. Serving S5's comparison transcript is the unrescored top beam, not a
separate S0 greedy result. See the [artifact contract](backend/assets/README.md).

Structured notes, unit-linked coverage, cited chat, and Studio revision outputs
are implemented. Their existence does not measure note fidelity, learning
gains, coverage correctness, or citation accuracy.

## 5. Supporting material reviewed

| Material | Contribution and limit |
| --- | --- |
| [Methods report](dump/all-10-methods-ran-on-qwen3-asr-06b.txt) | Historical FYRP setup and outcomes. Damaged table text and conversational fragments make saved JSON the numerical authority. |
| [Benchmark image](dump/image.png) | Corroborates the saved five-model result; not another experiment. |
| [Methods image](<dump/WhatsApp Image 2026-10-01 at 23.30.27.jpeg>) and [setup image](<dump/WhatsApp Image 2026-10-01 at 23.30.28.jpeg>) | Corroborate methods and reduced training scope for limited memory. Suggested reruns remain untested. |
| [Course-chat screenshot](<dump/image (1).png>) and [subject-library screenshot](<dump/image (2).png>) | Show subject organisation, mixed source types, and cited answers, not measured answer quality. |
| [Demonstration video](<dump/WhatsApp Video 2026-10-04 at 14.30.11.mp4>) | Shows a moving-average question, cited notes/transcripts, and timestamp-linked playback/navigation. Workflow evidence, not a labelled accuracy evaluation. |
| [Literature-review workbook](dump/Lecture-Note-Taking-Literature-Review_Enriched.xlsx) | Prior-paper summaries and assessed limitations; 32 entries include one duplicate. Actual worksheet row numbers are used in the gap mapping. |
| [Research tutorial](<dump/RE Tutorial 1.pdf>) | Original motivation and Whisper proposal. Publication years conflict with the workbook; its multi-perspective title exceeds implemented scope. |

The [research overview](research/README.md) maps supported gaps and flags
bibliography conflicts. The source PDF, workbook, images, video, historical
run log, and experimental JSON are preserved.

## 6. Outstanding evidence

- Serving S5 WER on long classroom recordings and new domains, including
  noise, accents, interruptions, and silence.
- Human-labelled note, question, competency-statement, coverage, and citation quality.
- Propagation of ASR errors into generated study material.
- Student learning gains, workload reduction, and accessibility outcomes.
- Qwen syllabus-conditioning gains and completed language-hint ablations.
- Export/reload and stability validation of fine-tuned S7-S10 variants.
- Student/tutor perspective-conditioned summarisation, continuous live capture,
  and public multi-user deployment.

See [replication](research/REPLICATION.md) and
[implementation status](backend/IMPLEMENTATION_STATUS.md).
