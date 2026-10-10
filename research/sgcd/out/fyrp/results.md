# FYRP: ten independent methods on Qwen3-ASR-0.6B

Experiment recorded: 2026-10-01. Documentation expanded: 2026-10-10.
Numerical values below are unchanged from [results.json](results.json).

## Design and primary finding

Each alternative is applied independently to the same Qwen3-ASR-0.6B baseline,
not stacked with other alternatives. Hyperparameters are tuned on 30 internal
DEV utterances and evaluated on 50 TEST utterances (every third item of the
frozen 150-utterance test subset). DEV and TEST lectures are disjoint.

The glossary and LM corpus use internal DEV reference text and research
syllabuses, excluding the tuning references. Fine-tuning uses 100 real DEV
examples, 80 synthetic examples, or both, according to the method. These are
internal splits of the SLR104 test tarball, not the official training corpus.

**S5 reduces WER from 58.86% to 55.69%**, a recorded paired delta of
**-3.18 percentage points**, or approximately **5.4% relative error reduction**.
Its reported 95% interval is **[-6.3, -0.2] points**. It is the only alternative
whose reported interval excludes zero in the improving direction. This is a
small-sample result, not measured serving accuracy or general classroom robustness.

## Ranked results

There are eleven rows: S0 plus ten alternatives. Error rates are percentages;
lower is better. Delta is method WER minus S0 WER, in percentage points.

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

## Metrics and uncertainty

- WER counts substitutions, deletions, and insertions after the frozen text
  normalisation. It can exceed 100% when generated text has many insertions.
- Word Acc is the reported convenience quantity 100 minus WER, not a separate
  measurement of the proportion of correctly recognised reference words.
- CER is character error rate. WER-sa uses lossy cross-script phonetic
  normalisation and is a secondary diagnostic, not semantic accuracy.
- K-WER and U-WER partition errors aligned to keyword and other reference words.
  Insertions are separate, so these buckets do not account for every WER error.
  This subset has 83 keyword and 515 other reference tokens.
- RTF is decoding time divided by audio duration; below 1 is faster than audio.
  S5 records 1.36, versus S0's 0.29. Five-beam search costs more than greedy decoding.
- The paired bootstrap uses 2,000 utterance resamples, seed 0. Its 95% intervals
  describe this subset and procedure, not variation across independent classrooms.
  The reported intervals are not a multiple-comparison-adjusted significance test
  across all ten alternatives.

## Interpreting the alternatives

| Method | Observation and limit |
| --- | --- |
| S1: fuzzy correction | Overall WER is unchanged at the reported precision; small keyword changes do not imply overall improvement. |
| S2: phonetic correction | Tuning WER improves by roughly 1.9 points in the historical account, but TEST WER worsens by 1.51 points. A small tuning set does not establish generalisation. |
| S3: context biasing | Top-80 syllabus terms produce a lower point estimate; its interval includes zero. This Qwen experiment is separate from Whisper SGCD narration. |
| S4: BM25 correction | Retrieval-constrained correction lowers WER by 0.84 points; the interval includes zero. |
| S5: N-best rescoring | Five candidates are rescored with a token-trigram domain LM, weight 0.2, and word-length bonus 0.5. The only reported overall-WER interval entirely below zero. |
| S6: shallow fusion | Applies the LM during decoding, unlike S5's candidate rescoring. The tuned 0.1 weight yields worse TEST WER and keyword error. |
| S7: embedding tuning | Fourteen new tokens are added; none is emitted on TEST. All 50 hypotheses match S0. |
| S8: synthetic-only partial tuning | Tunes the last four decoder layers on 80 TTS examples. Lowest K-WER (20.48%), but the overall-WER improvement interval includes zero. It is a trained alternative, not just a data-preparation step. |
| S9: real-plus-TTS partial tuning | Tunes the last four decoder layers on 100 real plus 80 TTS examples. Three reported repetition runaways accompany worse overall WER. |
| S10: real-plus-TTS LoRA | Rank-8 adapters on encoder/decoder attention use 100 real plus 80 TTS examples. Two reported repetition runaways accompany worse overall WER. |

The historical methods report and screenshots in [dump/](../../../../dump/)
corroborate the table. Claims that repetition controls would improve S8-S10
remain untested. Lower keyword error alone does not establish better notes.

## Recorded training scope and cost

| Variant | Trainable parameters | Examples | Epochs | Learning rate | Saved training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| S7 | 14,336 | 100 real | 2 | 0.0005 | 3128.7 |
| S8 | 62,923,776 | 80 TTS | 1 | 0.00002 | 52.7 |
| S9 | 62,923,776 | 100 real + 80 TTS | 1 | 0.00002 | 82.8 |
| S10 | 1,662,976; 92 adapter modules | 100 real + 80 TTS | 1 | 0.0002 | 210.7 |

Synthetic speech is generated with the macOS Hindi voice Lekha. Training was
restricted to fit the recorded 8 GB machine. The dump estimates about 25 minutes
for decoding and about 70 minutes for training under memory pressure; the JSON
records individual training-loop durations shown above. These are different
timing scopes and must not be advertised as a general end-to-end latency result.

## Serving integration and remaining limitations

S5 is now the application default, not merely a proposed replacement. The
[frozen LM artifact](../../../../backend/assets/README.md) has DEV provenance,
tokenizer identity, and per-job SHA-256 validation. It contains counts, not
fine-tuned model weights.

Serving uses roughly 25-second spans and a 512-token cap; research used a
200-token cap. S5 uses no syllabus prompt and repetition penalty 1.0. Its
retained serving comparison is the unrescored top beam, not a separate S0
greedy decode. This study has not measured that serving pipeline's WER.

The domain counts cover the research tutorial domains. New subjects, long
classroom recordings, noise, accents, and downstream notes require evaluation.
S7-S10 write training metadata and test hypotheses, but no exported reusable
trained checkpoints. This checkout contains metrics and training metadata;
the FYRP hypothesis directory is absent. Independent rescoring requires
recreating hypotheses, including retraining where necessary. Those variants
are not deployed models.

## Reproduction and report regeneration

See the [replication guide](../../../REPLICATION.md). The driver is
[run_fyrp.sh](../../run_fyrp.sh), with individual stages in
[fyrp_exp.py](../../src/fyrp_exp.py). Existing hypotheses allow cached stages
to be reused; missing training hypotheses require retraining.

The current report generator writes the numerical table and settings to this
Markdown file. Regenerating it overwrites the explanatory sections added here.
The generator code is unchanged; retain this documentation separately or
reapply its commentary after regeneration. Experimental JSON is the numerical
authority.

## Frozen settings

The following metadata is preserved from the original report. LM test
perplexity is a recorded diagnostic, not a tuning criterion or deployment result.

```json
{
 "S1": "rapidfuzz ratio >= 85",
 "S2": "phonetic-key ratio >= 85",
 "S4": "BM25 top-3 of 299 segs, thr 80",
 "S5": "k=5 beams, lam=0.2, len bonus=0.5",
 "LM": {
  "test_ppl": 13.75252033934461
 },
 "S6": "LM weight 0.1 (dev grid {'0.0': 0.36486486486486486, '0.1': 0.3621621621621622, '0.3': 0.41081081081081083})",
 "S3": "top-80 syllabus terms as system prompt",
 "S7": {
  "new_tokens": 14,
  "trained_params": 14336,
  "examples": 100,
  "epochs": 2,
  "lr": 0.0005,
  "train_s": 3128.7
 },
 "S8": {
  "trained_params": 62923776,
  "examples": 80,
  "epochs": 1,
  "lr": 2e-05,
  "train_s": 52.7
 },
 "S9": {
  "trained_params": 62923776,
  "examples": 180,
  "epochs": 1,
  "lr": 2e-05,
  "train_s": 82.8
 },
 "S10": {
  "lora_modules": 92,
  "trained_params": 1662976,
  "examples": 180,
  "epochs": 1,
  "lr": 0.0002,
  "train_s": 210.7
 }
}
```
