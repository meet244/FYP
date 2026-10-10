# Replication guide

Sections 0-8 reproduce the original Whisper comparison in `sgcd/COMPARISON.md`.
Sections 9-10 cover the later five-model benchmark and ten-method Qwen study.
See [results](../results.md) for the separate evaluation samples.

The original Whisper study requires no training or model API keys. Later FYRP
includes training variants. All studies target one laptop; their runtimes and
dependencies differ from the application.

---

## 0. What you need

| Requirement | Notes |
|---|---|
| Apple Silicon Mac | The decoder runs on MLX. On other hardware see §7 |
| Python 3.11+ | 3.11.9 used for the published run |
| ~3 GB free disk | 443 MB corpus + ~1.7 GB model weights + extracted audio |
| Internet | Corpus download and one-time model download |
| ~45 min wall clock | Mostly downloads; decoding is ~12 min |

---

## 1. Environment

```bash
cd research/sgcd
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install mlx-whisper jiwer soundfile numpy pandas \
                        scikit-learn scipy regex tiktoken indic-transliteration
```

All later commands are run from `research/sgcd/src`:

```bash
cd src
```

Use `../.venv/bin/python` so the virtual environment is used without activating it.

---

## 2. Get the corpus

MUCS 2021 subtask-2 Hindi–English, OpenSLR SLR104, CC BY-SA 4.0.
**Test tarball only** — the official training tarball is never used. The
internal DEV/TEST partition of this tarball is used below; FYRP later trains
on its internal DEV lectures.

```bash
mkdir -p ../data && cd ../data
curl -L -O https://openslr.elda.org/resources/104/Hindi-English_test.tar.gz
tar -xzf Hindi-English_test.tar.gz
cd ../src
```

Expected: 443,929,204 bytes, extracting to `data/test/` with 30 `.wav` files and
`data/test/transcripts/{text,segments,wav.scp,utt2spk,spk2utt}`.

Confirm the layout matches what we found:

```bash
../.venv/bin/python inspect_data.py | head -40
```

You should see 30 wav files, 3136 lines in `text`, and 3136 lines in `segments`.

---

## 3. Build the frozen evaluation set

```bash
../.venv/bin/python build_manifest.py
```

**Expected output — these numbers must match exactly.** They are deterministic
(seed 1337) and depend only on the corpus, not on your hardware:

```
utterances=3073  lectures=30  split={'dev': 967, 'test': 2106}  skipped={'duration': 17, 'too_few_words': 46}
eval subset: {'dev': 60, 'test': 150}  (dev lectures=9, test lectures=21)
total audio = 5.06 h   median dur = 5.0 s
```

If these differ, stop — the corpus you downloaded is not the one we used.

Then build the lecture-length span set used for the second table:

```bash
../.venv/bin/python build_concat_manifest.py
```

Expected: `pseudo-utterances=400  eval={'devcat': 40, 'testcat': 100}`,
mean duration 26.2 s, mean 55.1 reference words.

---

## 4. Check the syllabi

The seven course syllabi are committed in `syllabi/`, so you do not need to
regenerate them. Validate them and confirm every lecture maps to a course:

```bash
../.venv/bin/python make_syllabi.py --validate
../.venv/bin/python map_lectures.py --validate
```

Expected: `all syllabi valid` (7 files, 62 units, 74–105 tokens each) and
`mapping complete` (30 lectures over 7 courses).

**Leakage check.** The syllabi were written from lecture titles only. Titles were
read from each lecture's opening utterances, and `build_manifest.py` marks those
`title_source` and excludes them from the evaluation sample. To see the audit
trail:

```bash
../.venv/bin/python map_lectures.py --show | head -20
```

---

## 5. Decode

Model weights (~1.6 GB) download automatically on first use.

```bash
# Table 1 — utterance-level segments (150 utterances)
../.venv/bin/python decode.py --model turbo --split test --conditions C0 C2 C3

# Table 2 — lecture-length spans (100 spans)
../.venv/bin/python decode.py --model turbo --split testcat --conditions C0 C2 C3
```

Runtime on an M-series MacBook: roughly 2.5 min per condition on `test` and
3 min per condition on `testcat`, so about 17 min total plus the download.

Hypotheses are cached in `out/hyps/`. Re-running is free; delete the relevant
`.jsonl` or pass `--force` to redecode.

---

## 6. Score, and verify against our numbers

```bash
../.venv/bin/python score.py
../.venv/bin/python stats.py --model turbo --split test
../.venv/bin/python stats.py --model turbo --split testcat
../.venv/bin/python make_comparison.py            # regenerates COMPARISON.md
../.venv/bin/python make_comparison.py --verify   # checks it against our run
```

`--verify` performs two independent checks and exits non-zero if either fails.

**Numeric check** — every WER, term error, non-term error and script-fidelity
value against the published run, default tolerance ±1.0 WER point:

```
=== numeric check (tolerance ±1.00 WER points) ===
  all values within tolerance of the published run
```

**Qualitative check** — the nine claims the paper actually makes. These must hold
for any valid replication even if absolute values shift:

```
  PASS  [test]    B (terminology list) is WORSE than A on overall WER
  PASS  [test]    B improves technical-term error over A
  PASS  [test]    B degrades non-term error over A (the trade-off)
  PASS  [test]    C (narration) beats B on overall WER
  PASS  [test]    C does NOT degrade non-term error (unlike B)
  PASS  [testcat] C beats A on overall WER at lecture-length spans
  PASS  [testcat] B is still worse than A at lecture-length spans
  PASS  [testcat] C improves technical-term error at lecture-length spans
  PASS  [testcat] C improves script fidelity over A
RESULT: replication verified
```

Tighten the tolerance with `--tol 0.05` to demand near-exact agreement, which you
should get on the same hardware and library versions.

### The numbers you should obtain

| Setting | Method | WER | Term error | Non-term error | Script fidelity |
|---|---|---|---|---|---|
| Utterance-level | A no context | 85.69 | 65.53 | 43.50 | 32.6% |
| (150 utts, 5.7 s) | B terminology list | 136.22 | 31.55 | 53.18 | 60.9% |
| | C narration | 99.35 | 33.50 | 41.68 | 60.0% |
| Lecture-length | A no context | 43.46 | 48.04 | 34.64 | 48.0% |
| (100 spans, 26.2 s) | B terminology list | 62.86 | 23.00 | 46.42 | 70.3% |
| | C narration | **37.23** | **21.38** | **32.42** | **71.6%** |

---

## 7. Notes on exactness

**What is bit-exact.** The evaluation set, the dev/test split, the sampling, the
retrieval, the scoring and the bootstrap are all deterministic given seed 1337.
Anyone running §3 must get identical manifest counts.

**What may drift slightly.** Decoding is greedy (`temperature=0.0` as a scalar,
which disables the temperature-fallback loop), so it is deterministic for a fixed
model, runtime and hardware. Across MLX versions or different Apple Silicon
generations, floating-point differences can change a small number of tokens.
Because corpus WER here is sensitive to a handful of degenerate repetition
utterances (see below), a token-level difference can move aggregate WER by a few
tenths of a point. This is why `--verify` defaults to ±1.0 point and why the
qualitative claims are checked separately — those are the results the paper rests
on.

**Why absolute WER is high.** Two deliberate properties, both documented:

1. 99.6% of the corpus's segment boundaries are rounded to whole seconds, so
   fragments of adjacent utterances intrude into every clip. Verify this yourself:
   `build_manifest.py` reports it, and the 26 s spans in Table 2 roughly halve
   baseline WER precisely because the effect is amortised over longer spans.
2. The temperature-fallback loop that normally repairs repetition is disabled by
   design, so conditioning-induced instability remains measurable instead of
   being silently patched.

Both apply identically to all three methods, so the comparison is unaffected.
These numbers are **not** comparable to published MUCS leaderboard results.

**Non-Apple hardware.** Replace the `mlx_whisper.transcribe` call in
`src/decode.py` with `faster-whisper`; both expose the same decoder-context
interface and the same decoding parameters. Absolute values will differ; the
qualitative claims should not.

---

## 8. Going further

Reproduce the rest of the study, all cached and scored the same way:

```bash
# full condition set incl. the mismatched-syllabus control (C5) and oracle (C6)
../.venv/bin/python decode.py --model turbo --split test

# second model scale, for the generalisation row
../.venv/bin/python decode.py --model small  --split test

# development sweep that selected the frozen configuration
../.venv/bin/python sweep_dev.py --model turbo

# qualitative error analysis: where conditioning wins and loses
../.venv/bin/python error_analysis.py --model turbo --cond C4
```

Condition definitions are in `PREREGISTRATION.md`; every run and every number is
recorded in `RUNLOG.md`, including the hypotheses that failed.

---

## 9. Five-model baseline benchmark

This checkout contains saved benchmark metrics but not the Qwen or Parakeet
hypothesis caches. Original Whisper hypotheses are available. Recreating all
five model outputs is necessary to independently rescore the later benchmark;
the saved metrics alone are not a fresh reproduction.

Use the same frozen manifest, normalisation, and lecture split from sections
2-4. The recorded test subset has 150 utterances and 862 seconds of audio.
The benchmark ran one model per process on an M3 Air with 8 GB unified memory.

The original Whisper-only installation is insufficient for all later methods:

| Capability | Additional environment requirement |
| --- | --- |
| Qwen decoding/training | Torch and Transformers with the Qwen processor/model APIs used by the historical scripts; checkpoint weights |
| Script-tolerant scoring | indic-transliteration, already included in section 1 |
| FYRP text correction | rapidfuzz |
| Parakeet checkpoint | parakeet-mlx on Apple silicon |
| Driver downloads | Hugging Face CLI; the driver sets an HF transfer option, which may require its corresponding transport dependency |

The serving dependencies are documented in [backend requirements](../backend/requirements.txt)
and [optional ASR requirements](../backend/requirements-asr-extra.txt), but those
are not an exact lockfile for the historical experiments. Verify the APIs used
in the research scripts when recreating an environment. No fully pinned
historical research environment is supplied.

From the repository root, with the research virtual environment prepared:

~~~bash
cd research/sgcd/src
../.venv/bin/python bench_asr.py --model qwen06 --split test
../.venv/bin/python bench_asr.py --model qwen17 --split test
../.venv/bin/python bench_asr.py --model wlv3 --split test
../.venv/bin/python bench_asr.py --model turbo --split test
../.venv/bin/python bench_asr.py --model parakeet --split test
../.venv/bin/python bench_report.py --split test
~~~

These are sequential commands. Qwen defaults to the forced Hindi hint; Whisper
uses its frozen Hindi C0 settings; Parakeet is English-only. Caches are stored
under out/hyps with model-specific C0 names and runtime JSON under out/.

The convenience driver can also be run from the repository root:

~~~bash
bash research/sgcd/run_bench_all.sh
~~~

**Driver limitation:** it loops over turbo, wlv3, qwen17, and parakeet, assuming
qwen06 was already decoded. Run the explicit qwen06 command above before using
the driver for a complete five-model report. A missing cache otherwise produces
a report with fewer models. This documentation update does not change the driver.

Expected strict WER values, in model order qwen06, qwen17, wlv3, turbo, parakeet:
**59.74, 63.24, 74.17, 80.23, 97.45%**. Verify all five rows, sample counts, and
language settings against [saved JSON](sgcd/out/bench_scores__test.json) and the
[benchmark table](sgcd/out/tables/bench__test.md). Report regeneration rescores
cached hypotheses; it does not establish a fresh decode or runtime measurement.

The original SGCD turbo C0 is 85.69%, unlike the later benchmark's 80.23%.
Keep the experiment-specific baselines; the reviewed sources do not resolve
the exact cause of that difference. The original comparison verifier does not
validate this later benchmark.

## 10. FYRP: ten independent Qwen methods

The saved results and training metadata are present, but the FYRP hypothesis
directory is absent in this checkout. The commands below therefore require
decoding and training to recreate those outputs; do not assume a cached rerun.

The experiment uses 50 TEST utterances (every third of the frozen 150), 30 DEV
tuning utterances, and 100 real training examples from internal DEV lectures.
The glossary and LM text exclude tuning references. The training pool excludes
the 60-utterance DEV evaluation subset. No TEST reference trains the recogniser.

Preparation generates up to 120 TTS examples using macOS's Hindi voice Lekha;
the training variants use the first 80. Full S8-S10 reproduction therefore
requires macOS, that voice, and sufficient memory. Missing hypotheses for
trained variants cannot be rebuilt from exported trained weights: the current
scripts do not export those weights.

From the repository root:

~~~bash
bash research/sgcd/run_fyrp.sh
~~~

Or run stages individually, from research/sgcd/src:

~~~bash
../.venv/bin/python fyrp_exp.py prep
../.venv/bin/python fyrp_exp.py decode
../.venv/bin/python fyrp_exp.py train --method emb
../.venv/bin/python fyrp_exp.py train --method tts
../.venv/bin/python fyrp_exp.py train --method partial
../.venv/bin/python fyrp_exp.py train --method lora
../.venv/bin/python fyrp_exp.py report
~~~

Decode produces S0, S3, S5 beam candidates, and S6. Training produces S7-S10
hypotheses; report applies the text corrections and S5 rescoring, then writes
results.json and results.md. Caches are reused, but missing training results
require training again. Report generation still loads the tokenizer and builds
the LM; it is not a pure Markdown formatter.

Expected output is **11 rows**, baseline plus ten independent alternatives:

- S0 WER: **58.86%**.
- S5 WER: **55.69%**, delta **-3.18 percentage points**, reported 95% CI
  **[-6.3, -0.2]**.
- S8 WER: **55.85%**, interval includes zero.
- S5 is the only alternative with its reported interval entirely below zero.

Check every row and its frozen settings against
[results.json](sgcd/out/fyrp/results.json). The FYRP bootstrap uses 2,000
utterance resamples, seed 0; original SGCD uses 10,000, seed 1337.
The [detailed report](sgcd/out/fyrp/results.md) explains the metric and runtime
limits. Its narrative sections are overwritten by the current generator;
preserve/reapply that commentary after regeneration. Generator code is unchanged.

## 11. Serving is a separate validation target

The [application S5 artifact](../backend/assets/README.md) exports frozen LM
counts, not trained recogniser weights. Serving uses roughly 25-second spans
and a 512-token cap, whereas FYRP uses a 200-token cap. Its comparison output
is an unrescored beam, not an independent S0 greedy result.

Do not apply original Whisper gains to Qwen, compare absolute scores against
unmatched literature datasets, or present research RTF as application latency.
Long-recording WER, new-domain transfer, note fidelity, citation accuracy,
coverage correctness, and student learning outcomes need separate evaluation.
The demonstrations in [dump/](../dump/) establish workflows, not those metrics.
