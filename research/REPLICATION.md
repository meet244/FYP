# Replication guide

How to reproduce the three-way comparison table in `sgcd/COMPARISON.md` from
scratch, and how to check that what you got matches what we report.

Everything runs on one laptop. No training, no GPU cluster, no API keys.

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
**Test tarball only** — the training split is never used.

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
