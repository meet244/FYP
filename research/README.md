# Research

Research component of the ClassScribe final-year project: syllabus-grounded
contextual decoding for Hindi–English code-switched lecture transcription.

## Contents

| Path | What it is |
|---|---|
| [`REPLICATION.md`](REPLICATION.md) | **Start here to reproduce the results.** Step-by-step, with a verification script |
| [`paper/`](paper/) | The IEEE-format research paper (LaTeX) and its figures |
| [`sgcd/`](sgcd/) | Code, syllabi, cached decodes and all results |
| [`design/`](design/) | The original implementation plan, kept as the historical design record |
| [`reference/`](reference/) | Formatting reference paper |

## The result in one table

Three ways of supplying course-syllabus context to the decoder, on
lecture-length spans (100 spans, MUCS 2021 Hindi–English):

| Method | WER ↓ | Technical-term error ↓ | Everything-else error ↓ |
|---|---|---|---|
| A — no context | 43.46 | 48.04 | 34.64 |
| B — terminology list | 62.86 | 23.00 | 46.42 |
| C — syllabus as narration | **37.23** | **21.38** | **32.42** |

Enumerating terminology halves the error on the words it targets and makes
everything else worse, so overall accuracy drops. Rendering the same content as
fluent code-mixed narration achieves the same terminology gain with no collateral
damage. Full tables in [`sgcd/COMPARISON.md`](sgcd/COMPARISON.md).

The honest caveat, established by a mismatched-syllabus control: most of the
benefit comes from teaching the model the dual-script output convention rather
than from the syllabus content itself. A syllabus from the *wrong course*
reproduces roughly 81% of the gain. See [`sgcd/RUNLOG.md`](sgcd/RUNLOG.md).

## Reproducing

```bash
cd sgcd/src
../.venv/bin/python make_comparison.py --verify
```

Checks every published number against a fresh run and separately checks the nine
qualitative claims the paper makes. Exits non-zero on mismatch. Full instructions
in [`REPLICATION.md`](REPLICATION.md).

## Research discipline

- [`sgcd/PREREGISTRATION.md`](sgcd/PREREGISTRATION.md) — hypotheses and pass
  criteria, committed before the test set was decoded.
- [`sgcd/RUNLOG.md`](sgcd/RUNLOG.md) — append-only log of every run, including
  the pre-registered hypothesis that failed and why.
- The test split was decoded once, after the configuration was frozen on a
  lecture-disjoint development split.
