# Syllabus-Grounded Contextual Biasing for Code-Switched Lecture ASR

Implementation of `Research_Implementation_Plan.pdf` (v1.0): a controlled comparison of
**where** syllabus knowledge should be injected into an ASR pipeline for Hindi-English
code-switched technical lectures, and whether restricting that injection to regions where
the model is already uncertain avoids the over-biasing penalty.

Section numbers throughout the code and this file refer to that plan.

## What is being tested

| | Hypothesis |
|---|---|
| H1 | Injecting syllabus-derived context reduces WER relative to an unmodified baseline. |
| H2 | The reduction is *concentrated in terminology*; corpus WER understates it, so a term-restricted metric is required. |
| H3 | The injection points differ materially; token-level biasing and output-level correction are at least partially complementary. |
| H4 | Global grounding over-biases — terms are hallucinated into utterances that never contained them. Gating on model uncertainty keeps most of the terminology gain at a much lower penalty. **This is the principal claim.** |

The paper claims novelty for the *comparison*, the *gating strategy* and the *domain* —
not for prompting or contextual biasing as techniques.

## Data

OpenSLR **SLR104**, Hindi-English test portion (MUCS 2021 sub-task 2): 3,136 utterances,
5.18 h, 30 tutorial recordings, 16 kHz. Only the 443 MB test tarball is downloaded; no
fine-tuning is performed, so the 7.3 GB train tarball is not required. Nothing is ever
mined from the test transcripts.

### A data defect you must know about

The distributed `segments` file partitions each recording into **whole-second** windows
that tile the file exactly, so its boundaries cannot coincide with real speech edges.
`src/diagnose_segments.py` quantifies the damage: the WER-minimising window shift varies
per utterance from −2 s to +2 s (one sampled utterance drops from 5.62 to 0.88 WER at
−2 s), so no global offset fixes it. Cut windows routinely contain the tail of the
neighbouring sentence, which the model transcribes and which is then scored as
insertions.

`src/refine_segments.py` therefore snaps each shared internal boundary to the quietest
instant within a search radius, with a displacement penalty that keeps boundaries near
where the corpus put them. The transcript-to-utterance assignment is untouched — only
*where the audio is cut* changes, identically for every condition. Parameters are selected
on the Tier-1 sample by `src/tune_refinement.py`; the improvement is verified by
`src/diagnose_segments.py --validate`, and both records live in `report/`.

Anyone reporting utterance-level WER on this corpus without checking this will report an
inflated baseline and then measure their normaliser rather than their method.

## Layout

```
configs/config.yaml     the frozen experimental setup; edits invalidate the cache key
data/                   corpus, cut audio, manifests, frozen tiers        (gitignored)
cache/asr/              one JSON per (utterance, model, decode-config)    (gitignored)
runs/<tier>/<cond>/     hyps.jsonl, per_utt.jsonl, metrics.json, retrieval.json
report/                 pilots, corpus stats, results tables, figures, environment
syllabus/raw/           12 authored topic documents (the knowledge source)
syllabus/index/         chunk index, embeddings, frozen term lexicon + manifest
src/                    the harness
```

### The harness

| module | role |
|---|---|
| `prepare_slr104.py` | Kaldi dir → manifest; cuts each utterance once; §3.4 acceptance criteria |
| `diagnose_segments.py` | measures the boundary defect; `--validate` checks the fix |
| `refine_segments.py` / `tune_refinement.py` | boundary refinement and its parameter selection |
| `make_tiers.py` | freezes disjoint Tier 1 / 2 / 3, stratified by recording (§3.3) |
| `build_syllabus.py` | chunk index + **frozen** term lexicon with content hash (§5.3) |
| `backends.py` | faster-whisper wrapper; `DecodeConfig` hashes everything that changes output |
| `transcribe.py` | the decode cache (§4.4) — resumable, and what makes free conditions free |
| `normalize.py` | level-1 (headline) and level-2 (script-invariant) normalisation (§8.1) |
| `lexicon.py` | the frozen lexicon; defines B-WER / U-WER membership (§5.3, §8.2) |
| `score.py` | WER, **B-WER / U-WER**, CER, term P/R/F1, per-utterance edit counts (§8.2) |
| `retrieve.py` | two-pass retrieval, per-lecture and per-utterance, prose vs glossary context |
| `eval_retrieval.py` | top-1 topic accuracy, measured separately from WER (§6.3) |
| `conditions.py` | the §9.1 matrix: B0, C1, C2, C3, M1, M2, M3a, M3b, combinations |
| `correct_lexical.py` | M3a, including the `mat plot lib` → `matplotlib` merge case |
| `correct_llm.py` | M3b under hard constraints + the rewrite guard |
| `gating.py` | **G**: confidence gating, the threshold sweep and the frontier (§7.4) |
| `guards.py` | context-echo and rewrite guards, reported as measurements (§7.5) |
| `bootstrap.py` | paired bootstrap on WER, B-WER and U-WER + regression counts (§8.3) |
| `analyze_errors.py` | substitution taxonomy, WER by duration, **headroom estimate** (§8.4) |
| `sweeps.py` | Tier-1-only hyperparameter selection, one criterion stated in advance |
| `make_report.py` / `figures.py` | results tables and the paper's figures |
| `repro.py` | `report/environment.json` + the §13 checklist with pass/fail |
| `selftest.py` | hand-worked checks of scoring, correction, gating and guard logic |

## Running it

Each stage is a gate; do not proceed past a failing one (§11).

```bash
make setup                      # venv + pinned deps
make data                       # download, cut, verify counts vs published figures
make diagnose                   # measure the segment defect
make refine                     # tune + apply boundary refinement, validate the fix
make tiers syllabus selftest    # freeze tiers, freeze lexicon, check the scorer
make pilots                     # §4.2 turbo vs large-v3, §4.3 hi vs en vs auto
make baseline                   # B0 on Tier 1, validation gate, headroom estimate
make tune                       # every threshold chosen on Tier 1, and only there
make matrix                     # the full matrix on Tier 2 + stats + report + figures
make final                      # B0 and the best system on the complete test set
```

Single conditions and ablations:

```bash
$(PY) src/conditions.py M1 --tier tier2 --context-style glossary --name M1_glossary
$(PY) src/gating.py --tier tier2 --sweep --mechanism M2
$(PY) src/show_pairs.py --tier tier2 --run M2 --sort worst -n 20
```

## Rules the harness enforces rather than trusts

* **The lexicon is frozen before any grounded condition runs.** It defines B-WER/U-WER, so
  adding terms after seeing errors would make the metric a function of the method.
  `build_syllabus.py` refuses to overwrite it without `--refreeze`, and every
  `metrics.json` embeds the lexicon's size and content hash.
* **Hyperparameters are selected on Tier 1, which is disjoint from Tier 2 and Tier 3.**
* **`hotwords` and `prefix` are never set together** — faster-whisper silently drops the
  hints when a prefix is present (§7.2), so `DecodeConfig` raises instead.
* **The audio version enters the cache key**, so re-cutting the corpus cannot serve
  hypotheses decoded from the old cuts.
* **Guards are reported, not hidden**: firing rates and discard rates appear in the
  results table alongside the error rates.
* **Insertions of bias terms count against B-WER**, which is what makes over-biasing (H4)
  visible instead of flattering the method.

## Scope

Downstream note generation and summarisation from the corrected transcript is
deliberately **out of scope** (§1) so the ASR contribution can be evaluated in isolation.
