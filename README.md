# Syllabus-Grounded ASR for Code-Switched Classroom Lectures

Whisper large-v3 baseline → syllabus-grounded decoding → WER / term-F1 comparison, on
the Hindi–English code-switched test set of OpenSLR SLR104 (MUCS 2021 subtask-2).

**H1.** Injecting retrieved syllabus context into the decoding process of a
general-purpose ASR model reduces word error rate on code-switched technical lecture
speech, and the reduction is concentrated in domain-specific terminology rather than
distributed uniformly.

Note generation (downstream LLM summarisation) is out of scope for this repository.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Data (443 MB test tarball only — no fine-tuning, so the 7.3 GB train set is not needed)
cd data/raw/slr104
curl -C - -LO https://openslr.trmal.net/resources/104/Hindi-English_test.tar.gz
tar -xzf Hindi-English_test.tar.gz && cd -

# Model (~3 GB, cached in ~/.cache/huggingface)
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Systran/faster-whisper-large-v3')"
```

## Pipeline

```bash
# 1. Cut utterance WAVs + manifest (3136 utts, 5.18 h), freeze subsets
.venv/bin/python src/prepare_slr104.py --kaldi-dir data/raw/slr104/test/transcripts
.venv/bin/python src/make_subset.py          # 800 / dev200 / eval600, seed 1337

# 2. Syllabus index + term lexicon
.venv/bin/python src/build_syllabus.py

# 3. Language pilot (hi vs en vs auto), 100 utterances — fix the choice once
.venv/bin/python src/pilot_language.py --n 100

# 4. Baseline (S0) and error analysis
.venv/bin/python src/run_experiment.py --name S0_baseline --prompt-mode none
.venv/bin/python src/analyze_errors.py --hyps runs/S0_baseline/hyps.jsonl
.venv/bin/python src/show_pairs.py --hyps runs/S0_baseline/hyps.jsonl --sort worst

# 5. Ablation matrix
.venv/bin/python src/run_experiment.py --name S1_generic   --prompt-mode generic
.venv/bin/python src/run_experiment.py --name S2_random    --prompt-mode random
.venv/bin/python src/run_experiment.py --name S3_retrieved --prompt-mode retrieved
.venv/bin/python src/run_experiment.py --name S6_oracle    --prompt-mode oracle
.venv/bin/python src/correct_lexical.py --in-hyps runs/S3_retrieved/hyps.jsonl \
    --out-dir runs/S4_lexical
.venv/bin/python src/correct_llm.py --in-hyps runs/S3_retrieved/hyps.jsonl \
    --out-dir runs/S5_llm            # needs GROQ_API_KEY

# 6. Results table + paired bootstrap significance
.venv/bin/python src/make_report.py
.venv/bin/python src/bootstrap.py S0_baseline S3_retrieved
```

## Layout

| Path | Contents |
|---|---|
| `src/` | pipeline (prep, backends, decoding cache, normalisation, scoring, retrieval, correction) |
| `syllabus/raw/` | 12 hand-written course documents (8 in-domain, 4 distractors) |
| `syllabus/index/` | chunk embeddings, term lexicon, `rec2topic.json` (S6 oracle only) |
| `cache/asr/` | one JSON per (utterance, backend, config hash) — re-runs are free |
| `runs/<name>/` | `hyps.jsonl`, `per_utt.jsonl`, `metrics.json` |
| `report/` | write-up sections and the results table |

## Experiment matrix

| ID | System | Purpose |
|----|--------|---------|
| S0 | large-v3, no prompt | baseline |
| S1 | + generic prompt | controls for *any* prompt vs. *retrieved* prompt |
| S2 | + random syllabus doc | controls for retrieval quality |
| S3 | + retrieved prompt (k=3) | Method A |
| S4 | S3 + lexical correction | A + B1 |
| S5 | S3 + LLM correction | A + B2 |
| S6 | + oracle syllabus doc | upper bound on retrieval |

Reported per system: WER, CER, script-invariant WER, skeleton-WER lower bound,
term precision/recall/F1, % utterances improved, % regressed, and a paired bootstrap
p-value against S0.
