# Dataset, environment and evaluation harness

*Generated from measured artefacts by `src/make_setup_section.py`; do not edit by hand.*

## 1 Corpus

Evaluation uses the Hindi-English code-switched **test** portion of OpenSLR SLR104 (MUCS 2021 sub-task 2), drawn from Spoken Tutorial recordings on technical topics; the code-switching arises predominantly from that technical content. Only the test tarball (443 MB) was downloaded: no fine-tuning is performed, so the 7.3 GB train tarball is not required and no terminology is mined from any transcript.

| Property | Measured | Published |
|---|---|---|
| Utterances | 3136 | 3136 |
| Total audio | 5.178 h | 5.18 h |
| Recordings | 30 | 30 |
| Speakers (utt2spk) | 30 | — |
| Utterance duration mean / median | 5.94 s / 5.0 s | — |
| Utterance duration min / max | 1.0 s / 41.0 s | — |
| Sample rate | 16 kHz, 16-bit mono (no resampling) | 16 kHz, 16-bit |

Stage-1 gate (§3.4): counts and duration match the published figures (PASS); no utterance has zero duration; a random sample of 10 cut files is readable at the expected duration and sample rate.

## 2 A segmentation defect in the distributed test set

The distributed `segments` file partitions each recording into **whole-second** windows that tile the file exactly, so its boundaries cannot coincide with real speech edges. Cutting on them produces windows that contain the tail of the neighbouring sentence, which the decoder transcribes and which is then scored as insertions.

To quantify this, 20 Tier-1 utterances were decoded at each of several window shifts and scored against their references:

| Window shift | Mean WER | Median WER | Utterances best at this shift |
|---|---|---|---|
| -2 s | 1.0085 | 0.8286 | 3 |
| -1 s | 0.6725 | 0.6000 | 9 |
| +0 s | 0.8664 | 0.4473 | 5 |
| +1 s | 0.8221 | 0.7571 | 2 |
| +2 s | 1.5262 | 0.9437 | 1 |
| +3 s | 1.5226 | 1.1051 | 0 |

The WER-minimising shift **varies per utterance** across the whole range, so the defect is local boundary imprecision rather than a global offset and no constant correction can repair it.

### Boundary refinement

Each shared internal boundary is moved to the quietest instant within a search radius, scored by short-window frame energy plus a penalty per second of displacement that keeps boundaries near where the corpus put them. Utterance boundaries in continuous speech fall in inter-sentence pauses, so this recovers the true edge for the two utterances sharing it. The transcript-to-utterance assignment is never touched: only where the audio is cut changes, identically for every condition.

Parameters were selected on the Tier-1 sample (lowest mean per-utterance WER on the Tier-1 sample):

| Radius | Penalty (dB/s) | Mean WER | Median WER | Worse than distributed | Mean \|shift\| |
|---|---|---|---|---|---|
| 2.0 | 3.0 | 0.5813 | 0.4674 | 25% | 0.76 s |
| 1.5 | 4.0 | 0.6164 | 0.5528 | 35% | 0.67 s |
| 1.0 | 6.0 | 1.4004 | 0.4643 | 30% | 0.46 s |
| 2.5 **(chosen)** | 2.0 | 0.5501 | 0.4317 | 30% | 0.82 s |

| distributed windows | — | 0.8664 | 0.4473 | — | 0.00 s |

Applied corpus-wide, refinement moves a boundary by 0.75 s on average (median -0.13 s, p10 -1.25 s, p90 +1.23 s); 11.2% of boundaries do not move. Total audio is unchanged at 5.187 h.

Validation on the same utterance sample:

| Windows | Mean WER | Median WER |
|---|---|---|
| distributed | 0.8664 | 0.4473 |
| refined | 0.5813 | 0.4674 |

The refined cuts are identified by `data.audio_version` in the config, which enters the ASR cache key, so hypotheses decoded from the earlier cuts can never be served for the refined audio.

## 3 Evaluation tiers

| Tier | Utterances | Duration | Recordings | Purpose |
|---|---|---|---|---|
| tier1 | 200 | 18.5 min | 30 | tuning: thresholds, context format, retrieval depth |
| tier2 | 800 | 80.0 min | 30 | the full experiment matrix and all ablations |
| tier3 | 3136 | 311.2 min | 30 | final confirmation, two systems only |

Tier 1 and Tier 2 are disjoint samples, stratified by recording so every lecture topic appears in both, drawn once under seed 1337 and frozen as committed manifests.

> All development and ablation experiments use a fixed random subset of 800 utterances (seed 1337); hyperparameters were selected on a disjoint development slice of 200 utterances; final systems are additionally evaluated on the complete Hindi-English test set.

## 4 System under test

Whisper **large-v3-turbo** run locally through faster-whisper / CTranslate2 with `compute_type=int8` on `cpu`. Local execution is required because two of the three grounding mechanisms need decoder-level access and per-token confidences that a hosted API does not expose.

Fixed decoding: beam size 5, temperature 0.0 (deterministic, no temperature fallback), `condition_on_previous_text=False` (utterances are independent segments), `vad_filter=False` (audio is already sentence-segmented), `word_timestamps=True` (required for the per-token confidences used by confidence gating).

Every decode is cached under `cache/asr/<backend>/<config-hash>/<utt>.json`, keyed by the model identity, every decode parameter, the injected grounding payload and the audio version. An interrupted run resumes, output-level methods operate on cached text at zero decode cost, and metrics can be recomputed without touching the model.

### 4.1 Language configuration pilot (§4.3)

| Setting | WER | B-WER | U-WER | CER | WER (level 2) | Empty hyps |
|---|---|---|---|---|---|---|
| hi **(chosen)** | 1.1250 | 1.2500 | 1.1071 | 0.9677 | 1.1250 | 0 |
| en | 1.5000 | 1.5000 | 1.5000 | 1.1344 | 1.5000 | 0 |
| auto | 1.2188 | 0.7500 | 1.2857 | 1.0484 | 1.0000 | 0 |

'hi' gives the lowest WER on Tier 1 and is fixed for all subsequent experiments.

### 4.2 Model selection pilot (§4.2)

*(missing: run `python src/pilots.py model`)*

## 5 Normalisation and metrics

Two normalisation levels are defined before any modelling work (§8.1). **Level 1** — Unicode NFC, numeral unification (Devanagari digits, Latin digits and spelled-out numbers), punctuation removal, case folding, whitespace collapse — is script-preserving and produces the **headline WER**. **Level 2** additionally romanises Devanagari and applies a light, symmetric phonetic folding, so a technical term written in either script compares equal; it is reported alongside level 1 to quantify the orthographic share of total error and is never presented as the WER. Both foldings are applied to reference and hypothesis alike.

The primary metric is **decomposed WER**: every reference word is labelled B (a member of the frozen syllabus lexicon) or U (not), and error rates are reported separately. Substitutions and deletions are attributed to the class of the reference word; insertions to the class of the hypothesis word, so a syllabus term hallucinated into an utterance that never contained it is counted as a B insertion rather than hidden. Effective grounding lowers B-WER while leaving U-WER unchanged; over-biasing lowers B-WER and raises U-WER.

The lexicon holds **981 terms** over 12 authored topic documents (sha256 `96d72c87273a`), composed of 105 capitalised, 843 plain, 6 identifier like, 27 multi case. Construction: ASCII tokens of length >= 3 from the authored syllabus documents, minus a fixed general-English prose/function-word stop-list; first occurrence per document order retained. No term was added or removed after observing model output. It is frozen before any grounded condition runs, and every `metrics.json` embeds its size and content hash, so no metric can be traced to a different term list.

## 6 Environment

macOS-26.5.2-arm64-arm-64bit (arm64, 10 cores), Python 3.11.9. `faster-whisper` 1.1.1, `ctranslate2` 4.5.0, `sentence-transformers` 3.3.1, `jiwer` 3.0.4, `rapidfuzz` 3.10.1, `indic-transliteration` 2.3.68, `numpy` 1.26.4, `torch` 2.13.0, `soundfile` 0.12.1, `librosa` 0.10.2.post1, `matplotlib` 3.9.2, `onnxruntime` 1.28.0, `tokenizers` 0.22.2, `huggingface-hub` 0.36.2.

Code revision `34af60c129c4`; corpus tarball sha256 `93e358b3bf82`.
