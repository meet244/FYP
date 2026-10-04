# Implementation and research audit

Audit date: 2026-10-02. Scope: complete the existing single-machine ClassScribe
backend and connect it to the frontend, with the best supported research baseline
as the default. Preserve the original research experiment code and outputs.

## Why Qwen3-ASR-0.6B

The matched baseline comparison in
[`research/sgcd/RUNLOG.md`](../research/sgcd/RUNLOG.md) and
[`bench_scores__test.json`](../research/sgcd/out/bench_scores__test.json) uses 150
utterances. Strict WER, lower is better:

| Checkpoint | Strict WER (%) |
| --- | ---: |
| Qwen3-ASR-0.6B | **59.74** |
| Qwen3-ASR-1.7B | 63.24 |
| Whisper large-v3 | 74.17 |
| Whisper large-v3-turbo | 80.23 |
| Parakeet RNNT 1.1B | 97.45 |

Qwen 1.7B has better results under some script-normalized/semantic measures; its
forced-Hindi output transliterates English technical terms. Parakeet is an
English-only checkpoint. Therefore 0.6B is the defensible default for this
project's dual-script Hindi–English convention, rather than choosing the largest
checkpoint. Runtime generation is deterministic, bounded and uses a repetition
penalty; these settings and the native HF implementation need their own quality
assessment and are not claimed to reproduce cached research scores exactly.

The older Whisper long-span study is a different experiment: 100 concatenated
spans, baseline 43.46%, whole-syllabus narration 37.23%, retrieved narration
38.64%, keyword list 62.86%. Its gains cannot be attributed to Qwen or compared
directly with the 150-utterance baseline table. The mismatched-syllabus controls
also suggest much of the improvement comes from script/register cues, rather
than proving syllabus-semantic correction.

## Gaps closed

| Previous gap | Implemented behavior |
| --- | --- |
| Best benchmark checkpoint absent from backend | Native Transformers Qwen3-ASR adapter; 0.6B default |
| Model selection hidden in environment | Capability catalog, validated per-job selection and frontend controls |
| Every model treated as promptable Whisper | Runtime-specific Qwen/Whisper/Parakeet adapters, language/method validation |
| Whisper turbo thresholds applied across models | Fitted safeguards restricted to turbo; generic empty/repetition checks separate |
| Model switching can retain several large checkpoints | Serialized inference; one adapter and cleared Torch/MLX caches |
| Transcription lost on restart/retry | Durable queue recovery and atomic span checkpoints keyed to inputs |
| No cancel/retry controls | Cooperative cancellation, terminal state recovery and retry endpoints/UI |
| LLM notes failure requires full ASR rerun | Transcript preserved; notes-only regeneration job |
| Reprocess overwrites evidence | Per-run snapshot of effective configuration, spans and statistics |
| Stale notes survive changed transcript | Replace transcript and remove dependent notes before regeneration |
| Lecture summary discarded | Summary stored on lecture |
| Indistinct ASR failure state | Lecture/job failure with actionable error and retry |
| Unbounded or partial upload writes | Streaming file limit, partial cleanup and batch rollback |
| Partial normalized audio reused after failed conversion | Atomic WAV replacement |
| Image success contains only filename | Failed OCR exposes retryable error instead of fake extracted content |
| Chat holds database writer while answering | Detached eager-loaded snapshot; messages committed after answer |
| Explicit lecture query silently searches all lectures | Resolve chronological number/latest title or return no scoped result |
| Citations fabricated when model omitted markers | Return only citations actually referenced; strip unsupported markers |
| Coverage attribution treats all retrieved units equally | Attribute using strongest retrieved unit rather than ties |
| Syllabus and saved sessions absent in studio | Syllabus/coverage sidebar and selectable persistent chat history |
| UI job progress disappears after refresh | Recover jobs by subject/lecture with scoped polling |
| UI labels claim every decode is two-pass Whisper | Show selected pipeline stage and standard versus conditioned behavior |
| Old SQLite schema rejects new fields | Additive migration preserves existing lecture rows |
| Dependency conflicts with native Qwen | Compatible pinned Transformers, sentence-transformers and Chroma versions |

## Verification

Verified against the completed code on this Apple-silicon Mac:

- `python -m pytest tests/ -q`: **60 passed**, 9.39 seconds. One third-party
  Starlette/AnyIO deprecation warning. Tests use temporary storage and stub
  recognition/LLM calls; they do not establish recognition accuracy.
- Frontend TypeScript and `npm run build`: passed. `pip check`: no broken requirements.
- Real Qwen Hindi fixture (`test/voice/L01-perceptrons-hi.wav`): 24.93 seconds,
  one span, transcript/run persisted and indexed. Warm reprocessing took
  **10.9 seconds (2.29× audio speed)**. The first run included a weight download
  and is not a useful steady-state latency measurement.
- Real frontend English upload (`test/voice/L01-perceptrons-en.wav`): 96.39
  seconds, four spans, **14.0 seconds (6.9× audio speed)**. Language/model settings
  traveled from the browser to the saved job configuration.
- Real multilingual embeddings/Chroma: supporting Markdown uploaded through the
  browser and indexed into two chunks. Local extractive chat returned persisted
  source citations. A question about the latest recording cited only that recording.
- Browser: default Qwen selection, syllabus/coverage tab, saved chat reopening,
  historical transcript selection, historical notes disabled, timestamp seeking,
  notes-only failure/retry control, and reprocess submission verified. No browser
  page errors were reported.
- Restart recovery: interrupted the isolated server during its initial download;
  startup recovered the running job and completed it. Interrupted-span reuse and
  cancellation during inference failure are covered separately by automated tests.

The fixtures are demonstration/TTS recordings, not a held-out quality benchmark.
No real Gemini calls were possible without a credential; transcript-only completion,
extractive chat and actionable notes failure were verified instead. Optional
comparison adapters still need real-checkpoint verification on their target runtimes.

The Qwen adapter follows the [native Transformers processor API](https://huggingface.co/docs/transformers/model_doc/qwen3_asr).
The optional Parakeet adapter follows [parakeet-mlx](https://github.com/senstella/parakeet-mlx);
its [RNNT checkpoint is English-only](https://huggingface.co/nvidia/parakeet-rnnt-1.1b).

## Research still pending

1. **Classroom validation of the serving default.** Re-score bounded native Qwen
   inference on the same benchmark; then evaluate long real classroom recordings,
   silence/noise, interruptions, speaker accents and memory/latency. Demonstration
   TTS fixtures cannot establish deployment WER.
2. **Qwen 1.7B language/script control.** Compare auto language with forced Hindi
   on a held-out set, using both strict and transliteration-aware metrics.
3. **Qwen syllabus conditioning.** Run long-span baseline versus matched and
   mismatched narration, retrieved narration and register controls. Do not reuse
   Whisper's safeguards or published gains without calibration.
4. **FYRP rescoring.** The best smaller experiment is S5, 55.69% versus 58.86%
   baseline on 50 test utterances; paired delta -3.18 points, CI [-6.3, -0.2].
   It uses beam hypotheses and a domain trigram LM. This promising result needs a
   larger held-out/long-recording evaluation, an artifact/version contract and
   runtime cost analysis before replacing the standard decoder. See
   [`fyrp/results.md`](../research/sgcd/out/fyrp/results.md).
5. **Trainable variants.** S7–S10 produce training metadata and hypotheses but
   the current experiment code does not export reusable trained checkpoint
   weights. Repetition runaways affect partial fine-tuning/LoRA results. Export,
   reload tests, held-out evaluation and stability validation are still needed;
   these are not exposed as production models.
6. **Downstream evaluation and paper artifacts.** Quantify note fidelity,
   learning-outcome quality, unit attribution/coverage and citation correctness
   against human labels; complete placeholder paper figures and align claims
   with the matched/mismatched controls.

## Deployment/configuration still pending

- Set a valid Gemini credential to verify real generated notes, syllabus rewriting,
  composed chat and image OCR. No credential is present in the backend setup used
  for verification; those external calls are tested with stubs. Transcription and
  extractive source chat are local.
- Scanned PDF text extraction still requires OCR preprocessing. Image OCR is a
  Gemini integration, not an offline OCR model.
- Authentication, per-user authorization, an external queue and production schema
  migration tooling are needed if the scope changes to a public multi-user
  service. The implemented target is one local server process with SQLite.
- Historical jobs/checkpoint files are retained for audit/retry. Add a retention
  policy when storage growth matters. Optional comparison checkpoints require
  their own weights and runtime verification; they are not all loaded on startup.
