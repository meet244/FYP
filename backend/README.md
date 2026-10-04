# ClassScribe backend

FastAPI service for subjects, syllabus PDFs, lecture recordings, study notes,
learning outcomes, supporting files, coverage, and source-cited chat.

**Qwen3-ASR-0.6B is the default local speech model.** It had the lowest strict
WER among the five baseline checkpoints on the project's matched 150-utterance
Hindi–English benchmark. This is a choice for this corpus and its dual-script
transcript convention, not a claim of universal superiority. See
[implementation and research audit](IMPLEMENTATION_STATUS.md).

## Run locally

Use Python 3.11 and FFmpeg:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Set GEMINI_API_KEY in .env for generated notes, syllabus parsing, and image OCR.
# macOS: brew install ffmpeg
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run one Uvicorn process. The durable job table is coordinated by an in-process
worker; multiple server processes are not supported. Open `/docs` for the full
API contract. First use downloads model weights; later runs use the local cache.
If the Hugging Face Xet transport stalls, start with `HF_HUB_DISABLE_XET=1`.

Audio recognition, embeddings, SQLite, Chroma and files run locally. Gemini
receives the text needed for notes, syllabus rewriting, and composed chat;
image extraction sends image bytes. This is **not** a fully offline LLM pipeline.
Without a Gemini credential, transcripts remain available, chat can quote local
sources, and coverage works. Notes, syllabus parsing, and image extraction need
that credential. A failed notes stage can be retried without decoding audio.

## Recognition

The default is Qwen S5 domain-LM rescoring over roughly 25-second spans: a
five-beam candidate search followed by local rescoring, with no second decode.
For faster processing of long recordings, choose `baseline` (single-candidate
greedy decoding); transcript accuracy may differ from S5. Upload
`model_id`, `language` (`hi`, `en`, `auto`), and `method` (`baseline`, `s5`, `sgcd`) as
form fields, or send those fields as JSON to the reprocess endpoint. The resolved
configuration is saved with each job and successful transcript run.

On Apple silicon, Qwen uses the MPS GPU. Audio jobs share a single resident
model and run serially to bound unified-memory usage. Increasing worker count
does not parallelize speech decoding. Changing upload controls applies to new
uploads; existing jobs keep their saved configuration. Changing a running
recording's decoding configuration invalidates its span checkpoint.

| Model ID | Runtime | Role |
| --- | --- | --- |
| `qwen-0.6b` | Transformers / Torch, MPS, CUDA or CPU | Recommended default, Hindi–English |
| `whisper-turbo` | MLX on Apple silicon, faster-whisper elsewhere | Alternative; original SGCD checkpoint |
| `whisper-large-v3` | MLX / faster-whisper | Alternative; turbo safeguard thresholds disabled |
| `qwen-1.7b` | Transformers / Torch | Optional comparison; larger memory cost, script caveat |
| `parakeet-rnnt-1.1b` | parakeet-mlx on Apple silicon | Optional English-only comparison, no syllabus prompts |

Install `requirements-asr-extra.txt` only if you want the optional Parakeet
adapter. `/asr/models` reports installed runtimes and supported capabilities;
installed runtime does not mean weights have already downloaded. Only the
recommended model is loaded by default. Alternative models are explicit choices.

`sgcd` adds retrieved syllabus narration and a second decode. Whisper turbo's
fitted logprob/compression/length safeguards apply only to that checkpoint;
empty-output and repetition checks are separate. Qwen syllabus conditioning is
experimental: the Whisper experiments do not validate it. Prompts use prose,
not keyword lists. Short-span conditioning regressed in the original study.

## API flows

| Endpoint | Result |
| --- | --- |
| `GET /health`, `GET /asr/models` | Configuration, credential presence, model capabilities |
| `POST,GET /subjects`; `GET,DELETE /subjects/{id}` | Subject management |
| `POST,GET /subjects/{id}/syllabus` | PDF parsing job / parsed units |
| `PATCH /subjects/{id}/syllabus/units/{unit_id}` | Correct narration and refresh retrieval |
| `GET /subjects/{id}/coverage` | Note-linked coverage and outstanding units |
| `POST,GET /subjects/{id}/lectures` | Upload recording job / lecture library |
| `GET,DELETE /lectures/{id}` | Lecture metadata / removal |
| `GET /lectures/{id}/transcript`, `/notes`, `/audio` | Current transcript, notes, playable WAV |
| `POST /lectures/{id}/reprocess` | Decode stored recording with selected options |
| `POST /lectures/{id}/notes/regenerate` | Regenerate notes without another ASR pass |
| `GET /lectures/{id}/runs`; `GET /lectures/{id}/runs/{run_id}` | Saved configuration, statistics and historical spans |
| `POST,GET /subjects/{id}/materials` | Upload PDF, image, TXT, Markdown, CSV, DOCX or PPTX / list files |
| `GET /materials/{id}`; `GET /materials/{id}/file`; `DELETE /materials/{id}` | Extracted-text preview / inline original / remove file |
| `GET /subjects/{id}/syllabus/file` | Inline original syllabus PDF |
| `POST /subjects/{id}/chat` | Grounded answer and source citations |
| `GET /subjects/{id}/chat/sessions`; `GET,DELETE /chat/sessions/{id}` | Persistent conversation history |
| `GET /jobs`; `GET /jobs/{id}` | Durable status, stage, progress and errors |
| `POST /jobs/{id}/cancel`, `/retry` | Cancel at a safe stage boundary / retry failed or cancelled job |

Long operations return `202`. Poll jobs until `succeeded`, `failed`, or
`cancelled`. Running cancellation first reports `cancelling`; the current model
call may finish before the worker stops. Conflicting operations on the same
lecture and deletion during active work return `409`.

Jobs recover on process restart. Audio normalization and span checkpoints use
atomic replacement. Finished decode spans resume on retry when audio, syllabus,
and configuration match; changed inputs invalidate the checkpoint. Successful
ASR runs retain historical spans even when current transcripts are replaced.
Old notes are removed when their transcript changes. The frontend restores job
progress after a refresh and exposes retries, history, saved chats and syllabus.

Uploads stream with a configurable per-file limit (`CLASSSCRIBE_MAX_UPLOAD_MB`,
default 1024). Partial writes are removed; material batches roll back together.
Existing local databases get additive columns without resetting stored lectures.
CORS allows localhost frontend origins; configure `CLASSSCRIBE_CORS_ORIGINS` as
a JSON list if needed. This application currently has no multi-user authentication.

## Verification

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
```

Tests always use a temporary database and storage. They cover the upload → audio
conversion → ASR → notes → coverage pipeline with stubs, model validation,
checkpoint recovery, cancellation during failure, notes-only retries, historical
runs, chat persistence, batch rollback, and additive database migration. Stubbed
tests verify integration, not recognition quality. Research evaluation remains
in `../research`; fixtures in `../test` are demonstration audio and documents.

Known remaining requirements and the real-runtime verification record are in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
