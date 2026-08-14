# ClassScribe — backend

Records classroom lectures, transcribes them with **Syllabus-Grounded Contextual
Decoding (SGCD)**, turns the transcripts into structured study notes, and answers
questions over the resulting corpus with citations back to the recording.

The transcription core is the method from [`../research`](../research) — see
[`research/paper/ClassScribe_Research_Paper.tex`](../research/paper/ClassScribe_Research_Paper.tex).
Everything here is inference-time; nothing is trained.

## The pipeline

```
phone recording ──ffmpeg──▶ 16 kHz mono
                              │
                              ├─ segment into ~25 s spans        (segment.py)
                              ├─ pass 1: unconditioned decode    (backends.py)
                              ├─ retrieve k=3 syllabus units     (retrieve.py)
                              ├─ pass 2: conditioned on prose    (prompts.py)
                              └─ stability safeguard             (sgcd.py)
                                        │
                            transcript spans ──▶ notes + outcomes + terms
                                        │              │
                                        └──────┬───────┘
                                               ▼
                                     vector index (Chroma)
                                               ▼
                                    subject-scoped chat + citations
```

Cost is **one supplementary decode per span** — retrieval reuses the first-pass
hypothesis the baseline produces anyway.

## Three things that are load-bearing

**Span length is a precondition, not a tuning knob.** At the corpus's native 5.7 s
utterances, conditioning *regresses* WER by 5.11 points and needs the safeguard to
break even. At 26.2 s spans it gives −6.23 unaided. `span_target_s` defaults to
25 s, inside the 30 s encoder receptive field. Lowering it silently reverts the
system to the regime where the method does not work.

**The prompt must be narration, never a term list.** Feeding the syllabus as
comma-separated terminology reproduces the published failure mode: terminology
error halves, everything else degrades, aggregate WER +19.39. The same content as
fluent code-mixed prose gets the terminology gain with no collateral damage. This
is why syllabus ingestion is an LLM *rewrite* rather than PDF text extraction —
see [`app/ingest/syllabus.py`](app/ingest/syllabus.py).

**Safeguard thresholds do not transfer across checkpoints.** The defaults were
fitted on a development split for `whisper-large-v3-turbo`. Applied unchanged to a
smaller model they fired on 46% of utterances and made results worse. Change the
ASR model → refit or disable.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
brew install ffmpeg           # required — phone audio is m4a/aac
.venv/bin/uvicorn app.main:app --reload
```

Interactive API docs at `http://127.0.0.1:8000/docs`; `GET /health` echoes the
active SGCD configuration.

On Apple silicon the default backend is `mlx-whisper` (5.5–15.7× real time on a
consumer laptop, per the paper). Elsewhere set `CLASSSCRIBE_ASR_BACKEND=faster-whisper`
and `CLASSSCRIBE_ASR_MODEL=large-v3`. The first transcription downloads the model.

## Typical flow

```bash
# 1. create a subject
curl -X POST localhost:8000/subjects -H 'content-type: application/json' \
     -d '{"name":"Operating Systems","code":"ITC501"}'

# 2. upload the syllabus PDF  -> returns a job
curl -X POST localhost:8000/subjects/$SID/syllabus -F file=@syllabus.pdf

# 3. upload a recording       -> returns a job
curl -X POST localhost:8000/subjects/$SID/lectures \
     -F file=@lecture1.m4a -F title='Process scheduling'

# 4. poll
curl localhost:8000/jobs/$JOB_ID

# 5. ask
curl -X POST localhost:8000/subjects/$SID/chat -H 'content-type: application/json' \
     -d '{"question":"What is the difference between paging and segmentation?"}'
```

Upload the syllabus **before** the first lecture. Without one the pipeline runs a
single unconditioned pass — correct behaviour, but none of the method's benefit.
`POST /lectures/{id}/reprocess` re-decodes stored audio once a syllabus exists.

## API

| | |
|---|---|
| `POST /subjects` · `GET /subjects` · `GET,DELETE /subjects/{id}` | subjects |
| `POST /subjects/{id}/syllabus` | upload PDF → job |
| `GET /subjects/{id}/syllabus` | parsed units |
| `PATCH /subjects/{id}/syllabus/units/{unit_id}` | hand-correct a unit |
| `GET /subjects/{id}/coverage` | units delivered vs outstanding |
| `POST /subjects/{id}/lectures` | upload recording → job |
| `GET /subjects/{id}/lectures` · `GET,DELETE /lectures/{id}` | lectures |
| `GET /lectures/{id}/transcript` | spans + full text + ASR stats |
| `GET /lectures/{id}/notes` | notes, terms, learning outcomes |
| `GET /lectures/{id}/audio` | normalised WAV, for timestamp seeking |
| `POST /lectures/{id}/reprocess` | re-decode stored audio |
| `POST /subjects/{id}/chat` | ask a question |
| `GET /subjects/{id}/chat/sessions` · `GET,DELETE /chat/sessions/{id}` | history |
| `GET /jobs/{id}` · `GET /jobs` | job status |

Long operations return `202` with a job; poll `progress` (0–1), `stage`, `message`.

## Query routing

The chat endpoint classifies each question before retrieving, because the same
corpus answers different question shapes differently:

| type | behaviour |
|---|---|
| `lookup` | transcript-first, two or three sentences |
| `explain` | conceptual, built from the lecturer's framing |
| `summary` / `outline` | notes-first, structured |
| `compare` | point-by-point, table where dimensions are clean |
| `quiz` | practice questions, then an answers section |
| `coverage` | answered from the database, **no retrieval** — it is a computable fact |
| `smalltalk` | short reply, no retrieval |

Answers cite `[n]` markers resolving to `{lecture_id, timestamp}` so the frontend
can link straight into the audio.

## Layout

```
app/
  asr/        segment · backends · prompts · retrieve · sgcd · normalize
  ingest/     audio (ffmpeg) · syllabus (PDF → code-mixed units)
  notes/      synthesize (notes, terms, outcomes) · coverage
  rag/        store (Chroma) · indexer · router · answer
  jobs/       queue (worker thread) · handlers (pipelines)
  api/        subjects · lectures · chat · jobs
  models.py   ORM      schemas.py  API contract      views.py  detached rows
tests/        test_smoke.py     units + API surface
              test_pipeline.py  end-to-end with the model and LLM stubbed
```

`app/asr/normalize.py` is ported verbatim from the research so transcripts stay
comparable with the published numbers. Do not edit it.

## Tests

```bash
CLASSSCRIBE_DATA_DIR=./data/test CLASSSCRIBE_DB_URL="sqlite:///./data/test/test.db" \
  .venv/bin/python -m pytest tests/ -q
```

27 tests, ~16 s, no model download and no API key. Covers segmentation bounds,
retrieval ranking under a noisy first pass, prompt construction and
left-truncation, all three safeguard triggers, normalisation parity, the API
surface, and a full upload → ffmpeg → two-pass decode → notes → coverage run with
the ASR backend and LLM stubbed.

The stubs assert *plumbing*, not recognition quality — quality is what
[`../research`](../research) measures, against references this backend doesn't have.

## Known limits

- **Jobs are in-process.** A restart mid-transcription requeues the job from the
  start (`requeue_stale()`); there is no resume-from-span. Fine for one machine,
  not for a multi-worker deployment.
- **SQLite.** Single-writer. Long LLM calls are deliberately made outside any open
  session, but concurrent transcriptions will contend. Move to Postgres before
  serving a cohort.
- **Scanned syllabi fail.** `pypdf` extracts text, not images — OCR first.
- **Notes and chat need an API key**; transcription does not. Without a key,
  transcription still completes and the lecture stops at `transcribed`.
- **The evaluation harness lives in `../research`,** not here. This backend has no
  WER scoring — it has no references to score against.
