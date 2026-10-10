# ClassScribe

ClassScribe turns recorded lectures and course materials into a subject-based
study workspace. It supports Hindi-English transcription, structured notes,
source-cited questions and answers, and generated revision material.

## What is implemented

- Local lecture transcription with **Qwen3-ASR-0.6B and S5 domain-language-model
  rescoring** by default. A faster greedy baseline and optional model choices are
  available; syllabus-conditioned decoding is a separate method.
- Topic-organised notes with definitions, competency statements, recording
  timestamps, and optional syllabus-unit attribution and coverage.
- A shared retrieval index for transcripts, notes, syllabuses, PDFs, images,
  PPTX, DOCX, TXT, Markdown, and CSV files. Chat cites the retrieved course sources.
- Studio generation of quizzes, flashcards, mind maps, reports, slide decks,
  and infographics for a subject, syllabus unit, or recording.
- Saved chats, audio playback, source previews, transcript-run history, durable
  processing jobs, cancellation, retries, and notes-only regeneration.

Speech recognition, embeddings, SQLite, Chroma, and file storage run locally.
Gemini handles generated notes, syllabus parsing, composed chat, image extraction,
and Studio generation. A configured Gemini credential is required for those
features. Local transcription and source-quoted chat remain available without it.

## Run locally

Use Python 3.11 and FFmpeg for the backend. Follow the
[backend setup](backend/README.md) and [frontend setup](frontend/README.md) for
environment creation, dependencies, and credentials. With those environments
already prepared, start each service in its own terminal:

```bash
# Terminal 1, from the repository root
cd backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2, from the repository root
cd frontend
npm run dev
```

Open `http://localhost:3000`. Backend health is at
`http://127.0.0.1:8000/health`; API documentation is at
`http://127.0.0.1:8000/docs`. Run one backend process: the application targets a
single local machine and has no multi-user authentication.

## Research results

Three separate experiments inform the application:

| Experiment | Evaluation | Supported finding |
| --- | --- | --- |
| Five-model ASR benchmark | 150 Hindi-English utterances | Qwen3-ASR-0.6B has the lowest recorded strict WER, **59.74%**, among the five tested checkpoints. |
| Ten independent Qwen improvement methods | 50 test utterances; 30 tuning utterances | S5 reduces WER from **58.86% to 55.69%**; paired delta **-3.18 percentage points**, reported 95% CI **[-6.3, -0.2]**. |
| Original Whisper syllabus-context study | 100 concatenated spans, averaging 26.2 seconds | Whole-syllabus narration reduces WER from **43.46% to 37.23%**; mismatched-syllabus controls indicate much of the benefit is format/register priming. |

These numbers describe different experiments, not the accuracy of the running
application. The serving pipeline uses longer spans and different generation
limits. Its WER, generated-note quality, citation correctness, and educational
benefits still need dedicated evaluation.

## Documentation

| Document | Purpose |
| --- | --- |
| [Results summary](results.md) | Experiment tables, interpretation, evidence sources, and claim boundaries |
| [Research overview](research/README.md) | Research questions and seven literature gaps addressed by the project |
| [Replication guide](research/REPLICATION.md) | Commands and requirements for all three experiments |
| [Detailed Qwen results](research/sgcd/out/fyrp/results.md) | All ten methods, settings, uncertainty, and runtime limitations |
| [Implementation status](backend/IMPLEMENTATION_STATUS.md) | Implemented behaviour, historical checks, demonstrations, and outstanding validation |
| [Test fixtures](test/README.md) | Demonstration course materials and recordings |

The documents and demonstrations in [dump/](dump/) are supporting evidence.
Their older proposals and informal observations are distinguished from current
implementation and saved experimental measurements. Multiple Studio output
formats do not establish a student/tutor perspective-conditioned summarisation
system.
