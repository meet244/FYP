# ClassScribe — frontend

A NotebookLM-style client for the [`../backend`](../backend) API: upload a
syllabus and a recording, watch the SGCD pipeline run, read the transcript
against the audio, and ask questions that cite the moment each claim was made.

The backend is the whole application. This is a typed client over its REST
surface and nothing else — no database, no auth, no LLM calls of its own.

## Setup

```bash
pnpm install
cp .env.example .env.local     # only if the backend is not on :8000
pnpm dev
```

The backend must be running:

```bash
cd ../backend && .venv/bin/uvicorn app.main:app --reload
```

Open `http://localhost:3000`. The dot in the header turns red when the API is
unreachable and amber when the configuration is one that silently breaks the
method — see below.

## Stack

Next.js 16 (App Router) · React 19 · Tailwind v4 · shadcn/ui on Radix ·
framer-motion · SWR · lucide-react.

Deliberately absent: there is no auth provider, no ORM and no AI SDK. Audio
staying on one machine is the point of the system, so the client talks to
`127.0.0.1:8000` directly — `app.main` sets `allow_origins=["*"]` for exactly
this reason.

## Layout

```
app/
  page.tsx                        subjects
  subjects/[subjectId]/page.tsx   workspace — lectures · syllabus · coverage + chat
  lectures/[lectureId]/page.tsx   transcript · notes · audio · ASR stats
components/
  chat-panel        routed Q&A, citations that seek the audio
  transcript-view   spans, safeguard reverts, retrieved units
  notes-view        notes, key terms, learning outcomes
  syllabus-panel    upload + the unit prose editor
  coverage-panel    units delivered vs outstanding
  job-progress      pipeline stages for a running job
  audio-player      seekable, exposes a `seek()` handle to the views above
  health-badge      live /health, with warnings
  ui/               shadcn primitives, unmodified
lib/
  api/types.ts      mirror of backend/app/schemas.py
  api/client.ts     fetch wrapper, one function per endpoint
  api/hooks.ts      SWR bindings + the job poller
```

## Three things the UI is deliberate about

**Job progress shows stages, not just a percentage.** Transcription is two
decode passes and runs for minutes. A bar sitting at 40% is indistinguishable
from a hang, so `job-progress.tsx` renders the `stage` field from
`jobs/handlers.py` as a track: audio → SGCD decode → notes → index.

**Configurations that silently break the method are surfaced.** Span target
below 20 s puts conditioning back in the regime where it *regresses* WER, and a
missing ffmpeg makes every upload fail. Both are warnings in the health popover
rather than something you discover from bad transcripts. Per-lecture, a
safeguard revert rate above 25% is flagged the same way — it means the
thresholds need refitting for the current checkpoint.

**The unit prose editor says what prose is for.** `prose` is the field SGCD
conditions on, and rewriting it as a term list reproduces the published failure
mode. The editor says so next to the textarea, because the field looks like
free-form metadata and is not.

## Polling, not sockets

`useJobTracker` polls `/jobs/{id}` every 1.5 s while a job is unsettled. The
backend queue is a worker thread with a `jobs` table and no push channel, so
polling is the available mechanism; at minute-scale jobs it is also the cheap
one. Finished jobs invalidate the SWR keys they could have changed
(`revalidateAfterJob`).

## Notes without a key

Transcription needs no API key; notes, syllabus parsing and chat do. When the
key is missing the backend still finishes the transcript and stops the lecture
at `transcribed` — the notes tab explains that state rather than showing an
empty list.
