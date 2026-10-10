# ClassScribe frontend

Next.js 16 / React 19 client for the [backend](../backend/README.md), using SWR,
Tailwind and shadcn components. It sends no model API requests directly.

```bash
npm ci
# If the API isn't on port 8000, set NEXT_PUBLIC_API_URL in .env.local.
npm run dev
```

Start the backend separately, then open `http://localhost:3000`.

- Subject studio: Chats, Files, and Audio recordings. Page-wide drops route documents
  and recordings to their separate libraries.
- Files: PDFs (including syllabuses), PPTX, DOCX, TXT, Markdown, CSV and images.
  Text is extracted, chunked and embedded for chat. Optional syllabus setup and
  coverage are available under Files.
- Recording upload: Qwen3-ASR-0.6B selected by default with S5 five-beam decoding.
  Choose **Fast · single pass** to skip domain-LM rescoring for new uploads;
  accuracy may differ. Upload settings do not change jobs already queued.
- Jobs: a minimizable processing panel shows the source, actual progress, current
  stage and queue count while chat stays usable. It respects reduced motion and
  starts minimized on small screens. Status is restored after refresh, with
  cancellation, failure messages and retries.
  During transcription, percentages count completed segments in the named pass;
  notes and indexing follow afterward. They are not estimates of time remaining.
- Lecture: playable transcript, notes, notes-only regeneration, model reprocessing,
  per-run transcript history and statistics.
- Chat: subject-scoped sources and citations that open an in-page preview without
  leaving the conversation or automatically downloading files. PDFs and images
  show the original; Office and text documents show extracted text. Recording
  citations show the transcript, cited notes, and audio at the cited timestamp.
- Studio: generate quizzes, flashcards, mind maps, reports, slide decks, and
  infographics from a whole subject, syllabus unit, or recording. Quizzes have
  hints, answers, explanations, and source references; flashcards support recall
  practice. Reports offer study guide, briefing, FAQ, and glossary formats.
  Use the customization controls for focus, supported item counts, and quiz
  difficulty. Items are saved and support reopening, failure retry, and deletion.
  Chat requests such as "make a quiz on unit 2" create the same Studio items.

The header reports the configured model and missing FFmpeg/Gemini configuration.
Standard recognition is one pass. Syllabus narration adds a second pass and is
experimental on Qwen; published Whisper gains are not displayed as Qwen results.

Without a Gemini key, local transcription and source-quoted chat work. Generated
notes, syllabus parsing, image extraction, composed answers, and Studio
generation require Gemini configuration. Generation
runs separately from transcription, so a quiz does not queue behind ASR.
Multiple output formats do not constitute student/tutor perspective-conditioned
summarisation or evidence of improved learning outcomes.

The API allows localhost frontend origins by default; see the backend configuration for
other origins. This is a single-machine application without user authentication.

Verify types and production output:

```bash
npx tsc --noEmit
npm run build
```

See [implementation and research audit](../backend/IMPLEMENTATION_STATUS.md) for
what was missing, what is now connected, verification and remaining research.

The [results summary](../results.md) distinguishes ASR experiments from the
workflow demonstrations in [dump/](../dump/). These demonstrations show source
previews and timestamp navigation; they do not measure citation correctness or
note quality.
