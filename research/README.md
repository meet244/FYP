# ClassScribe research

ClassScribe studies Hindi-English code-switched lecture transcription and
implements a course study workspace around transcripts, notes, uploaded
materials, and source-cited revision outputs.

Documentation review: 2026-10-10. The original Whisper proposal, later Qwen
experiments, and current application are separate evidence categories.

## Research questions and findings

| Study | Question | Supported answer |
| --- | --- | --- |
| Original Whisper SGCD | Does syllabus context improve recognition, and is the benefit content-specific? | On 100 long spans, whole-syllabus narration reduces WER from 43.46% to 37.23%. Mismatched context captures much of the retrieved-narration gain; semantic grounding is not established. |
| Five-model benchmark | Which tested model best matches the corpus's Hindi/Latin convention? | Qwen3-ASR-0.6B has the lowest strict WER, 59.74%, on the shared 150-utterance sample. |
| Ten-method FYRP comparison | Which independent adaptation improves Qwen 0.6B over its own baseline? | S5 reduces WER from 58.86% to 55.69% on 50 utterances, delta -3.18 points, reported 95% CI [-6.3, -0.2]. |

See the [consolidated results](../results.md) for tables, metric caveats, and
the difference between research settings and serving defaults.

## Seven literature gaps addressed

The source is the [enriched literature-review workbook](../dump/Lecture-Note-Taking-Literature-Review_Enriched.xlsx),
particularly its Data sheet summaries, pipelines, and Drawbacks / Cons column.
Rows below are actual worksheet rows. Limitations are the review's assessments,
often based on abstracts, not independently verified full-paper claims.
These gaps describe experiments or implemented capabilities, not demonstrated
superiority over every prior system.

| Gap | Prior work and recorded limitation | Project response and boundary |
| --- | --- | --- |
| 1. Limited measured Hindi-English code-switching support | *Automated Text Note Generator* (2026, row 5) reports approximate bilingual accuracy without a dataset, WER, or matched baselines. | [Five models on the same 150 utterances](sgcd/out/tables/bench__test.md), with strict/script-tolerant errors and runtime. Supplies bounded bilingual evaluation, not general classroom robustness. |
| 2. Recognition without lecture-domain adaptation | *Transcription, Translation and Summarization to Improve Educational Understanding* (2024, row 12) lacks domain adaptation; *Automated Tagging to Enable Fine-Grained Browsing of Lecture Videos* (2011, row 15) does not address technical-vocabulary errors. | [S5 domain-LM rescoring](sgcd/out/fyrp/results.md) improves overall and keyword error over S0 on 50 utterances. New-subject gains are unmeasured. |
| 3. Commercial transcription API dependency | *Automated Text Note Generator* (row 5) uses Deepgram; *AUTOMATED LECTURE NOTE-TAKING SYSTEM* (2024, row 29) uses external Whisper-1/GPT-4 APIs. | [Local Qwen recognition](../backend/README.md) removes commercial ASR dependency. Gemini generation means the complete system is not offline. |
| 4. Slides, captions, or custom hardware required | *Automatic Lecture Annotation* (2020, row 3) assumes slides; *Towards Automated Study Guides for MOOCs* (2015, row 16) requires captions; *Let's Reinvent Note Taking* (2018, row 25) requires hardware. | [Recorded-audio processing](../backend/app/jobs/handlers.py) creates transcripts and notes without those prerequisites. Automatic blackboard-video understanding is not implemented. |
| 5. Extractive summaries with limited content restructuring | *Automatic Running Notes Generation from Audio Lecture using NLP for Comprehensive Learning* (2024, row 6) uses TextRank; *Automated Extraction and Augmentation of Key Information from Audio using Speech Recognition and Text Summarization* (undated, row 11) uses TF-IDF. | [Note synthesis](../backend/app/notes/synthesize.py) generates topic sections, definitions, competency statements, timestamps, and optional unit links. Better fidelity than extractive methods has not been measured. |
| 6. Separate input paths rather than shared source access | *Unified AI-Driven Multimodel Framework for Real-Time Lecture Understanding and Study Support* (2026, row 31) separately summarises YouTube audio and PDFs, with limited formats. | [Subject retrieval](../backend/app/rag/answer.py) combines transcripts, notes, syllabus units, and document/image text in cited answers. Shared text retrieval is not temporal audio-video fusion. |
| 7. Practice generation needs prepared notes or is deferred | *Automated generation of practice questions from semi-structured lecture notes* (2012, row 17) needs existing notes; the unified framework (row 31) leaves questions for future extension. | [Studio](../backend/app/studio/generators.py) connects lecture processing to quizzes and flashcards for a subject, unit, or recording. Question correctness and learning benefit need evaluation. |

The [tutorial](<../dump/RE Tutorial 1.pdf>) also motivates structured notes,
revision features, and timestamp navigation. Its noise/accent motivation is
not evidence that those recognition problems have been solved.

## Source reconciliation

The workbook has 32 entries and 31 distinct titles: *Echonotes* appears in rows
**2 and 30**. Some cross-reference notes use inconsistent numbering.
The workbook is preserved.

| Paper | Tutorial year | Workbook year |
| --- | --- | --- |
| Automatic Lecture Summarizer | 2020 | Not supplied |
| Automated Extraction and Augmentation of Key Information from Audio | 2022 | Not supplied |
| Pinch Hitter | 2024 | 2017 |
| Automated Tagging to Enable Fine-Grained Browsing of Lecture Videos | 2025 | 2011 |
| Towards Automated Study Guides for MOOCs | 2020 | 2015 |
| Automated generation of practice questions from semi-structured lecture notes | 2021 | 2012 |

Gap-table years follow workbook metadata. Resolve these conflicts against
original publications before submitting a formal bibliography. Source
documents have not been edited.

The tutorial title includes **multi-perspective summarisation**. Current
Studio report formats (study guide, briefing, FAQ, glossary) are different
outputs, not a validated student/tutor perspective-conditioned summariser.
The perspective-based dialogue paper (row 33) is adjacent literature, not
evidence that ClassScribe implements or outperforms that method.

## Contents and reproduction

| Path | Purpose |
| --- | --- |
| [REPLICATION.md](REPLICATION.md) | Reproduce all three experiments |
| [sgcd/README.md](sgcd/README.md) | Conditions, metrics, layout, and caches |
| [sgcd/COMPARISON.md](sgcd/COMPARISON.md) | Generated original Whisper comparison |
| [sgcd/out/fyrp/results.md](sgcd/out/fyrp/results.md) | Detailed Qwen methods and settings |
| [sgcd/RUNLOG.md](sgcd/RUNLOG.md) | Append-only historical experiment record |
| [paper/](paper/) | IEEE-format paper sources; some figures/evaluations remain incomplete |
| [design/](design/) | Historical design record, not current implementation specification |
| Formatting reference | Earlier documentation refers to a reference directory that is not supplied in this checkout. |
| [../dump/](../dump/) | Literature sources and demonstrations |

Verify the original SGCD table against saved scores, from the repository root:

    cd research/sgcd/src
    ../.venv/bin/python make_comparison.py --verify

This checks cached scores and nine qualitative claims; it does not perform a
fresh decode. See the [replication guide](REPLICATION.md) for decoding and reports.

## Research discipline and remaining evidence

- Original SGCD hypotheses are in [PREREGISTRATION.md](sgcd/PREREGISTRATION.md).
  The later benchmark and FYRP are separate experiments.
- Tune on internal DEV lectures and evaluate on lecture-disjoint TEST lectures.
  FYRP uses 30 tuning and 50 test utterances; uncertainty is based on utterances,
  not independent classrooms.
- Preserve logs and experimental JSON. Screenshots corroborate outputs;
  demonstrations are not additional accuracy measurements.
- Absolute WER cannot establish superiority over papers using different
  datasets, segmentation, scripts, language hints, or decoding settings.
- Classroom robustness, error propagation, note/citation quality, learning
  outcomes, and production-scale deployment remain open.
