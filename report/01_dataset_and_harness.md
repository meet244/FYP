# 1. Dataset, environment and evaluation harness

## 1.1 Data

Evaluation uses the Hindi–English code-switched **test** portion of OpenSLR SLR104
(MUCS 2021 subtask-2). The material is drawn from Spoken Tutorial recordings on
technical topics, and the code-switching arises predominantly from that technical
content. Only the test tarball (443 MB) was downloaded; no fine-tuning is performed, so
the 7.3 GB train tarball is not required.

Distribution is Kaldi-style (`wav.scp`, `text`, `utt2spk`, `segments`). Measured
contents after preparation:

| Property | Value |
|---|---|
| Recordings (long tutorials) | 30 |
| Utterances (from `segments`) | 3,136 |
| Total audio | 5.18 h |
| Mean / median utterance | 5.9 s / 5.0 s |
| Min / max utterance | 1.0 s / 41.0 s |
| Sample rate | 16 kHz, 16-bit mono (no resampling needed) |

`src/prepare_slr104.py` cuts each utterance once into `data/audio/test/<utt_id>.wav`
(576 MB) and writes `data/manifests/test.jsonl`. Cutting is grouped by recording so
each source file is decoded a single time rather than once per utterance.

**Lecture topics present in the test set** (identified from the lecture titles):
LibreOffice Impress (9 recordings), LibreOffice Writer (5), Mozilla Thunderbird (4),
gedit text editor (4), JChemPaint (3), Xfig (3), C programming (2).

### Evaluation subsets

All development and ablation experiments use a fixed random subset of **800 utterances
(seed 1337, 77.9 min)**; final systems are additionally evaluated on the complete
Hindi–English test set. The subset is further split into a **200-utterance dev slice**
(`dev200.jsonl`, used only for threshold and *k* selection) and a **600-utterance
evaluation slice** (`eval600.jsonl`), so no hyper-parameter is tuned on the utterances
used to report a result.

## 1.2 Environment

MacBook Air, Apple Silicon, 16 GB RAM, 10 cores, macOS 15. Python 3.11.9.
Whisper **large-v3** runs locally through `faster-whisper` 1.1.1 / CTranslate2 with
`compute_type="int8"`. CTranslate2 does not use the Apple Silicon GPU, so all decoding
is CPU-bound. Decoding is cached per `(utterance, backend, config-hash)` in `cache/asr/`,
which makes a re-run of an already-decoded condition effectively free.

Fixed decoding settings: `beam_size=5`, `temperature=0.0`,
`condition_on_previous_text=False` (utterances are independent),
`vad_filter=False` (audio is already sentence-segmented).

## 1.3 Normalisation

On code-switched data a large fraction of apparent errors are formatting mismatches
rather than recognition failures, so scoring is defined before any modelling work.
Three normalisation levels are reported:

1. **`basic_norm` — the headline metric.** NFC normalisation, Devanagari→Latin digit
   mapping, punctuation stripping (including `।` and `॥`), lower-casing, whitespace
   collapse. Script-preserving, and therefore directly comparable with the MUCS
   baselines and published Whisper numbers.
2. **`script_invariant_norm` — secondary.** Devanagari is romanised (ITRANS) and both
   sides pass through a light phonetic fold: `ph→f`, `c→k`, `w→v`, `x→ks`, `z→j`,
   aspirate reduction, doubled-letter collapse, inherent-schwa deletion, anusvara→`n`,
   plus English-only rules for soft *c*/*g* and `-tion`. The candra vowels used to
   write English loanwords (`ऑफिस`, `फॉन्ट`, `कॉपी`) have no ITRANS equivalent and
   survive transliteration unconverted, so they are folded onto `ओ`/`ए` first — without
   this step romanisation silently drops the vowel.
3. **`consonant_skeleton_norm` — orthography-agnostic lower bound.** Vowels are removed
   from the romanised form. English and Devanagari spellings of a loanword agree on
   consonants but rarely on vowels (`स्लाइड` *slaid* vs *slide*; `सिलेक्ट` *silekt* vs
   *select*). On a 24-pair loanword probe this matches 21/24, against 5/24 for level 2.
   It is deliberately lenient — it also collapses genuinely distinct Hindi words
   (`कर`/`कोर`, `बात`/`बीत`) — so it is reported as a **lower bound on WER**, never as
   the WER.

The gap between levels 1 and 3 quantifies how much of the error is orthographic rather
than acoustic, which is a reportable finding in its own right.

## 1.4 Metrics

Corpus WER (headline), CER, script-invariant WER, skeleton-WER lower bound, and
**term-level precision / recall / F1** restricted to the syllabus lexicon. The term
metric is necessary because compound splitting of a technical term
(`matplotlib` → `mat plot lib`) is a real failure that barely moves corpus WER.
Per-utterance edit counts and reference lengths are stored for every run
(`runs/<name>/per_utt.jsonl`) to support the regression analysis and the paired
bootstrap test.

## 1.5 Syllabus resource

Twelve hand-written syllabus documents (`syllabus/raw/*.md`, 200–400 words each) cover
the course topics: LibreOffice Impress, LibreOffice Writer, C programming, Mozilla
Thunderbird, gedit, JChemPaint, Xfig, Gmail/web, GNU-Linux shell, Python, LaTeX and
Scilab. The last four are **distractors** — they are not present in the test audio, and
exist so that retrieval is a real decision rather than a formality.

Documents are chunked (120 words, 30 overlap → 34 chunks) and embedded with
`paraphrase-multilingual-MiniLM-L12-v2`. The term lexicon (977 terms) is extracted from
the syllabus documents only, in document order, with a generic-English stop list.
**No terms are mined from the test transcripts**; doing so would be test-set leakage.

On a probe of the seven lecture titles, top-3 retrieval selects the correct topic 7/7.
