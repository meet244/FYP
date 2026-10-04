# Frozen FYRP S5 domain LM

`fyrp_s5_lm.json.gz` contains sparse token n-gram counts, not model weights.
It is built using the same corpus as `research/sgcd/src/fyrp_exp.py`:
syllabus units plus four-utterance windows from DEV lectures, excluding all
30 tuning references. No TEST reference contributes to its counts.

The source speech dataset is [OpenSLR SLR104](https://www.openslr.org/104/),
**CC BY-SA 4.0**. Attribution: the MUCS 2021 multilingual/code-switching ASR
challenge organizers and the contributors listed on the dataset page. The
derived count artifact is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The remaining inputs are this project's research syllabus files.

Rebuild from `backend/` with `.venv/bin/python scripts/export_s5_lm.py`.
Export does not decode audio or tune against TEST. The artifact records the
corpus digest, tokenizer vocabulary digest, DEV source IDs, disjoint lecture
IDs, and frozen settings. Jobs retain its SHA-256 digest; a changed artifact
or mismatched tokenizer fails explicitly rather than silently using another LM.

S5 uses five beams, weight 0.2, word-length bonus 0.5, absolute discount 0.75,
unigram smoothing 0.01, and no repetition penalty. Research used a 200-token
generation cap on short utterances. Serving defaults to a 512-token cap to avoid
cutting off the longer application spans; this difference is saved in each run.
It is independent of SGCD and S6. `baseline_text` for S5 is the unrescored top
beam, **not** a separate S0 greedy decode. The application splits recordings
into roughly 25-second spans, so its deployment WER has not been measured by
the original 50-utterance experiment. These tutorial-domain counts also do
not establish improved recognition for arbitrary new classroom subjects.
