Follow-up queued. Meanwhile, four of five done:

┌────────────────────────┬───────┬──────────┬───────┬─────────┬─────────┬──────────────┬───────┬───────┬─────────┐
│         Model          │ WER % │ WER-sa % │ CER % │ K-WER % │ U-WER % │ script fid % │  RTF  │ speed │   RAM   │
├────────────────────────┼───────┼──────────┼───────┼─────────┼─────────┼──────────────┼───────┼───────┼─────────┤
│ Qwen3-ASR-0.6B         │ 59.74 │ 50.39    │ 50.49 │ 44.66   │ 39.92   │ 47.2         │ 0.339 │ 2.95× │ 1.41 GB │
├────────────────────────┼───────┼──────────┼───────┼─────────┼─────────┼──────────────┼───────┼───────┼─────────┤
│ Qwen3-ASR-1.7B         │ 63.24 │ 47.94    │ 57.26 │ 92.23   │ 39.45   │ 8.4          │ 1.209 │ 0.83× │ 1.26 GB │
├────────────────────────┼───────┼──────────┼───────┼─────────┼─────────┼──────────────┼───────┼───────┼─────────┤
│ Whisper large-v3       │ 74.17 │ 61.94    │ 62.39 │ 63.59   │ 42.63   │ 34.4         │ 0.761 │ 1.31× │ 0.61 GB │
├────────────────────────┼───────┼──────────┼───────┼─────────┼─────────┼──────────────┼───────┼───────┼─────────┤
│ Whisper large-v3-turbo │ 80.23 │ 66.61    │ 65.62 │ 65.53   │ 43.71   │ 32.8         │ 0.229 │ 4.36× │ 0.58 GB │
└────────────────────────┴───────┴──────────┴───────┴─────────┴─────────┴──────────────┴───────┴───────┴─────────┘

Read the 1.7B row carefully — it is not what it looks like. Script fidelity 8.4% and K-WER 92% with U-WER 39.45% (the best of any model) is a contradiction that only resolves one way. Sample output:

REF: 1 index 1 पर argument को दिखाता है
1.7B: वन इंडेक्स वन पर आर्गुमेंट को दिखाता है।     ← "one index one … argument", in Devanagari
0.6B: One index one पर argument को दिखाता है      ← same words, Latin script

Forced to language="Hindi", the 1.7B commits hard to Devanagari and transliterates every English technical term. It heard the words correctly — its WER-sa of 47.94 is the best score in the table — but the SLR104 reference writes those terms in Latin, so strict WER punishes every one.

- Fixable at inference, not by retraining: the language hint is the lever. I'm re-running the 1.7B with auto-detect (--language none) to see if it preserves Latin script.
  ─────────────────────────────────────────────────

Parakeet (4.3 GB) downloading; the 1.7B auto-detect rerun follows it.

  Ran 1 shell command

Waiting on Parakeet.

✻ Brewed for 1m 2s · done 3:00 PM

※ recap: Benchmarking 5 ASR models on OpenSLR 104 Hindi-English test set. Qwen3-ASR-0.6B leads at 59.7% WER, both Whispers trail at 74-80%. Waiting on Parakeet 1.1B to finish downloading and decoding, then a Qwen 1.7B auto-language rerun. (disable recaps in /config)

⏺ Background command "Prefetch all four remaining models sequentially" completed (exit code 0)

  Ran 1 shell command

All weigh

  Ran 1 shell command

All five complete.

  Ran 1 shell command

⏺ Background command "Wait for Parakeet result" completed (exit code 0)

All five models done. Parakeet behaves exactly as its language coverage predicts — and produces a genuinely interesting artifact.

┌────────────────────────┬───────┬──────────┬───────┬─────────┬─────────┬──────────────┬───────┬────────┬─────────┐
│         Model          │ WER % │ WER-sa % │ CER % │ K-WER % │ U-WER % │ script fid % │  RTF  │ speed  │   RAM   │
├────────────────────────┼───────┼──────────┼───────┼─────────┼─────────┼──────────────┼───────┼────────┼─────────┤
│ Qwen3-ASR-0.6B         │ 59.74 │ 50.39    │ 50.49 │ 44.66   │ 39.92   │ 47.2         │ 0.339 │ 2.95×  │ 1.41 GB │
├────────────────────────┼───────┼──────────┼───────┼─────────┼─────────┼──────────────┼───────┼────────┼─────────┤
│ Qwen3-ASR-1.7B         │ 63.24 │ 47.94    │ 57.26 │ 92.23   │ 39.45   │ 8.4          │ 1.209 │ 0.83×  │ 1.26 GB │
├────────────────────────┼───────┼──────────┼───────┼─────────┼─────────┼──────────────┼───────┼────────┼─────────┤
│ Whisper large-v3       │ 74.17 │ 61.94    │ 62.39 │ 63.59   │ 42.63   │ 34.4         │ 0.761 │ 1.31×  │ 0.61 GB │
├────────
REF:  यह कार्य गतिविधि पर निर्भर होता है
HYP:  mouse ko hata nahi sakthi with kakaran hair            ← pure Hindi span: noise

It nails every English token and phonetically hallucinates the Devanagari spans into Latin gibberish. K-WER only counts syllabus keywords, which are ~all English — so an English-only model wins that column by construction. Do not read that 37.86% as Parakeet being good here. Its 91.54% U-WER is the honest number.

★ Insight ─────────────────────────────────────
- This is why the K-WER/U-WER split matters more than corpus WER. A single aggregate would have hidden that Parakeet's errors are perfectly segregated by language, while Whisper's are diffuse. Different failures need different fixes.
- 13.18× realtime is 3× faster than turbo and 17× faster than Qwen 1.7B. RNNT is a streaming transducer — no autoregressive attention over a decoder LM, no 30-second padded windows. Whisper burns compute padding a 5.75 s utterance to 30 s; Parakeet processes only the audio present.
- Parakeet's high script fidelity is an artifact of never having another script available. It scores 58.8% because it can't emit Devanagari — a model with only one output alphabet trivially wins an alphabet-agreement metric. Metrics have preconditions.
  ─────────────────────────────────────────────────

Language-hint ablation on both Qwen models running now (~25 min) — that decides whether the 1.7B's real accuracy is being masked.

  Ran 1 shell command


Three results that need the caveat attached:

1. The 1.7B is the best recognizer but the wrong-alphabet writer. Forced to Hindi it transliterates English terms into Devanagari — "वन इंडेक्स वन" for "one index one". That's why its K-WER is 92% and script fidelity 8.4%, while its WER-sa (47.94) and U-WER (39.45) are the best in the table. It hears the most; it writes it in the script SLR104 doesn't use. Possibly fixable with auto-detect instead of a forced hint — that ablation script is written (run_lang_variants.sh) but the run got interrupted, so it's untested.
2. Parakeet's "best K-WER" and "best script fidelity" are artifacts, not quality. It's English-only, and no Parakeet supports Hindi. K-WER counts syllabus keywords, which are ~all English; script fidelity rewards a model that physically cannot emit Devanagari. Its real number is U-WER 91.54%.
3. Turbo vs large-v3: +6.1 WER for 3.3× the throughput. Scaling within Whisper buys 6 points; switching family buys 14.

For ClassScribe: your backend is currently Whisper turbo at 80.23 WER. Moving to Qwen3-ASR-0.6B is worth ~20 WER points at comparable speed. Qwen exposes a prompt= context slot, so your SGCD syllabus-grounding method transfers in principle — but every SGCD number is Whisper-initial_prompt-specific and would need re-running.

Absolute WERs are inflated by the 2–28 s segmentation (your testcat run shows baseline roughly halving on 26 s spans). Rankings hold; levels don't.

Written up in research/sgcd/RUNLOG.md (appended, per the repo's discipline). New code: src/bench_asr.py, src/bench_report.py, run_bench_all.sh; table at out/tables/bench__test.md.

Two things I can do next — say which: finish the interrupted language-hint ablation (~25 min, decides whether the 1.7B actually beats the 0.6B), or publish this as a shareable report page for your FYP writeup.