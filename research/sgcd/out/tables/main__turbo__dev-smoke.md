# Main results — turbo, dev-smoke split (N=8)

| Cond | Prompt | WER | WER-sa | K-WER | U-WER | CER | Script fid. | ΔWER vs C0 [95% CI] | Worse % | Fallback % |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | no prompt (baseline) | 115.25 | 98.31 | 40.00 | 34.26 | 93.05 | 42.9 | ref | — | — |
| C2 | syllabus keywords, comma-separated (naive baseline) | 170.34 | 145.76 | 10.00 | 58.33 | 153.48 | 64.3 | +55.08 [-105.27, +264.96] | 62 | — |
| C4 | retrieved k units, prose (SGCD) | 66.95 | 50.00 | 20.00 | 40.74 | 65.56 | 64.3 | -48.31 [-144.53, +11.83] | 50 | — |

Negative ΔWER = improvement. CI from a paired bootstrap over utterances (10k resamples, seed 1337). WER-sa is the transliteration-tolerant variant (Devanagari and Latin reduced to a common consonant skeleton).

## Contrasts

_run stats.py to populate_
