# Main results — turbo, testcat split (N=100)

| Cond | Prompt | WER | WER-sa | K-WER | U-WER | CER | Script fid. | ΔWER vs C0 [95% CI] | Worse % | Fallback % |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | no prompt (baseline) | 43.46 | 28.05 | 48.04 | 34.64 | 33.73 | 48.0 | ref | — | — |
| C4 | retrieved k units, prose (SGCD) | 38.64 | 25.53 | 21.24 | 33.94 | 28.72 | 70.0 | -4.83 [-9.21, -0.53] | 32 | — |
| C5 | retrieved from a different course (content-specificity control) | 39.56 | 26.40 | 25.44 | 33.91 | 29.96 | 67.3 | -3.91 [-8.53, +0.61] | 37 | — |

Negative ΔWER = improvement. CI from a paired bootstrap over utterances (10k resamples, seed 1337). WER-sa is the transliteration-tolerant variant (Devanagari and Latin reduced to a common consonant skeleton).

## Contrasts

- **H3  matched vs mismatched syllabus** (C5->C4): ΔWER -0.92 [95% CI -3.53, +1.34], P(improve)=0.768
