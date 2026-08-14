# Main results — turbo, test split (N=150)

| Cond | Prompt | WER | WER-sa | K-WER | U-WER | CER | Script fid. | ΔWER vs C0 [95% CI] | Worse % | Fallback % |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | no prompt (baseline) | 85.69 | 71.87 | 65.53 | 43.50 | 67.92 | 32.6 | ref | — | — |
| C1 | generic code-mixed sentence, no course content (style control) | 102.32 | 90.48 | 55.83 | 44.38 | 82.29 | 41.6 | +16.63 [-3.61, +36.66] | 31 | — |
| C2 | syllabus keywords, comma-separated (naive baseline) | 136.22 | 118.43 | 31.55 | 53.18 | 114.16 | 60.9 | +50.53 [+16.55, +90.94] | 46 | — |
| C3 | whole-syllabus prose | 99.35 | 88.81 | 33.50 | 41.68 | 82.89 | 60.0 | +13.66 [-7.70, +35.35] | 32 | — |
| C4 | retrieved k units, prose (SGCD) | 90.80 | 81.27 | 35.44 | 41.95 | 75.97 | 58.8 | +5.11 [-13.24, +23.28] | 34 | — |
| C5 | retrieved from a different course (content-specificity control) | 87.71 | 77.62 | 35.92 | 40.19 | 74.33 | 58.6 | +2.02 [-13.98, +18.98] | 29 | — |
| C6 | retrieved using the reference (oracle retrieval topline) **(topline, oracle)** | 95.61 | 85.64 | 29.61 | 40.53 | 78.73 | 62.1 | +9.92 [-10.49, +30.23] | 33 | — |
| C7 | C4 + confidence guard (full system) | 71.50 | 61.28 | 35.44 | 41.14 | 56.22 | 58.1 | -14.19 [-26.55, -4.79] | 24 | 12.7 |

Negative ΔWER = improvement. CI from a paired bootstrap over utterances (10k resamples, seed 1337). WER-sa is the transliteration-tolerant variant (Devanagari and Latin reduced to a common consonant skeleton).

## Contrasts

- **H4  prose rendering vs keyword list** (C2->C3): ΔWER -36.88 [95% CI -75.64, -5.18], P(improve)=0.990
- **H5  retrieval vs whole syllabus** (C3->C4): ΔWER -8.55 [95% CI -22.89, +5.71], P(improve)=0.875
- **H3  matched vs mismatched syllabus** (C5->C4): ΔWER +3.09 [95% CI -9.82, +16.16], P(improve)=0.327
- **guard effect** (C4->C7): ΔWER -19.30 [95% CI -33.92, -5.90], P(improve)=0.999
