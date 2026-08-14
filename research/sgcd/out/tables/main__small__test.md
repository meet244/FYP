# Main results — small, test split (N=150)

| Cond | Prompt | WER | WER-sa | K-WER | U-WER | CER | Script fid. | ΔWER vs C0 [95% CI] | Worse % | Fallback % |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | no prompt (baseline) | 171.50 | 159.72 | 99.51 | 75.17 | 147.73 | 1.6 | ref | — | — |
| C1 | generic code-mixed sentence, no course content (style control) | 247.57 | 239.02 | 99.51 | 77.33 | 160.28 | 1.6 | +76.07 [+42.65, +113.00] | 48 | — |
| C2 | syllabus keywords, comma-separated (naive baseline) | 330.76 | 313.29 | 40.29 | 93.30 | 344.80 | 46.7 | +159.26 [+90.94, +230.38] | 49 | — |
| C3 | whole-syllabus prose | 148.81 | 138.66 | 80.58 | 70.70 | 117.43 | 20.5 | -22.68 [-55.99, +10.58] | 33 | — |
| C4 | retrieved k units, prose (SGCD) | 143.23 | 131.72 | 74.27 | 69.15 | 122.57 | 23.7 | -28.27 [-60.77, +3.10] | 35 | — |
| C5 | retrieved from a different course (content-specificity control) | 160.75 | 149.13 | 88.35 | 69.08 | 128.49 | 16.0 | -10.75 [-47.43, +26.30] | 32 | — |
| C6 | retrieved using the reference (oracle retrieval topline) **(topline, oracle)** | 168.88 | 158.71 | 67.48 | 69.15 | 138.48 | 25.6 | -2.61 [-42.13, +37.63] | 35 | — |
| C7 | C4 + confidence guard (full system) | 157.78 | 146.68 | 79.13 | 73.14 | 129.62 | 17.9 | -13.72 [-23.47, -5.73] | 16 | 46.0 |

Negative ΔWER = improvement. CI from a paired bootstrap over utterances (10k resamples, seed 1337). WER-sa is the transliteration-tolerant variant (Devanagari and Latin reduced to a common consonant skeleton).

## Contrasts

- **H4  prose rendering vs keyword list** (C2->C3): ΔWER -181.95 [95% CI -252.58, -116.53], P(improve)=1.000
- **H5  retrieval vs whole syllabus** (C3->C4): ΔWER -5.58 [95% CI -32.84, +21.22], P(improve)=0.651
- **H3  matched vs mismatched syllabus** (C5->C4): ΔWER -17.52 [95% CI -45.96, +10.66], P(improve)=0.888
- **guard effect** (C4->C7): ΔWER +14.55 [95% CI -15.44, +45.84], P(improve)=0.177
