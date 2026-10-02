| Rank | Step | Method | WER % | Word Acc % | ΔWER vs base (95% CI) | CER % | WER-sa % | K-WER % | U-WER % | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S5 | N-best rescoring with domain n-gram LM | 55.69 | 44.31 | -3.18 [-6.3, -0.2] | 43.84 | 47.14 | 42.17 | 34.95 | 1.36 |
| 2 | S8 | TTS synthetic data (partial FT on TTS only) | 55.85 | 44.15 | -3.01 [-6.8, +0.8] | 43.78 | 50.34 | 20.48 | 39.03 | 0.61 |
| 3 | S3 | Context biasing via system prompt | 57.36 | 42.64 | -1.51 [-4.8, +1.3] | 46.73 | 48.99 | 49.40 | 34.95 | 0.66 |
| 4 | S4 | BM25 retrieval-constrained span correction | 58.03 | 41.97 | -0.84 [-2.0, +0.2] | 46.29 | 49.83 | 48.19 | 35.73 | 0.29 |
| 5 | S0 | Baseline (Qwen3-ASR-0.6B, greedy, no context) | 58.86 | 41.14 | — | 46.70 | 50.00 | 50.60 | 36.31 | 0.29 |
| 6 | S7 | Embedding-only tuning (tokenizer expansion) | 58.86 | 41.14 | +0.00 [+0.0, +0.0] | 46.70 | 50.00 | 50.60 | 36.31 | 0.63 |
| 7 | S1 | Fuzzy / edit-distance lexicon correction | 58.86 | 41.14 | +0.00 [-0.8, +0.8] | 46.79 | 49.83 | 49.40 | 36.50 | 0.29 |
| 8 | S6 | Shallow fusion (n-gram LM in decoding) | 60.03 | 39.97 | +1.17 [-1.8, +4.4] | 49.65 | 50.67 | 62.65 | 36.12 | 0.29 |
| 9 | S2 | Phonetic matching for code-switched OOV | 60.37 | 39.63 | +1.51 [-0.5, +3.5] | 46.79 | 51.01 | 42.17 | 39.42 | 0.29 |
| 10 | S10 | LoRA r=8 enc+dec attention (real+TTS) | 62.21 | 37.79 | +3.34 [-9.0, +22.3] | 48.67 | 56.06 | 39.76 | 38.83 | 0.65 |
| 11 | S9 | Partial FT, last-4 decoder layers (real+TTS) | 67.22 | 32.78 | +8.36 [-0.6, +18.9] | 55.56 | 58.59 | 31.33 | 40.39 | 0.65 |

Settings:
{
 "S1": "rapidfuzz ratio >= 85",
 "S2": "phonetic-key ratio >= 85",
 "S4": "BM25 top-3 of 299 segs, thr 80",
 "S5": "k=5 beams, lam=0.2, len bonus=0.5",
 "LM": {
  "test_ppl": 13.75252033934461
 },
 "S6": "LM weight 0.1 (dev grid {'0.0': 0.36486486486486486, '0.1': 0.3621621621621622, '0.3': 0.41081081081081083})",
 "S3": "top-80 syllabus terms as system prompt",
 "S7": {
  "new_tokens": 14,
  "trained_params": 14336,
  "examples": 100,
  "epochs": 2,
  "lr": 0.0005,
  "train_s": 3128.7
 },
 "S8": {
  "trained_params": 62923776,
  "examples": 80,
  "epochs": 1,
  "lr": 2e-05,
  "train_s": 52.7
 },
 "S9": {
  "trained_params": 62923776,
  "examples": 180,
  "epochs": 1,
  "lr": 2e-05,
  "train_s": 82.8
 },
 "S10": {
  "lora_modules": 92,
  "trained_params": 1662976,
  "examples": 180,
  "epochs": 1,
  "lr": 0.0002,
  "train_s": 210.7
 }
}
