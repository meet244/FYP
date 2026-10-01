### SLR104 Hindi–English test, n=150, zero-prompt

| Model | WER % | WER-sa % | CER % | K-WER % | U-WER % | script fid % | RTF | x realtime | peak RAM GB |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR-0.6B (torch/MPS) | 59.74 | 50.39 | 50.49 | 44.66 | 39.92 | 47.2 | 0.3388 | 2.95 | 1.41 |
| Qwen3-ASR-1.7B (torch/MPS) | 63.24 | 47.94 | 57.26 | 92.23 | 39.45 | 8.4 | 1.2093 | 0.83 | 1.26 |
| Whisper large-v3 (fp16 MLX) | 74.17 | 61.94 | 62.39 | 63.59 | 42.63 | 34.4 | 0.7609 | 1.31 | 0.61 |
| Whisper large-v3-turbo (MLX) | 80.23 | 66.61 | 65.62 | 65.53 | 43.71 | 32.8 | 0.2291 | 4.36 | 0.58 |
| NVIDIA Parakeet RNNT 1.1B (MLX) | 97.45 | 81.93 | 81.53 | 37.86 | 91.54 | 58.8 | 0.0759 | 13.18 | 0.99 |
