#!/bin/bash
# Drive the cross-model benchmark: fetch each checkpoint, decode, move on.
# One model per process so weights are never co-resident in the 8 GB of unified
# memory. `hf download` blocks on the prefetcher's blob lock if it is mid-flight,
# so running this alongside a prefetch chain is safe.
# (macOS ships bash 3.2 — no associative arrays, hence the case block.)
set -u
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
PY=.venv/bin/python

repo_of() {
  case "$1" in
    turbo)    echo mlx-community/whisper-large-v3-turbo ;;
    wlv3)     echo mlx-community/whisper-large-v3-mlx ;;
    qwen17)   echo Qwen/Qwen3-ASR-1.7B-hf ;;
    qwen06)   echo Qwen/Qwen3-ASR-0.6B-hf ;;
    parakeet) echo mlx-community/parakeet-rnnt-1.1b ;;
  esac
}

for k in turbo wlv3 qwen17 parakeet; do
  if [ -f out/hyps/${k}__C0__test.jsonl ]; then echo "[skip] $k already decoded"; continue; fi
  repo=$(repo_of "$k")
  echo "=== $k  fetch $repo  $(date +%H:%M:%S)"
  .venv/bin/hf download "$repo" >/dev/null 2>&1 || { echo "FAIL fetch $k"; continue; }
  echo "=== $k  decode  $(date +%H:%M:%S)"
  (cd src && ../$PY -u bench_asr.py --model "$k" --split test 2>&1 | grep -v processor_kwargs)
done

echo "=== report $(date +%H:%M:%S)"
(cd src && ../$PY bench_report.py --split test)
