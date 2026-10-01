#!/bin/bash
# Language-hint ablation: both Qwen models with auto-detect instead of a forced
# "Hindi" hint. Isolates output-script convention from recognition accuracy.
set -u
cd "$(dirname "$0")"
while [ ! -f out/hyps/parakeet__C0__test.jsonl ]; do sleep 20; done
for k in qwen17 qwen06; do
  echo "=== $k auto  $(date +%H:%M:%S)"
  (cd src && ../.venv/bin/python -u bench_asr.py --model "$k" --split test \
      --language none --suffix auto 2>&1 | grep -v processor_kwargs)
done
echo "=== done $(date +%H:%M:%S)"
