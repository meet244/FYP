#!/bin/bash
# FYRP: ten independent improvement methods on Qwen3-ASR-0.6B (SLR104 Hindi-English).
# One process per stage so fp16 decode weights and fp32 training weights are never
# co-resident in 8 GB unified memory. Every stage caches; re-running is near-free.
set -eu
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
F='warn|Loading weights|processor_kwargs'
$PY -u fyrp_exp.py prep
$PY -u fyrp_exp.py decode 2>&1 | grep -v -i -E "$F"
for m in emb tts partial lora; do
  $PY -u fyrp_exp.py train --method $m 2>&1 | grep -v -i -E "$F"
done
$PY -u fyrp_exp.py report
