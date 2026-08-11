"""Print ref/hyp pairs side by side. Look at these before believing any WER."""
import argparse
import json
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent))
from normalize import basic_norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hyps", default="runs/S0_baseline/hyps.jsonl")
ap.add_argument("-n", type=int, default=20)
ap.add_argument("--sort", choices=["none", "worst", "best"], default="none")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.hyps, encoding="utf-8")]
for r in rows:
    ref, hyp = basic_norm(r["ref"]), basic_norm(r["hyp"])
    r["_wer"] = jiwer.wer([ref], [hyp]) if ref.strip() else None
if a.sort != "none":
    rows = sorted([r for r in rows if r["_wer"] is not None],
                  key=lambda r: r["_wer"], reverse=(a.sort == "worst"))

for r in rows[:a.n]:
    print(f"--- {r['utt_id']}  ({r.get('duration')}s, WER={r['_wer']:.2f})")
    print(f"  REF: {r['ref']}")
    print(f"  HYP: {r['hyp']}")
    if r.get("prompt"):
        print(f"  PROMPT: {r['prompt'][:120]}...")
