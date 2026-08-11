"""Week-2 pilot: language='hi' vs 'en' vs auto-detect on the first N subset utterances.

The choice changes WER substantially on code-switched audio, so fix it once, here,
and report the pilot table.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backends import DecodeConfig, get_backend  # noqa: E402
from score import score_file  # noqa: E402
from transcribe import transcribe_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/manifests/subset.jsonl")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--compute-type", default="int8")
a = ap.parse_args()

backend = get_backend("local", model_size="large-v3", compute_type=a.compute_type)
results = {}
for lang in ["hi", "en", None]:
    tag = lang or "auto"
    out = Path("runs") / f"pilot_lang_{tag}" / "hyps.jsonl"
    transcribe_manifest(backend, a.manifest,
                        lambda row, l=lang: DecodeConfig(language=l),
                        out, limit=a.n)
    m = score_file(out)
    results[tag] = m
    print(f"[{tag}] WER={m['wer']:.4f}  CER={m['cer']:.4f}  "
          f"scriptWER={m['wer_script_invariant']:.4f}")

Path("runs/pilot_language.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nlanguage  WER     CER     script-invariant WER")
for k, m in results.items():
    print(f"{k:8s}  {m['wer']:.4f}  {m['cer']:.4f}  {m['wer_script_invariant']:.4f}")
