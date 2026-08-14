"""Print (ref, normalised ref, hyp, normalised hyp) quadruples for manual reading.

Plan §4.1 calls this non-negotiable: look for numerals ("5" vs "पाँच"), English
words written in Devanagari, and stray tokens before freezing the normaliser.

    python src/sanity_check.py                 # references only (pre-decode)
    python src/sanity_check.py turbo C0 test   # with hypotheses
"""
import json
import sys

from config import HYP, OUT
from normalize import normalize, script_mix


def main():
    if len(sys.argv) >= 4:
        model, cond, split = sys.argv[1:4]
        path = HYP / f"{model}__{cond}__{split}.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        rows = [
            json.loads(l)
            for l in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        rows = [r for r in rows if r.get("in_eval")][:20]

    n = min(20, len(rows))
    for i, r in enumerate(rows[:n], 1):
        print(f"--- {i}. {r['utt_id']}")
        print(f"  REF   : {r['ref']}")
        print(f"  nREF  : {normalize(r['ref'])}")
        if "hyp" in r:
            print(f"  HYP   : {r['hyp']}")
            print(f"  nHYP  : {normalize(r['hyp'])}")

    mix = [script_mix(r["ref"]) for r in rows]
    if mix:
        print(
            f"\nreference script mix over {len(mix)} utts: "
            + "  ".join(f"{k}={sum(m[k] for m in mix)/len(mix)*100:.1f}%" for k in ("dev", "lat", "other"))
        )
    digits = sum(any(c.isdigit() for c in r["ref"]) for r in rows)
    print(f"references containing digits: {digits}/{len(rows)}")


if __name__ == "__main__":
    main()
