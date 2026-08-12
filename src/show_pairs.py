"""The §8.1 validation gate: print reference/hypothesis pairs side by side.

"Before accepting any baseline number, print twenty reference/hypothesis pairs side by
side and inspect them manually. A baseline WER above roughly 60% on this corpus is far
more likely to be a normalisation defect than a model failure."

Each pair is shown at all three normalisation levels together with its per-utterance
WER and its B/U error split, so a suspicious corpus WER can be attributed to the right
cause: level-1 high but level-2 low means script convention, both high means genuine
recognition failure, and a large insertion count on short utterances means the decoder
ran past the audio. The output is written to report/validation_pairs.md so the manual
inspection is part of the record rather than a transient terminal session.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, load_config, read_jsonl
from lexicon import load_lexicon
from normalize import level1, level2
from score import decompose_utterance


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--run", default="B0")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--sort", choices=["none", "worst", "best", "random"],
                    default="none")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    rows = read_jsonl(ROOT / "runs" / a.tier / a.run / "hyps.jsonl")
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    for r in rows:
        r["_m"] = decompose_utterance(level1(r["ref"]), level1(r.get("hyp") or ""), lex)
    scored = [r for r in rows if r["_m"]["wer"] is not None]
    if a.sort == "worst":
        scored.sort(key=lambda r: -r["_m"]["wer"])
    elif a.sort == "best":
        scored.sort(key=lambda r: r["_m"]["wer"])
    elif a.sort == "random":
        import random
        random.Random(a.seed).shuffle(scored)
    sample = scored[:a.n]

    lines = [f"# Validation pairs — {a.tier}/{a.run} ({a.sort} {a.n})", "",
             "Manual inspection gate (§8.1). L1 = script-preserving (headline metric), "
             "L2 = script-invariant (secondary).", ""]
    for r in sample:
        m = r["_m"]
        l1r, l1h = level1(r["ref"]), level1(r.get("hyp") or "")
        lines += [
            f"### {r['utt_id']}  ({r.get('duration')}s)",
            f"* WER {m['wer']:.2f} — sub {m['sub']} ins {m['ins']} del {m['del']}; "
            f"B-errors {m['b_errors']}/{m['b_ref_len']} "
            f"U-errors {m['u_errors']}/{m['u_ref_len']}",
            f"* `REF L1` {l1r}",
            f"* `HYP L1` {l1h}",
            f"* `REF L2` {level2(r['ref'])}",
            f"* `HYP L2` {level2(r.get('hyp') or '')}",
        ]
        if r.get("context"):
            lines.append(f"* `CONTEXT` {r['context'][:160]}…")
        if r.get("hotwords"):
            lines.append(f"* `HOTWORDS` {r['hotwords'][:160]}…")
        if r.get("hyp_before_correction"):
            lines.append(f"* `PRE-CORRECTION` {level1(r['hyp_before_correction'])}")
        if r.get("guard_context_echo"):
            lines.append(f"* context-echo guard FIRED "
                         f"(score {r.get('context_echo_score')})")
        lines.append("")
        print(f"--- {r['utt_id']}  ({r.get('duration')}s, WER={m['wer']:.2f}, "
              f"ins={m['ins']})")
        print(f"  REF: {l1r}")
        print(f"  HYP: {l1h}")

    out = ROOT / "report" / f"validation_pairs_{a.tier}_{a.run}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
