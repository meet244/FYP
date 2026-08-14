"""DEV-only tuning sweep. Nothing here may ever be run on the TEST split.

Tunes, in order:
  1. language="hi" vs language=None          (C0 baseline WER on DEV)
  2. k in {1,2,3} retrieved units            (C4 WER on DEV)
  3. prompt token cap in {120, 200}          (C4 WER on DEV)
  4. guard thresholds                        (post-hoc over cached C0/C4 DEV hyps)

Prints a frozen config block to paste into RUNLOG.md and decode.py.

    python src/sweep_dev.py --model small
"""
import argparse
import itertools
import json

from config import HYP, MODELS
import decode
from decode import apply_guard, run
from score import keyword_set, score_rows


def wer_of(rows, kws):
    return score_rows(rows, kws)[0]["wer"]


def read(model, cond, split):
    p = HYP / f"{model}__{cond}__{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=list(MODELS))
    a = ap.parse_args()
    kws = keyword_set()
    m = a.model
    log = {}

    print("=== 1. language ===")
    lang_wer = {}
    for lang in ("hi", None):
        decode.DECODE["language"] = lang
        rows = run(m, "C0", "dev", tag_suffix=f"-lang-{lang or 'auto'}")
        lang_wer[lang] = wer_of(rows, kws)
        print(f"  language={lang!r:6}  C0 WER={lang_wer[lang]*100:.2f}")
    best_lang = min(lang_wer, key=lang_wer.get)
    decode.DECODE["language"] = best_lang
    log["language"] = best_lang
    print(f"  -> chosen language={best_lang!r}")

    base_tag = f"dev-lang-{best_lang or 'auto'}"
    c0 = read(m, "C0", base_tag)
    fp = {x["utt_id"]: x["hyp"] for x in c0}
    c0_wer = lang_wer[best_lang]

    print("\n=== 2/3. k and prompt cap (C4) ===")
    grid = {}
    for k, cap in itertools.product((1, 2, 3), (120, 200)):
        rows = run(m, "C4", "dev", first_pass=fp, k=k, max_tokens=cap,
                   tag_suffix=f"-k{k}-cap{cap}")
        grid[(k, cap)] = wer_of(rows, kws)
        print(f"  k={k} cap={cap}:  C4 WER={grid[(k,cap)]*100:.2f}  (C0={c0_wer*100:.2f})")
    best_k, best_cap = min(grid, key=grid.get)
    log["k"], log["max_prompt_tokens"] = best_k, best_cap
    print(f"  -> chosen k={best_k}, cap={best_cap}")

    print("\n=== 4. guard thresholds ===")
    c4 = read(m, "C4", f"dev-k{best_k}-cap{best_cap}")
    # C4's rows were decoded on the dev manifest but cached under a sweep tag;
    # C0 above uses the same manifest rows, so utt_ids line up.
    best = None
    for dlp, cr, lr in itertools.product((0.05, 0.10, 0.15, 0.25), (2.0, 2.4, 100.0), (1.5, 2.0, 100.0)):
        guarded = apply_guard(c0, c4, d_logprob=dlp, max_cr=cr, len_ratio=lr)
        w = wer_of(guarded, kws)
        fb = sum(x["fallback"] for x in guarded) / len(guarded)
        if best is None or w < best[0]:
            best = (w, dlp, cr, lr, fb)
    w, dlp, cr, lr, fb = best
    log["guard"] = dict(d_logprob=dlp, max_cr=cr, len_ratio=lr)
    print(f"  best guard: d_logprob={dlp} max_cr={cr} len_ratio={lr} -> "
          f"C7 WER={w*100:.2f} (C4={grid[(best_k,best_cap)]*100:.2f}, C0={c0_wer*100:.2f}), "
          f"fallback={fb*100:.1f}%")

    print("\n=== FROZEN CONFIG (paste into decode.py / prompts.py and RUNLOG.md) ===")
    print(json.dumps(log, indent=2))
    print("\nDEV WERs:", json.dumps(
        {"C0": round(c0_wer * 100, 2),
         **{f"C4_k{k}_cap{c}": round(v * 100, 2) for (k, c), v in grid.items()},
         "C7_best": round(w * 100, 2)}, indent=2))


if __name__ == "__main__":
    main()
