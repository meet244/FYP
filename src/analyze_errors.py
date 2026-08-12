"""Baseline error analysis (§8.4). Produces the paper's discussion section.

Three deliverables:

  1. WER as a function of utterance duration.
  2. The hundred most frequent substitution pairs, each classified as orthographic,
     terminology, function-word, or hallucination error.
  3. The share of total word errors falling on lexicon terms — the **headroom estimate**.

The headroom estimate is computed and stated *before* presenting any grounded result. If
terminology accounts for only a small fraction of total errors, the maximum achievable
WER gain is correspondingly bounded. Predicting a modest result in advance and then
observing it is a far stronger position than reporting a modest result unexplained.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import jiwer

from common import ROOT, load_config, read_jsonl, write_json
from lexicon import load_lexicon
from normalize import DEVANAGARI, level1, level2

# Frequent Hindi function words: errors on these are grammatical/acoustic noise, not
# terminology failures, and they dominate raw substitution counts.
FUNCTION_WORDS = set("""
का की के को में से पर है हैं हो होता होती होते था थी थे और या भी ही तो जो यह वह इस उस
इसे उसे कि कर करें करते करना किया गया गयी गये हम आप वे एक अब यहाँ यहां वहाँ वहां जब तब
अगर लिए साथ बाद पहले तरह ऐसे कुछ सब बहुत नहीं ना क्या कैसे क्यों जैसे तक द्वारा वाला
""".split())


def classify(ref_w: str, hyp_w: str, lex) -> str:
    """Coarse bucket for one substitution pair."""
    if level2(ref_w) == level2(hyp_w):
        return "orthographic"           # same word, different script/spelling convention
    if lex.in_bias(ref_w) or lex.in_bias(hyp_w):
        return "terminology"
    if ref_w in FUNCTION_WORDS or hyp_w in FUNCTION_WORDS:
        return "function_word"
    if not DEVANAGARI.search(ref_w) and not DEVANAGARI.search(hyp_w):
        return "english_nonterm"
    return "other"


def analyse(tier: str, run: str, cfg, top: int = 100) -> dict:
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    rows = read_jsonl(ROOT / "runs" / tier / run / "hyps.jsonl")

    pairs = collections.Counter()
    buckets = collections.Counter()
    dels, inss = collections.Counter(), collections.Counter()
    hallucinated_terms = collections.Counter()
    by_dur = collections.defaultdict(lambda: [0, 0])
    tot_err = term_err = 0
    ins_total = 0

    for r in rows:
        ref, hyp = level1(r["ref"]), level1(r.get("hyp") or "")
        if not ref.strip():
            continue
        o = jiwer.process_words([ref], [hyp if hyp.strip() else ""])
        rw, hw = o.references[0], o.hypotheses[0]
        n_err = o.substitutions + o.insertions + o.deletions
        tot_err += n_err
        d = r.get("duration") or 0
        bucket = "0-2s" if d < 2 else "2-4s" if d < 4 else "4-7s" if d < 7 else "7s+"
        by_dur[bucket][0] += n_err
        by_dur[bucket][1] += len(rw)

        for ch in o.alignments[0]:
            if ch.type == "substitute":
                for i, j in zip(range(ch.ref_start_idx, ch.ref_end_idx),
                                range(ch.hyp_start_idx, ch.hyp_end_idx)):
                    pairs[(rw[i], hw[j])] += 1
                    buckets[classify(rw[i], hw[j], lex)] += 1
                    if lex.in_bias(rw[i]) or lex.in_bias(hw[j]):
                        term_err += 1
            elif ch.type == "delete":
                for i in range(ch.ref_start_idx, ch.ref_end_idx):
                    dels[rw[i]] += 1
                    if lex.in_bias(rw[i]):
                        term_err += 1
            elif ch.type == "insert":
                for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                    inss[hw[j]] += 1
                    ins_total += 1
                    if lex.in_bias(hw[j]):
                        term_err += 1
                        hallucinated_terms[hw[j]] += 1

    sub_total = sum(buckets.values()) or 1
    out = {
        "tier": tier, "run": run, "n_utts": len(rows),
        "total_errors": tot_err,
        "term_touching_errors": term_err,
        "headroom_estimate": {
            "share_of_errors_on_lexicon_terms": term_err / max(1, tot_err),
            "statement": (
                f"{term_err} of {tot_err} word errors ({100*term_err/max(1,tot_err):.1f}%) "
                f"touch a syllabus lexicon term. Terminology biasing can therefore "
                f"reduce overall WER by at most that share, and only to the extent it "
                f"repairs those errors without introducing new ones."),
        },
        "substitution_categories": {
            k: {"count": v, "share": v / sub_total} for k, v in buckets.most_common()},
        "top_substitutions": [
            {"ref": a, "hyp": b, "count": c,
             "category": classify(a, b, lex)} for (a, b), c in pairs.most_common(top)],
        "top_deletions": dels.most_common(30),
        "top_insertions": inss.most_common(30),
        "hallucinated_lexicon_terms": hallucinated_terms.most_common(30),
        "insertions_total": ins_total,
        "wer_by_duration": {
            k: {"errors": v[0], "ref_words": v[1],
                "wer": v[0] / v[1] if v[1] else None}
            for k, v in sorted(by_dur.items())},
    }

    print(f"\n=== headroom estimate (§8.4) ===\n{out['headroom_estimate']['statement']}")
    print(f"\n=== substitution categories ({sub_total} substitutions) ===")
    for k, v in out["substitution_categories"].items():
        print(f"  {v['count']:6d}  {100*v['share']:5.1f}%  {k}")
    print(f"\n=== WER by utterance duration ===")
    for k, v in out["wer_by_duration"].items():
        if v["wer"] is not None:
            print(f"  {k:>5s}: WER={v['wer']:.4f}  ({v['ref_words']} ref words)")
    print(f"\n=== top 20 substitution pairs (ref -> hyp) ===")
    for p in out["top_substitutions"][:20]:
        print(f"  {p['count']:4d}  {p['ref']:>22s}  ->  {p['hyp']:<22s} [{p['category']}]")

    write_json(ROOT / "runs" / tier / run / "error_analysis.json", out)
    print(f"\n-> runs/{tier}/{run}/error_analysis.json")
    return out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier2")
    ap.add_argument("--run", default="B0")
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()
    analyse(a.tier, a.run, cfg, a.top)


if __name__ == "__main__":
    main()
