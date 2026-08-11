"""Baseline error analysis: substitution table, WER vs duration, term headroom."""
import argparse
import collections
import json
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent))
from normalize import DEVANAGARI, basic_norm, script_invariant_norm  # noqa: E402


def classify(ref_w, hyp_w, terms):
    """Coarse bucket for a substitution pair."""
    if script_invariant_norm(ref_w) == script_invariant_norm(hyp_w):
        return "orthographic/script"
    if ref_w in terms or hyp_w in terms:
        return "technical-term"
    if not DEVANAGARI.search(ref_w) and not DEVANAGARI.search(hyp_w):
        return "english-word"
    if len(ref_w) <= 3:
        return "function-word"
    return "other-hindi"


def main(hyp_jsonl, terms_path, top=100):
    rows = [json.loads(l) for l in open(hyp_jsonl, encoding="utf-8")]
    terms = set()
    if Path(terms_path).exists():
        terms = {t.strip().lower() for t in open(terms_path, encoding="utf-8")
                 if t.strip()}

    pairs = collections.Counter()
    buckets = collections.Counter()
    dels = collections.Counter()
    inss = collections.Counter()
    term_errors = tot_errors = 0
    by_dur = collections.defaultdict(lambda: [0, 0])  # bucket -> [errors, ref_words]

    for r in rows:
        ref, hyp = basic_norm(r["ref"]), basic_norm(r["hyp"])
        if not ref.strip():
            continue
        o = jiwer.process_words([ref], [hyp])
        ref_w, hyp_w = o.references[0], o.hypotheses[0]
        n_err = o.substitutions + o.insertions + o.deletions
        tot_errors += n_err
        d = r.get("duration") or 0
        b = "0-2s" if d < 2 else "2-4s" if d < 4 else "4-7s" if d < 7 else "7s+"
        by_dur[b][0] += n_err
        by_dur[b][1] += len(ref_w)
        for ch in o.alignments[0]:
            if ch.type == "substitute":
                for i, j in zip(range(ch.ref_start_idx, ch.ref_end_idx),
                                range(ch.hyp_start_idx, ch.hyp_end_idx)):
                    pairs[(ref_w[i], hyp_w[j])] += 1
                    buckets[classify(ref_w[i], hyp_w[j], terms)] += 1
                    if ref_w[i] in terms or hyp_w[j] in terms:
                        term_errors += 1
            elif ch.type == "delete":
                for i in range(ch.ref_start_idx, ch.ref_end_idx):
                    dels[ref_w[i]] += 1
                    if ref_w[i] in terms:
                        term_errors += 1
            elif ch.type == "insert":
                for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                    inss[hyp_w[j]] += 1
                    if hyp_w[j] in terms:
                        term_errors += 1

    print(f"\n=== top {top} substitution pairs (ref -> hyp) ===")
    for (a, b), c in pairs.most_common(top):
        print(f"{c:4d}  {a}  ->  {b}")
    print("\n=== substitution categories ===")
    tot_sub = sum(buckets.values()) or 1
    for k, v in buckets.most_common():
        print(f"{v:6d}  {100*v/tot_sub:5.1f}%  {k}")
    print("\n=== top deletions / insertions ===")
    print("DEL:", ", ".join(f"{w}({c})" for w, c in dels.most_common(20)))
    print("INS:", ", ".join(f"{w}({c})" for w, c in inss.most_common(20)))
    print("\n=== WER by utterance duration ===")
    for b in ["0-2s", "2-4s", "4-7s", "7s+"]:
        e, n = by_dur[b]
        if n:
            print(f"{b:>5}: WER={e/n:.4f}  ({n} ref words, {e} errors)")
    if terms:
        print(f"\n=== headroom ===\n{term_errors}/{tot_errors} word errors touch a "
              f"syllabus term = {100*term_errors/max(1,tot_errors):.1f}% of all errors")

    out = Path(hyp_jsonl).parent / "error_analysis.json"
    out.write_text(json.dumps({
        "top_substitutions": [{"ref": a, "hyp": b, "count": c}
                              for (a, b), c in pairs.most_common(top)],
        "categories": dict(buckets),
        "top_deletions": dels.most_common(30),
        "top_insertions": inss.most_common(30),
        "wer_by_duration": {k: {"errors": v[0], "ref_words": v[1],
                                "wer": v[0] / v[1] if v[1] else None}
                            for k, v in by_dur.items()},
        "total_errors": tot_errors, "term_touching_errors": term_errors,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", default="runs/S0_baseline/hyps.jsonl")
    ap.add_argument("--terms", default="syllabus/index/terms.txt")
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()
    main(a.hyps, a.terms, a.top)
