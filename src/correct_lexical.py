"""Method B1 — deterministic phonetic/lexical correction against retrieved terms."""
import argparse
import json
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent))
from score import score_file  # noqa: E402


def correct(hyp: str, candidate_terms: list[str], threshold: int = 88) -> str:
    """Replace out-of-lexicon ASCII tokens with the closest syllabus term."""
    if not candidate_terms:
        return hyp
    lower = {t.lower() for t in candidate_terms}
    out = []
    for tok in hyp.split():
        raw = tok
        if tok.lower() in lower or not tok.isascii() or len(tok) < 4:
            out.append(raw)
            continue
        m = process.extractOne(tok.lower(), candidate_terms, scorer=fuzz.ratio)
        out.append(m[0] if m and m[1] >= threshold else raw)
    return " ".join(out)


def apply_to_run(in_hyps, out_dir, pass1, k=3, threshold=88, terms_path=None):
    from retrieve import SyllabusRetriever
    r = SyllabusRetriever()
    q = {j["utt_id"]: j["hyp"] for j in
         (json.loads(l) for l in open(pass1, encoding="utf-8"))}
    rows = [json.loads(l) for l in open(in_hyps, encoding="utf-8")]
    changed = 0
    for row in rows:
        cands = r.candidate_terms(q.get(row["utt_id"], row["hyp"]), k=k)
        new = correct(row["hyp"], cands, threshold)
        changed += new != row["hyp"]
        row["hyp_before_correction"] = row["hyp"]
        row["hyp"] = new
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hyps.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    m = score_file(out_dir / "hyps.jsonl", terms_path)
    m["utts_changed"] = changed
    m["threshold"] = threshold
    m["k"] = k
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-hyps", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pass1", default="runs/S0_baseline/hyps.jsonl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--threshold", type=int, default=88)
    ap.add_argument("--sweep", default=None,
                    help="comma-separated thresholds, e.g. 80,84,88,92,96")
    ap.add_argument("--terms", default="syllabus/index/terms.txt")
    a = ap.parse_args()

    if a.sweep:
        results = []
        for th in [int(x) for x in a.sweep.split(",")]:
            m = apply_to_run(a.in_hyps, f"{a.out_dir}_th{th}", a.pass1, a.k, th, a.terms)
            results.append({"threshold": th, "wer": m["wer"],
                            "term_f1": m.get("term_f1"), "changed": m["utts_changed"]})
            print(f"th={th}: WER={m['wer']:.4f} termF1={m.get('term_f1', 0):.4f} "
                  f"changed={m['utts_changed']}")
        Path("runs/threshold_sweep.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
    else:
        m = apply_to_run(a.in_hyps, a.out_dir, a.pass1, a.k, a.threshold, a.terms)
        print(json.dumps(m, indent=2, ensure_ascii=False))
