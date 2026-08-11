"""Scoring: corpus WER/CER, script-invariant WER, term-level P/R/F1, per-utterance stats."""
import argparse
import json
from collections import Counter
from pathlib import Path

import jiwer

from normalize import basic_norm, consonant_skeleton_norm, script_invariant_norm

DEFAULT_TERMS = "syllabus/index/terms.txt"


def load_terms(path=DEFAULT_TERMS):
    return {t.strip().lower() for t in open(path, encoding="utf-8") if t.strip()}


def term_metrics(refs, hyps, terms):
    """Recall/precision restricted to syllabus terminology (multiset match)."""
    tp = fp = fn = 0
    for r, h in zip(refs, hyps):
        rc = Counter(w for w in r.split() if w in terms)
        hc = Counter(w for w in h.split() if w in terms)
        for t in set(rc) | set(hc):
            tp += min(rc[t], hc[t])
            fn += max(0, rc[t] - hc[t])
            fp += max(0, hc[t] - rc[t])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"term_tp": tp, "term_fp": fp, "term_fn": fn,
            "term_precision": prec, "term_recall": rec, "term_f1": f1}


def corpus_measures(refs, hyps):
    """WER + edit-op counts over the whole corpus (jiwer skips empty references)."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    out = jiwer.process_words([r for r, _ in pairs], [h for _, h in pairs])
    return {"wer": out.wer, "sub": out.substitutions, "ins": out.insertions,
            "del": out.deletions, "hits": out.hits}


def per_utterance(rows):
    """Per-utterance edit counts and reference lengths (needed for §10 bootstrap)."""
    recs = []
    for r in rows:
        ref, hyp = basic_norm(r["ref"]), basic_norm(r["hyp"])
        if not ref.strip():
            continue
        o = jiwer.process_words([ref], [hyp])
        errs = o.substitutions + o.insertions + o.deletions
        recs.append({"utt_id": r["utt_id"], "duration": r.get("duration"),
                     "ref_len": len(ref.split()), "errors": errs, "wer": o.wer,
                     "sub": o.substitutions, "ins": o.insertions, "del": o.deletions})
    return recs


def score_file(hyp_jsonl, terms_path=None, write_per_utt=True):
    rows = [json.loads(l) for l in open(hyp_jsonl, encoding="utf-8")]
    refs = [basic_norm(r["ref"]) for r in rows]
    hyps = [basic_norm(r["hyp"]) for r in rows]
    si_refs = [script_invariant_norm(r["ref"]) for r in rows]
    si_hyps = [script_invariant_norm(r["hyp"]) for r in rows]
    nonempty = [(r, h) for r, h in zip(refs, hyps) if r.strip()]

    out = {"n_utts": len(rows), "n_scored": len(nonempty)}
    out.update(corpus_measures(refs, hyps))
    out["cer"] = jiwer.cer([r for r, _ in nonempty], [h for _, h in nonempty])
    si = [(r, h) for r, h in zip(si_refs, si_hyps) if r.strip()]
    out["wer_script_invariant"] = jiwer.wer([r for r, _ in si], [h for _, h in si])
    sk = [(consonant_skeleton_norm(r["ref"]), consonant_skeleton_norm(r["hyp"]))
          for r in rows]
    sk = [(r, h) for r, h in sk if r.strip()]
    out["wer_skeleton_lower_bound"] = jiwer.wer([r for r, _ in sk], [h for _, h in sk])
    out["empty_hyps"] = sum(1 for h in hyps if not h.strip())

    if terms_path and Path(terms_path).exists():
        out.update(term_metrics(refs, hyps, load_terms(terms_path)))

    pu = per_utterance(rows)
    if write_per_utt:
        p = Path(hyp_jsonl).parent / "per_utt.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in pu:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out["total_ref_words"] = sum(r["ref_len"] for r in pu)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps")
    ap.add_argument("--terms", default=DEFAULT_TERMS)
    a = ap.parse_args()
    m = score_file(a.hyps, a.terms)
    print(json.dumps(m, indent=2, ensure_ascii=False))
    out = Path(a.hyps).parent / "metrics.json"
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", out)
