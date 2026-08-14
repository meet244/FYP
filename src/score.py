"""Scoring (§8.2).

Primary metric: **decomposed WER**. Every reference word is labelled B (member of the
frozen syllabus lexicon) or U (not), and the error rate is reported separately for the
two classes:

    B-WER = (B substitutions + B deletions + B insertions) / (# B reference words)
    U-WER = (U substitutions + U deletions + U insertions) / (# U reference words)

Attribution follows the convention used in the contextual-biasing literature:
substitutions and deletions are attributed to the class of the *reference* word;
insertions are attributed to the class of the *hypothesis* word, so a syllabus term
hallucinated into an utterance that never contained it lands in B insertions and
raises U-WER's counterpart rather than hiding. This is what makes over-biasing (H4)
visible: effective grounding lowers B-WER while leaving U-WER flat; over-biasing
lowers B-WER *and* raises U-WER.

Supporting metrics: overall WER with sub/ins/del breakdown, CER, level-2
(script-invariant) WER, terminology precision/recall/F1, empty-hypothesis count, and
per-utterance edit counts and reference lengths, which §8.3's paired bootstrap
requires and which cannot be recovered from corpus aggregates.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jiwer

from common import read_jsonl, write_json, write_jsonl
from lexicon import Lexicon, load_lexicon
from normalize import level1, level2, skeleton


def _wer_from_counts(errors: int, ref_len: int) -> float | None:
    return errors / ref_len if ref_len else None


def decompose_utterance(ref: str, hyp: str, lex: Lexicon) -> dict:
    """Per-utterance edit counts, split into B and U classes.

    `ref`/`hyp` must already be level-1 normalised.
    """
    ref_w, hyp_w = ref.split(), hyp.split()
    out = {
        "ref_len": len(ref_w), "hyp_len": len(hyp_w),
        "sub": 0, "ins": 0, "del": 0, "hits": 0,
        "b_ref_len": 0, "u_ref_len": 0,
        "b_sub": 0, "b_del": 0, "b_ins": 0, "b_hits": 0,
        "u_sub": 0, "u_del": 0, "u_ins": 0, "u_hits": 0,
    }
    ref_is_b = lex.bias_flags(ref_w)
    hyp_is_b = lex.bias_flags(hyp_w)
    out["b_ref_len"] = sum(ref_is_b)
    out["u_ref_len"] = len(ref_w) - out["b_ref_len"]

    if not ref_w:
        # Unscorable: jiwer cannot compute WER against an empty reference.
        out["errors"] = out["b_errors"] = out["u_errors"] = 0
        out["wer"] = None
        return out

    o = jiwer.process_words([ref], [hyp] if hyp_w else [""])
    out["sub"], out["ins"], out["del"] = o.substitutions, o.insertions, o.deletions
    out["hits"] = o.hits

    for chunk in o.alignments[0]:
        if chunk.type == "substitute":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                key = "b_sub" if ref_is_b[i] else "u_sub"
                out[key] += 1
        elif chunk.type == "delete":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                out["b_del" if ref_is_b[i] else "u_del"] += 1
        elif chunk.type == "insert":
            for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                out["b_ins" if hyp_is_b[j] else "u_ins"] += 1
        elif chunk.type == "equal":
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                out["b_hits" if ref_is_b[i] else "u_hits"] += 1

    out["errors"] = out["sub"] + out["ins"] + out["del"]
    out["b_errors"] = out["b_sub"] + out["b_del"] + out["b_ins"]
    out["u_errors"] = out["u_sub"] + out["u_del"] + out["u_ins"]
    out["wer"] = out["errors"] / len(ref_w)
    return out


def term_metrics(refs: list[str], hyps: list[str], lex: Lexicon) -> dict:
    """Terminology precision / recall / F1, multiset-matched per utterance."""
    tp = fp = fn = 0
    for r, h in zip(refs, hyps):
        rc = Counter(w for w in r.split() if lex.in_bias(w))
        hc = Counter(w for w in h.split() if lex.in_bias(w))
        for t in set(rc) | set(hc):
            tp += min(rc[t], hc[t])
            fn += max(0, rc[t] - hc[t])
            fp += max(0, hc[t] - rc[t])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"term_tp": tp, "term_fp": fp, "term_fn": fn,
            "term_precision": prec, "term_recall": rec, "term_f1": f1}


def score_rows(rows: list[dict], lex: Lexicon) -> tuple[dict, list[dict]]:
    """Score in-memory hypothesis rows. Returns (aggregate metrics, per-utterance)."""
    refs1 = [level1(r["ref"]) for r in rows]
    hyps1 = [level1(r.get("hyp") or "") for r in rows]

    per_utt = []
    for row, ref, hyp in zip(rows, refs1, hyps1):
        d = decompose_utterance(ref, hyp, lex)
        d["utt_id"] = row["utt_id"]
        d["duration"] = row.get("duration")
        d["rec"] = row.get("rec")
        per_utt.append(d)

    scorable = [d for d in per_utt if d["ref_len"] > 0]
    tot = {k: sum(d[k] for d in scorable) for k in
           ("ref_len", "errors", "sub", "ins", "del", "hits",
            "b_ref_len", "u_ref_len", "b_errors", "u_errors",
            "b_sub", "b_del", "b_ins", "u_sub", "u_del", "u_ins", "b_hits", "u_hits")}

    m: dict = {
        "n_utts": len(rows),
        "n_scored": len(scorable),
        "total_ref_words": tot["ref_len"],
        # --- headline (level 1, script-preserving) ---
        "wer": _wer_from_counts(tot["errors"], tot["ref_len"]),
        "sub": tot["sub"], "ins": tot["ins"], "del": tot["del"], "hits": tot["hits"],
        # --- primary decomposition ---
        "b_wer": _wer_from_counts(tot["b_errors"], tot["b_ref_len"]),
        "u_wer": _wer_from_counts(tot["u_errors"], tot["u_ref_len"]),
        "b_ref_words": tot["b_ref_len"],
        "u_ref_words": tot["u_ref_len"],
        "b_sub": tot["b_sub"], "b_del": tot["b_del"], "b_ins": tot["b_ins"],
        "u_sub": tot["u_sub"], "u_del": tot["u_del"], "u_ins": tot["u_ins"],
        "bias_token_rate": (tot["b_ref_len"] / tot["ref_len"]) if tot["ref_len"] else 0.0,
    }

    # --- supporting metrics ---
    pairs = [(r, h) for r, h in zip(refs1, hyps1) if r.strip()]
    m["cer"] = jiwer.cer([r for r, _ in pairs], [h for _, h in pairs])
    l2 = [(level2(r["ref"]), level2(r.get("hyp") or "")) for r in rows]
    l2 = [(r, h) for r, h in l2 if r.strip()]
    m["wer_level2"] = jiwer.wer([r for r, _ in l2], [h for _, h in l2])
    sk = [(skeleton(r["ref"]), skeleton(r.get("hyp") or "")) for r in rows]
    sk = [(r, h) for r, h in sk if r.strip()]
    m["wer_skeleton_lower_bound"] = jiwer.wer([r for r, _ in sk], [h for _, h in sk])
    m["orthographic_error_share"] = (
        1 - m["wer_level2"] / m["wer"] if m["wer"] else None)
    m["empty_hyps"] = sum(1 for h in hyps1 if not h.strip())
    m.update(term_metrics(refs1, hyps1, lex))
    m.update(lex.stamp())

    # --- guard / instrumentation counters carried on the hypothesis rows (§7.5) ---
    fired = sum(1 for r in rows if r.get("guard_context_echo"))
    if any("guard_context_echo" in r for r in rows):
        m["guard_context_echo_fired"] = fired
        m["guard_context_echo_rate"] = fired / len(rows) if rows else 0.0
    fired_r = sum(1 for r in rows if r.get("guard_runaway"))
    if any("guard_runaway" in r for r in rows):
        m["guard_runaway_fired"] = fired_r
        m["guard_runaway_rate"] = fired_r / len(rows) if rows else 0.0
    discarded = sum(1 for r in rows if r.get("guard_rewrite_discarded"))
    if any("guard_rewrite_discarded" in r for r in rows):
        m["guard_rewrite_discarded"] = discarded
        m["guard_rewrite_discard_rate"] = discarded / len(rows) if rows else 0.0
    changed = sum(1 for r in rows if r.get("corrected"))
    if any("corrected" in r for r in rows):
        m["utts_corrected"] = changed
    grounded = sum(1 for r in rows if r.get("grounded"))
    if any("grounded" in r for r in rows):
        m["utts_grounded"] = grounded
        m["grounded_rate"] = grounded / len(rows) if rows else 0.0
    return m, per_utt


def score_run(run_path: str | Path, lexicon_path: str | Path | None = None,
              write: bool = True) -> dict:
    """Score a run directory's hyps.jsonl, writing per_utt.jsonl and metrics.json."""
    run_path = Path(run_path)
    hyps = run_path / "hyps.jsonl" if run_path.is_dir() else run_path
    out_dir = hyps.parent
    lex = load_lexicon(lexicon_path or "syllabus/index/terms.txt")
    rows = read_jsonl(hyps)
    m, per_utt = score_rows(rows, lex)
    if write:
        write_jsonl(out_dir / "per_utt.jsonl", per_utt)
        prev = {}
        p = out_dir / "metrics.json"
        if p.exists():
            prev = {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                    if k.startswith("_")}
        write_json(p, {**m, **prev})
    return m


def summary_line(m: dict) -> str:
    def f(k, n=4):
        v = m.get(k)
        return "-" if v is None else f"{v:.{n}f}"
    return (f"WER={f('wer')}  B-WER={f('b_wer')}  U-WER={f('u_wer')}  "
            f"CER={f('cer')}  WER-L2={f('wer_level2')}  termF1={f('term_f1')}  "
            f"n={m.get('n_scored')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score a run directory or hyps.jsonl")
    ap.add_argument("run")
    ap.add_argument("--lexicon", default="syllabus/index/terms.txt")
    a = ap.parse_args()
    m = score_run(a.run, a.lexicon)
    print(json.dumps(m, indent=2, ensure_ascii=False))
    print("\n" + summary_line(m))
