"""G — confidence-gated biasing (§7.4). The study's principal contribution.

Every mechanism in §7.1-7.3 is applied *globally*: the syllabus is pushed at every
utterance and every token, including the many the model already transcribes correctly.
That is the source of the over-biasing penalty in H4.

The gating idea: apply grounding only where the model signals uncertainty.

  1. During pass 1, retain per-segment and per-token confidence scores, which the
     runtime returns alongside word-level timing.
  2. Identify low-confidence utterances and low-confidence token spans.
  3. Apply the grounding mechanism selectively: use the grounded decode only for
     flagged utterances, and restrict output-level correction to flagged spans.
  4. Leave confidently decoded regions entirely untouched.

Because decodes are cached per (utterance, config), the whole threshold sweep is free
once the mechanism has been decoded on the tier: gating is a per-utterance *selection*
between the unbiased and the grounded hypothesis, not a new decode. Where the grounded
decode is absent, only the flagged utterances are decoded — which is exactly the partial
cost §9.2 predicts.

Testable prediction: gated grounding achieves most of the terminology gain of global
grounding at a materially lower penalty on non-terminology words. That is a curve, not a
point, so `sweep()` traces the trade-off frontier by sweeping the threshold and
reporting B-WER against U-WER at each setting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from backends import model_spec_from_config
from common import (ROOT, load_config, manifest_for_tier, read_jsonl, run_dir,
                    write_json, write_jsonl)
from lexicon import load_lexicon
from normalize import level1
from score import score_rows, summary_line
from transcribe import cached_result


# --- confidence extraction ---------------------------------------------------

def utterance_confidence(row: dict) -> float | None:
    """Utterance-level confidence: mean per-word probability from the pass-1 decode."""
    conf = row.get("conf") or {}
    v = conf.get("mean_word_prob")
    if v is None:
        # Fall back to avg_logprob mapped into (0, 1] if word timings are unavailable.
        lp = conf.get("avg_logprob")
        if lp is None:
            return None
        import math
        return math.exp(lp)
    return v


def word_probability_spans(cfg, row: dict, word_threshold: float) -> list[int]:
    """Indices of level-1 tokens whose source word was decoded with low probability.

    Whisper returns confidences per decoded word; scoring and correction operate on
    level-1 normalised tokens. One decoded word can normalise to zero or several
    tokens, so the probability is propagated to each token it produced, keeping the
    span indices aligned with the tokens `correct_lexical` will edit.
    """
    spec = model_spec_from_config(cfg)
    res = cached_result(spec.name, row.get("cfg_key", ""), row["utt_id"])
    if not res:
        return []
    flagged: list[int] = []
    idx = 0
    for seg in res.get("segments") or []:
        for w in seg.get("words") or []:
            toks = level1(w.get("word") or "").split()
            p = w.get("prob")
            for _ in toks:
                if p is not None and p < word_threshold:
                    flagged.append(idx)
                idx += 1
    return flagged


def confidence_table(cfg, base_rows: list[dict], word_threshold: float) -> dict:
    """Per-utterance {conf, low_conf_token_indices} from the pass-1 run."""
    out = {}
    for row in base_rows:
        out[row["utt_id"]] = {
            "conf": utterance_confidence(row),
            "spans": word_probability_spans(cfg, row, word_threshold),
            "n_tokens": len(level1(row.get("hyp") or "").split()),
        }
    return out


# --- the gated condition -----------------------------------------------------

def _load_run(tier: str, name: str) -> dict[str, dict]:
    p = ROOT / "runs" / tier / name / "hyps.jsonl"
    if not p.exists():
        raise SystemExit(
            f"G needs the mechanism run {name} on {tier} first (§9.2 execution order): "
            f"missing {p.relative_to(ROOT)}")
    return {r["utt_id"]: r for r in read_jsonl(p)}


def compose_gated(base: dict[str, dict], grounded: dict[str, dict], conf: dict,
                  threshold: float, rows: list[dict],
                  correct_spans: bool = False, cfg=None,
                  plan=None) -> tuple[list[dict], dict]:
    """Select, per utterance, the unbiased or the grounded hypothesis.

    threshold = 0.0 leaves everything unbiased (equals B0); threshold = 1.0 grounds
    everything (equals the global mechanism). The interesting region is in between.
    """
    out, n_flagged = [], 0
    for r in rows:
        u = r["utt_id"]
        b = base.get(u)
        if b is None:
            continue
        c = (conf.get(u) or {}).get("conf")
        flagged = (c is None) or (c < threshold)
        row = dict(grounded[u] if (flagged and u in grounded) else b)
        row["gate_conf"] = c
        row["grounded"] = bool(flagged and u in grounded)
        row["gate_threshold"] = threshold
        n_flagged += bool(flagged)

        if correct_spans and flagged and cfg is not None and plan is not None:
            from correct_lexical import correct_utterance
            spans = (conf.get(u) or {}).get("spans") or []
            if spans:
                new, edits = correct_utterance(
                    row["hyp"], plan.candidates(u),
                    cfg["correction"]["fuzzy_threshold"],
                    cfg["correction"]["min_token_len"], spans=spans)
                row["edits"] = edits
                row["corrected"] = bool(edits) and new != row["hyp"]
                if row["corrected"]:
                    row["hyp_before_correction"] = row["hyp"]
                    row["hyp"] = new
        out.append(row)
    meta = {"threshold": threshold, "n_flagged": n_flagged,
            "flagged_rate": n_flagged / len(out) if out else 0.0}
    return out, meta


def run_gated(cfg, rows: list[dict], tier: str, opts: dict):
    """Entry point used by conditions.run_condition for the `G` condition."""
    mech = opts.get("gate_mechanism") or cfg["gating"]["mechanism"]
    threshold = opts.get("gate_threshold")
    if threshold is None:
        threshold = cfg["gating"]["utt_conf_threshold"]
    word_th = cfg["gating"]["word_conf_threshold"]
    correct_spans = bool(opts.get("gate_correct_spans", False))

    base = _load_run(tier, "B0")
    grounded = _load_run(tier, mech)
    conf = confidence_table(cfg, list(base.values()), word_th)

    plan = None
    if correct_spans:
        from retrieve import RetrievalPlan, SyllabusRetriever
        r = SyllabusRetriever(cfg)
        plan = RetrievalPlan(r, rows, base,
                             granularity=cfg["retrieval"]["granularity"],
                             k=cfg["retrieval"]["top_k"])

    out, meta = compose_gated(base, grounded, conf, threshold, rows,
                              correct_spans=correct_spans, cfg=cfg, plan=plan)
    meta.update({"condition": "G", "mechanism": mech,
                 "utt_conf_threshold": threshold, "word_conf_threshold": word_th,
                 "span_restricted_correction": correct_spans,
                 "decodes_paid_for": "cached M-condition decodes reused (§9.2)"})
    return out, meta, None


# --- threshold sweep and trade-off frontier (§7.4) ---------------------------

def sweep(cfg, tier: str, mechanism: str, thresholds: list[float],
          correct_spans: bool = False) -> dict:
    rows = read_jsonl(manifest_for_tier(cfg, tier))
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    base = _load_run(tier, "B0")
    grounded = _load_run(tier, mechanism)
    conf = confidence_table(cfg, list(base.values()),
                            cfg["gating"]["word_conf_threshold"])

    plan = None
    if correct_spans:
        from retrieve import RetrievalPlan, SyllabusRetriever
        plan = RetrievalPlan(SyllabusRetriever(cfg), rows, base,
                             granularity=cfg["retrieval"]["granularity"],
                             k=cfg["retrieval"]["top_k"])

    points = []
    for th in thresholds:
        out, meta = compose_gated(base, grounded, conf, th, rows,
                                  correct_spans=correct_spans, cfg=cfg, plan=plan)
        m, _ = score_rows(out, lex)
        points.append({"threshold": th, "flagged_rate": meta["flagged_rate"],
                       "n_flagged": meta["n_flagged"],
                       "wer": m["wer"], "b_wer": m["b_wer"], "u_wer": m["u_wer"],
                       "term_f1": m["term_f1"], "term_recall": m["term_recall"],
                       "term_precision": m["term_precision"],
                       "wer_level2": m["wer_level2"]})
        print(f"  th={th:<4} flagged={meta['flagged_rate']*100:5.1f}%  "
              f"{summary_line(m)}", flush=True)

    out = {"tier": tier, "mechanism": mechanism, "thresholds": thresholds,
           "span_restricted_correction": correct_spans, "points": points}
    # Best threshold by the gating criterion: lowest B-WER subject to U-WER not
    # exceeding the unbiased baseline's U-WER (that is what "at no cost" means).
    b0 = next((p for p in points if p["threshold"] == 0.0), None)
    if b0:
        eligible = [p for p in points if p["u_wer"] <= b0["u_wer"] + 1e-9]
        out["baseline_u_wer"] = b0["u_wer"]
        out["chosen_threshold"] = (min(eligible, key=lambda p: p["b_wer"])["threshold"]
                                  if eligible else None)
    write_json(ROOT / "runs" / tier / f"G_sweep_{mechanism}.json", out)
    return out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Confidence-gated biasing (§7.4)")
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--mechanism", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the confidence threshold and write the frontier")
    ap.add_argument("--correct-spans", action="store_true",
                    help="also restrict M3a correction to flagged spans")
    a = ap.parse_args()
    mech = a.mechanism or cfg["gating"]["mechanism"]
    if a.sweep:
        res = sweep(cfg, a.tier, mech, cfg["gating"]["sweep"], a.correct_spans)
        print(f"\nchosen threshold (lowest B-WER with U-WER <= baseline): "
              f"{res.get('chosen_threshold')}")
    else:
        from conditions import run_condition
        run_condition("G", a.tier, cfg, gate_mechanism=mech,
                      gate_correct_spans=a.correct_spans)


if __name__ == "__main__":
    main()
