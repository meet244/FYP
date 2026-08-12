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


def select_flagged(conf: dict, rows: list[dict], mode: str = "percentile",
                   value: float = 0.0) -> tuple[set[str], float | None]:
    """Decide which utterances the grounding mechanism is allowed to touch.

    `percentile` mode flags the least-confident `value` percent of utterances by rank.
    Rank selection is the only calibration-free way to gate: an absolute threshold on
    Tier 1 flagged either nothing (<=0.6) or 19% (0.9), so the sweep could not trace a
    trade-off curve at all. Rank also makes the endpoints exact — 0 grounds nothing and
    reproduces B0, 100 grounds everything and reproduces the global mechanism — which is
    what makes the frontier interpretable.

    Utterances with no confidence signal are always flagged: no signal means no evidence
    of confidence, and leaving them unbiased would be an unstated decision.

    Returns (flagged utterance ids, the effective confidence cut-off if one exists).
    """
    pairs = [(r["utt_id"], (conf.get(r["utt_id"]) or {}).get("conf")) for r in rows
             if r["utt_id"] in conf]
    unknown = {u for u, c in pairs if c is None}
    known = sorted(((c, u) for u, c in pairs if c is not None))

    if mode == "absolute":
        return {u for c, u in known if c < value} | unknown, value
    if mode != "percentile":
        raise ValueError(f"unknown gating mode {mode!r}")

    k = int(round(value / 100.0 * len(known)))
    k = max(0, min(len(known), k))
    return ({u for _, u in known[:k]} | unknown,
            known[k - 1][0] if k > 0 else None)


def compose_gated(base: dict[str, dict], grounded: dict[str, dict],
                  flagged: set[str], rows: list[dict],
                  correct_spans: bool = False, cfg=None, plan=None,
                  conf: dict | None = None,
                  label: dict | None = None) -> tuple[list[dict], dict]:
    """Select, per utterance, the unbiased or the grounded hypothesis."""
    conf = conf or {}
    out, n_flagged = [], 0
    for r in rows:
        u = r["utt_id"]
        b = base.get(u)
        if b is None:
            continue
        is_flagged = u in flagged
        row = dict(grounded[u] if (is_flagged and u in grounded) else b)
        row["gate_conf"] = (conf.get(u) or {}).get("conf")
        row["grounded"] = bool(is_flagged and u in grounded)
        if label:
            row.update(label)
        n_flagged += bool(is_flagged)

        if correct_spans and is_flagged and cfg is not None and plan is not None:
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
    meta = {"n_flagged": n_flagged,
            "flagged_rate": n_flagged / len(out) if out else 0.0}
    return out, meta


def run_gated(cfg, rows: list[dict], tier: str, opts: dict):
    """Entry point used by conditions.run_condition for the `G` condition."""
    g = cfg["gating"]
    mech = opts.get("gate_mechanism") or g["mechanism"]
    mode = opts.get("gate_mode") or g.get("mode", "percentile")
    if mode == "percentile":
        value = opts.get("gate_percentile")
        if value is None:
            value = g.get("percentile", 40)
    else:
        value = opts.get("gate_threshold")
        if value is None:
            value = g["utt_conf_threshold"]
    word_th = g["word_conf_threshold"]
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

    flagged, cut = select_flagged(conf, rows, mode, value)
    out, meta = compose_gated(base, grounded, flagged, rows,
                              correct_spans=correct_spans, cfg=cfg, plan=plan,
                              conf=conf,
                              label={"gate_mode": mode, "gate_value": value})
    meta.update({"condition": "G", "mechanism": mech, "gate_mode": mode,
                 "gate_value": value, "effective_conf_cutoff": cut,
                 "word_conf_threshold": word_th,
                 "span_restricted_correction": correct_spans,
                 "decode_fraction": meta["flagged_rate"],
                 "decodes_paid_for": "cached M-condition decodes reused (§9.2)"})
    return out, meta, None


# --- threshold sweep and trade-off frontier (§7.4) ---------------------------

def sweep(cfg, tier: str, mechanism: str, values: list[float],
          correct_spans: bool = False, mode: str | None = None) -> dict:
    """Trace the gating frontier by sweeping how much of the tier gets grounded.

    Two things are being traded off and both are reported per point: the share of the
    global mechanism's B-WER gain that is retained, and the share of utterances that had
    to be re-decoded to get it. On this corpus global biasing turned out to carry almost
    no U-WER penalty, so gating's contribution is the second axis — cost — rather than
    the first. Reporting both keeps the claim honest whichever way the data falls.
    """
    mode = mode or cfg["gating"].get("mode", "percentile")
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
    for v in values:
        flagged, cut = select_flagged(conf, rows, mode, v)
        out_rows, meta = compose_gated(base, grounded, flagged, rows,
                                       correct_spans=correct_spans, cfg=cfg,
                                       plan=plan, conf=conf)
        m, _ = score_rows(out_rows, lex)
        points.append({"value": v, "percentile": v if mode == "percentile" else None,
                       "threshold": v, "mode": mode,
                       "effective_conf_cutoff": cut,
                       "flagged_rate": meta["flagged_rate"],
                       "decode_fraction": meta["flagged_rate"],
                       "n_flagged": meta["n_flagged"],
                       "wer": m["wer"], "b_wer": m["b_wer"], "u_wer": m["u_wer"],
                       "term_f1": m["term_f1"], "term_recall": m["term_recall"],
                       "term_precision": m["term_precision"],
                       "wer_level2": m["wer_level2"]})
        print(f"  {mode}={v:<5} grounded={meta['flagged_rate']*100:5.1f}%  "
              f"{summary_line(m)}", flush=True)

    res = {"tier": tier, "mechanism": mechanism, "mode": mode, "values": values,
           "span_restricted_correction": correct_spans, "points": points}

    # Endpoints: nothing grounded (== B0) and everything grounded (== the mechanism).
    lo = min(points, key=lambda p: p["flagged_rate"])
    hi = max(points, key=lambda p: p["flagged_rate"])
    gain_full = (lo["b_wer"] - hi["b_wer"]) if (lo["b_wer"] is not None
                                               and hi["b_wer"] is not None) else None
    retain = cfg["gating"].get("retain_target", 0.90)
    res.update({"baseline_b_wer": lo["b_wer"], "baseline_u_wer": lo["u_wer"],
                "global_b_wer": hi["b_wer"], "global_u_wer": hi["u_wer"],
                "full_b_wer_gain": gain_full, "retain_target": retain})

    for p in points:
        p["retained_gain"] = ((lo["b_wer"] - p["b_wer"]) / gain_full
                              if gain_full and gain_full > 1e-12 else None)

    # Chosen operating point: the cheapest gate that keeps `retain_target` of the gain
    # without letting U-WER drift above the more permissive of the two endpoints.
    u_cap = max(lo["u_wer"], hi["u_wer"]) + 1e-9
    eligible = [p for p in points
                if p["retained_gain"] is not None and p["retained_gain"] >= retain
                and p["u_wer"] <= u_cap]
    chosen = min(eligible, key=lambda p: p["flagged_rate"]) if eligible else None
    res["chosen"] = chosen
    res["chosen_value"] = chosen["value"] if chosen else None
    res["chosen_threshold"] = res["chosen_value"]          # back-compat for figures
    res["statement_for_report"] = (
        f"Gating {chosen['flagged_rate']*100:.0f}% of utterances retains "
        f"{chosen['retained_gain']*100:.0f}% of the B-WER gain of grounding all of them"
        if chosen else
        "No gate retained the target share of the gain; grounding is not separable from "
        "cost on this tier.")
    write_json(ROOT / "runs" / tier / f"G_sweep_{mechanism}.json", res)
    print("\n" + res["statement_for_report"])
    return res


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Confidence-gated biasing (§7.4)")
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--mechanism", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the gate and write the trade-off frontier")
    ap.add_argument("--mode", choices=["percentile", "absolute"], default=None)
    ap.add_argument("--correct-spans", action="store_true",
                    help="also restrict M3a correction to flagged spans")
    a = ap.parse_args()
    mech = a.mechanism or cfg["gating"]["mechanism"]
    mode = a.mode or cfg["gating"].get("mode", "percentile")
    if a.sweep:
        values = (cfg["gating"]["sweep_percentiles"] if mode == "percentile"
                  else cfg["gating"]["sweep"])
        res = sweep(cfg, a.tier, mech, values, a.correct_spans, mode=mode)
        print(f"chosen {mode}: {res.get('chosen_value')}")
    else:
        from conditions import run_condition
        run_condition("G", a.tier, cfg, gate_mechanism=mech,
                      gate_correct_spans=a.correct_spans)


if __name__ == "__main__":
    main()
