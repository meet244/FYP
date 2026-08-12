"""The experiment matrix (§9.1) and the grounding mechanisms it is built from (§7).

Four mechanisms, driven by the same retrieved context, differing only in *where* the
knowledge enters the pipeline. That is the core experimental variable.

    B0       baseline, no grounding                    reference point
    C1       generic non-syllabus context              effect of conditioning per se
    C2       randomly chosen syllabus document         whether the *correct* syllabus matters
    C3       oracle syllabus document                  upper bound given perfect retrieval
    M1       retrieved context conditioning            mechanism 1 (§7.1)
    M2       retrieved token-level biasing             mechanism 2 (§7.2)
    M3a      lexical/phonetic correction on B0         mechanism 3, deterministic (§7.3)
    M3b      constrained model-based correction on B0  mechanism 3, model-based (§7.3)
    M2+M3a   best decode-time mechanism plus correction  complementarity (H3)
    G        confidence-gated biasing                  principal contribution (H4)

Cost classes (§9.2): B0/C1/C2/C3/M1/M2 are one decode each; every M3 variant and every
combination is free because it operates on cached text; G is partial, re-using decodes
it has already paid for.

Every condition writes runs/<tier>/<name>/{hyps,per_utt}.jsonl and metrics.json, plus
retrieval.json where retrieval was involved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backends import decode_config_from_config, get_backend, model_spec_from_config
from common import (ROOT, load_config, manifest_for_tier, read_jsonl, run_dir,
                    write_json, write_jsonl)
from guards import apply_context_echo_guard, apply_rewrite_guard
from lexicon import load_lexicon
from retrieve import GENERIC_CONTEXT, RetrievalPlan, SyllabusRetriever, random_topic_plan
from score import score_rows, summary_line
from transcribe import decode_rows

DECODE_CONDITIONS = ("B0", "C1", "C2", "C3", "M1", "M2")
TEXT_CONDITIONS = ("M3a", "M3b", "M2+M3a", "M1+M3a")
ALL_CONDITIONS = DECODE_CONDITIONS + TEXT_CONDITIONS + ("G",)


# --- helpers ----------------------------------------------------------------

def base_run_path(tier: str, name: str) -> Path:
    return ROOT / "runs" / tier / name / "hyps.jsonl"


def require_base(tier: str, name: str) -> list[dict]:
    p = base_run_path(tier, name)
    if not p.exists():
        raise SystemExit(
            f"condition needs {name} on {tier} first (execution order, §9.2): "
            f"missing {p.relative_to(ROOT)}")
    return read_jsonl(p)


def pass1_index(tier: str, name: str = "B0") -> dict[str, dict]:
    return {r["utt_id"]: r for r in require_base(tier, name)}


def _guard_all(rows: list[dict], fallback: dict[str, str], cfg) -> list[dict]:
    n = cfg["guards"]["context_echo_ngram"]
    th = cfg["guards"]["context_echo_threshold"]
    for row in rows:
        # M2 injects hint terms rather than a context string, but they occupy the same
        # prompt slot and can be echoed the same way, so both are guarded.
        injected = row.get("context") or row.get("hotwords")
        if injected:
            apply_context_echo_guard(row, fallback.get(row["utt_id"], ""), n, th,
                                     injected=injected)
        else:
            row["guard_context_echo"] = False
    return rows


# --- decode-time conditions --------------------------------------------------

def build_decode_condition(name: str, cfg, rows: list[dict], tier: str,
                           opts: dict) -> tuple:
    """Returns (cfg_fn, meta, retrieval_dump)."""
    style = opts.get("context_style") or cfg["context"]["style"]
    max_words = cfg["context"]["max_words"]
    n_terms = opts.get("hotword_terms") or cfg["hotwords"]["n_terms"]
    gran = opts.get("granularity") or cfg["retrieval"]["granularity"]
    k = opts.get("top_k") or cfg["retrieval"]["top_k"]
    meta = {"condition": name, "context_style": style,
            "context_max_words": max_words}

    if name == "B0":
        return (lambda row: decode_config_from_config(cfg)), meta, None

    if name == "C1":
        ctx = GENERIC_CONTEXT
        meta["context"] = ctx
        return ((lambda row: decode_config_from_config(cfg, context=ctx)), meta, None)

    r = SyllabusRetriever(cfg)

    if name == "C2":
        assign = random_topic_plan(r, rows, seed=cfg["data"]["seed"])
        ctxs = {rec: r.context_for_topic(t, style, max_words)
                for rec, t in assign.items()}
        meta.update({"assignment": assign, "seed": cfg["data"]["seed"]})

        def cfg_fn(row):
            rec = row.get("rec") or row["utt_id"].split("_")[1]
            return decode_config_from_config(cfg, context=ctxs.get(rec))
        return cfg_fn, meta, {"random_topic_per_recording": assign}

    if name == "C3":
        topics = {row["utt_id"]: r.oracle_topic(row) for row in rows}
        missing = [u for u, t in topics.items() if not t]
        if missing:
            print(f"  WARN: no oracle topic for {len(missing)} utterances "
                  f"(rec2topic.json incomplete); they decode unbiased")
        ctxs = {t: r.context_for_topic(t, style, max_words)
                for t in set(topics.values()) if t}
        meta["n_missing_oracle"] = len(missing)

        def cfg_fn(row):
            return decode_config_from_config(cfg, context=ctxs.get(topics[row["utt_id"]]))
        return cfg_fn, meta, {"oracle_topic": topics}

    # M1 / M2 both need retrieval against the pass-1 transcript
    plan = RetrievalPlan(r, rows, pass1_index(tier, opts.get("pass1", "B0")),
                         granularity=gran, k=k)
    meta.update({"granularity": gran, "top_k": k})

    if name == "M1":
        ctxs = {u: plan.context(u, style, max_words) for u in plan.by_utt}

        def cfg_fn(row):
            return decode_config_from_config(cfg, context=ctxs[row["utt_id"]])
        return cfg_fn, meta, plan.dump()

    if name == "M2":
        meta["hotword_terms"] = n_terms
        hws = {u: plan.hotwords(u, n_terms) for u in plan.by_utt}

        def cfg_fn(row):
            # §7.2: hotwords only. Setting a prefix would silently disable them, and
            # DecodeConfig raises if both are supplied.
            return decode_config_from_config(cfg, hotwords=hws[row["utt_id"]])
        return cfg_fn, meta, plan.dump()

    raise ValueError(f"not a decode condition: {name}")


# --- output-level conditions (§7.3) -----------------------------------------

def run_text_condition(name: str, cfg, rows: list[dict], tier: str,
                       opts: dict) -> tuple[list[dict], dict, dict | None]:
    base = {"M3a": "B0", "M3b": "B0", "M2+M3a": "M2", "M1+M3a": "M1"}[name]
    base_rows = require_base(tier, base)
    by_utt = {r["utt_id"]: r for r in base_rows}
    out = [dict(by_utt[r["utt_id"]]) for r in rows if r["utt_id"] in by_utt]

    r = SyllabusRetriever(cfg)
    plan = RetrievalPlan(r, rows, pass1_index(tier, "B0"),
                         granularity=opts.get("granularity")
                         or cfg["retrieval"]["granularity"],
                         k=opts.get("top_k") or cfg["retrieval"]["top_k"])
    meta = {"condition": name, "base_run": base}

    if name.endswith("M3a") or name == "M3a":
        from correct_lexical import correct_utterance
        th = opts.get("fuzzy_threshold") or cfg["correction"]["fuzzy_threshold"]
        min_len = cfg["correction"]["min_token_len"]
        meta.update({"fuzzy_threshold": th, "min_token_len": min_len})
        spans_by_utt = opts.get("spans")     # confidence gating restricts M3a (§7.4)
        n_edits = 0
        for row in out:
            cands = plan.candidates(row["utt_id"])
            new, edits = correct_utterance(
                row["hyp"], cands, th, min_len,
                spans=(spans_by_utt or {}).get(row["utt_id"]) if spans_by_utt else None)
            row["edits"] = edits
            row["corrected"] = bool(edits) and new != row["hyp"]
            if row["corrected"]:
                row["hyp_before_correction"] = row["hyp"]
                row["hyp"] = new
                n_edits += len(edits)
        meta["n_edits"] = n_edits
        return out, meta, plan.dump()

    if name == "M3b":
        from correct_llm import correct_run
        return correct_run(out, plan, cfg, meta)

    raise ValueError(name)


# --- runner ------------------------------------------------------------------

def run_condition(name: str, tier: str, cfg=None, out_name: str | None = None,
                  **opts) -> dict:
    cfg = cfg or load_config()
    rows = read_jsonl(manifest_for_tier(cfg, tier))
    if opts.get("limit"):
        rows = rows[:opts["limit"]]
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    out_name = out_name or name
    d = run_dir(out_name, tier)
    print(f"\n=== {out_name} on {tier} ({len(rows)} utts) ===", flush=True)

    if name in DECODE_CONDITIONS:
        cfg_fn, meta, retr = build_decode_condition(name, cfg, rows, tier, opts)
        backend = get_backend("local", model_spec_from_config(
            cfg, override_size=opts.get("model")))
        hyps = decode_rows(backend, rows, cfg_fn, desc=out_name)
        if name != "B0":
            fallback = {u: r["hyp"] for u, r in pass1_index(tier, "B0").items()}
            _guard_all(hyps, fallback, cfg)
    elif name in TEXT_CONDITIONS:
        hyps, meta, retr = run_text_condition(name, cfg, rows, tier, opts)
    elif name == "G":
        from gating import run_gated
        hyps, meta, retr = run_gated(cfg, rows, tier, opts)
    else:
        raise ValueError(f"unknown condition {name}")

    write_jsonl(d / "hyps.jsonl", hyps)
    if retr:
        write_json(d / "retrieval.json", retr)
    m, per_utt = score_rows(hyps, lex)
    write_jsonl(d / "per_utt.jsonl", per_utt)
    m["_condition"] = meta
    m["_tier"] = tier
    m["_model"] = cfg["model"]["size"]
    m["_decode"] = dict(cfg["decode"])
    write_json(d / "metrics.json", m)
    print(f"  {summary_line(m)}")
    if m.get("guard_context_echo_fired"):
        print(f"  context-echo guard fired on {m['guard_context_echo_fired']} utts "
              f"({100*m['guard_context_echo_rate']:.1f}%)")
    return m


def main():
    ap = argparse.ArgumentParser(description="Run one condition of the §9.1 matrix")
    ap.add_argument("condition", choices=list(ALL_CONDITIONS))
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--name", default=None, help="run directory name (default: condition)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="override model size")
    ap.add_argument("--context-style", choices=["prose", "glossary"], default=None)
    ap.add_argument("--granularity", choices=["lecture", "utterance"], default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--hotword-terms", type=int, default=None)
    ap.add_argument("--fuzzy-threshold", type=int, default=None)
    ap.add_argument("--gate-threshold", type=float, default=None)
    ap.add_argument("--gate-mechanism", default=None)
    a = ap.parse_args()
    opts = {k: v for k, v in vars(a).items()
            if v is not None and k not in ("condition", "tier", "name")}
    run_condition(a.condition, a.tier, out_name=a.name, **opts)


if __name__ == "__main__":
    main()
