"""Tier-1 hyperparameter sweeps.

Every hyperparameter in this study is selected on Tier 1 and on Tier 1 only (§3.3,
§13). Each sweep writes runs/<tier>/sweep_<name>.json containing every point measured
and the chosen value, so the report can state both the selected setting and the curve it
came from.

  m3a    fuzzy-match threshold for lexical correction (§7.3). Free: text-only.
  m2     number of hint terms supplied to token-level biasing (§7.2). One decode per
         point — it interacts strongly with over-biasing, so it must be swept.
  style  prose vs glossary context for M1 (§7.1). One decode per point; the plan calls
         this a cheap and informative ablation row.
  topk   retrieval depth for M1 (§6.2). One decode per point.

Selection criterion, stated in advance and identical for every sweep: the lowest B-WER
among the points whose U-WER does not exceed the unbiased baseline's U-WER. This encodes
the study's hypothesis (H4) that terminology gain must not be bought with
non-terminology damage. If no point qualifies, the criterion falls back to lowest
overall WER and the report says so.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, load_config, read_json, write_json
from conditions import run_condition


def _baseline_u_wer(tier: str) -> float | None:
    p = ROOT / "runs" / tier / "B0" / "metrics.json"
    return read_json(p).get("u_wer") if p.exists() else None


def _choose(points: list[dict], key: str, base_u: float | None) -> tuple:
    eligible = [p for p in points
                if base_u is None or (p["u_wer"] is not None and p["u_wer"] <= base_u)]
    if eligible:
        best = min(eligible, key=lambda p: p["b_wer"])
        return best[key], "lowest B-WER with U-WER <= baseline U-WER"
    best = min(points, key=lambda p: p["wer"])
    return best[key], ("no point kept U-WER at or below the baseline; fell back to "
                       "lowest overall WER")


def _point(m: dict, key: str, value) -> dict:
    return {key: value, "wer": m["wer"], "b_wer": m["b_wer"], "u_wer": m["u_wer"],
            "term_f1": m["term_f1"], "term_precision": m["term_precision"],
            "term_recall": m["term_recall"], "wer_level2": m["wer_level2"],
            "echo_guard_rate": m.get("guard_context_echo_rate")}


def sweep_m3a(cfg, tier: str, values: list[int]) -> dict:
    points = []
    for th in values:
        m = run_condition("M3a", tier, cfg, out_name=f"_sweep_M3a_th{th}",
                          fuzzy_threshold=th)
        points.append(_point(m, "threshold", th))
    chosen, why = _choose(points, "threshold", _baseline_u_wer(tier))
    out = {"sweep": "m3a_fuzzy_threshold", "tier": tier, "points": points,
           "chosen": chosen, "criterion": why}
    write_json(ROOT / "runs" / tier / f"sweep_m3a_{tier}.json", out)
    print(f"\nM3a threshold chosen on {tier}: {chosen} ({why})")
    return out


def sweep_m2(cfg, tier: str, values: list[int]) -> dict:
    points = []
    for n in values:
        m = run_condition("M2", tier, cfg, out_name=f"_sweep_M2_n{n}",
                          hotword_terms=n)
        points.append(_point(m, "hotword_terms", n))
    chosen, why = _choose(points, "hotword_terms", _baseline_u_wer(tier))
    out = {"sweep": "m2_hotword_terms", "tier": tier, "points": points,
           "chosen": chosen, "criterion": why}
    write_json(ROOT / "runs" / tier / f"sweep_m2_{tier}.json", out)
    print(f"\nM2 hint-term count chosen on {tier}: {chosen} ({why})")
    return out


def sweep_style(cfg, tier: str) -> dict:
    points = []
    for style in ("prose", "glossary"):
        m = run_condition("M1", tier, cfg, out_name=f"_sweep_M1_{style}",
                          context_style=style)
        p = _point(m, "style", style)
        points.append(p)
    chosen, why = _choose(points, "style", _baseline_u_wer(tier))
    out = {"sweep": "m1_context_style", "tier": tier, "points": points,
           "chosen": chosen, "criterion": why}
    write_json(ROOT / "runs" / tier / f"sweep_style_{tier}.json", out)
    print(f"\nM1 context style chosen on {tier}: {chosen} ({why})")
    return out


def sweep_topk(cfg, tier: str, values: list[int]) -> dict:
    points = []
    for k in values:
        m = run_condition("M1", tier, cfg, out_name=f"_sweep_M1_k{k}", top_k=k)
        points.append(_point(m, "top_k", k))
    chosen, why = _choose(points, "top_k", _baseline_u_wer(tier))
    out = {"sweep": "m1_top_k", "tier": tier, "points": points,
           "chosen": chosen, "criterion": why}
    write_json(ROOT / "runs" / tier / f"sweep_topk_{tier}.json", out)
    print(f"\nM1 retrieval depth chosen on {tier}: {chosen} ({why})")
    return out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["m3a", "m2", "style", "topk", "all"])
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--m3a", default="80,84,88,92,96")
    ap.add_argument("--m2", default="10,20,30,50")
    ap.add_argument("--topk", default="1,3,5")
    a = ap.parse_args()

    if a.which in ("m3a", "all"):
        sweep_m3a(cfg, a.tier, [int(x) for x in a.m3a.split(",")])
    if a.which in ("style", "all"):
        sweep_style(cfg, a.tier)
    if a.which in ("m2", "all"):
        sweep_m2(cfg, a.tier, [int(x) for x in a.m2.split(",")])
    if a.which in ("topk", "all"):
        sweep_topk(cfg, a.tier, [int(x) for x in a.topk.split(",")])


if __name__ == "__main__":
    main()
