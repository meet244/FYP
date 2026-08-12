"""Run the experiment matrix in the order §9.2 prescribes.

Execution order matters and is not negotiable: B0 is decoded first and completely on
every tier, because every free condition depends on its cached output and confidence
gating requires its token-level confidence values.

Cost classes (§9.2):
    one decode each   B0, C1, C2, C3, M1, M2
    free              every M3 variant and every combination — operates on cached text
    partial           G — re-uses the mechanism decodes it has already paid for

So the full matrix is about six and a fraction decodes, not nine.

Stages, each gated on the previous one (§11):

    python src/run_matrix.py baseline --tier tier1     B0 + validation gate + analysis
    python src/run_matrix.py tune                      Tier-1 sweeps, thresholds chosen
    python src/run_matrix.py matrix --tier tier2       the full matrix + stats + report
    python src/run_matrix.py final --tier tier3        B0 and the single best system
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, load_config, read_json, write_json
from conditions import run_condition
from lexicon import load_lexicon


def _py(*args: str) -> None:
    """Run one of this project's scripts as a subprocess, inheriting stdout."""
    cmd = [sys.executable, *args]
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def lexicon_coverage(cfg, tier: str) -> dict:
    """§5.4: the fraction of reference word tokens that are lexicon terms.

    This sets the ceiling on achievable gain and belongs in the results section.
    """
    from common import manifest_for_tier, read_jsonl
    rows = read_jsonl(manifest_for_tier(cfg, tier))
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    cov = lex.coverage([r["ref"] for r in rows])
    cov.update({"tier": tier, **lex.stamp()})
    write_json(ROOT / "report" / "lexicon_coverage.json", cov)
    print(f"lexicon coverage on {tier}: {cov['bias_token_rate']*100:.2f}% of "
          f"{cov['ref_tokens']} reference tokens are lexicon terms "
          f"({cov['bias_token_rate_same_script']*100:.2f}% written in Latin script)")
    return cov


def stage_baseline(cfg, tier: str) -> None:
    """Stage 3-4 of §11: baseline locked, validation gate, headroom stated."""
    run_condition("B0", tier, cfg)
    lexicon_coverage(cfg, tier)
    _py("src/show_pairs.py", "--tier", tier, "--run", "B0", "-n", "20",
        "--sort", "random")
    _py("src/analyze_errors.py", "--tier", tier, "--run", "B0")
    m = read_json(ROOT / "runs" / tier / "B0" / "metrics.json")
    print(f"\n=== §8.1 validation gate ===")
    ok = m["wer"] < 0.60
    print(f"[{'PASS' if ok else 'CHECK'}] baseline WER = {m['wer']:.4f} "
          f"({'below' if ok else 'ABOVE'} the 0.60 plausibility threshold)")
    print(f"    level-2 (script-invariant) WER = {m['wer_level2']:.4f}; "
          f"orthographic share of error = "
          f"{100*(m['orthographic_error_share'] or 0):.1f}%")
    if not ok:
        print("    §8.1: a baseline above ~60% is more likely a normalisation defect "
              "than a model failure — inspect report/validation_pairs_*.md before "
              "proceeding.")


def stage_tune(cfg, tier: str = "tier1") -> None:
    """Stage 8-9 of §11: every threshold chosen on the development tier."""
    _py("src/eval_retrieval.py", "--tier", tier)
    _py("src/sweeps.py", "style", "--tier", tier)
    _py("src/sweeps.py", "m2", "--tier", tier)
    _py("src/sweeps.py", "m3a", "--tier", tier)
    # Gating needs a global mechanism run on the same tier to gate against.
    mech = cfg["gating"]["mechanism"]
    run_condition(mech, tier, cfg)
    _py("src/gating.py", "--tier", tier, "--sweep", "--mechanism", mech)


def stage_matrix(cfg, tier: str, skip_m3b: bool = False) -> None:
    """Stage 10 of §11: the full matrix, then statistics and the report."""
    # one decode each, B0 first and completely (§9.2 execution order)
    for cond in ("B0", "C1", "C2", "C3", "M1", "M2"):
        run_condition(cond, tier, cfg)
    # ablation rows (decodes, but cheap and informative)
    run_condition("M1", tier, cfg, out_name="M1_glossary", context_style="glossary")
    run_condition("M1", tier, cfg, out_name="M1_utterance", granularity="utterance")
    # free: operate on cached text
    run_condition("M3a", tier, cfg)
    run_condition("M2+M3a", tier, cfg)
    run_condition("M1+M3a", tier, cfg)
    if not skip_m3b:
        try:
            run_condition("M3b", tier, cfg)
        except SystemExit as exc:
            print(f"\n{exc}\n(continuing without M3b)")
    # partial: gated biasing, plus the frontier sweep
    mech = cfg["gating"]["mechanism"]
    run_condition("G", tier, cfg, gate_mechanism=mech)
    _py("src/gating.py", "--tier", tier, "--sweep", "--mechanism", mech)

    lexicon_coverage(cfg, tier)
    _py("src/eval_retrieval.py", "--tier", tier)
    _py("src/analyze_errors.py", "--tier", tier, "--run", "B0")
    _py("src/bootstrap.py", "--tier", tier)
    _py("src/make_report.py", "--tier", tier)
    _py("src/figures.py", "--tier", tier)
    _py("src/repro.py")


def stage_final(cfg, tier: str, best: str) -> None:
    """Stage 11 of §11: B0 and the single best system on the complete test set."""
    run_condition("B0", tier, cfg)
    if best in ("M3a", "M2+M3a", "M1+M3a", "G"):
        # these need their base decode on this tier first
        base = {"M2+M3a": "M2", "M1+M3a": "M1", "G": cfg["gating"]["mechanism"]}.get(best)
        if base:
            run_condition(base, tier, cfg)
    run_condition(best, tier, cfg)
    _py("src/bootstrap.py", "--tier", tier)
    _py("src/make_report.py", "--tier", tier)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["baseline", "tune", "matrix", "final"])
    ap.add_argument("--tier", default=None)
    ap.add_argument("--best", default="G", help="best system for the final tier")
    ap.add_argument("--skip-m3b", action="store_true")
    a = ap.parse_args()

    if a.stage == "baseline":
        stage_baseline(cfg, a.tier or "tier1")
    elif a.stage == "tune":
        stage_tune(cfg, a.tier or "tier1")
    elif a.stage == "matrix":
        stage_matrix(cfg, a.tier or "tier2", a.skip_m3b)
    else:
        stage_final(cfg, a.tier or "tier3", a.best)


if __name__ == "__main__":
    main()
