"""Where is the study up to? Reads the artefacts on disk and reports against §11.

Run it any time, including while `run_all.sh` is mid-flight:

    PYTHONPATH=src .venv/bin/python src/status.py
"""
from __future__ import annotations

from pathlib import Path

from common import ROOT, load_config, read_json


def _exists(p: Path) -> bool:
    return p.exists()


def _metrics(tier: str, run: str) -> dict | None:
    p = ROOT / "runs" / tier / run / "metrics.json"
    return read_json(p) if p.exists() else None


def main() -> None:
    cfg = load_config()
    R, RUNS = ROOT / "report", ROOT / "runs"

    print("=" * 78)
    print("Syllabus-grounded contextual biasing — study status")
    print("=" * 78)

    corpus = read_json(R / "corpus_stats.json") if (R / "corpus_stats.json").exists() \
        else None
    tiers = (read_json(ROOT / "data" / "manifests" / "tiers.json")
             if (ROOT / "data" / "manifests" / "tiers.json").exists() else None)
    lex = (read_json(ROOT / "syllabus" / "index" / "lexicon_manifest.json")
           if (ROOT / "syllabus" / "index" / "lexicon_manifest.json").exists() else None)
    refine = (read_json(R / "segment_refinement.json")
              if (R / "segment_refinement.json").exists() else None)

    print(f"\nfrozen setup: model={cfg['model']['size']} "
          f"({cfg['model']['compute_type']}, beam {cfg['decode']['beam_size']}, "
          f"temp {cfg['decode']['temperature']}), language={cfg['decode']['language']}, "
          f"audio={cfg['data'].get('audio_version')}")
    if corpus:
        print(f"corpus: {corpus['n_utts']} utts / {corpus['total_hours']} h, "
              f"published-figure gate "
              f"{'PASS' if corpus['all_criteria_passed'] else 'FAIL'}")
    if refine:
        print(f"segments: refined (r={refine['radius_s']}s, "
              f"lambda={refine['lambda_db_per_s']}), |shift| "
              f"{refine['boundary_shift_abs_mean']:.2f}s, acceptance gate "
              f"{'PASS' if refine.get('acceptance_gate_passed') else 'FAIL'}")
    if tiers:
        print("tiers: " + ", ".join(f"{t['tier']}={t['n_utts']}u/{t['minutes']}min"
                                   for t in tiers["tiers"]))
    if lex:
        print(f"lexicon: {lex['n_terms']} terms, frozen, sha256 "
              f"{lex['terms_sha256_12']}")

    # --- §11 stage gates ---------------------------------------------------
    stages = [
        ("1  corpus downloaded, manifest built, tiers frozen",
         bool(corpus and tiers)),
        ("2  model running locally, caching verified, pilots complete",
         _exists(R / "pilot_model.json") and _exists(R / "pilot_language.json")),
        ("3  normalisation + scoring harness; baseline B0 locked",
         bool(_metrics("tier1", "B0"))),
        ("4  error analysis; headroom estimate stated",
         _exists(RUNS / "tier1" / "B0" / "error_analysis.json")),
        ("5  syllabus documents authored; chunk index + lexicon frozen", bool(lex)),
        ("6  retrieval implemented; top-1 accuracy measured",
         _exists(RUNS / "tier1" / "retrieval_accuracy.json")
         or _exists(RUNS / "tier2" / "retrieval_accuracy.json")),
        ("7  M1 and M2 implemented with guards",
         bool(_metrics("tier1", "M1") or _metrics("tier2", "M1"))),
        ("8  M3a/M3b thresholds swept on Tier 1",
         bool(list((RUNS / "tier1").glob("sweep_*.json"))
              if (RUNS / "tier1").exists() else False)),
        ("9  confidence gating; trade-off frontier plotted",
         bool(list((RUNS / "tier1").glob("G_sweep_*.json"))
              if (RUNS / "tier1").exists() else False)),
        ("10 full matrix on Tier 2; bootstrap p-values",
         _exists(RUNS / "tier2" / "bootstrap.json")),
        ("11 B0 and best system on Tier 3",
         bool(_metrics("tier3", "B0"))),
        ("12 figures, tables, transcripts handed on",
         _exists(R / "results_tier2.md")),
    ]
    print("\n--- §11 stage gates " + "-" * 58)
    for label, ok in stages:
        print(f"[{'x' if ok else ' '}] {label}")

    # --- pilots -------------------------------------------------------------
    for name, path in (("§4.3 language", R / "pilot_language.json"),
                       ("§4.2 model", R / "pilot_model.json")):
        if path.exists():
            d = read_json(path)
            # n_utts is printed because a pilot recorded on a handful of utterances is
            # not a pilot; without it a smoke-test record reads like a result.
            print(f"\n{name} pilot on {d.get('tier')} "
                  f"({d.get('n_utts')} utts): decision = {d['decision']}")
            for k, v in d["results"].items():
                print(f"    {k:16s} WER={v['wer']:.4f} B-WER={v['b_wer']:.4f} "
                      f"U-WER={v['u_wer']:.4f}"
                      + (f" wall={v['wall_clock_min']}min" if "wall_clock_min" in v
                         else ""))

    # --- per-tier runs ------------------------------------------------------
    for tier in ("tier1", "tier2", "tier3"):
        d = RUNS / tier
        if not d.exists():
            continue
        runs = sorted(x.name for x in d.iterdir()
                      if x.is_dir() and (x / "metrics.json").exists())
        real = [r for r in runs if not r.startswith("_")]
        sweeps = [r for r in runs if r.startswith("_")]
        if not runs:
            continue
        print(f"\n--- {tier}: {len(real)} conditions"
              + (f" (+{len(sweeps)} sweep runs)" if sweeps else "") + " " + "-" * 40)
        print(f"    {'condition':14s} {'WER':>8s} {'B-WER':>8s} {'U-WER':>8s} "
              f"{'termF1':>7s} {'echo':>6s}")
        b0 = _metrics(tier, "B0")
        for r in real:
            m = _metrics(tier, r)
            if not m or m.get("wer") is None:
                continue
            delta = ""
            if b0 and r != "B0" and b0.get("wer") is not None:
                delta = f"  ({m['wer']-b0['wer']:+.4f})"
            echo = m.get("guard_context_echo_rate")
            print(f"    {r:14s} {m['wer']:8.4f} {m['b_wer']:8.4f} "
                  f"{m['u_wer']:8.4f} {m['term_f1']:7.4f} "
                  f"{('-' if echo is None else f'{echo:.3f}'):>6s}{delta}")

    # --- what is running / what is next ------------------------------------
    logs = ROOT / "logs" / "run_all.log"
    if logs.exists():
        lines = [l for l in logs.read_text().splitlines() if l.strip()]
        print(f"\n--- run_all.sh progress " + "-" * 54)
        for l in lines[-8:]:
            print("    " + l)

    print()


if __name__ == "__main__":
    main()
