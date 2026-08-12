"""Paired bootstrap significance testing (§8.3).

Mean WER differences on a few hundred utterances are easily noise. The test resamples
utterances with replacement many times, recomputes the aggregate error rate for both
systems on each resample, and reports the proportion of resamples in which the grounded
system wins. A p-value is reported for every system against the baseline.

The test is run on all three headline quantities — WER, B-WER and U-WER — because the
central claim (H4) is directional: B-WER should fall while U-WER does not rise, and each
half of that statement needs its own test.

This is only possible because per-utterance edit counts and reference lengths are stored
rather than corpus aggregates: recomputing them from cached hypotheses is exactly what
`score.py` writes to per_utt.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import ROOT, read_jsonl, write_json

METRICS = {
    "wer": ("errors", "ref_len"),
    "b_wer": ("b_errors", "b_ref_len"),
    "u_wer": ("u_errors", "u_ref_len"),
}


def load_per_utt(tier: str, run: str) -> dict[str, dict]:
    p = ROOT / "runs" / tier / run / "per_utt.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    return {r["utt_id"]: r for r in read_jsonl(p)}


def paired_bootstrap(err_a, den_a, err_b, den_b, n: int = 10000,
                     seed: int = 0) -> dict:
    """Fraction of resamples in which system B has the lower error rate than A."""
    rng = np.random.default_rng(seed)
    ea, da, eb, db = (np.asarray(x, dtype=float) for x in (err_a, den_a, err_b, den_b))
    n_utt = len(ea)
    if n_utt == 0:
        return {"p_value": None, "b_wins": None}
    idx = rng.integers(0, n_utt, size=(n, n_utt))
    sa, sb = da[idx].sum(1), db[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_a = np.where(sa > 0, ea[idx].sum(1) / np.where(sa > 0, sa, 1), np.nan)
        rate_b = np.where(sb > 0, eb[idx].sum(1) / np.where(sb > 0, sb, 1), np.nan)
    valid = ~(np.isnan(rate_a) | np.isnan(rate_b))
    if not valid.any():
        return {"p_value": None, "b_wins": None}
    wins = float((rate_b[valid] < rate_a[valid]).mean())
    # One-sided p-value for "B is better than A".
    return {"b_wins": wins, "p_value": 1.0 - wins,
            "resamples": int(valid.sum())}


def compare(tier: str, run_a: str, run_b: str, n: int = 10000,
            seed: int = 0) -> dict:
    a, b = load_per_utt(tier, run_a), load_per_utt(tier, run_b)
    keys = sorted(set(a) & set(b))
    out = {"tier": tier, "a": run_a, "b": run_b, "n_utts": len(keys),
           "resamples": n, "seed": seed}

    for metric, (ekey, dkey) in METRICS.items():
        ea = [a[k][ekey] for k in keys]
        da = [a[k][dkey] for k in keys]
        eb = [b[k][ekey] for k in keys]
        db = [b[k][dkey] for k in keys]
        rate_a = sum(ea) / sum(da) if sum(da) else None
        rate_b = sum(eb) / sum(db) if sum(db) else None
        res = paired_bootstrap(ea, da, eb, db, n=n, seed=seed)
        out[metric] = {
            "a": rate_a, "b": rate_b,
            "abs_delta": (rate_b - rate_a) if (rate_a is not None
                                               and rate_b is not None) else None,
            "rel_delta_pct": (100 * (rate_b - rate_a) / rate_a
                              if rate_a else None),
            "p_value": res["p_value"], "b_wins": res["b_wins"],
        }

    # Regression counter (§7.5): count utterances whose error increased, not just the
    # mean. As important as the mean.
    improved = sum(1 for k in keys if b[k]["errors"] < a[k]["errors"])
    regressed = sum(1 for k in keys if b[k]["errors"] > a[k]["errors"])
    unchanged = len(keys) - improved - regressed
    out["utterances"] = {
        "improved": improved, "regressed": regressed, "unchanged": unchanged,
        "pct_improved": 100 * improved / len(keys) if keys else None,
        "pct_regressed": 100 * regressed / len(keys) if keys else None,
    }
    return out


def compare_all(tier: str, baseline: str = "B0", n: int = 10000,
                seed: int = 0) -> dict:
    runs = [d.name for d in sorted((ROOT / "runs" / tier).iterdir())
            if (d / "per_utt.jsonl").exists() and d.name != baseline]
    out = {r: compare(tier, baseline, r, n=n, seed=seed) for r in runs}
    write_json(ROOT / "runs" / tier / "bootstrap.json", out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Paired bootstrap test (§8.3)")
    ap.add_argument("--tier", default="tier2")
    ap.add_argument("--baseline", default="B0")
    ap.add_argument("--run", default=None, help="single run; default: all runs")
    ap.add_argument("-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.run:
        print(json.dumps(compare(a.tier, a.baseline, a.run, a.n, a.seed), indent=2))
    else:
        res = compare_all(a.tier, a.baseline, a.n, a.seed)
        print(f"{'run':10s} {'WER':>8s} {'ΔWER':>8s} {'p':>7s} "
              f"{'B-WER':>8s} {'Δ':>8s} {'p':>7s} {'U-WER':>8s} {'Δ':>8s} {'p':>7s} "
              f"{'%impr':>6s} {'%regr':>6s}")
        for r, c in res.items():
            def f(m, k, nd=4):
                v = c[m][k]
                return "-" if v is None else f"{v:.{nd}f}"
            print(f"{r:10s} {f('wer','b')} {f('wer','abs_delta')} "
                  f"{f('wer','p_value',3):>7s} {f('b_wer','b')} "
                  f"{f('b_wer','abs_delta')} {f('b_wer','p_value',3):>7s} "
                  f"{f('u_wer','b')} {f('u_wer','abs_delta')} "
                  f"{f('u_wer','p_value',3):>7s} "
                  f"{c['utterances']['pct_improved']:6.1f} "
                  f"{c['utterances']['pct_regressed']:6.1f}")
        print(f"\n-> runs/{a.tier}/bootstrap.json")


if __name__ == "__main__":
    main()
