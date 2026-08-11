"""Paired bootstrap significance test over per-utterance edit counts."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_bootstrap(err_a, len_a, err_b, len_b, n=10000, seed=0):
    """err_*: per-utterance edit counts; len_*: per-utterance ref lengths.
    Returns the fraction of resamples in which system B beats system A."""
    rng = np.random.default_rng(seed)
    ea, la, eb, lb = map(np.asarray, (err_a, len_a, err_b, len_b))
    n_utt = len(ea)
    idx = rng.integers(0, n_utt, size=(n, n_utt))
    wer_a = ea[idx].sum(1) / la[idx].sum(1)
    wer_b = eb[idx].sum(1) / lb[idx].sum(1)
    return float((wer_b < wer_a).mean())


def load(run):
    p = Path("runs") / run / "per_utt.jsonl"
    return {json.loads(l)["utt_id"]: json.loads(l) for l in open(p, encoding="utf-8")}


def compare(run_a, run_b, n=10000):
    a, b = load(run_a), load(run_b)
    keys = sorted(set(a) & set(b))
    ea = [a[k]["errors"] for k in keys]
    la = [a[k]["ref_len"] for k in keys]
    eb = [b[k]["errors"] for k in keys]
    lb = [b[k]["ref_len"] for k in keys]
    wins = paired_bootstrap(ea, la, eb, lb, n=n)
    improved = sum(1 for k in keys if b[k]["errors"] < a[k]["errors"])
    regressed = sum(1 for k in keys if b[k]["errors"] > a[k]["errors"])
    return {
        "a": run_a, "b": run_b, "n_utts": len(keys),
        "wer_a": sum(ea) / sum(la), "wer_b": sum(eb) / sum(lb),
        "abs_delta": sum(eb) / sum(lb) - sum(ea) / sum(la),
        "rel_delta_pct": 100 * (sum(eb) / sum(lb) - sum(ea) / sum(la)) / (sum(ea) / sum(la)),
        "pct_improved": 100 * improved / len(keys),
        "pct_regressed": 100 * regressed / len(keys),
        "p_value": 1 - wins,   # H1: B better than A
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("-n", type=int, default=10000)
    a = ap.parse_args()
    print(json.dumps(compare(a.run_a, a.run_b, a.n), indent=2))
