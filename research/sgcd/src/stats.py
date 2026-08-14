"""Paired bootstrap over utterances: ΔWER with 95% CI, plus degradation rate.

Two estimators are reported because they answer different questions:
  - corpus ΔWER  = sum(errors)/sum(ref_len) difference, resampling utterances.
    This is the number that belongs in the results table.
  - macro ΔWER   = mean of per-utterance WER differences (what §4.3 of the plan
    specifies). Reported alongside; it weights short utterances equally.
"""
import argparse
import json

import numpy as np

from config import OUT, SEED


def load(model, cond, split="test"):
    p = OUT / f"perutt__{model}__{cond}__{split}.json"
    if not p.exists():
        return None
    d = json.load(p.open())
    return {x["utt_id"]: x for x in d}


def _align(a, b):
    keys = [k for k in a if k in b]
    e_a = np.array([a[k]["errors"] for k in keys], float)
    e_b = np.array([b[k]["errors"] for k in keys], float)
    n = np.array([a[k]["ref_len"] for k in keys], float)
    return e_a, e_b, n


def paired_bootstrap(a, b, n_boot=10000, seed=SEED):
    """a = baseline per-utt stats, b = system per-utt stats (dicts keyed by utt_id)."""
    e_a, e_b, n = _align(a, b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))

    corpus = e_b.sum() / n.sum() - e_a.sum() / n.sum()
    bs = (e_b[idx].sum(axis=1) - e_a[idx].sum(axis=1)) / n[idx].sum(axis=1)

    w_a, w_b = e_a / np.maximum(n, 1), e_b / np.maximum(n, 1)
    d = w_b - w_a
    macro, macro_bs = d.mean(), d[idx].mean(axis=1)

    return dict(
        n=len(n),
        corpus_delta=corpus,
        ci=(np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
        p_improve=float((bs < 0).mean()),
        macro_delta=macro,
        macro_ci=(np.percentile(macro_bs, 2.5), np.percentile(macro_bs, 97.5)),
        degraded=float((d > 0).mean()),
        improved=float((d < 0).mean()),
    )


def report(base_name, sys_name, r):
    lo, hi = r["ci"]
    sig = "*" if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else " "
    print(
        f"{base_name} -> {sys_name}: ΔWER = {r['corpus_delta']*100:+.2f}  "
        f"95% CI [{lo*100:+.2f}, {hi*100:+.2f}]  P(improve)={r['p_improve']:.3f} {sig}   "
        f"| macro Δ={r['macro_delta']*100:+.2f}  worse on {r['degraded']*100:.0f}% / "
        f"better on {r['improved']*100:.0f}% of utts  (n={r['n']})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    base = load(a.model, "C0", a.split)
    if base is None:
        raise SystemExit(f"no C0 per-utterance scores for {a.model}/{a.split} — run score.py first")

    print(f"=== {a.model} / {a.split} — paired bootstrap vs C0 (10k resamples, seed {SEED}) ===")
    out = {}
    for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
        s = load(a.model, c, a.split)
        if s is None:
            continue
        out[f"C0->{c}"] = r = paired_bootstrap(base, s)
        report("C0", c, r)

    pairs = [
        ("C2", "C3", "H4: prose rendering beats keyword list"),
        ("C3", "C4", "H5: retrieval beats whole-syllabus"),
        ("C5", "C4", "H3: matched syllabus beats mismatched (content-specificity)"),
        ("C4", "C7", "guard effect"),
    ]
    print()
    for x, y, label in pairs:
        sx, sy = load(a.model, x, a.split), load(a.model, y, a.split)
        if sx is None or sy is None:
            continue
        out[f"{x}->{y}"] = r = paired_bootstrap(sx, sy)
        print(f"[{label}]")
        report(x, y, r)

    ser = {
        k: {kk: (list(vv) if isinstance(vv, tuple) else float(vv)) for kk, vv in v.items()}
        for k, v in out.items()
    }
    (OUT / f"stats__{a.model}__{a.split}.json").write_text(json.dumps(ser, indent=2))
    print(f"\nwrote {OUT}/stats__{a.model}__{a.split}.json")


if __name__ == "__main__":
    main()
