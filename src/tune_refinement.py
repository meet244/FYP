"""Select the boundary-refinement parameters (search radius, displacement penalty).

`refine_segments.py` has two knobs and they trade off against each other: a wide search
radius with a weak penalty rescues badly misplaced boundaries but also moves boundaries
that were already correct, while a tight radius with a strong penalty is conservative and
leaves the catastrophic cases unfixed.

This script measures both effects on the *same* Tier-1 utterance sample the offset
diagnostic used, cutting and decoding only those utterances, so a setting can be chosen
on evidence rather than taste. Selection happens on Tier 1 — the development slice — and
the chosen values are recorded in report/refinement_tuning.json.

Reported per setting: mean and median per-utterance WER, and the fraction of utterances
whose WER is worse than with the distributed windows (the cost of touching boundaries
that did not need it).
"""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

from backends import decode_config_from_config, get_backend, model_spec_from_config
from common import ROOT, load_config, read_json, read_jsonl, write_json
from lexicon import load_lexicon
from normalize import level1
from refine_segments import energy_db, load_kaldi, refine_boundaries
from score import decompose_utterance

TMP = ROOT / "data" / "audio" / "_tune"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--grid", default="2.0:3.0,1.5:4.0,1.0:6.0,2.5:2.0",
                    help="comma-separated radius:lambda pairs")
    a = ap.parse_args()

    # Same sample as diagnose_segments.py: same tier, filter and seed.
    rows = read_jsonl(ROOT / cfg["data"]["tiers"][a.tier])
    rows = [r for r in rows if len(r["ref"].split()) >= 5]
    sample = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    want = {r["utt_id"]: r for r in sample}

    kd, text, segs, _ = load_kaldi(cfg)
    by_rec: dict[str, list] = {}
    for u, (rec, s, e) in segs.items():
        if u in text:
            by_rec.setdefault(rec, []).append((u, s, e))
    for rec in by_rec:
        by_rec[rec].sort(key=lambda t: t[1])
    recs_needed = sorted({segs[u][0] for u in want})

    backend = get_backend("local", model_spec_from_config(cfg))
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    dc = decode_config_from_config(cfg)
    TMP.mkdir(parents=True, exist_ok=True)

    prev = read_json(ROOT / "report" / "segment_offset_diagnostic.json")
    baseline_per_utt = {d["utt_id"]: d["by_offset"]["0.0"]["wer"]
                        for d in prev["detail"]}
    baseline_mean = prev["summary"]["0.0"]["mean_wer"]
    baseline_median = prev["summary"]["0.0"]["median_wer"]

    grid = [tuple(float(x) for x in g.split(":")) for g in a.grid.split(",")]
    results = []
    energy_cache: dict[str, np.ndarray] = {}
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for radius, lam in grid:
        wers, worse, moved = [], 0, []
        for rec in recs_needed:
            if rec not in audio_cache:
                x, sr = sf.read(str(kd.parent / f"{rec}.wav"), dtype="float32",
                                always_2d=True)
                x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
                audio_cache[rec] = (x, sr)
                energy_cache[rec] = energy_db(x, sr)[0]
            x, sr = audio_cache[rec]
            items = by_rec[rec]
            nominal = [items[0][1]] + [it[2] for it in items]
            refined = refine_boundaries(energy_cache[rec], nominal, radius, lam)
            for i, (u, _, _) in enumerate(items):
                if u not in want:
                    continue
                s, e = refined[i], refined[i + 1]
                moved.append(abs(s - nominal[i]))
                aa, bb = int(round(s * sr)), int(round(e * sr))
                seg = x[max(0, aa):min(len(x), bb)]
                wav = TMP / f"{u}.wav"
                sf.write(str(wav), seg.astype(np.float32), sr, subtype="PCM_16")
                res = backend.transcribe(str(wav), dc)
                m = decompose_utterance(level1(want[u]["ref"]),
                                        level1(res["text"]), lex)
                wers.append(m["wer"])
                if m["wer"] > baseline_per_utt.get(u, 9e9) + 1e-9:
                    worse += 1
                wav.unlink(missing_ok=True)
        entry = {
            "radius": radius, "lambda": lam,
            "mean_wer": statistics.mean(wers), "median_wer": statistics.median(wers),
            "pct_worse_than_distributed": 100 * worse / len(wers),
            "mean_abs_shift_s": float(np.mean(moved)),
        }
        results.append(entry)
        print(f"radius={radius:<4} lambda={lam:<4} mean={entry['mean_wer']:.4f} "
              f"median={entry['median_wer']:.4f} "
              f"worse={entry['pct_worse_than_distributed']:.0f}% "
              f"|shift|={entry['mean_abs_shift_s']:.2f}s", flush=True)

    # Selection: lowest mean WER. Corpus WER aggregates total errors, so the mean over
    # utterances is the quantity that tracks it; the median is reported as a check that
    # the typical utterance is not being sacrificed for the outliers.
    best = min(results, key=lambda r: r["mean_wer"])
    out = {"n_utts": len(sample), "seed": a.seed, "tier": a.tier,
           "distributed_mean_wer": baseline_mean,
           "distributed_median_wer": baseline_median,
           "grid": results, "chosen": {"radius": best["radius"],
                                       "lambda": best["lambda"]},
           "criterion": "lowest mean per-utterance WER on the Tier-1 sample"}
    write_json(ROOT / "report" / "refinement_tuning.json", out)
    print(f"\ndistributed windows: mean {baseline_mean:.4f} median {baseline_median:.4f}")
    print(f"chosen: radius={best['radius']} lambda={best['lambda']} "
          f"(mean {best['mean_wer']:.4f}, median {best['median_wer']:.4f})")


if __name__ == "__main__":
    main()
