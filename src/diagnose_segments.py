"""Data validation: are the distributed segment boundaries usable for utterance-level
scoring? (§3.4 acceptance criteria, §8.1 validation gate.)

The SLR104 test distribution's `segments` file is a contiguous partition of each
recording into whole-second windows. Whole-second boundaries cannot coincide with real
speech edges, so before trusting any baseline WER we must know how far the text-to-audio
assignment is off and in which direction.

Method: for a sample of utterances, cut the window shifted by each candidate offset
delta, decode it, and score against the reference. If the corpus alignment is sound the
WER curve is minimised at delta = 0. A minimum elsewhere means the transcript lags or
leads the audio and the window must be corrected before any experiment runs.

Run: python src/diagnose_segments.py --n 24 --offsets -2,-1,0,1,2,3
"""
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

from backends import decode_config_from_config, get_backend, model_spec_from_config
from common import ROOT, load_config, read_jsonl, write_json
from lexicon import load_lexicon
from normalize import level1
from score import decompose_utterance

TMP = ROOT / "data" / "audio" / "_diag"


def cut(rec_path: Path, start: float, end: float, out: Path) -> float:
    x, sr = sf.read(str(rec_path), dtype="float32", always_2d=True)
    if x.shape[1] > 1:
        x = x.mean(axis=1, keepdims=True)
    x = x[:, 0]
    a, b = int(round(start * sr)), int(round(end * sr))
    seg = x[max(0, a):min(len(x), b)]
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), seg, sr, subtype="PCM_16")
    return len(seg) / sr


def validate(cfg, a) -> dict:
    """Score the refined cuts on the same sample the offset diagnostic used.

    Acceptance criterion: the refined windows must beat the distributed windows
    (offset 0) on both mean and median WER over the identical utterance sample.
    """
    rows = read_jsonl(ROOT / cfg["data"]["tiers"][a.tier])
    rows = [r for r in rows if len(r["ref"].split()) >= 5]
    sample = random.Random(a.seed).sample(rows, min(a.n, len(rows)))

    backend = get_backend("local", model_spec_from_config(cfg))
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    dc = decode_config_from_config(cfg)

    wers, detail = [], []
    for i, row in enumerate(sample, 1):
        res = backend.transcribe(row["audio"], dc)
        ref, hyp = level1(row["ref"]), level1(res["text"])
        m = decompose_utterance(ref, hyp, lex)
        wers.append(m["wer"])
        detail.append({"utt_id": row["utt_id"], "wer": round(m["wer"], 3),
                       "ref": ref, "hyp": hyp})
        print(f"[{i}/{len(sample)}] {row['utt_id']} WER={m['wer']:.2f}", flush=True)

    prev_path = ROOT / "report" / "segment_offset_diagnostic.json"
    prev = None
    if prev_path.exists():
        prev = __import__("json").loads(prev_path.read_text())["summary"].get("0.0")

    out = {
        "n_utts": len(sample), "audio_version": cfg["data"].get("audio_version"),
        "refined": {"mean_wer": statistics.mean(wers),
                    "median_wer": statistics.median(wers)},
        "distributed_offset0": prev,
    }
    print(f"\nrefined cuts     mean WER {out['refined']['mean_wer']:.4f}  "
          f"median {out['refined']['median_wer']:.4f}")
    if prev:
        print(f"distributed cuts mean WER {prev['mean_wer']:.4f}  "
              f"median {prev['median_wer']:.4f}")
        better = (out["refined"]["mean_wer"] < prev["mean_wer"]
                  and out["refined"]["median_wer"] < prev["median_wer"])
        out["refinement_improves_both"] = better
        print(f"[{'PASS' if better else 'FAIL'}] refined windows beat the distributed "
              f"windows on mean and median WER")
    write_json(ROOT / "report" / "segment_refinement_validation.json", out)
    return out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--offsets", default="-2,-1,0,1,2,3")
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--validate", action="store_true",
                    help="score the current (refined) cuts on the same sample and "
                         "compare against the distributed windows")
    a = ap.parse_args()

    if a.validate:
        validate(cfg, a)
        return

    offsets = [float(x) for x in a.offsets.split(",")]
    rows = read_jsonl(ROOT / cfg["data"]["tiers"][a.tier])
    # Only utterances with at least 5 reference words: a 3-word utterance cannot
    # discriminate between offsets.
    rows = [r for r in rows if len(r["ref"].split()) >= 5]
    sample = random.Random(a.seed).sample(rows, min(a.n, len(rows)))

    seg_map = {}
    for line in open(ROOT / cfg["data"]["kaldi_dir"] / "segments", encoding="utf-8"):
        u, rec, s, e = line.split()
        seg_map[u] = (rec, float(s), float(e))

    backend = get_backend("local", model_spec_from_config(cfg))
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    dc = decode_config_from_config(cfg)

    per_offset: dict[float, list[float]] = {d: [] for d in offsets}
    detail = []
    for i, row in enumerate(sample, 1):
        rec, s, e = seg_map[row["utt_id"]]
        rec_path = ROOT / cfg["data"]["kaldi_dir"] / ".." / f"{rec}.wav"
        rec_path = rec_path.resolve()
        ref = level1(row["ref"])
        entry = {"utt_id": row["utt_id"], "ref": ref, "window": [s, e], "by_offset": {}}
        for d in offsets:
            wav = TMP / f"{row['utt_id']}_off{d:+.0f}.wav"
            cut(rec_path, max(0.0, s + d), e + d, wav)
            res = backend.transcribe(str(wav), dc)
            hyp = level1(res["text"])
            m = decompose_utterance(ref, hyp, lex)
            per_offset[d].append(m["wer"])
            entry["by_offset"][str(d)] = {"wer": round(m["wer"], 3), "hyp": hyp}
            wav.unlink(missing_ok=True)
        detail.append(entry)
        best = min(entry["by_offset"], key=lambda k: entry["by_offset"][k]["wer"])
        print(f"[{i}/{len(sample)}] {row['utt_id']} best offset={best} "
              f"WER={entry['by_offset'][best]['wer']:.2f} "
              f"(offset 0: {entry['by_offset']['0.0']['wer']:.2f})", flush=True)

    print(f"\n{'offset':>8s} {'mean WER':>10s} {'median WER':>11s} {'best-for-n':>11s}")
    summary = {}
    for d in offsets:
        wers = per_offset[d]
        n_best = sum(1 for e in detail
                     if min(e["by_offset"], key=lambda k: e["by_offset"][k]["wer"])
                     == str(d))
        summary[str(d)] = {"mean_wer": statistics.mean(wers),
                           "median_wer": statistics.median(wers),
                           "n_utts_best": n_best}
        print(f"{d:+8.0f} {statistics.mean(wers):10.4f} "
              f"{statistics.median(wers):11.4f} {n_best:11d}")

    best_offset = min(summary, key=lambda k: summary[k]["mean_wer"])
    verdict = ("distributed segments are usable as-is" if float(best_offset) == 0.0
               else f"transcript/audio alignment is offset by {best_offset}s; "
                    f"windows must be corrected before any experiment")
    print(f"\nbest mean-WER offset: {best_offset}s -> {verdict}")
    write_json(ROOT / "report" / "segment_offset_diagnostic.json",
               {"n_utts": len(sample), "offsets": offsets, "summary": summary,
                "best_offset": float(best_offset), "verdict": verdict,
                "detail": detail})


if __name__ == "__main__":
    main()
