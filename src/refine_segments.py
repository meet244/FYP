"""Boundary refinement for the SLR104 test segments (data preparation, §3.4).

Why this step exists
--------------------
The distributed `segments` file partitions each recording into **whole-second** windows
that tile the file exactly. Whole-second boundaries cannot coincide with real speech
edges, and `diagnose_segments.py` measures the consequence: for a sampled utterance the
WER-minimising window shift is scattered between -2 s and +2 s, with large individual
gains (one utterance drops from 5.62 to 0.88 WER at -2 s). A cut window therefore often
contains the tail of the neighbouring sentence, which the model dutifully transcribes and
which is then scored as insertions. Because the best shift differs per utterance, no
global offset fixes it.

What this step does
-------------------
It keeps the distributed transcript-to-window assignment — same utterances, same order,
same count — and only moves each *shared internal boundary* to the quietest instant
within a search radius. Utterance boundaries in continuous speech fall in inter-sentence
pauses, so snapping a boundary to nearby silence recovers the true edge for the two
utterances that share it.

Boundaries are chosen by minimising

    score(t) = mean_energy_db over a short window centred on t  +  lambda * |t - t_nominal|

The regularisation term keeps a boundary near where the corpus put it: without it a
boundary would migrate to the longest pause in the search radius, which may belong to a
different sentence break.

This is a documented, deterministic preprocessing step applied identically to every
condition, and it touches only *where the audio is cut* — never the reference text. Run
`--validate` afterwards to confirm the refined windows score better than the distributed
ones on the same sample of utterances.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ROOT, load_config, write_json, write_jsonl

FRAME = 0.010          # 10 ms energy frames
SMOOTH = 0.050         # 50 ms smoothing
CUT_WIN = 0.30         # a boundary is judged by the energy in this window around it
MIN_UTT = 0.40         # absolute floor on any utterance duration
MIN_KEEP = 0.60        # an utterance must retain at least this share of its nominal
                       # duration: a reference of sixteen words cannot be spoken in the
                       # 0.4 s that an unconstrained greedy repair once left it


def energy_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Frame-wise RMS energy in dB relative to the recording's loudest frame."""
    hop = max(1, int(round(FRAME * sr)))
    n = len(x) // hop
    frames = x[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    k = max(1, int(round(SMOOTH / FRAME)))
    if k > 1:
        kernel = np.ones(k) / k
        rms = np.convolve(rms, kernel, mode="same")
    db = 20 * np.log10(rms / max(rms.max(), 1e-12) + 1e-12)
    return db, FRAME


def refine_boundaries(db: np.ndarray, nominal: list[float], radius: float,
                      lam: float) -> list[float]:
    """Choose all internal boundaries jointly, subject to minimum-duration constraints.

    Selecting each boundary independently and then repairing the ordering greedily is
    unsound: a chain of boundaries that all drift left leaves the utterances between them
    squeezed onto the minimum-length floor, and a sixteen-word reference ends up with 0.4 s
    of audio. That failure was caught by the §8.1 validation gate — 382 of 3,136
    utterances had been collapsed.

    This is instead a shortest-path problem. Boundary j may sit anywhere within `radius`
    of its nominal position; utterance i must keep at least
    `max(MIN_UTT, MIN_KEEP * nominal duration)` seconds; and the total cost is the sum of
    each boundary's frame energy plus `lam` per second of displacement. Minimising that
    subject to the constraints is exact and linear in the number of frames, because the
    constraint `t_j >= t_{j-1} + floor` makes the inner minimisation a prefix minimum.
    """
    n_frames = len(db)
    half = max(1, int(round((CUT_WIN / 2) / FRAME)))
    cum = np.concatenate([[0.0], np.cumsum(db)])

    # Mean energy in a CUT_WIN window centred on every frame, vectorised.
    idx_all = np.arange(n_frames)
    a = np.maximum(0, idx_all - half)
    b = np.minimum(n_frames, idx_all + half + 1)
    energy = (cum[b] - cum[a]) / np.maximum(1, b - a)

    nb = len(nominal)                       # boundaries: 0 .. nb-1 (nb-1 utterances)
    nom_f = [int(round(t / FRAME)) for t in nominal]
    r = int(round(radius / FRAME))
    floors = [max(int(round(MIN_UTT / FRAME)),
                  int(round(MIN_KEEP * (nom_f[i + 1] - nom_f[i]))))
              for i in range(nb - 1)]

    INF = np.inf
    # f[j] holds, for every frame, the best total cost of placing boundary j there.
    f: list[np.ndarray] = []
    windows: list[tuple[int, int]] = []

    # Boundary 0 is fixed at its nominal position (start of the first utterance).
    f0 = np.full(n_frames, INF)
    f0[nom_f[0]] = 0.0
    f.append(f0)
    windows.append((nom_f[0], nom_f[0]))

    for j in range(1, nb):
        fixed_last = (j == nb - 1)          # end of the last utterance is fixed too
        if fixed_last:
            lo = hi = min(nom_f[j], n_frames - 1)
        else:
            lo = max(1, nom_f[j] - r)
            hi = min(n_frames - 1, nom_f[j] + r)
        prev = f[j - 1]
        pref = np.minimum.accumulate(prev)   # pref[t] = best cost with t_{j-1} <= t
        cur = np.full(n_frames, INF)
        floor = floors[j - 1]
        ts = np.arange(lo, hi + 1)
        src = ts - floor                     # latest allowed position of boundary j-1
        ok = src >= 0
        if ok.any():
            base = np.full(ts.shape, INF)
            base[ok] = pref[src[ok]]
            step = energy[ts] + lam * np.abs(ts - nom_f[j]) * FRAME
            cur[ts] = base + step
        f.append(cur)
        windows.append((lo, hi))

    if not np.isfinite(f[-1]).any():         # constraints infeasible; keep the corpus's
        return list(nominal)                 # own boundaries rather than invent any

    # Backtrack: choose each boundary as the best position consistent with its successor.
    out_f = [0] * nb
    out_f[nb - 1] = int(np.nanargmin(np.where(np.isfinite(f[-1]), f[-1], np.nan)))
    for j in range(nb - 2, -1, -1):
        limit = out_f[j + 1] - floors[j]
        if limit < 0:
            return list(nominal)
        cand = f[j][:limit + 1]
        if not np.isfinite(cand).any():
            return list(nominal)
        out_f[j] = int(np.nanargmin(np.where(np.isfinite(cand), cand, np.nan)))
    return [t * FRAME for t in out_f]


def load_kaldi(cfg):
    kd = ROOT / cfg["data"]["kaldi_dir"]
    text, segs, utt2spk = {}, {}, {}
    for line in open(kd / "text", encoding="utf-8"):
        p = line.strip().split(maxsplit=1)
        if len(p) == 2:
            text[p[0]] = p[1]
    for line in open(kd / "segments", encoding="utf-8"):
        u, rec, s, e = line.split()
        segs[u] = (rec, float(s), float(e))
    if (kd / "utt2spk").exists():
        for line in open(kd / "utt2spk", encoding="utf-8"):
            p = line.strip().split(maxsplit=1)
            if len(p) == 2:
                utt2spk[p[0]] = p[1]
    return kd, text, segs, utt2spk


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=2.0,
                    help="search radius in seconds around the nominal boundary")
    ap.add_argument("--lam", type=float, default=3.0,
                    help="dB penalty per second of displacement")
    ap.add_argument("--audio-out", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    a = ap.parse_args()

    kd, text, segs, utt2spk = load_kaldi(cfg)
    audio_out = a.audio_out or ROOT / cfg["data"]["audio_dir"]
    manifest = a.manifest or ROOT / cfg["data"]["manifest"]
    audio_out.mkdir(parents=True, exist_ok=True)

    by_rec: dict[str, list[tuple[str, float, float]]] = {}
    for u, (rec, s, e) in segs.items():
        if u in text:
            by_rec.setdefault(rec, []).append((u, s, e))
    for rec in by_rec:
        by_rec[rec].sort(key=lambda t: t[1])

    rows, shifts = [], []
    recs = sorted(by_rec)
    for n, rec in enumerate(recs, 1):
        items = by_rec[rec]
        wav = (kd.parent / f"{rec}.wav")
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        db, _ = energy_db(x, sr)

        nominal = [items[0][1]] + [it[2] for it in items]
        refined = refine_boundaries(db, nominal, a.radius, a.lam)
        shifts.extend(r - o for r, o in zip(refined[1:-1], nominal[1:-1]))

        for i, (u, s_nom, e_nom) in enumerate(items):
            s, e = refined[i], refined[i + 1]
            aa, bb = int(round(s * sr)), int(round(e * sr))
            seg = x[max(0, aa):min(len(x), bb)]
            if len(seg) == 0:
                continue
            sf.write(str(audio_out / f"{u}.wav"), seg.astype(np.float32), sr,
                     subtype="PCM_16")
            rows.append({"utt_id": u, "audio": str(audio_out / f"{u}.wav"),
                         "ref": text[u], "duration": round(len(seg) / sr, 3),
                         "rec": rec, "spk": utt2spk.get(u),
                         "start": round(s, 3), "end": round(e, 3),
                         "start_nominal": s_nom, "end_nominal": e_nom})
        print(f"  refined {n}/{len(recs)} recordings", flush=True)

    rows.sort(key=lambda r: r["utt_id"])

    # --- acceptance gate ---------------------------------------------------
    # A boundary optimiser that squeezes an utterance too small is worse than no
    # refinement at all: the audio no longer contains the words the reference claims.
    # An earlier greedy version did exactly that to 382 utterances and the defect only
    # surfaced in the §8.1 pair inspection. It is checked here instead.
    ratios = [r["duration"] / max(1e-9, r["end_nominal"] - r["start_nominal"])
              for r in rows]
    wps = [len(r["ref"].split()) / max(1e-9, r["duration"]) for r in rows]
    too_small = [r["utt_id"] for r, q in zip(rows, ratios) if q < MIN_KEEP - 1e-6]
    too_fast = [r["utt_id"] for r, w in zip(rows, wps) if w > 8.0]
    gate_ok = not too_small and not too_fast
    print(f"\n=== refinement acceptance gate ===")
    print(f"[{'OK' if not too_small else 'FAIL'}] utterances below "
          f"{MIN_KEEP:.0%} of nominal duration: {len(too_small)}")
    print(f"[{'OK' if not too_fast else 'FAIL'}] utterances implying >8 reference "
          f"words/second: {len(too_fast)}"
          + (f" e.g. {too_fast[:5]}" if too_fast else ""))
    if not gate_ok:
        print("refusing to overwrite the manifest — fix the optimiser first")
        raise SystemExit(1)

    write_jsonl(manifest, rows)
    sh = np.array(shifts)
    stats = {
        "radius_s": a.radius, "lambda_db_per_s": a.lam,
        "n_utts": len(rows), "n_boundaries_refined": int(len(sh)),
        "total_hours": round(sum(r["duration"] for r in rows) / 3600, 3),
        "boundary_shift_mean": float(sh.mean()), "boundary_shift_abs_mean":
            float(np.abs(sh).mean()), "boundary_shift_median": float(np.median(sh)),
        "boundary_shift_p10": float(np.percentile(sh, 10)),
        "boundary_shift_p90": float(np.percentile(sh, 90)),
        "boundary_shift_unmoved_pct": float(100 * (np.abs(sh) < 0.02).mean()),
        "min_duration_ratio": float(min(ratios)),
        "min_duration_s": float(min(r["duration"] for r in rows)),
        "max_ref_words_per_second": float(max(wps)),
        "acceptance_gate_passed": gate_ok,
        "constraints": {"min_utt_s": MIN_UTT, "min_share_of_nominal": MIN_KEEP},
        "method": ("internal boundaries snapped to the quietest instant within the "
                   "search radius, with a dB-per-second displacement penalty; "
                   "transcript-to-utterance assignment unchanged"),
    }
    write_json(ROOT / "report" / "segment_refinement.json", stats)
    print(f"\n{len(rows)} utterances, {stats['total_hours']} h -> {manifest}")
    print(f"boundary shift: mean {sh.mean():+.2f}s, |mean| {np.abs(sh).mean():.2f}s, "
          f"p10 {np.percentile(sh,10):+.2f}s, p90 {np.percentile(sh,90):+.2f}s, "
          f"unmoved {stats['boundary_shift_unmoved_pct']:.1f}%")


if __name__ == "__main__":
    main()
