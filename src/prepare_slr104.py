"""Corpus preparation (§3.4).

Reads the Kaldi-style SLR104 Hindi-English test directory (`text`, `wav.scp`,
`segments`, `utt2spk`), cuts each utterance out of the long tutorial recording **once**
to mono 16 kHz, and writes a flat manifest with one record per utterance:
{utt_id, audio, ref, duration, rec, spk}.

Cutting once and reusing the files guarantees every experimental condition sees
byte-identical audio. Cutting is grouped by recording so each source WAV is read a
single time rather than once per utterance.

Acceptance criteria, all checked by `--verify` (§3.4):
  * utterance count and total duration match the published figures for the test set;
  * a random sample of ten cut files is readable and matches its reference;
  * no utterance has zero duration.
"""
from __future__ import annotations

import argparse
import io
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from common import ROOT, load_config, read_jsonl, write_json, write_jsonl

TARGET_SR = 16000
# Published figures for the SLR104 Hindi-English test portion (OpenSLR record).
PUBLISHED_HOURS = 5.18
PUBLISHED_UTTS = 3136


def read_kaldi_map(path: Path) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def read_segments(path: Path) -> dict[str, tuple[str, float, float]]:
    segs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            utt, rec, start, end = line.strip().split()
            segs[utt] = (rec, float(start), float(end))
    return segs


def _resolve(src: str, kaldi_dir: Path) -> str:
    """wav.scp paths are relative to the distribution root, not to `transcripts/`."""
    if src.strip().endswith("|") or Path(src).is_absolute():
        return src
    for base in (kaldi_dir, kaldi_dir.parent, kaldi_dir.parent.parent, Path.cwd()):
        cand = base / src
        if cand.exists():
            return str(cand)
    return src


def load_recording(src: str) -> tuple[np.ndarray, int]:
    """Return (mono float32 samples, sr). Handles plain paths and Kaldi pipe entries."""
    if src.strip().endswith("|"):
        raw = subprocess.run(src.strip()[:-1], shell=True, check=True,
                             capture_output=True).stdout
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    elif Path(src).suffix.lower() != ".wav":
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", src,
             "-ac", "1", "-ar", str(TARGET_SR), "-f", "wav", "pipe:1"],
            check=True, capture_output=True).stdout
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    else:
        data, sr = sf.read(src, dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    x = data[:, 0]
    if sr != TARGET_SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    return x, sr


def prepare(kaldi_dir: Path, audio_out: Path, manifest: Path) -> list[dict]:
    audio_out.mkdir(parents=True, exist_ok=True)
    text = read_kaldi_map(kaldi_dir / "text")
    wavscp = read_kaldi_map(kaldi_dir / "wav.scp")
    utt2spk = (read_kaldi_map(kaldi_dir / "utt2spk")
               if (kaldi_dir / "utt2spk").exists() else {})
    seg_path = kaldi_dir / "segments"
    segments = read_segments(seg_path) if seg_path.exists() else None
    print(f"{len(text)} text entries, {len(wavscp)} recordings, "
          f"segments={'yes' if segments else 'no'}")

    by_rec: dict[str, list] = defaultdict(list)
    for utt in text:
        if segments:
            if utt not in segments:
                print(f"WARN: no segment for {utt}", file=sys.stderr)
                continue
            rec, start, end = segments[utt]
            by_rec[rec].append((utt, start, end))
        else:
            by_rec[utt].append((utt, None, None))

    rows, skipped = [], 0
    recs = sorted(by_rec)
    for n, rec in enumerate(recs, 1):
        items = by_rec[rec]
        todo = [it for it in items if not (audio_out / f"{it[0]}.wav").exists()]
        if todo:
            if rec not in wavscp:
                print(f"WARN: recording {rec} missing from wav.scp", file=sys.stderr)
                skipped += len(items)
                continue
            x, sr = load_recording(_resolve(wavscp[rec], kaldi_dir))
            for utt, start, end in todo:
                if start is None:
                    seg = x
                else:
                    a, b = int(round(start * sr)), int(round(end * sr))
                    seg = x[max(0, a):min(len(x), b)]
                if len(seg) == 0:
                    print(f"WARN: empty segment {utt}", file=sys.stderr)
                    continue
                sf.write(audio_out / f"{utt}.wav", seg.astype(np.float32), sr,
                         subtype="PCM_16")
            print(f"  cut {n}/{len(recs)} recordings", flush=True)

        for utt, start, end in items:
            wav = audio_out / f"{utt}.wav"
            if not wav.exists():
                continue
            dur = (round(end - start, 3) if start is not None
                   else round(sf.info(str(wav)).duration, 3))
            rows.append({"utt_id": utt, "audio": str(wav), "ref": text[utt],
                         "duration": dur, "rec": rec,
                         "spk": utt2spk.get(utt)})

    rows.sort(key=lambda r: r["utt_id"])
    write_jsonl(manifest, rows)
    total_h = sum(r["duration"] for r in rows) / 3600
    print(f"{len(rows)} utterances, {total_h:.2f} h -> {manifest}"
          + (f" ({skipped} skipped)" if skipped else ""))
    return rows


def verify(manifest: Path, n_sample: int = 10, seed: int = 1337) -> dict:
    rows = read_jsonl(manifest)
    durs = [r["duration"] for r in rows]
    total_h = sum(durs) / 3600
    zero = [r["utt_id"] for r in rows if not r["duration"] or r["duration"] <= 0]

    report = {
        "n_utts": len(rows),
        "published_n_utts": PUBLISHED_UTTS,
        "total_hours": round(total_h, 3),
        "published_hours": PUBLISHED_HOURS,
        "hours_within_1pct": abs(total_h - PUBLISHED_HOURS) / PUBLISHED_HOURS < 0.01,
        "n_recordings": len({r["rec"] for r in rows}),
        "n_speakers": len({r.get("spk") for r in rows if r.get("spk")}),
        "zero_duration_utts": zero,
        "duration_mean": round(sum(durs) / len(durs), 2),
        "duration_median": round(sorted(durs)[len(durs) // 2], 2),
        "duration_min": min(durs),
        "duration_max": max(durs),
    }

    print(f"\n=== acceptance criteria (§3.4) ===")
    ok_count = report["n_utts"] == PUBLISHED_UTTS
    print(f"[{'OK' if ok_count else 'FAIL'}] utterance count "
          f"{report['n_utts']} vs published {PUBLISHED_UTTS}")
    print(f"[{'OK' if report['hours_within_1pct'] else 'FAIL'}] total duration "
          f"{total_h:.2f} h vs published {PUBLISHED_HOURS} h")
    print(f"[{'OK' if not zero else 'FAIL'}] zero-duration utterances: {len(zero)}")

    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_sample, len(rows)))
    print(f"\n=== random sample of {len(sample)} cut files ===")
    sample_ok = True
    checks = []
    for r in sample:
        try:
            info = sf.info(r["audio"])
            match = (abs(info.duration - r["duration"]) < 0.15
                     and info.samplerate == TARGET_SR and info.channels == 1)
        except Exception as exc:                       # noqa: BLE001
            info, match = None, False
            print(f"  ERROR reading {r['audio']}: {exc}")
        sample_ok &= bool(match)
        checks.append({"utt_id": r["utt_id"], "ok": bool(match),
                       "duration": r["duration"],
                       "file_duration": round(info.duration, 3) if info else None,
                       "sr": info.samplerate if info else None})
        if info:
            print(f"  [{'OK' if match else 'FAIL'}] {r['utt_id']}  "
                  f"{info.duration:.2f}s @{info.samplerate}Hz x{info.channels}")
            print(f"        REF: {r['ref'][:100]}")
    report["sample_checks"] = checks
    report["sample_ok"] = sample_ok
    report["all_criteria_passed"] = bool(ok_count and report["hours_within_1pct"]
                                         and not zero and sample_ok)
    print(f"\n[{'PASS' if report['all_criteria_passed'] else 'FAIL'}] "
          f"stage-1 gate (§11): counts and duration match published figures")
    write_json(ROOT / "report" / "corpus_stats.json", report)
    return report


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaldi-dir", type=Path,
                    default=ROOT / cfg["data"]["kaldi_dir"])
    ap.add_argument("--audio-out", type=Path, default=ROOT / cfg["data"]["audio_dir"])
    ap.add_argument("--manifest", type=Path, default=ROOT / cfg["data"]["manifest"])
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        prepare(a.kaldi_dir, a.audio_out, a.manifest)
    verify(a.manifest)


if __name__ == "__main__":
    main()
