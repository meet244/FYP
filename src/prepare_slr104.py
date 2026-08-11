"""Build data/manifests/test.jsonl from the SLR104 Kaldi-style test directory.

Cuts per-utterance WAVs once (grouped by recording, so each source file is decoded
a single time) so every later experiment reads byte-identical audio.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def read_kaldi_map(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def read_segments(path):
    segs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            utt, rec, start, end = line.strip().split()
            segs[utt] = (rec, float(start), float(end))
    return segs


def load_recording(src: str):
    """Return (mono float32 samples, sr). Handles plain paths and Kaldi pipe entries."""
    if src.strip().endswith("|"):
        cmd = src.strip()[:-1]
        raw = subprocess.run(cmd, shell=True, check=True, capture_output=True).stdout
        import io

        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    else:
        p = Path(src)
        if p.suffix.lower() != ".wav":
            out = subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(p),
                 "-ac", "1", "-ar", str(TARGET_SR), "-f", "wav", "pipe:1"],
                check=True, capture_output=True).stdout
            import io

            data, sr = sf.read(io.BytesIO(out), dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return data[:, 0], sr


def resample_if_needed(x, sr):
    if sr == TARGET_SR:
        return x, sr
    import librosa

    return librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR), TARGET_SR


def main(kaldi_dir: Path, audio_out: Path, manifest: Path):
    audio_out.mkdir(parents=True, exist_ok=True)
    text = read_kaldi_map(kaldi_dir / "text")
    wavscp = read_kaldi_map(kaldi_dir / "wav.scp")
    seg_path = kaldi_dir / "segments"
    segments = read_segments(seg_path) if seg_path.exists() else None
    print(f"{len(text)} text entries, {len(wavscp)} wav.scp entries, "
          f"segments={'yes' if segments else 'no'}")

    # Resolve relative paths in wav.scp against the Kaldi dir and its parents.
    def resolve(src: str) -> str:
        if src.strip().endswith("|") or Path(src).is_absolute():
            return src
        for base in (kaldi_dir, kaldi_dir.parent, kaldi_dir.parent.parent, Path.cwd()):
            cand = base / src
            if cand.exists():
                return str(cand)
        return src

    by_rec = defaultdict(list)
    if segments:
        for utt in text:
            if utt not in segments:
                print(f"WARN: no segment for {utt}", file=sys.stderr)
                continue
            rec, start, end = segments[utt]
            by_rec[rec].append((utt, start, end))
    else:
        for utt in text:
            by_rec[utt].append((utt, None, None))

    rows, missing = [], 0
    recs = sorted(by_rec)
    for n, rec in enumerate(recs, 1):
        items = by_rec[rec]
        todo = [it for it in items if not (audio_out / f"{it[0]}.wav").exists()]
        if todo:
            if rec not in wavscp:
                print(f"WARN: recording {rec} missing from wav.scp", file=sys.stderr)
                missing += len(items)
                continue
            x, sr = load_recording(resolve(wavscp[rec]))
            x, sr = resample_if_needed(x, sr)
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
        if n % 25 == 0 or n == len(recs):
            print(f"  cut {n}/{len(recs)} recordings", flush=True)

        for utt, start, end in items:
            wav = audio_out / f"{utt}.wav"
            if not wav.exists():
                continue
            dur = round(end - start, 3) if start is not None else round(
                sf.info(str(wav)).duration, 3)
            rows.append({"utt_id": utt, "audio": str(wav), "ref": text[utt],
                         "duration": dur, "rec": rec})

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["utt_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(r["duration"] or 0 for r in rows)
    print(f"{len(rows)} utterances, {total/3600:.2f} h -> {manifest}"
          + (f" ({missing} skipped)" if missing else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kaldi-dir", type=Path, required=True)
    p.add_argument("--audio-out", type=Path, default=Path("data/audio/test"))
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/test.jsonl"))
    a = p.parse_args()
    main(a.kaldi_dir, a.audio_out, a.manifest)
