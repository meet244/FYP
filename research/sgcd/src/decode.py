"""Decode every condition, cache every hypothesis. Re-runs are near-free.

Usage:
    python src/decode.py --model turbo --split test
    python src/decode.py --model small --split dev --conditions C0 C4 --k 3
"""
import argparse
import json
import pathlib
import sys
import time

import mlx_whisper
import numpy as np
import soundfile as sf

from config import HYP, MODELS, N_DEV, N_TEST, OUT, ORACLE_CONDITIONS, CONDITIONS
from courses import course_of, get_course
from prompts import build, n_tokens, MAX_PROMPT_TOKENS
from retrieve import get_index

DECODE = dict(
    language="hi",                     # DEV-tunable: "hi" vs None
    task="transcribe",
    temperature=0.0,                   # scalar -> disables fallback -> deterministic + fast
    condition_on_previous_text=False,  # prompt is the ONLY context: no cross-utterance drift
    word_timestamps=False,
)

# Guard thresholds — FIT ON DEV, then frozen here. See RUNLOG.md.
GUARD = dict(d_logprob=0.25, max_cr=2.0, len_ratio=1.5)  # FROZEN from DEV sweep 2026-08-14

LIMIT = 0  # debug only: cap utterances per condition


def load_manifest(split):
    paths = [OUT / "manifest.jsonl", OUT / "manifest_concat.jsonl"]
    if not paths[0].exists():
        sys.exit(f"{paths[0]} missing — run build_manifest.py first")
    rows = [json.loads(l) for p in paths if p.exists()
            for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split and r.get("in_eval", True)]
    return rows[:LIMIT] if LIMIT else rows


def load_audio(row):
    info = sf.info(row["wav"])
    sr = info.samplerate
    assert sr == 16000, f"expected 16 kHz, got {sr} for {row['wav']}"  # corpus is 16 kHz
    a, _ = sf.read(
        row["wav"], start=int(row["start"] * sr), stop=int(row["end"] * sr), dtype="float32"
    )
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a


def _seg_stats(o):
    """Mean avg_logprob / max compression_ratio over the produced segments."""
    segs = o.get("segments") or []
    lp = [s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
    cr = [s["compression_ratio"] for s in segs if s.get("compression_ratio") is not None]
    return (float(np.mean(lp)) if lp else None), (float(max(cr)) if cr else None)


def run(model_key, condition, split="test", first_pass=None, k=3, max_tokens=None, force=False,
        tag_suffix=""):
    """`split` selects manifest rows; `tag_suffix` only names the cache file, so a
    sweep can decode the same DEV rows under several configs without collisions."""
    tag = f"{model_key}__{condition}__{split}{tag_suffix}"
    path = HYP / f"{tag}.jsonl"
    if path.exists() and not force:
        print(f"[cached] {tag}")
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = load_manifest(split)
    if not rows:
        sys.exit(f"no manifest rows for split={split}")
    if condition in ("C4", "C5", "C7") and first_pass is None:
        raise ValueError(f"{condition} needs first_pass hypotheses (run C0 first)")

    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        cid = course_of(r["lecture_id"])
        course = get_course(cid, condition)
        units, unit_ids = None, None
        if condition in ("C4", "C5", "C6", "C7"):
            idx = get_index(course, k=k)
            # Leakage guard is STRUCTURAL, on where the query comes from — not on
            # string equality, because a first-pass hypothesis that happens to be
            # perfect equals the reference without any leakage having occurred.
            if condition in ORACLE_CONDITIONS:
                q = r["ref"]  # topline: oracle retrieval, clearly labelled
            else:
                assert condition not in ORACLE_CONDITIONS
                q = first_pass.get(r["utt_id"], "")
            units = idx.query(q)
            unit_ids = [u["unit_id"] for u in units]
        prompt = build("C4" if condition == "C7" else condition, course, units, max_tokens)

        o = mlx_whisper.transcribe(
            load_audio(r), path_or_hf_repo=MODELS[model_key], initial_prompt=prompt, **DECODE
        )
        lp, cr = _seg_stats(o)
        res.append(
            dict(
                utt_id=r["utt_id"],
                lecture_id=r["lecture_id"],
                course_id=cid,
                ref=r["ref"],
                hyp=o["text"].strip(),
                avg_logprob=lp,
                compression_ratio=cr,
                prompt_tokens=n_tokens(prompt) if prompt else 0,
                retrieved=unit_ids,
                prompt_course=course["course_id"],
            )
        )
        if i % 25 == 0:
            print(f"    {tag}: {i}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    el = time.time() - t0
    audio_s = sum(r["dur"] for r in rows)
    with path.open("w", encoding="utf-8") as f:
        for x in res:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[done] {tag}  {el:.1f}s  RTF={el/audio_s:.3f}  ({audio_s/el:.1f}x realtime)")
    return res


def apply_guard(base, prompted, d_logprob=None, max_cr=None, len_ratio=None):
    """Fall back to the unprompted hypothesis when prompting destabilised decoding."""
    d_logprob = GUARD["d_logprob"] if d_logprob is None else d_logprob
    max_cr = GUARD["max_cr"] if max_cr is None else max_cr
    len_ratio = GUARD["len_ratio"] if len_ratio is None else len_ratio

    b = {x["utt_id"]: x for x in base}
    out, n_fb = [], 0
    for p in prompted:
        bb = b.get(p["utt_id"])
        bad = False
        if bb and p["avg_logprob"] is not None and bb["avg_logprob"] is not None:
            bad |= p["avg_logprob"] < bb["avg_logprob"] - d_logprob
        if p["compression_ratio"] is not None:
            bad |= p["compression_ratio"] > max_cr
        if bb and len(bb["hyp"].split()):
            bad |= len(p["hyp"].split()) > len_ratio * len(bb["hyp"].split())
        if bad and bb:
            out.append({**p, "hyp": bb["hyp"], "fallback": True})
            n_fb += 1
        else:
            out.append({**p, "fallback": False})
    rate = 100 * n_fb / len(prompted) if prompted else 0.0
    print(f"[guard] fell back on {n_fb}/{len(prompted)} ({rate:.1f}%)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo", choices=list(MODELS))
    ap.add_argument("--split", default="test", choices=["dev", "test", "devcat", "testcat"])
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--k", type=int, default=3, help="retrieved syllabus units (FROZEN: 3)")
    ap.add_argument("--max-tokens", type=int, default=MAX_PROMPT_TOKENS)
    ap.add_argument("--language", default="hi", help='"hi" or "none"')
    ap.add_argument("--limit", type=int, default=0, help="debug: cap utterances")
    ap.add_argument("--force", action="store_true", help="ignore cached hypotheses")
    ap.add_argument("--suffix", default="", help="tag suffix for sweep runs")
    a = ap.parse_args()

    global LIMIT
    LIMIT = a.limit
    DECODE["language"] = None if a.language.lower() in ("none", "auto", "") else a.language

    conds = list(a.conditions)
    if "C0" not in conds and any(c in conds for c in ("C4", "C5", "C7")):
        conds = ["C0"] + conds

    fp = None
    results = {}
    for c in conds:
        if c == "C7":
            continue
        results[c] = run(a.model, c, a.split, first_pass=fp, k=a.k, max_tokens=a.max_tokens,
                         force=a.force, tag_suffix=a.suffix)
        if c == "C0":
            fp = {x["utt_id"]: x["hyp"] for x in results["C0"]}

    if "C7" in conds:
        c4 = results.get("C4") or run(a.model, "C4", a.split, first_pass=fp, k=a.k,
                                      max_tokens=a.max_tokens, tag_suffix=a.suffix)
        guarded = apply_guard(results["C0"], c4)
        tag = f"{a.model}__C7__{a.split}{a.suffix}"
        with (HYP / f"{tag}.jsonl").open("w", encoding="utf-8") as f:
            for x in guarded:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"[done] {tag} (derived from C4, no extra decode)")


if __name__ == "__main__":
    main()
