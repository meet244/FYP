"""Cross-model ASR benchmark on the frozen SLR104 manifest (zero-prompt).

One process per model — Apple Silicon unified memory is the binding constraint,
so weights are never co-resident. Hypotheses land in the same cache layout the
SGCD study uses (`out/hyps/<model>__C0__<split>.jsonl`), so `score.py`,
`stats.py` and `make_tables.py` all work on them unchanged.

    python bench_asr.py --model wlv3   --split test
    python bench_asr.py --model qwen17 --split test --limit 10
"""
import argparse
import json
import os
import pathlib
import resource
import sys
import time

import numpy as np
import soundfile as sf

from config import HYP, OUT
from decode import load_manifest, load_audio

SEGWAV = OUT / "segwav"
SEGWAV.mkdir(parents=True, exist_ok=True)

# backend, hf repo, note
BENCH = {
    "wlv3":     ("mlx_whisper", "mlx-community/whisper-large-v3-mlx",      "Whisper large-v3 (fp16 MLX)"),
    "turbo":    ("mlx_whisper", "mlx-community/whisper-large-v3-turbo",    "Whisper large-v3-turbo (MLX)"),
    "qwen17":   ("qwen",        "Qwen/Qwen3-ASR-1.7B-hf",                  "Qwen3-ASR-1.7B (torch/MPS)"),
    "qwen06":   ("qwen",        "Qwen/Qwen3-ASR-0.6B-hf",                  "Qwen3-ASR-0.6B (torch/MPS)"),
    "parakeet": ("parakeet",    "mlx-community/parakeet-rnnt-1.1b",        "NVIDIA Parakeet RNNT 1.1B (MLX)"),
}

WHISPER_DECODE = dict(  # identical to the SGCD study's frozen C0 settings
    language="hi", task="transcribe", temperature=0.0,
    condition_on_previous_text=False, word_timestamps=False,
)


def seg_path(row):
    """Materialise the utterance as its own 16 kHz wav — parakeet-mlx and the
    Qwen processor both want a file/array, and cutting once keeps every model on
    byte-identical audio."""
    p = SEGWAV / f"{row['utt_id']}.wav"
    if not p.exists():
        sf.write(p, load_audio(row), 16000)
    return p


# ---------------------------------------------------------------- backends ---
def make_whisper(repo):
    import mlx_whisper

    def fn(row):
        o = mlx_whisper.transcribe(load_audio(row), path_or_hf_repo=repo, **WHISPER_DECODE)
        segs = o.get("segments") or []
        lp = [s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
        cr = [s["compression_ratio"] for s in segs if s.get("compression_ratio") is not None]
        return o["text"].strip(), (float(np.mean(lp)) if lp else None), (float(max(cr)) if cr else None)

    return fn


def make_qwen(repo, language):
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(repo)
    model = AutoModelForMultimodalLM.from_pretrained(repo, dtype=torch.float16).to(dev).eval()
    print(f"[load] {repo} on {model.device} dtype={model.dtype}", flush=True)

    def fn(row):
        kw = {"language": language} if language else {}
        # feature extractor is fixed at 16 kHz, which is the corpus rate — passing
        # sampling_rate here would land in **kwargs and be rejected by the processor
        inputs = proc.apply_transcription_request(
            audio=load_audio(row), **kw
        ).to(model.device, model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return proc.decode(gen, return_format="transcription_only")[0].strip(), None, None

    return fn


def make_parakeet(repo):
    from parakeet_mlx import from_pretrained

    model = from_pretrained(repo)

    def fn(row):
        r = model.transcribe(str(seg_path(row)))
        return (r.text or "").strip(), None, None

    return fn


def build(key, language):
    backend, repo, _ = BENCH[key]
    if backend == "mlx_whisper":
        return make_whisper(repo)
    if backend == "qwen":
        return make_qwen(repo, language)
    return make_parakeet(repo)


# -------------------------------------------------------------------- run ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(BENCH))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--language", default="Hindi", help='Qwen language hint; "none" = auto-detect')
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--suffix", default="", help="tag suffix, e.g. 'auto' for a language-hint variant")
    a = ap.parse_args()

    rows = load_manifest(a.split)
    if a.limit:
        rows = rows[: a.limit]
    # a capped run is a smoke test: tag it apart so it can never masquerade as the real run
    tag = f"{a.model}__C0__{a.split}{a.suffix}" + (f"smoke{a.limit}" if a.limit else "")
    out_p = HYP / f"{tag}.jsonl"
    part_p = HYP / f".{tag}.partial.jsonl"

    if out_p.exists() and not a.force:
        print(f"[cached] {tag}")
        return

    # resume: an interrupted run keeps every utterance it already paid for
    done = {}
    if part_p.exists() and not a.force:
        for l in part_p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                done[d["utt_id"]] = d
        print(f"[resume] {len(done)} utterances already decoded")
    elif a.force:
        part_p.unlink(missing_ok=True)

    lang = None if a.language.lower() in ("none", "auto", "") else a.language
    transcribe = build(a.model, lang)

    res, t0, audio_s, dec_s = [], time.time(), 0.0, 0.0
    with part_p.open("a", encoding="utf-8") as pf:
        for i, r in enumerate(rows, 1):
            audio_s += r["dur"]
            if r["utt_id"] in done:
                res.append(done[r["utt_id"]])
                dec_s += done[r["utt_id"]].get("dec_s", 0.0)
                continue
            t1 = time.time()
            hyp, lp, cr = transcribe(r)
            dt = time.time() - t1
            dec_s += dt
            rec = dict(
                utt_id=r["utt_id"], lecture_id=r["lecture_id"], course_id=None,
                ref=r["ref"], hyp=hyp, avg_logprob=lp, compression_ratio=cr,
                prompt_tokens=0, retrieved=None, prompt_course=None,
                dur=r["dur"], dec_s=round(dt, 3),
            )
            res.append(rec)
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.flush()
            if i % 10 == 0:
                print(f"    {tag}: {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    with out_p.open("w", encoding="utf-8") as f:
        for x in res:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    part_p.unlink(missing_ok=True)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes
    meta = dict(
        model=a.model, repo=BENCH[a.model][1], backend=BENCH[a.model][0], note=BENCH[a.model][2],
        split=a.split, n=len(res), audio_s=round(audio_s, 1),
        decode_s=round(dec_s, 1), wall_s=round(time.time() - t0, 1),
        rtf=round(dec_s / audio_s, 4) if audio_s else None,
        xrealtime=round(audio_s / dec_s, 2) if dec_s else None,
        peak_rss_gb=round(peak_gb, 2), language=lang,
    )
    (OUT / f"bench_runtime__{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {tag}  {meta['decode_s']}s decode over {meta['audio_s']}s audio  "
          f"RTF={meta['rtf']}  ({meta['xrealtime']}x realtime)  peakRSS={meta['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
