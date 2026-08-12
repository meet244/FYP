"""Throughput benchmark for the decode configuration (§9.3).

§9.3 asks for batched inference to be benchmarked against sequential decoding on a small
sample before committing to the long runs, with the worker thread count set to the
available core count.

Two findings this script establishes rather than assumes:

1. **faster-whisper's batched pipeline does not help this corpus.**
   `BatchedInferencePipeline` batches the VAD-derived chunks *within a single audio
   file*. Our audio is already sentence-segmented into short utterances and §4.1 fixes
   `vad_filter=False`, so each file yields one chunk and there is nothing to batch. The
   benchmark measures it anyway so the claim is evidence-based.

2. **Concurrency across utterances is where the throughput is.** CTranslate2 exposes
   intra-op threading (`cpu_threads`) and inter-op workers (`num_workers`). One decode
   using all cores scales poorly; several concurrent decodes with fewer threads each
   usually beat it. `cpu_threads` and `num_workers` are deliberately excluded from the
   ASR cache key on the grounds that they change speed and not output — this benchmark
   verifies that by comparing the hypotheses each setting produces, and reports any
   mismatch as a warning, because if it were false the cache would be unsound.

Results go to report/inference_benchmark.json. Nothing here changes the frozen decode
configuration; it informs `model.cpu_threads` / `model.num_workers`, which are the only
two settings that may be tuned for speed.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backends import DecodeConfig, LocalWhisper, ModelSpec, decode_config_from_config
from common import ROOT, load_config, read_jsonl, write_json


def _decode_all(model, rows, cfg_obj: DecodeConfig, concurrency: int) -> list[str]:
    def one(row):
        segments, _ = model.model.transcribe(
            row["audio"], language=cfg_obj.language, beam_size=cfg_obj.beam_size,
            temperature=cfg_obj.temperature, condition_on_previous_text=False,
            vad_filter=False, word_timestamps=cfg_obj.word_timestamps)
        return "".join(s.text for s in segments).strip()

    if concurrency <= 1:
        return [one(r) for r in rows]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(one, rows))


def _batched(model, rows, cfg_obj: DecodeConfig) -> tuple[list[str], str | None]:
    try:
        from faster_whisper import BatchedInferencePipeline
    except ImportError:
        return [], "BatchedInferencePipeline unavailable in this faster-whisper build"
    pipe = BatchedInferencePipeline(model=model.model)
    out = []
    for row in rows:
        segments, _ = pipe.transcribe(
            row["audio"], language=cfg_obj.language, batch_size=8,
            beam_size=cfg_obj.beam_size, temperature=cfg_obj.temperature)
        out.append("".join(s.text for s in segments).strip())
    return out, None


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--configs", default="1x0,2x4,4x2",
                    help="comma-separated concurrency x cpu_threads pairs")
    ap.add_argument("--skip-batched", action="store_true")
    a = ap.parse_args()

    rows = read_jsonl(ROOT / cfg["data"]["tiers"][a.tier])[:a.n]
    audio_s = sum(r["duration"] for r in rows)
    dc = decode_config_from_config(cfg)
    print(f"benchmark on {len(rows)} utts, {audio_s/60:.1f} min audio, "
          f"model={cfg['model']['size']} {cfg['model']['compute_type']}\n")

    results, reference_hyps = [], None
    for spec_str in a.configs.split(","):
        conc, threads = (int(x) for x in spec_str.split("x"))
        spec = ModelSpec(size=cfg["model"]["size"],
                         compute_type=cfg["model"]["compute_type"],
                         device=cfg["model"]["device"],
                         cpu_threads=threads,
                         num_workers=max(1, conc))
        model = LocalWhisper(spec)
        t0 = time.time()
        hyps = _decode_all(model, rows, dc, conc)
        wall = time.time() - t0
        if reference_hyps is None:
            reference_hyps = hyps
            identical = True
        else:
            identical = hyps == reference_hyps
        entry = {"mode": "sequential" if conc == 1 else f"concurrent x{conc}",
                 "concurrency": conc, "cpu_threads": threads,
                 "wall_clock_s": round(wall, 1),
                 "rtf": round(wall / audio_s, 3),
                 "utts_per_min": round(60 * len(rows) / wall, 2),
                 "hypotheses_identical_to_first": identical}
        results.append(entry)
        print(f"concurrency={conc} cpu_threads={threads}: {wall:.1f}s "
              f"RTF={entry['rtf']} ({entry['utts_per_min']} utt/min)"
              + ("" if identical else "  ** OUTPUT DIFFERS **"), flush=True)
        del model

    if not a.skip_batched:
        spec = ModelSpec(size=cfg["model"]["size"],
                         compute_type=cfg["model"]["compute_type"],
                         device=cfg["model"]["device"])
        model = LocalWhisper(spec)
        t0 = time.time()
        hyps, err = _batched(model, rows, dc)
        wall = time.time() - t0
        entry = {"mode": "batched pipeline (batch_size=8)", "error": err,
                 "wall_clock_s": round(wall, 1) if not err else None,
                 "rtf": round(wall / audio_s, 3) if not err else None,
                 "utts_per_min": round(60 * len(rows) / wall, 2) if not err else None,
                 "hypotheses_identical_to_first": (hyps == reference_hyps) if hyps
                 else None,
                 "note": ("batches VAD chunks within one file; this corpus is already "
                          "sentence-segmented and vad_filter is off (§4.1), so there is "
                          "nothing to batch")}
        results.append(entry)
        print(f"batched pipeline: " + (err or f"{wall:.1f}s RTF={entry['rtf']} "
                                             f"({entry['utts_per_min']} utt/min)"),
              flush=True)

    timed = [r for r in results if r.get("wall_clock_s")]
    best = min(timed, key=lambda r: r["wall_clock_s"]) if timed else None
    mismatches = [r for r in results if r.get("hypotheses_identical_to_first") is False]
    out = {
        "n_utts": len(rows), "audio_minutes": round(audio_s / 60, 2),
        "model": cfg["model"]["size"], "compute_type": cfg["model"]["compute_type"],
        "results": results,
        "fastest": best,
        "speedup_over_sequential": (
            round(timed[0]["wall_clock_s"] / best["wall_clock_s"], 2)
            if best and timed else None),
        "cache_key_assumption_holds": not mismatches,
        "cache_key_note": (
            "cpu_threads and num_workers are excluded from the ASR cache key on the "
            "grounds that they change speed, not output. "
            + ("Verified: every setting produced identical hypotheses."
               if not mismatches else
               "VIOLATED: settings produced different hypotheses — they must be added "
               "to the cache key before mixing them.")),
    }
    write_json(ROOT / "report" / "inference_benchmark.json", out)
    print(f"\nfastest: {best['mode'] if best else '—'} "
          f"(speedup {out['speedup_over_sequential']}x over sequential)")
    print(out["cache_key_note"])


if __name__ == "__main__":
    main()
