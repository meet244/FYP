"""Cached decoding (§4.4).

Every decode result is written to cache/asr/<backend>/<config-hash>/<utt_id>.json.
This is not an optimisation, it is what makes the study feasible:

  * an interrupted run resumes instead of restarting;
  * re-computing metrics on a finished condition costs seconds, not hours;
  * every output-level method (§7.3) and every combination operates on cached text at
    zero additional decode cost;
  * confidence gating (§7.4) re-uses the decodes it needs and pays only for the
    flagged utterances it has not already seen;
  * results stay reproducible after the fact without re-running the model.

The cached payload keeps the full segment list, per-word probabilities, avg_logprob and
no_speech_prob, because §7.4 needs them and re-decoding to recover them would cost
another full pass.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from backends import DecodeConfig, ModelSpec
from common import ROOT, write_jsonl

CACHE = ROOT / "cache" / "asr"


def cache_path(backend_name: str, cfg_key: str, utt_id: str) -> Path:
    d = CACHE / backend_name / cfg_key
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{utt_id}.json"


def cached_result(backend_name: str, cfg_key: str, utt_id: str) -> dict | None:
    p = cache_path(backend_name, cfg_key, utt_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        p.unlink(missing_ok=True)   # truncated by an interrupted write; re-decode
        return None


def confidence_summary(res: dict) -> dict:
    """Utterance-level confidence signals distilled from the cached segments (§7.4)."""
    segs = res.get("segments") or []
    words = [w for s in segs for w in (s.get("words") or [])
             if w.get("prob") is not None]
    probs = [w["prob"] for w in words]
    dur = sum((s.get("end") or 0) - (s.get("start") or 0) for s in segs)
    return {
        "mean_word_prob": sum(probs) / len(probs) if probs else None,
        "min_word_prob": min(probs) if probs else None,
        "n_words": len(words),
        "avg_logprob": (sum(s.get("avg_logprob") or 0.0 for s in segs) / len(segs)
                        if segs else None),
        "no_speech_prob": (max((s.get("no_speech_prob") or 0.0) for s in segs)
                           if segs else None),
        "seg_duration": dur,
    }


def decode_rows(backend, rows: list[dict], cfg_fn: Callable[[dict], DecodeConfig],
                desc: str = "decode", progress: bool = True) -> list[dict]:
    """Decode (or read from cache) every row. `cfg_fn(row) -> DecodeConfig`.

    Returns hypothesis records. Never calls the model twice for the same
    (utterance, model, decode-config) triple.
    """
    out: list[dict] = []
    n_new = 0
    t0 = time.time()
    audio_seconds_decoded = 0.0
    it: Iterable[dict] = tqdm(rows, desc=desc, unit="utt") if progress else rows
    for row in it:
        cfg = cfg_fn(row)
        key = cfg.key(backend.spec)
        res = cached_result(backend.name, key, row["utt_id"])
        if res is None:
            res = backend.transcribe(row["audio"], cfg)
            res["_cfg"] = cfg.describe()
            res["_model"] = backend.name
            cache_path(backend.name, key, row["utt_id"]).write_text(
                json.dumps(res, ensure_ascii=False), encoding="utf-8")
            n_new += 1
            audio_seconds_decoded += row.get("duration") or 0.0
        out.append({
            "utt_id": row["utt_id"],
            "rec": row.get("rec"),
            "duration": row.get("duration"),
            "ref": row["ref"],
            "hyp": res["text"],
            "language": res.get("language"),
            "language_prob": res.get("language_prob"),
            "context": cfg.context,
            "hotwords": cfg.hotwords,
            "cfg_key": key,
            "conf": confidence_summary(res),
        })
    elapsed = time.time() - t0
    rtf = (elapsed / audio_seconds_decoded) if audio_seconds_decoded else None
    print(f"  {len(out)} utts, {n_new} newly decoded in {elapsed/60:.1f} min"
          + (f" (RTF={rtf:.2f})" if rtf else " (all cached)"), flush=True)
    return out


def decode_manifest(backend, manifest_rows: list[dict], cfg_fn, out_path: Path,
                    desc: str = "decode") -> list[dict]:
    rows = decode_rows(backend, manifest_rows, cfg_fn, desc=desc)
    write_jsonl(out_path, rows)
    return rows


def load_pass1(path: str | Path) -> dict[str, dict]:
    """Index a pass-1 hypothesis file by utterance id (used by retrieval and gating)."""
    from common import read_jsonl
    return {r["utt_id"]: r for r in read_jsonl(path)}


def warm_cache_stats() -> dict:
    """How many decodes are cached, per (backend, config). Useful for §9.2 accounting."""
    stats = {}
    if not CACHE.exists():
        return stats
    for backend_dir in sorted(CACHE.iterdir()):
        for cfg_dir in sorted(backend_dir.iterdir()):
            stats[f"{backend_dir.name}/{cfg_dir.name}"] = len(
                list(cfg_dir.glob("*.json")))
    return stats


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(warm_cache_stats(), indent=2))
