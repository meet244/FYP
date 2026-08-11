"""Cached transcription. Never call a model twice for the same (utterance, config)."""
import json
import time
from pathlib import Path

from tqdm import tqdm

from backends import DecodeConfig

CACHE = Path("cache/asr")


def _cache_path(backend_name, utt_id, cfg: DecodeConfig) -> Path:
    d = CACHE / backend_name / cfg.key()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{utt_id}.json"


def transcribe_manifest(backend, manifest_path, cfg_fn, out_path, limit=None):
    """cfg_fn(row) -> DecodeConfig, so per-utterance prompts are supported."""
    rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    out, n_new, t0 = [], 0, time.time()
    for row in tqdm(rows, desc=f"{backend.name} -> {Path(out_path).parent.name}"):
        cfg = cfg_fn(row)
        cp = _cache_path(backend.name, row["utt_id"], cfg)
        if cp.exists():
            res = json.loads(cp.read_text(encoding="utf-8"))
        else:
            res = backend.transcribe(row["audio"], cfg)
            res["_cfg"] = cfg.__dict__
            cp.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            n_new += 1
        out.append({"utt_id": row["utt_id"], "ref": row["ref"], "hyp": res["text"],
                    "duration": row.get("duration"),
                    "prompt": cfg.prompt, "language": res.get("language")})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    audio_s = sum(r.get("duration") or 0 for r in rows)
    el = time.time() - t0
    print(f"{len(out)} utts ({n_new} newly decoded) in {el/60:.1f} min"
          + (f", RTF={el/audio_s:.2f}" if n_new and audio_s else "")
          + f" -> {out_path}")
    return out
