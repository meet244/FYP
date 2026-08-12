"""The two required pilots (§4.2, §4.3), run on Tier 1 only.

model    large-v3-turbo vs large-v3 under identical settings. The turbo variant is
         optimised for speed and is known to be weaker than large-v3 on non-English
         audio; since this study is entirely about non-English code-switched audio the
         choice must be justified empirically rather than assumed. Records WER and
         wall-clock time for each. §4.2 must not be skipped.

language forcing Hindi vs forcing English vs automatic detection. Language selection
         materially affects WER on code-switched audio; the setting is fixed for all
         later experiments and the choice is reported.

Both write a table to report/ and leave their decodes in the shared cache, so the
language pilot's `hi` arm and the model pilot's turbo arm are the same decode.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from backends import decode_config_from_config, get_backend, model_spec_from_config
from common import ROOT, load_config, manifest_for_tier, read_jsonl, write_json
from score import score_rows, summary_line
from lexicon import load_lexicon
from transcribe import decode_rows


def _run(backend, rows, cfg, lex, label, **overrides) -> dict:
    t0 = time.time()
    hyps = decode_rows(backend, rows,
                       lambda row: decode_config_from_config(cfg, **overrides),
                       desc=label)
    wall = time.time() - t0
    m, _ = score_rows(hyps, lex)
    audio_s = sum(r["duration"] for r in rows)
    m["_wall_clock_s"] = round(wall, 1)
    m["_audio_s"] = round(audio_s, 1)
    m["_rtf"] = round(wall / audio_s, 3) if audio_s else None
    print(f"[{label}] {summary_line(m)}  wall={wall/60:.1f} min "
          f"RTF={m['_rtf']}", flush=True)
    return m


def pilot_model(cfg, rows, lex) -> dict:
    """§4.2 — turbo vs large-v3 under identical settings."""
    results = {}
    for size in ("large-v3-turbo", "large-v3"):
        backend = get_backend("local", model_spec_from_config(cfg, override_size=size))
        results[size] = _run(backend, rows, cfg, lex, f"model={size}")
        del backend

    turbo, big = results["large-v3-turbo"], results["large-v3"]
    d_wer = turbo["wer"] - big["wer"]
    speedup = (big["_wall_clock_s"] / turbo["_wall_clock_s"]
               if turbo["_wall_clock_s"] else None)
    # §4.2's three outcomes, decided by a threshold stated in advance: 1 WER point.
    if d_wer <= 0.01:
        decision = "turbo"
        rationale = ("turbo is within 1 WER point of large-v3 and much faster; proceed "
                     "with turbo and report this pilot as the justification.")
    elif d_wer > 0.03:
        decision = "large-v3"
        rationale = ("turbo is substantially worse on this language pair; use large-v3 "
                     "as the primary system so the baseline is not knowingly weakened.")
    else:
        decision = "mixed"
        rationale = ("results are mixed; run the full matrix on turbo and confirm the "
                     "headline result on large-v3 as a robustness argument.")
    out = {
        "pilot": "model_selection", "tier": "tier1", "n_utts": len(rows),
        "results": {k: {"wer": v["wer"], "b_wer": v["b_wer"], "u_wer": v["u_wer"],
                        "cer": v["cer"], "wer_level2": v["wer_level2"],
                        "term_f1": v["term_f1"], "wall_clock_min":
                        round(v["_wall_clock_s"] / 60, 1), "rtf": v["_rtf"]}
                    for k, v in results.items()},
        "wer_delta_turbo_minus_largev3": d_wer,
        "turbo_speedup": round(speedup, 2) if speedup else None,
        "decision": decision, "rationale": rationale,
    }
    print(f"\n§4.2 DECISION: {decision} — {rationale}")
    write_json(ROOT / "report" / "pilot_model.json", out)
    return out


def pilot_language(cfg, rows, lex) -> dict:
    """§4.3 — forcing Hindi vs forcing English vs automatic detection."""
    size = cfg["model"]["size"]
    backend = get_backend("local", model_spec_from_config(cfg))
    results = {}
    for lang in ("hi", "en", None):
        tag = lang or "auto"
        results[tag] = _run(backend, rows, cfg, lex, f"lang={tag}", language=lang)

    best = min(results, key=lambda k: results[k]["wer"])
    out = {
        "pilot": "language_configuration", "tier": "tier1", "model": size,
        "n_utts": len(rows),
        "results": {k: {"wer": v["wer"], "b_wer": v["b_wer"], "u_wer": v["u_wer"],
                        "cer": v["cer"], "wer_level2": v["wer_level2"],
                        "empty_hyps": v["empty_hyps"]} for k, v in results.items()},
        "decision": best,
        "rationale": (f"'{best}' gives the lowest WER on Tier 1 and is fixed for all "
                      f"subsequent experiments."),
    }
    print(f"\n§4.3 DECISION: language={best}")
    write_json(ROOT / "report" / "pilot_language.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["model", "language", "both"])
    ap.add_argument("--tier", default="tier1")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config()
    rows = read_jsonl(manifest_for_tier(cfg, a.tier))
    if a.limit:
        rows = rows[:a.limit]
    lex = load_lexicon(cfg["scoring"]["lexicon"])
    print(f"pilot on {a.tier}: {len(rows)} utts, "
          f"{sum(r['duration'] for r in rows)/60:.1f} min audio\n")

    if a.which in ("model", "both"):
        pilot_model(cfg, rows, lex)
    if a.which in ("language", "both"):
        pilot_language(cfg, rows, lex)


if __name__ == "__main__":
    main()
