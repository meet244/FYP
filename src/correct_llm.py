"""M3b — constrained model-based output-level correction (§7.3).

A language model repairs only misrecognised terminology, under hard constraints:
preserve all other words exactly, preserve script, do not translate, punctuate,
reorder, add or delete words. A post-check (the rewrite guard, §7.5) discards the
correction whenever it alters more than a fixed fraction of tokens, and the **discard
rate is reported** — an unconstrained rewrite destroys WER, and demonstrating that this
was detected and prevented is a point in the paper's favour.

M3b is worth emphasising as the only mechanism deployable against a hosted ASR API,
which is a practical contribution independent of its accuracy.

Requests are cached per (utterance, model, prompt hash) so a repeat run costs nothing.
If no provider is configured the condition is skipped loudly rather than silently
substituting the uncorrected baseline, so it can never be mistaken for a real result.
"""
from __future__ import annotations

import os
from pathlib import Path

from common import ROOT, stable_hash, write_json
from guards import apply_rewrite_guard

SYSTEM = (
    "You correct ASR transcripts of Hindi-English code-switched technical lectures. "
    "Fix ONLY misrecognised technical terms, using the supplied reference term list. "
    "Rules: preserve every other word exactly, including disfluencies and grammar. "
    "Preserve the original script of each word. Do not translate, transliterate, "
    "punctuate or reorder. Do not add or delete words. If nothing is clearly a "
    "misrecognised technical term, return the transcript unchanged. "
    "Output only the corrected transcript.")


def build_user(hyp: str, terms: list[str]) -> str:
    return f"Reference terms: {', '.join(terms)}\n\nTranscript: {hyp}"


class GroqCorrector:
    def __init__(self, model: str):
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=key)
        self.model = model

    def __call__(self, hyp: str, terms: list[str]) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=512,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": build_user(hyp, terms)}])
        return (r.choices[0].message.content or "").strip()


def get_corrector(cfg):
    provider = cfg["correction"]["llm_provider"]
    if provider in (None, "none", ""):
        return None, "no provider configured"
    if provider == "groq":
        try:
            return GroqCorrector(cfg["correction"]["llm_model"]), None
        except Exception as exc:                                   # noqa: BLE001
            return None, str(exc)
    return None, f"unknown provider {provider!r}"


def correct_run(rows: list[dict], plan, cfg, meta: dict):
    """Apply M3b to `rows` in place. Signature matches conditions.run_text_condition."""
    model = cfg["correction"]["llm_model"]
    max_change = cfg["correction"]["llm_max_token_change"]
    max_terms = cfg["correction"]["llm_max_terms"]
    corrector, err = get_corrector(cfg)
    if corrector is None:
        raise SystemExit(
            f"M3b skipped: {err}. Set GROQ_API_KEY (or correction.llm_provider: none "
            f"in configs/config.yaml to exclude M3b from the matrix). The condition is "
            f"deliberately not falling back to the uncorrected baseline, which would "
            f"look like a real result.")

    cache_dir = ROOT / "cache" / "llm" / model
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_api = 0
    for i, row in enumerate(rows, 1):
        terms = plan.candidates(row["utt_id"])[:max_terms]
        prompt_key = stable_hash({"hyp": row["hyp"], "terms": terms, "sys": SYSTEM})
        cp = cache_dir / f"{row['utt_id']}_{prompt_key}.json"
        if cp.exists():
            corrected = __import__("json").loads(cp.read_text(encoding="utf-8"))["out"]
        else:
            corrected = corrector(row["hyp"], terms)
            cp.write_text(__import__("json").dumps(
                {"in": row["hyp"], "out": corrected, "terms": terms},
                ensure_ascii=False), encoding="utf-8")
            n_api += 1
        apply_rewrite_guard(row, corrected, max_change)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} corrected "
                  f"(api calls={n_api})", flush=True)

    meta.update({
        "llm_model": model, "llm_max_token_change": max_change,
        "llm_max_terms": max_terms, "api_calls": n_api,
        "n_discarded": sum(1 for r in rows if r.get("guard_rewrite_discarded")),
        "n_changed": sum(1 for r in rows if r.get("corrected")),
    })
    return rows, meta, plan.dump()
