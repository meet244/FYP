"""End-to-end integration test of the condition matrix, with the model stubbed out.

`selftest.py` checks the scoring and correction logic in isolation. This checks that the
*pipeline* holds together: that every condition in §9.1 builds its grounding payload,
routes it to the decoder, applies its guards, scores, and lands in the report and figures
— including the free conditions and confidence gating, which depend on artefacts written
by earlier conditions.

The decoder is replaced by a stub that:

  * derives a hypothesis from the reference by corrupting the syllabus terms in it
    (`font` -> `fond`, and one term split into pieces, mimicking `matplotlib` ->
    `mat plot lib`), so terminology errors exist to be repaired;
  * repairs a corrupted term if that term appears in the injected context or hint list,
    so grounded conditions genuinely differ from the baseline;
  * emits per-word probabilities, low for corrupted words, so confidence gating has a
    signal to gate on;
  * echoes the injected context for one designated utterance, so the context-echo guard
    is exercised rather than assumed.

Everything runs in a throwaway tier so no real run directory is touched. This costs no
decode time and is worth running before any expensive stage.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from common import ROOT, Config, load_config, write_json, write_jsonl

TIER = "_selftest"
FAILS: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"[{'ok' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(label)


# --- the stub decoder --------------------------------------------------------

CORRUPT = {"font": "fond", "impress": "impres", "matplotlib": "mat plot lib",
           "slide": "slaid", "insert": "insart"}


def StubSpec():
    """A real ModelSpec so the cache key is computed exactly as in production."""
    from backends import ModelSpec
    return ModelSpec(size="stub")


class StubBackend:
    """Deterministic fake decoder; see the module docstring."""

    def __init__(self):
        self.spec = StubSpec()
        self.name = self.spec.name
        self.calls = 0

    def transcribe(self, audio_path: str, cfg) -> dict:
        self.calls += 1
        utt = Path(audio_path).stem
        ref = REFS[utt]
        injected = " ".join(filter(None, [cfg.context or "", cfg.hotwords or ""]))

        # One utterance echoes the context, to exercise the context-echo guard.
        if utt == ECHO_UTT and injected.strip():
            text = injected
            words = [{"word": w, "start": 0.0, "end": 0.1, "prob": 0.9}
                     for w in text.split()]
            return {"text": text, "segments": [{"start": 0.0, "end": 1.0, "text": text,
                                                "avg_logprob": -0.2,
                                                "no_speech_prob": 0.01,
                                                "compression_ratio": 1.0,
                                                "words": words}],
                    "language": "hi", "language_prob": 0.9}

        out, words = [], []
        for tok in ref.split():
            low = tok.lower()
            if low in CORRUPT and low not in injected.lower():
                for piece in CORRUPT[low].split():
                    out.append(piece)
                    words.append({"word": piece, "prob": 0.25})
            else:
                out.append(tok)
                words.append({"word": tok, "prob": 0.95})
        text = " ".join(out)
        for w in words:
            w["start"], w["end"] = 0.0, 0.1
        return {"text": text,
                "segments": [{"start": 0.0, "end": 1.0, "text": text,
                              "avg_logprob": -0.3, "no_speech_prob": 0.01,
                              "compression_ratio": 1.0, "words": words}],
                "language": "hi", "language_prob": 0.9}


REFS: dict[str, str] = {}
ECHO_UTT = ""


def build_tier(cfg) -> Config:
    """A throwaway tier whose references contain syllabus terminology."""
    global ECHO_UTT
    rows = [
        {"utt_id": "t_406yMKxIdSDHRf8H_0001",
         "ref": "इस slide में font को insert करें", "duration": 4.0},
        {"utt_id": "t_406yMKxIdSDHRf8H_0002",
         "ref": "libreoffice impress में slide insert करें", "duration": 5.0},
        {"utt_id": "t_4oLp3bc9OSJbDrwM_0003",
         "ref": "python में matplotlib import करें", "duration": 3.0},
        {"utt_id": "t_4oLp3bc9OSJbDrwM_0004",
         "ref": "अब font size बदलें और slide देखें", "duration": 6.0},
        {"utt_id": "t_406yMKxIdSDHRf8H_0005",
         "ref": "यह एक सामान्य वाक्य है", "duration": 2.0},
    ]
    for r in rows:
        r["rec"] = r["utt_id"].split("_")[1]
        r["audio"] = f"/dev/null/{r['utt_id']}.wav"
        REFS[r["utt_id"]] = r["ref"]
    ECHO_UTT = rows[-1]["utt_id"]

    man = ROOT / "data" / "manifests" / f"{TIER}.jsonl"
    write_jsonl(man, rows)
    c = Config({k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()})
    c["data"] = dict(cfg["data"])
    c["data"]["tiers"] = dict(cfg["data"]["tiers"])
    c["data"]["tiers"][TIER] = str(man.relative_to(ROOT))
    # rec2topic covers the real recording ids, which the fake utt_ids reuse, so the
    # oracle condition (C3) resolves a topic.
    return c


def main() -> int:
    cfg = build_tier(load_config())
    run_root = ROOT / "runs" / TIER
    if run_root.exists():
        shutil.rmtree(run_root)

    import conditions
    stub = StubBackend()
    conditions.get_backend = lambda name, spec=None: stub          # type: ignore
    import gating
    gating.model_spec_from_config = lambda c, **k: StubSpec()      # type: ignore

    from conditions import run_condition

    # --- decode-time conditions -----------------------------------------
    m_b0 = run_condition("B0", TIER, cfg)
    check("B0 scores and writes metrics", m_b0["wer"] is not None,
          f"WER={m_b0['wer']:.3f} B-WER={m_b0['b_wer']:.3f}")
    check("B0 finds bias words in the references", m_b0["b_ref_words"] > 0,
          f"{m_b0['b_ref_words']} bias reference words")
    check("baseline has terminology errors to repair", m_b0["b_wer"] > 0)

    for cond in ("C1", "C2", "C3", "M1", "M2"):
        m = run_condition(cond, TIER, cfg)
        check(f"{cond} runs and scores", m["wer"] is not None,
              f"WER={m['wer']:.3f} B-WER={m['b_wer']:.3f} U-WER={m['u_wer']:.3f}")

    m_m1 = __import__("json").loads(
        (run_root / "M1" / "metrics.json").read_text())
    m_m2 = __import__("json").loads(
        (run_root / "M2" / "metrics.json").read_text())
    check("M1 improves B-WER over B0 when the context names the terms",
          m_m1["b_wer"] < m_b0["b_wer"], f"{m_b0['b_wer']:.3f} -> {m_m1['b_wer']:.3f}")
    check("M2 improves B-WER over B0 via hint terms",
          m_m2["b_wer"] < m_b0["b_wer"], f"{m_b0['b_wer']:.3f} -> {m_m2['b_wer']:.3f}")
    check("context-echo guard fired and was counted",
          m_m1.get("guard_context_echo_fired", 0) >= 1,
          f"{m_m1.get('guard_context_echo_fired')} firing(s)")
    check("retrieval plan was dumped", (run_root / "M1" / "retrieval.json").exists())

    # --- ablation rows ---------------------------------------------------
    run_condition("M1", TIER, cfg, out_name="M1_glossary", context_style="glossary")
    run_condition("M1", TIER, cfg, out_name="M1_utterance", granularity="utterance")
    check("glossary ablation produced a run",
          (run_root / "M1_glossary" / "metrics.json").exists())
    check("per-utterance retrieval ablation produced a run",
          (run_root / "M1_utterance" / "metrics.json").exists())

    # --- free (text-level) conditions ------------------------------------
    m_m3a = run_condition("M3a", TIER, cfg)
    check("M3a runs on cached text and repairs terms",
          m_m3a["b_wer"] < m_b0["b_wer"],
          f"{m_b0['b_wer']:.3f} -> {m_m3a['b_wer']:.3f}")
    check("M3a made no decoder calls (free condition, §9.2)",
          stub.calls == PRE_FREE_CALLS[0], f"calls={stub.calls}")
    m_comb = run_condition("M2+M3a", TIER, cfg)
    check("M2+M3a runs", m_comb["wer"] is not None,
          f"WER={m_comb['wer']:.3f} B-WER={m_comb['b_wer']:.3f}")
    run_condition("M1+M3a", TIER, cfg)

    # --- confidence gating ------------------------------------------------
    m_g = run_condition("G", TIER, cfg, gate_mechanism="M2")
    check("G runs and reports how much was grounded", "grounded_rate" in m_g,
          f"grounded_rate={m_g.get('grounded_rate')}")
    # The configured default threshold need not flag anything on the stub's confidence
    # distribution — that is what the Tier-1 sweep is for. What must hold is that some
    # threshold produces partial grounding.
    m_g_partial = run_condition("G", TIER, cfg, out_name="G_partial",
                                gate_mechanism="M2", gate_threshold=0.9)
    check("G grounds some but not all utterances at an intermediate threshold",
          0 < (m_g_partial.get("grounded_rate") or 0) < 1,
          f"grounded_rate={m_g_partial.get('grounded_rate')}")

    from gating import sweep
    sw = sweep(cfg, TIER, "M2", [0.0, 0.5, 0.9, 1.0])
    pts = {p["threshold"]: p for p in sw["points"]}
    check("gate threshold 0 reproduces the baseline exactly",
          abs(pts[0.0]["wer"] - m_b0["wer"]) < 1e-12)
    check("gate threshold 1 reproduces the global mechanism exactly",
          abs(pts[1.0]["wer"] - m_m2["wer"]) < 1e-12)
    check("sweep traces a monotone flagged rate",
          pts[0.0]["flagged_rate"] <= pts[0.5]["flagged_rate"] <= pts[1.0]["flagged_rate"])

    # --- statistics, report, figures --------------------------------------
    import bootstrap
    c = bootstrap.compare(TIER, "B0", "M2", n=200)
    check("bootstrap compares B-WER and U-WER separately",
          {"wer", "b_wer", "u_wer"} <= set(c))
    check("bootstrap counts improved/regressed utterances",
          "improved" in c["utterances"])
    bootstrap.compare_all(TIER, "B0", n=200)

    import analyze_errors
    ea = analyze_errors.analyse(TIER, "B0", cfg, top=10)
    check("headroom estimate is produced",
          "share_of_errors_on_lexicon_terms" in ea["headroom_estimate"])

    import eval_retrieval
    ra = eval_retrieval.evaluate(cfg, TIER, "B0", ks=(1, 3))
    check("retrieval accuracy is measured for both granularities",
          {"lecture_k1", "utterance_k1"} <= set(ra["results"]))

    import make_report
    rows = make_report.collect(TIER, resamples=200)
    md = make_report.to_markdown(rows, TIER, cfg)
    check("report table contains every run",
          all(any(r["run"] == n for r in rows)
              for n in ("B0", "C1", "C2", "C3", "M1", "M2", "M3a", "G")))
    check("report renders B-WER and U-WER columns", "B-WER" in md and "U-WER" in md)

    import figures
    check("matrix figure renders", figures.fig_matrix(TIER) is not None)
    check("frontier figure renders", figures.fig_frontier(TIER, "M2") is not None)
    check("wer-by-duration figure renders",
          figures.fig_wer_duration(TIER, "B0") is not None)
    check("pipeline figure renders", figures.fig_pipeline() is not None)

    # The throwaway tier's artefacts must not survive: a stub run directory sitting
    # beside the real ones, or a figure built from fake hypotheses, would be mistaken
    # for a result. Pass --keep to inspect them.
    if "--keep" not in sys.argv:
        shutil.rmtree(run_root, ignore_errors=True)
        (ROOT / "data" / "manifests" / f"{TIER}.jsonl").unlink(missing_ok=True)
        shutil.rmtree(ROOT / "cache" / "asr" / stub.name, ignore_errors=True)
        for f in (ROOT / "report" / "figures").glob(f"*{TIER}*"):
            f.unlink()
        for name in ("lexicon_coverage.json",):
            pass  # written only by the real pipeline

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print(f"pipeline self-test passed ({stub.calls} stub decodes)")
    return 0


PRE_FREE_CALLS = [0]

if __name__ == "__main__":
    # Record the decoder call count at the point the free conditions begin, so the
    # "free means free" assertion is checked rather than assumed.
    import conditions as _c

    _orig = _c.run_text_condition

    def _wrapped(name, cfg, rows, tier, opts):
        import conditions as cc
        PRE_FREE_CALLS[0] = getattr(cc.get_backend("local"), "calls", 0)
        return _orig(name, cfg, rows, tier, opts)

    _c.run_text_condition = _wrapped
    sys.exit(main())
