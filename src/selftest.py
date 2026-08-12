"""Self-tests for the parts of the harness that do not need the model.

Scoring, normalisation, correction and gating logic decide every number the paper
reports, so they are checked against hand-worked examples rather than trusted. Run:

    PYTHONPATH=src python src/selftest.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

FAILS: list[str] = []


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"[{'ok' if ok else 'FAIL'}] {label}: got {got!r}" +
          ("" if ok else f", want {want!r}"))
    if not ok:
        FAILS.append(label)


def close(label: str, got, want, tol=1e-9) -> None:
    ok = got is not None and abs(got - want) <= tol
    print(f"[{'ok' if ok else 'FAIL'}] {label}: got {got!r}" +
          ("" if ok else f", want ~{want!r}"))
    if not ok:
        FAILS.append(label)


def test_normalise() -> None:
    from normalize import level1, level2, skeleton
    check("level1 strips punctuation and folds case",
          level1("Font, Style और SIZE?"), "font style और size")
    check("level1 unifies Devanagari digits", level1("वर्जन ३.३.४"), "वर्जन 3 3 4")
    check("level1 unifies spelled-out numerals", level1("दो files"), "2 files")
    check("level2 makes script irrelevant (font)",
          level2("फॉन्ट") == level2("font"), True)
    check("level2 makes script irrelevant (slide)",
          level2("स्लाइड") == level2("slide"), True)
    check("level2 keeps different words distinct",
          level2("font") == level2("size"), False)
    check("skeleton drops vowels", skeleton("slide"), "sld")


def test_lexicon(tmp: Path) -> None:
    from lexicon import Lexicon
    p = tmp / "terms.txt"
    p.write_text("font\nslide\nmatplotlib\nprintf\n", encoding="utf-8")
    lex = Lexicon(p)
    check("lexicon size", len(lex), 4)
    check("Latin term is a bias word", lex.in_bias("font"), True)
    check("Devanagari spelling of a term is a bias word", lex.in_bias("फॉन्ट"), True)
    check("non-term is not a bias word", lex.in_bias("करें"), False)
    check("stamp records size and hash",
          set(lex.stamp()) >= {"lexicon_size", "lexicon_sha256_12"}, True)


def test_score_decomposition(tmp: Path) -> None:
    from lexicon import Lexicon
    from score import decompose_utterance, score_rows
    p = tmp / "terms2.txt"
    p.write_text("font\nslide\nmatplotlib\n", encoding="utf-8")
    lex = Lexicon(p)

    # ref: 4 words, 1 bias term (font); hyp substitutes the bias term.
    d = decompose_utterance("इस font को बदलें", "इस size को बदलें", lex)
    check("ref length", d["ref_len"], 4)
    check("bias reference words", d["b_ref_len"], 1)
    check("non-bias reference words", d["u_ref_len"], 3)
    check("substitution attributed to B", d["b_sub"], 1)
    check("no U errors", d["u_errors"], 0)
    close("B error rate is 1.0", d["b_errors"] / d["b_ref_len"], 1.0)

    # A hallucinated bias term is an insertion attributed to B: this is how
    # over-biasing becomes visible (§8.2).
    d2 = decompose_utterance("इस को बदलें", "इस slide को बदलें", lex)
    check("hallucinated term counted as a B insertion", d2["b_ins"], 1)
    check("hallucination leaves U errors untouched", d2["u_errors"], 0)
    check("hallucination inflates B numerator with a zero B denominator",
          (d2["b_ref_len"], d2["b_errors"]), (0, 1))

    # Deletion of a non-bias word is a U error.
    d3 = decompose_utterance("इस font को बदलें", "इस font बदलें", lex)
    check("deletion attributed to U", (d3["u_del"], d3["b_del"]), (1, 0))

    # Aggregate path, including the guard counters.
    rows = [
        {"utt_id": "a", "ref": "इस font को बदलें", "hyp": "इस size को बदलें",
         "duration": 3.0, "guard_context_echo": True},
        {"utt_id": "b", "ref": "slide insert करें", "hyp": "slide insert करें",
         "duration": 2.0, "guard_context_echo": False},
    ]
    m, per = score_rows(rows, lex)
    check("aggregate scored count", m["n_scored"], 2)
    close("aggregate WER = 1 error / 7 ref words", m["wer"], 1 / 7)
    # Bias words across the two references: 'font' and 'slide'. 'insert' is not in this
    # fixture's lexicon, so it counts towards U.
    check("aggregate bias reference words", m["b_ref_words"], 2)
    close("aggregate B-WER = 1 error / 2 bias words", m["b_wer"], 1 / 2)
    close("aggregate U-WER = 0", m["u_wer"], 0.0)
    check("guard firing counted", m["guard_context_echo_fired"], 1)
    check("per-utterance records written for the bootstrap", len(per), 2)
    check("per-utterance keeps B/U denominators",
          {"b_ref_len", "u_ref_len", "b_errors", "u_errors"} <= set(per[0]), True)


def test_guards() -> None:
    from guards import (apply_context_echo_guard, apply_rewrite_guard,
                        context_echo_score, token_change_ratio)
    ctx = "यह libreoffice impress पर एक spoken tutorial है"
    close("echo score is 1.0 when the output is the context",
          context_echo_score(ctx, ctx, 3), 1.0)
    close("echo score is 0.0 for unrelated output",
          context_echo_score("save button पर click करें", ctx, 3), 0.0)

    row = {"utt_id": "x", "hyp": ctx, "context": ctx}
    apply_context_echo_guard(row, "save button पर click करें", 3, 0.5)
    check("echo guard fires and falls back to the unbiased hypothesis",
          (row["guard_context_echo"], row["hyp"]),
          (True, "save button पर click करें"))

    close("token change ratio", token_change_ratio("a b c d", "a b c e"), 0.25)
    row2 = {"utt_id": "y", "hyp": "font size बदलें"}
    apply_rewrite_guard(row2, "पूरा वाक्य पूरी तरह बदल दिया गया है", 0.20)
    check("rewrite guard discards an over-aggressive correction",
          (row2["guard_rewrite_discarded"], row2["hyp"]), (True, "font size बदलें"))
    row3 = {"utt_id": "z", "hyp": "font sise बदलें"}
    apply_rewrite_guard(row3, "font size बदलें", 0.40)
    check("rewrite guard accepts a small correction",
          (row3["guard_rewrite_discarded"], row3["hyp"]), (False, "font size बदलें"))


def test_correct_lexical() -> None:
    from correct_lexical import correct_utterance
    terms = ["matplotlib", "printf", "libreoffice", "impress", "font"]

    out, edits = correct_utterance("import mat plot lib को", terms, 88, 4)
    check("split compound term is merged back", "matplotlib" in out.split(), True)
    check("merge recorded as an edit",
          any(e["type"] == "merge" for e in edits), True)

    out2, _ = correct_utterance("printff को चलाएं", terms, 88, 4)
    check("near-miss single token is repaired", "printf" in out2.split(), True)

    out3, edits3 = correct_utterance("इस को बदलें", terms, 88, 4)
    check("short function words are left alone", (out3, edits3), ("इस को बदलें", []))

    out4, _ = correct_utterance("font को बदलें", terms, 88, 4)
    check("a token already in the lexicon is untouched", out4, "font को बदलें")

    # Span restriction: only index 1 is editable, so a term at index 3 is left alone.
    out5, edits5 = correct_utterance("printff और printff को", terms, 88, 4, spans=[0])
    check("span restriction confines edits to flagged tokens",
          (out5.split()[0], out5.split()[2]), ("printf", "printff"))


def test_gating() -> None:
    from gating import compose_gated, select_flagged
    rows = [{"utt_id": "a"}, {"utt_id": "b"}, {"utt_id": "c"}, {"utt_id": "d"}]
    base = {u: {"utt_id": u, "hyp": f"base {u}", "ref": f"r {u}"} for u in "abcd"}
    grounded = {u: {"utt_id": u, "hyp": f"grounded {u}", "ref": f"r {u}"}
                for u in "abcd"}
    conf = {"a": {"conf": 0.30, "spans": []}, "b": {"conf": 0.55, "spans": []},
            "c": {"conf": 0.80, "spans": []}, "d": {"conf": 0.95, "spans": []}}

    def hyps(mode, value):
        flagged, _ = select_flagged(conf, rows, mode, value)
        out, meta = compose_gated(base, grounded, flagged, rows, conf=conf)
        return [r["hyp"] for r in out], meta

    h, _ = hyps("percentile", 0)
    check("0th percentile grounds nothing (equals B0)",
          h, [f"base {u}" for u in "abcd"])
    h, _ = hyps("percentile", 100)
    check("100th percentile grounds everything (equals the global mechanism)",
          h, [f"grounded {u}" for u in "abcd"])
    h, meta = hyps("percentile", 50)
    check("50th percentile grounds the least-confident half",
          h, ["grounded a", "grounded b", "base c", "base d"])
    close("re-decode fraction is reported", meta["flagged_rate"], 0.5)

    # Rank selection is what makes the gate calibration-free: an absolute threshold that
    # sits below the whole distribution flags nothing, which is the failure observed on
    # Tier 1 with large-v3.
    h, _ = hyps("absolute", 0.20)
    check("an absolute threshold below the distribution flags nothing",
          h, [f"base {u}" for u in "abcd"])
    h, _ = hyps("percentile", 25)
    check("the same data still gates by rank", h[0], "grounded a")

    # No confidence signal means no evidence of confidence: always gated.
    conf_missing = dict(conf)
    conf_missing["d"] = {"conf": None, "spans": []}
    flagged, _ = select_flagged(conf_missing, rows, "percentile", 0)
    check("utterances without a confidence signal are always gated", flagged, {"d"})


def test_bootstrap() -> None:
    from bootstrap import paired_bootstrap
    # B is uniformly better: it should win essentially every resample.
    n = 60
    err_a = [3] * n
    err_b = [1] * n
    den = [10] * n
    res = paired_bootstrap(err_a, den, err_b, den, n=500, seed=0)
    check("a uniformly better system gets a significant p-value",
          res["p_value"] < 0.01, True)
    res2 = paired_bootstrap(err_a, den, err_a, den, n=500, seed=0)
    check("identical systems are not significant", res2["p_value"] >= 0.5, True)


def test_decode_config() -> None:
    from backends import DecodeConfig, ModelSpec
    spec = ModelSpec()
    a = DecodeConfig(context="ctx")
    b = DecodeConfig(hotwords="a, b")
    c = DecodeConfig()
    check("different grounding payloads give different cache keys",
          len({a.key(spec), b.key(spec), c.key(spec)}), 3)
    check("identical configs share a cache key",
          DecodeConfig(context="ctx").key(spec), a.key(spec))
    check("audio version enters the cache key",
          DecodeConfig(audio_version="raw").key(spec)
          != DecodeConfig(audio_version="refined").key(spec), True)
    try:
        DecodeConfig(hotwords="x", prefix="y")
        check("§7.2 guard rejects hotwords+prefix", False, True)
    except ValueError:
        check("§7.2 guard rejects hotwords+prefix", True, True)


def test_retrieval_assembly() -> None:
    from common import load_config
    from retrieve import SyllabusRetriever
    cfg = load_config()
    r = SyllabusRetriever(cfg)
    hits = [(d, 1.0) for d in r.docs if d["topic"] == "libreoffice_impress"][:2]

    prose = r.assemble_context(hits, "prose", 60)
    gloss = r.assemble_context(hits, "glossary", 60)
    check("prose context stays within the word budget",
          len(prose.split()) <= 60, True)
    check("glossary context stays within the word budget",
          len(gloss.split()) <= 60, True)
    check("prose context is not a comma-separated term dump (§7.1)",
          prose.count(",") < len(prose.split()) / 4, True)
    check("glossary context is a term list", "Technical terms:" in gloss, True)
    check("prose context names the topic", "impress" in prose.lower(), True)
    hw = r.hotword_terms(["libreoffice_impress"], 10)
    check("hint terms are limited to the requested count", len(hw) <= 10, True)
    check("oracle topic map is loaded", len(r.rec2topic) > 0, True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    for fn in (test_normalise, test_decode_config, test_guards,
               test_correct_lexical, test_gating, test_bootstrap,
               test_retrieval_assembly):
        print(f"\n--- {fn.__name__}")
        fn()
    print("\n--- test_lexicon")
    test_lexicon(tmp)
    print("\n--- test_score_decomposition")
    test_score_decomposition(tmp)

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("all self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
