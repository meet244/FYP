"""Collect every run on a tier into the results table (§9.1, §12).

Emits Markdown and CSV with the metrics §8.2 requires: overall WER with its
substitution/insertion/deletion breakdown, the B-WER / U-WER decomposition, level-2
script-invariant WER, terminology precision/recall/F1, the percentage of utterances
improved and regressed, guard firing rates, and a bootstrap p-value for every system
against the baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import bootstrap
from common import ROOT, load_config, read_json, write_json

ORDER = ["B0", "C1", "C2", "C3", "M1", "M1_glossary", "M1_utterance", "M2",
         "M3a", "M3b", "M1+M3a", "M2+M3a", "G"]
LABEL = {
    "B0": "B0  baseline, no grounding",
    "C1": "C1  generic non-syllabus context",
    "C2": "C2  random syllabus document",
    "C3": "C3  oracle syllabus document",
    "M1": "M1  retrieved context conditioning",
    "M1_glossary": "M1' glossary-style context (ablation)",
    "M1_utterance": "M1'' per-utterance retrieval (ablation)",
    "M2": "M2  retrieved token-level biasing",
    "M3a": "M3a lexical correction on B0",
    "M3b": "M3b constrained model correction on B0",
    "M1+M3a": "M1+M3a  context + correction",
    "M2+M3a": "M2+M3a  biasing + correction",
    "G": "G   confidence-gated biasing",
}
ISOLATES = {
    "B0": "reference point", "C1": "effect of conditioning per se",
    "C2": "whether the correct syllabus matters",
    "C3": "upper bound given perfect retrieval",
    "M1": "mechanism 1", "M2": "mechanism 2", "M3a": "mechanism 3 (deterministic)",
    "M3b": "mechanism 3 (model-based)", "M2+M3a": "complementarity (H3)",
    "M1+M3a": "complementarity (H3)", "G": "principal contribution (H4)",
    "M1_glossary": "prose vs glossary (§7.1)",
    "M1_utterance": "retrieval granularity (§6.4)",
}


def collect(tier: str, baseline: str = "B0", resamples: int = 10000) -> list[dict]:
    tier_dir = ROOT / "runs" / tier
    runs = [d.name for d in sorted(tier_dir.iterdir())
            if d.is_dir() and (d / "metrics.json").exists()]
    runs = [r for r in ORDER if r in runs] + [r for r in runs if r not in ORDER]

    rows = []
    for r in runs:
        m = read_json(tier_dir / r / "metrics.json")
        row = {
            "run": r, "label": LABEL.get(r, r), "isolates": ISOLATES.get(r, ""),
            "n": m.get("n_scored"), "wer": m.get("wer"),
            "b_wer": m.get("b_wer"), "u_wer": m.get("u_wer"),
            "cer": m.get("cer"), "wer_level2": m.get("wer_level2"),
            "sub": m.get("sub"), "ins": m.get("ins"), "del": m.get("del"),
            "term_p": m.get("term_precision"), "term_r": m.get("term_recall"),
            "term_f1": m.get("term_f1"),
            "echo_guard_rate": m.get("guard_context_echo_rate"),
            "rewrite_discard_rate": m.get("guard_rewrite_discard_rate"),
            "grounded_rate": m.get("grounded_rate"),
        }
        if r != baseline and (tier_dir / baseline / "per_utt.jsonl").exists():
            try:
                c = bootstrap.compare(tier, baseline, r, n=resamples)
                row.update({
                    "d_wer": c["wer"]["abs_delta"], "p_wer": c["wer"]["p_value"],
                    "d_b_wer": c["b_wer"]["abs_delta"], "p_b_wer": c["b_wer"]["p_value"],
                    "d_u_wer": c["u_wer"]["abs_delta"], "p_u_wer": c["u_wer"]["p_value"],
                    "pct_improved": c["utterances"]["pct_improved"],
                    "pct_regressed": c["utterances"]["pct_regressed"],
                })
            except FileNotFoundError:
                pass
        rows.append(row)
    return rows


def _f(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def to_markdown(rows: list[dict], tier: str, cfg) -> str:
    lex = rows[0] if rows else {}
    head = [
        f"# Results — {tier}",
        "",
        f"Model `{cfg['model']['size']}` ({cfg['model']['compute_type']}, "
        f"beam {cfg['decode']['beam_size']}, temperature "
        f"{cfg['decode']['temperature']}, language `{cfg['decode']['language']}`). "
        f"Headline WER is level-1 (script-preserving) normalisation; WER-L2 is the "
        f"script-invariant secondary metric. B-WER / U-WER split errors by whether the "
        f"reference word is in the frozen syllabus lexicon. p-values are one-sided "
        f"paired bootstrap against B0.",
        "",
        "| System | What it isolates | N | WER | ΔWER | p | B-WER | ΔB | p | U-WER | ΔU | p | CER | WER-L2 | Term F1 | %impr | %regr |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        head.append("| " + " | ".join([
            r["label"], r["isolates"], _f(r["n"]),
            _f(r["wer"]), _f(r.get("d_wer")), _f(r.get("p_wer"), 3),
            _f(r["b_wer"]), _f(r.get("d_b_wer")), _f(r.get("p_b_wer"), 3),
            _f(r["u_wer"]), _f(r.get("d_u_wer")), _f(r.get("p_u_wer"), 3),
            _f(r["cer"]), _f(r["wer_level2"]), _f(r["term_f1"]),
            _f(r.get("pct_improved"), 1), _f(r.get("pct_regressed"), 1),
        ]) + " |")

    head += ["", "## Error composition and guards", "",
             "| System | sub | ins | del | Term P | Term R | echo-guard rate | rewrite-discard rate | grounded rate |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        head.append("| " + " | ".join([
            r["run"], _f(r["sub"]), _f(r["ins"]), _f(r["del"]),
            _f(r["term_p"]), _f(r["term_r"]),
            _f(r.get("echo_guard_rate"), 3), _f(r.get("rewrite_discard_rate"), 3),
            _f(r.get("grounded_rate"), 3)]) + " |")

    # Context from the other measured artefacts, so the table is readable alone.
    extras = []
    ra = ROOT / "runs" / tier / "retrieval_accuracy.json"
    if ra.exists():
        d = read_json(ra)
        extras.append(f"* Retrieval top-1 topic accuracy ({d['primary_condition']}): "
                      f"**{d['primary_top1_accuracy']:.3f}** (§6.3).")
    ea = ROOT / "runs" / tier / "B0" / "error_analysis.json"
    if ea.exists():
        d = read_json(ea)
        extras.append("* Headroom: " + d["headroom_estimate"]["statement"])
    cov = ROOT / "report" / "lexicon_coverage.json"
    if cov.exists():
        d = read_json(cov)
        extras.append(
            f"* Lexicon coverage on this tier: {d.get('bias_token_rate', 0)*100:.1f}% "
            f"of reference word tokens are lexicon terms "
            f"({d.get('bias_tokens')}/{d.get('ref_tokens')}); §5.4.")
    if extras:
        head += ["", "## Measured context", ""] + extras
    return "\n".join(head) + "\n"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier2")
    ap.add_argument("--baseline", default="B0")
    ap.add_argument("-n", "--resamples", type=int, default=10000)
    a = ap.parse_args()

    rows = collect(a.tier, a.baseline, a.resamples)
    md = to_markdown(rows, a.tier, cfg)
    out_md = ROOT / "report" / f"results_{a.tier}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")

    out_csv = ROOT / "report" / f"results_{a.tier}.csv"
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run"] + [k for k in keys if k != "run"])
        w.writeheader()
        w.writerows(rows)

    print(md)
    print(f"-> {out_md.relative_to(ROOT)}\n-> {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
