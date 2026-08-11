"""Collect runs/*/metrics.json into the results table (markdown + csv)."""
import argparse
import json
from pathlib import Path

import bootstrap

ORDER = ["S0_baseline", "S1_generic", "S2_random", "S3_retrieved", "S4_lexical",
         "S5_llm", "S6_oracle"]
LABEL = {
    "S0_baseline": "S0  large-v3, no prompt",
    "S1_generic": "S1  + generic prompt",
    "S2_random": "S2  + random syllabus doc",
    "S3_retrieved": "S3  + retrieved prompt (k=3)",
    "S4_lexical": "S4  S3 + lexical correction",
    "S5_llm": "S5  S3 + LLM correction",
    "S6_oracle": "S6  + oracle syllabus doc",
}


def main(baseline="S0_baseline", out="report/results.md"):
    runs = [d.name for d in sorted(Path("runs").iterdir())
            if (d / "metrics.json").exists()]
    runs = [r for r in ORDER if r in runs] + [r for r in runs if r not in ORDER]

    rows = []
    for r in runs:
        m = json.loads((Path("runs") / r / "metrics.json").read_text(encoding="utf-8"))
        row = {"run": LABEL.get(r, r), "n": m.get("n_scored"), "wer": m.get("wer"),
               "cer": m.get("cer"), "wer_si": m.get("wer_script_invariant"),
               "wer_skel": m.get("wer_skeleton_lower_bound"),
               "term_f1": m.get("term_f1"), "term_recall": m.get("term_recall"),
               "term_precision": m.get("term_precision")}
        if r != baseline and (Path("runs") / baseline / "per_utt.jsonl").exists() \
                and (Path("runs") / r / "per_utt.jsonl").exists():
            c = bootstrap.compare(baseline, r)
            row.update({"d_wer": c["abs_delta"], "improved": c["pct_improved"],
                        "regressed": c["pct_regressed"], "p": c["p_value"]})
        rows.append(row)

    def f(v, n=4):
        return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))

    hdr = ("| System | N | WER | CER | WER(script-inv) | WER(skeleton, LB) | "
           "Term P | Term R | Term F1 | ΔWER | % impr | % regr | p |")
    sep = "|" + "---|" * 13
    lines = [hdr, sep]
    for r in rows:
        lines.append("| " + " | ".join([
            r["run"], f(r["n"]), f(r["wer"]), f(r["cer"]), f(r["wer_si"]),
            f(r["wer_skel"]), f(r.get("term_precision")), f(r.get("term_recall")),
            f(r.get("term_f1")), f(r.get("d_wer")), f(r.get("improved"), 1),
            f(r.get("regressed"), 1), f(r.get("p"), 4)]) + " |")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="S0_baseline")
    ap.add_argument("--out", default="report/results.md")
    a = ap.parse_args()
    main(a.baseline, a.out)
