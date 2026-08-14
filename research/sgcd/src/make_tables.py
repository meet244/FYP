"""Render the paper's main table (markdown + csv) from scores.csv and stats JSON."""
import argparse
import csv
import json

from config import OUT, TABLES, CONDITION_DOC


def pct(x, nd=2):
    return "—" if x in (None, "", "None") else f"{float(x)*100:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    scores = {}
    with (OUT / "scores.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["model"] == a.model and r["split"] == a.split:
                scores[r["cond"]] = r
    if not scores:
        raise SystemExit(f"no rows for model={a.model} split={a.split} in scores.csv")

    sp = OUT / f"stats__{a.model}__{a.split}.json"
    st = json.loads(sp.read_text()) if sp.exists() else {}

    hdr = ["Cond", "Prompt", "WER", "WER-sa", "K-WER", "U-WER", "CER", "Script fid.", "ΔWER vs C0 [95% CI]", "Worse %", "Fallback %"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for c in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
        r = scores.get(c)
        if not r:
            continue
        s = st.get(f"C0->{c}")
        if s:
            lo, hi = s["ci"]
            delta = f"{s['corpus_delta']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]"
            worse = f"{s['degraded']*100:.0f}"
        else:
            delta, worse = ("—", "—") if c != "C0" else ("ref", "—")
        label = CONDITION_DOC[c] + (" **(topline, oracle)**" if c == "C6" else "")
        lines.append(
            "| " + " | ".join([
                c, label, pct(r["wer"]), pct(r["wer_sa"]), pct(r["k_wer"]), pct(r["u_wer"]), pct(r["cer"]),
                pct(r["script_fidelity"], 1), delta, worse,
                pct(r["fallback_rate"], 1) if r["fallback_rate"] not in ("", "None") else "—",
            ]) + " |"
        )

    extra = []
    for key, label in [
        ("C2->C3", "H4  prose rendering vs keyword list"),
        ("C3->C4", "H5  retrieval vs whole syllabus"),
        ("C5->C4", "H3  matched vs mismatched syllabus"),
        ("C4->C7", "guard effect"),
    ]:
        if key in st:
            s = st[key]
            lo, hi = s["ci"]
            extra.append(f"- **{label}** ({key}): ΔWER {s['corpus_delta']*100:+.2f} "
                         f"[95% CI {lo*100:+.2f}, {hi*100:+.2f}], P(improve)={s['p_improve']:.3f}")

    n = scores["C0"]["n"]
    md = (
        f"# Main results — {a.model}, {a.split} split (N={n})\n\n"
        + "\n".join(lines)
        + "\n\nNegative ΔWER = improvement. CI from a paired bootstrap over utterances "
        "(10k resamples, seed 1337). WER-sa is the transliteration-tolerant variant "
        "(Devanagari and Latin reduced to a common consonant skeleton).\n\n## Contrasts\n\n"
        + ("\n".join(extra) if extra else "_run stats.py to populate_")
        + "\n"
    )
    out = TABLES / f"main__{a.model}__{a.split}.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
