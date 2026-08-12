"""Figures for the paper.

  fig_frontier      B-WER against U-WER as the gating threshold is swept (§7.4). The
                    gating claim is a curve, not a point: this traces the trade-off
                    frontier between terminology gain and non-terminology penalty, with
                    the unbiased baseline and the globally grounded system marked as the
                    two endpoints.
  fig_sweep         Tier-1 hyperparameter sweeps (M3a threshold, M2 hint-term count).
  fig_wer_duration  WER as a function of utterance duration (§8.4).
  fig_matrix        Overall / B-WER / U-WER per condition, the main results figure.
  fig_pipeline      The two-pass architecture (§6.2), which §12 asks for as a figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from common import ROOT, load_config, read_json   # noqa: E402

FIGDIR = ROOT / "report" / "figures"
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p = FIGDIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p.relative_to(ROOT)}")
    return p


def fig_frontier(tier: str, mechanism: str) -> Path | None:
    p = ROOT / "runs" / tier / f"G_sweep_{mechanism}.json"
    if not p.exists():
        print(f"skip frontier: {p.relative_to(ROOT)} not found")
        return None
    d = read_json(p)
    pts = sorted(d["points"], key=lambda x: x.get("flagged_rate", 0))
    b = [x["b_wer"] for x in pts]
    u = [x["u_wer"] for x in pts]
    # Label each point by the share of utterances it re-decoded: on this corpus the
    # interesting axis of the gate is cost, not the raw threshold value.
    th = [100 * x.get("flagged_rate", 0) for x in pts]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(u, b, "-o", color="#2b6cb0", ms=4, lw=1.4, zorder=3)
    for x, y, t in zip(u, b, th):
        ax.annotate(f"{t:.0f}%", (x, y), textcoords="offset points", xytext=(4, 4),
                    fontsize=7, color="#4a5568")
    ax.plot(u[0], b[0], "s", color="#718096", ms=8, label="unbiased (B0)", zorder=4)
    ax.plot(u[-1], b[-1], "^", color="#c53030", ms=8,
            label=f"global {mechanism}", zorder=4)
    chosen = d.get("chosen") or None
    if chosen:
        ax.plot(chosen["u_wer"], chosen["b_wer"], "*", color="#2f855a", ms=15,
                label=(f"chosen gate: {100*chosen['flagged_rate']:.0f}% re-decoded, "
                       f"{100*(chosen.get('retained_gain') or 0):.0f}% of gain"),
                zorder=5)
    ax.set_xlabel("U-WER  (non-terminology words) →  worse")
    ax.set_ylabel("B-WER  (syllabus terms) →  worse")
    ax.set_title(f"Confidence-gating trade-off frontier ({tier}, {mechanism})")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, f"frontier_{tier}_{mechanism}.png")


def fig_sweep(tier: str) -> list[Path]:
    out = []
    specs = [
        (f"sweep_m3a_{tier}.json", "M3a fuzzy-match threshold", "threshold"),
        (f"sweep_m2_{tier}.json", "M2 number of hint terms", "hotword_terms"),
        (f"sweep_topk_{tier}.json", "Retrieval depth k", "top_k"),
    ]
    for fname, title, xkey in specs:
        p = ROOT / "runs" / tier / fname
        if not p.exists():
            continue
        d = read_json(p)
        pts = sorted(d["points"], key=lambda x: x[xkey])
        xs = [x[xkey] for x in pts]
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        ax.plot(xs, [x["wer"] for x in pts], "-o", ms=4, label="WER", color="#2d3748")
        ax.plot(xs, [x["b_wer"] for x in pts], "-o", ms=4, label="B-WER",
                color="#2b6cb0")
        ax.plot(xs, [x["u_wer"] for x in pts], "-o", ms=4, label="U-WER",
                color="#c53030")
        if d.get("chosen") is not None:
            ax.axvline(d["chosen"], color="#2f855a", ls="--", lw=1,
                       label=f"chosen = {d['chosen']:g}")
        ax.set_xlabel(title)
        ax.set_ylabel("error rate")
        ax.set_title(f"{title} — swept on {tier}")
        ax.legend(frameon=False, fontsize=8)
        out.append(_save(fig, f"{p.stem}.png"))
    return out


def fig_wer_duration(tier: str, run: str = "B0") -> Path | None:
    p = ROOT / "runs" / tier / run / "error_analysis.json"
    if not p.exists():
        print(f"skip wer-duration: {p.relative_to(ROOT)} not found")
        return None
    d = read_json(p)["wer_by_duration"]
    order = ["0-2s", "2-4s", "4-7s", "7s+"]
    keys = [k for k in order if k in d and d[k]["wer"] is not None]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.bar(keys, [d[k]["wer"] for k in keys], color="#4299e1", width=0.6)
    for i, k in enumerate(keys):
        ax.text(i, d[k]["wer"], f"{d[k]['wer']:.2f}", ha="center", va="bottom",
                fontsize=8)
    ax.set_xlabel("utterance duration")
    ax.set_ylabel("WER")
    ax.set_title(f"WER by utterance duration ({tier}, {run})")
    return _save(fig, f"wer_by_duration_{tier}_{run}.png")


def fig_matrix(tier: str) -> Path | None:
    import make_report
    rows = make_report.collect(tier, resamples=2000)
    rows = [r for r in rows if r.get("wer") is not None]
    if not rows:
        return None
    names = [r["run"] for r in rows]
    x = range(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(5.5, 0.85 * len(names)), 3.6))
    ax.bar([i - w for i in x], [r["wer"] for r in rows], w, label="WER",
           color="#2d3748")
    ax.bar(list(x), [r["b_wer"] for r in rows], w, label="B-WER", color="#2b6cb0")
    ax.bar([i + w for i in x], [r["u_wer"] for r in rows], w, label="U-WER",
           color="#c53030")
    b0 = next((r for r in rows if r["run"] == "B0"), None)
    if b0:
        ax.axhline(b0["wer"], color="#718096", ls=":", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("error rate")
    ax.set_title(f"Experiment matrix ({tier})")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    return _save(fig, f"matrix_{tier}.png")


def fig_pipeline() -> Path:
    """The two-pass architecture figure (§6.2)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axis("off")
    boxes = [
        (0.02, "audio"), (0.17, "pass 1\nunbiased decode"),
        (0.35, "retrieval\nover syllabus index"), (0.54, "context\nassembly"),
        (0.72, "pass 2\ngrounded decode"), (0.89, "output-level\ncorrection"),
    ]
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, 0.42), 0.125, 0.32, fill=True,
                                   facecolor="#ebf8ff", edgecolor="#2b6cb0", lw=1.2))
        ax.text(x + 0.0625, 0.58, label, ha="center", va="center", fontsize=8)
    for x, _ in boxes[:-1]:
        ax.annotate("", xy=(x + 0.145, 0.58), xytext=(x + 0.127, 0.58),
                    arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.1))
    ax.annotate("same audio", xy=(0.785, 0.42), xytext=(0.09, 0.2),
                arrowprops=dict(arrowstyle="->", color="#c53030", lw=1,
                                connectionstyle="arc3,rad=-0.18"),
                fontsize=8, color="#c53030")
    ax.text(0.5, 0.94, "Two-pass syllabus-grounded pipeline (§6.2)",
            ha="center", fontsize=9.5)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0.05, 1.0)
    return _save(fig, "pipeline.png")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier2")
    ap.add_argument("--mechanism", default=None)
    a = ap.parse_args()
    mech = a.mechanism or cfg["gating"]["mechanism"]
    fig_pipeline()
    fig_matrix(a.tier)
    fig_wer_duration(a.tier)
    fig_frontier(a.tier, mech)
    fig_sweep(a.tier)
    fig_sweep("tier1")


if __name__ == "__main__":
    main()
