"""Score every cross-model bench run and emit one comparison table.

Accuracy comes from score.score_rows (the study's frozen normalisation, so these
numbers sit on the same scale as the SGCD tables); speed and memory come from the
per-run bench_runtime JSON written by bench_asr.py.
"""
import argparse
import json

from config import HYP, OUT
from bench_asr import BENCH
from score import keyword_set, score_rows

ORDER = ["wlv3", "turbo", "qwen17", "qwen06", "parakeet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    kws = keyword_set()
    rows = []
    for key in ORDER:
        tag = f"{key}__C0__{a.split}"
        p = HYP / f"{tag}.jsonl"
        if not p.exists():
            print(f"[missing] {tag}")
            continue
        recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        s, per = score_rows(recs, kws)
        meta_p = OUT / f"bench_runtime__{tag}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        empty = sum(1 for r in recs if not r["hyp"].strip())
        rows.append({**meta, **s, 'key': key, 'note': BENCH[key][2],
                     'repo': BENCH[key][1], 'empty': empty})
        json.dump(per, (OUT / f"perutt__{tag}.json").open("w"))

    if not rows:
        print("no bench runs found — run bench_asr.py first")
        return

    hdr = ("| Model | WER % | WER-sa % | CER % | K-WER % | U-WER % | script fid % | "
           "RTF | x realtime | peak RAM GB |")
    sep = "|" + "---|" * 10
    lines = [f"### SLR104 Hindi–English test, n={rows[0]['n']}, zero-prompt\n", hdr, sep]
    for r in sorted(rows, key=lambda x: x["wer"]):
        lines.append(
            f"| {r['note']} | {100*r['wer']:.2f} | {100*r['wer_sa']:.2f} | {100*r['cer']:.2f} | "
            f"{100*(r['k_wer'] or 0):.2f} | {100*(r['u_wer'] or 0):.2f} | "
            f"{100*(r['script_fidelity'] or 0):.1f} | {r.get('rtf','—')} | "
            f"{r.get('xrealtime','—')} | {r.get('peak_rss_gb','—')} |"
        )
    md = "\n".join(lines) + "\n"
    (OUT / f"tables/bench__{a.split}.md").write_text(md, encoding="utf-8")
    json.dump(rows, (OUT / f"bench_scores__{a.split}.json").open("w"), indent=2, default=str)
    print(md)
    print(f"wrote {OUT/'tables'/f'bench__{a.split}.md'}")


if __name__ == "__main__":
    main()
