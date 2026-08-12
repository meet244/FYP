"""Retrieval accuracy, measured separately from end-to-end WER (§6.3).

How often does the top-1 retrieved topic match the true topic of the utterance? This
decomposes end-to-end failure into *retrieval failure* and *biasing failure*, which is
the difference between a discussion section that explains results and one that merely
reports them.

Gold topics come from `syllabus/index/rec2topic.json`, which maps each recording to its
lecture topic. That mapping is lecture metadata a real deployment would already hold
(course title / lecture title); it is not mined from the reference transcripts, so using
it here is not test-set leakage.

Both granularities from §6.4 are evaluated: per-lecture (all utterances of a recording
form one query) and per-utterance (each pass-1 transcript is its own query).
"""
from __future__ import annotations

import argparse

from common import ROOT, load_config, manifest_for_tier, read_jsonl, write_json
from retrieve import RetrievalPlan, SyllabusRetriever


def evaluate(cfg, tier: str, pass1_run: str = "B0",
             ks=(1, 3, 5)) -> dict:
    rows = read_jsonl(manifest_for_tier(cfg, tier))
    p1 = {r["utt_id"]: r for r in
          read_jsonl(ROOT / "runs" / tier / pass1_run / "hyps.jsonl")}
    r = SyllabusRetriever(cfg)
    gold = {row["utt_id"]: r.oracle_topic(row) for row in rows}
    n_gold = sum(1 for v in gold.values() if v)

    out = {"tier": tier, "pass1_run": pass1_run, "n_utts": len(rows),
           "n_with_gold_topic": n_gold,
           "n_topics_in_index": len(r.topics), "results": {}}

    for gran in ("lecture", "utterance"):
        for k in ks:
            plan = RetrievalPlan(r, rows, p1, granularity=gran, k=k)
            top1 = hit_any = tot = 0
            per_rec_correct = {}
            for row in rows:
                g = gold.get(row["utt_id"])
                if not g:
                    continue
                tot += 1
                topics = plan.topics(row["utt_id"])
                if topics and topics[0] == g:
                    top1 += 1
                if g in topics:
                    hit_any += 1
                rec = row.get("rec")
                per_rec_correct.setdefault(rec, [0, 0])
                per_rec_correct[rec][1] += 1
                if topics and topics[0] == g:
                    per_rec_correct[rec][0] += 1
            key = f"{gran}_k{k}"
            out["results"][key] = {
                "granularity": gran, "top_k": k, "n_scored": tot,
                "top1_accuracy": top1 / tot if tot else None,
                "gold_in_topk": hit_any / tot if tot else None,
                "recordings_fully_correct": sum(
                    1 for c, n in per_rec_correct.values() if c == n),
                "n_recordings": len(per_rec_correct),
            }
            print(f"{key:16s} top-1 acc={out['results'][key]['top1_accuracy']:.4f}  "
                  f"gold in top-{k}={out['results'][key]['gold_in_topk']:.4f}  "
                  f"({tot} utts)", flush=True)

    primary = f"{cfg['retrieval']['granularity']}_k{cfg['retrieval']['top_k']}"
    out["primary_condition"] = primary
    out["primary_top1_accuracy"] = out["results"].get(primary, {}).get("top1_accuracy")
    write_json(ROOT / "runs" / tier / "retrieval_accuracy.json", out)
    print(f"\nprimary ({primary}) top-1 retrieval accuracy: "
          f"{out['primary_top1_accuracy']:.4f}  -> "
          f"runs/{tier}/retrieval_accuracy.json")
    return out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="tier2")
    ap.add_argument("--pass1", default="B0")
    a = ap.parse_args()
    evaluate(cfg, a.tier, a.pass1)


if __name__ == "__main__":
    main()
