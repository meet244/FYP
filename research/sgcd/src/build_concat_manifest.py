"""Secondary experiment: concatenate consecutive segments into ~25 s pseudo-utterances.

Motivation (plan §Part 7, the response prescribed for "C4 ≈ C5"): if matched and
mismatched syllabi perform alike, one explanation is that 5.7 s utterances are too
short for *content* context to matter — there is barely any content to predict.
This rebuilds the evaluation set from longer spans and re-runs the H3 contrast.

Rules kept identical to the frozen manifest where they still apply:
  - only strictly adjacent segments of the same lecture are merged (prev end == next start)
  - pseudo-utterances stay <= MAX_DUR so the whole span fits one Whisper window
  - title_source utterances are still excluded
  - same lecture-disjoint dev/test assignment, same seed

Writes out/manifest_concat.jsonl with splits "devcat"/"testcat"; the frozen
manifest.jsonl is never modified.
"""
import collections
import json

from build_manifest import stratified_sample
from config import MAX_DUR, OUT, SEED

TARGET_DUR = 25.0          # aim for ~25 s spans
MIN_CONCAT_DUR = 18.0      # discard tails shorter than this
N_TESTCAT, N_DEVCAT = 100, 40


def main():
    rows = [
        json.loads(l)
        for l in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rows = [r for r in rows if not r.get("title_source")]
    by_lec = collections.defaultdict(list)
    for r in rows:
        by_lec[r["lecture_id"]].append(r)

    out = []
    for lec, rs in sorted(by_lec.items()):
        rs.sort(key=lambda x: x["utt_id"])
        cur = []
        for r in rs:
            # start a new span if this segment is not contiguous with the previous
            if cur and abs(r["start"] - cur[-1]["end"]) > 1e-6:
                cur = _flush(cur, out) or []
            cur.append(r)
            span = cur[-1]["end"] - cur[0]["start"]
            if span >= TARGET_DUR or span >= MAX_DUR:
                cur = _flush(cur, out) or []
        _flush(cur, out)

    for r in out:
        r["split"] = "devcat" if r["_split"] == "dev" else "testcat"
        del r["_split"]

    sampled = set()
    for split, n in (("devcat", N_DEVCAT), ("testcat", N_TESTCAT)):
        pool = [r for r in out if r["split"] == split]
        sampled |= {r["utt_id"] for r in stratified_sample(pool, n, seed=SEED)}
    for r in out:
        r["in_eval"] = r["utt_id"] in sampled

    with (OUT / "manifest_concat.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ev = collections.Counter(r["split"] for r in out if r["in_eval"])
    durs = [r["dur"] for r in out if r["in_eval"]]
    print(f"pseudo-utterances={len(out)}  eval={dict(ev)}")
    print(f"eval audio = {sum(durs)/60:.1f} min   mean dur = {sum(durs)/len(durs):.1f} s")
    print(f"mean ref words = {sum(len(r['ref'].split()) for r in out if r['in_eval'])/len(durs):.1f}")


def _flush(cur, out):
    """Emit one pseudo-utterance from the buffered segments, if long enough."""
    if not cur:
        return None
    dur = cur[-1]["end"] - cur[0]["start"]
    if dur < MIN_CONCAT_DUR or dur > MAX_DUR:
        return None
    out.append(
        dict(
            utt_id=f"{cur[0]['utt_id']}+{len(cur)}",
            lecture_id=cur[0]["lecture_id"],
            wav=cur[0]["wav"],
            start=cur[0]["start"],
            end=cur[-1]["end"],
            dur=round(dur, 2),
            ref=" ".join(r["ref"] for r in cur),
            n_segments=len(cur),
            _split=cur[0]["split"],
        )
    )
    return None


if __name__ == "__main__":
    main()
