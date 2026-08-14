"""Inspect lecture openers and validate the lecture -> course mapping.

SLR104 recording IDs are opaque hashes, so the course each lecture belongs to was
identified by reading each lecture's opening utterances, which announce the topic
("... के इस spoken tutorial में आपका स्वागत है"). Those utterances are marked
title_source in the manifest and excluded from evaluation, so nothing that feeds
a syllabus is ever scored.

    python src/map_lectures.py --show       # openers per lecture (the audit trail)
    python src/map_lectures.py --validate   # every lecture mapped and has a syllabus
"""
import argparse
import collections
import json

from config import OUT, SYL
from courses import MAP_PATH, course_of


def manifest():
    return [
        json.loads(l)
        for l in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if not (a.show or a.validate):
        a.validate = True

    rows = manifest()
    by_lec = collections.defaultdict(list)
    for r in sorted(rows, key=lambda x: x["utt_id"]):
        by_lec[r["lecture_id"]].append(r)

    if a.show:
        for lec, rs in sorted(by_lec.items()):
            titles = [r for r in rs if r.get("title_source")]
            print(f"\n### {lec}  ({len(rs)} utts, course={course_of(lec)})")
            for r in titles:
                print(f"    [title_source] {r['ref'][:140]}")

    if a.validate:
        have = {p.stem for p in SYL.glob("*.json")} - {"lecture_map", "lecture_titles"}
        mapping = json.loads(MAP_PATH.read_text(encoding="utf-8")) if MAP_PATH.exists() else {}
        ok = True
        counts = collections.Counter()
        for lec, rs in sorted(by_lec.items()):
            cid = course_of(lec)
            n_eval = sum(1 for r in rs if r.get("in_eval"))
            counts[cid] += n_eval
            explicit = lec in mapping
            has_syl = cid in have
            if not (explicit and has_syl):
                ok = False
            print(
                f"{lec:20} -> {cid:22} eval_utts={n_eval:4} "
                f"{'mapped' if explicit else 'NOT-IN-MAP':10} "
                f"{'syllabus-ok' if has_syl else 'NO-SYLLABUS'}"
            )
        print(f"\n{len(by_lec)} lectures over {len(counts)} courses:")
        for c, n in counts.most_common():
            print(f"   {n:4} eval utts   {c}")
        if len(have) < 2:
            ok = False
            print("!! need >= 2 courses for the mismatched-syllabus control (C5)")
        print("\nmapping complete" if ok else "\nfix the gaps above before decoding")


if __name__ == "__main__":
    main()
