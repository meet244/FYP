"""Build the frozen evaluation manifest from the SLR104 Kaldi-style test dir.

Applies the fixed rules of plan §1.4:
  1. duration filter 2.0-28.0 s (whole utterance inside one Whisper window)
  2. >= 4 reference words
  3. lecture-disjoint DEV (30% of lectures) / TEST (70%) split
  4. stratified sample, proportional per lecture, seed 1337
  5. N_DEV=60, N_TEST=150

Plus one leakage rule specific to this corpus: SLR104 IDs carry no topic name, so
each lecture's title was read from its opening utterances ("... के इस spoken
tutorial में आपका स्वागत है"). Those N_TITLE_UTTS utterances are therefore marked
title_source and excluded from the evaluation set, so no scored utterance's
reference contributed anything to the syllabus.

Every filtered utterance is written; `in_eval` marks the sampled subset that the
runner actually decodes. Nothing here ever looks at a hypothesis.
"""
import collections
import json
import random
import re
import statistics
import sys

import soundfile as sf

from config import DATA, OUT, SEED, MIN_DUR, MAX_DUR, MIN_WORDS, DEV_LECTURE_FRAC, N_DEV, N_TEST

N_TITLE_UTTS = 2  # opening utterances per lecture: title source, never scored


def find(name):
    hits = sorted(DATA.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {DATA} — re-check Step 0")
    return hits[0]


def load_kaldi():
    text = {}
    for line in find("text").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt, _, tr = line.partition(" ")
        text[utt] = tr.strip()

    wavscp = {}
    for line in find("wav.scp").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec, _, rest = line.partition(" ")
        m = re.search(r"\S+\.wav", rest)  # handles both plain paths and pipe commands
        if m:
            wavscp[rec] = m.group(0)

    segs = {}
    segf = sorted(DATA.rglob("segments"))
    if segf:
        for line in segf[0].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt, rec, s, e = line.split()
            segs[utt] = (rec, float(s), float(e))
    else:
        # No segments file: every wav IS one utterance
        for utt in text:
            if utt in wavscp:
                segs[utt] = (utt, 0.0, None)
    return text, wavscp, segs


_wav_cache = {}


def resolve_wav(p):
    from pathlib import Path

    if p in _wav_cache:
        return _wav_cache[p]
    q = Path(p)
    if not q.is_absolute():
        q = DATA / p
    if not q.exists():
        hits = sorted(DATA.rglob(Path(p).name))
        if not hits:
            raise FileNotFoundError(f"wav {Path(p).name} not found under {DATA}")
        q = hits[0]
    _wav_cache[p] = str(q)
    return str(q)


def stratified_sample(rows, n, seed=SEED):
    """Sample n rows proportionally per lecture, so no lecture dominates."""
    by_lec = collections.defaultdict(list)
    for r in rows:
        by_lec[r["lecture_id"]].append(r)
    rng = random.Random(seed)
    lecs = sorted(by_lec)
    for l in lecs:
        by_lec[l].sort(key=lambda r: r["utt_id"])
        rng.shuffle(by_lec[l])
    if n >= len(rows):
        return rows

    total = len(rows)
    quota, remainder = {}, {}
    for l in lecs:
        exact = n * len(by_lec[l]) / total
        quota[l] = min(int(exact), len(by_lec[l]))
        remainder[l] = exact - int(exact)
    # largest-remainder top-up, ties broken by lecture id for determinism
    for l in sorted(lecs, key=lambda x: (-remainder[x], x)):
        if sum(quota.values()) >= n:
            break
        if quota[l] < len(by_lec[l]):
            quota[l] += 1
    # if still short (lectures exhausted), fill from whatever remains
    picked = [r for l in lecs for r in by_lec[l][: quota[l]]]
    if len(picked) < n:
        rest = [r for l in lecs for r in by_lec[l][quota[l]:]]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]
    picked.sort(key=lambda r: r["utt_id"])
    return picked


def main():
    text, wavscp, segs = load_kaldi()
    rows, skipped = [], collections.Counter()
    for utt, tr in sorted(text.items()):
        if utt not in segs:
            skipped["no_segment"] += 1
            continue
        rec, s, e = segs[utt]
        if rec not in wavscp:
            skipped["no_wav_entry"] += 1
            continue
        wav = resolve_wav(wavscp[rec])
        if e is None:
            info = sf.info(wav)
            e = info.frames / info.samplerate
        dur = e - s
        if not (MIN_DUR <= dur <= MAX_DUR):
            skipped["duration"] += 1
            continue
        if len(tr.split()) < MIN_WORDS:
            skipped["too_few_words"] += 1
            continue
        rows.append(
            dict(utt_id=utt, lecture_id=rec, wav=wav, start=s, end=e, dur=round(dur, 2), ref=tr)
        )
    if not rows:
        sys.exit(f"no utterances survived filtering; skipped={dict(skipped)}")

    lectures = sorted({r["lecture_id"] for r in rows})
    rng = random.Random(SEED)
    shuffled = list(lectures)
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * DEV_LECTURE_FRAC))
    dev = set(shuffled[:n_dev])
    for r in rows:
        r["split"] = "dev" if r["lecture_id"] in dev else "test"

    # Title-source utterances: the opening N_TITLE_UTTS of each lecture, by utt_id
    # order (IDs end in a zero-padded sequence number). Excluded from evaluation.
    seen = collections.Counter()
    for r in sorted(rows, key=lambda x: x["utt_id"]):
        r["title_source"] = seen[r["lecture_id"]] < N_TITLE_UTTS
        seen[r["lecture_id"]] += 1

    sampled = set()
    for split, n in (("dev", N_DEV), ("test", N_TEST)):
        pool = [r for r in rows if r["split"] == split and not r["title_source"]]
        sampled |= {r["utt_id"] for r in stratified_sample(pool, n)}
    for r in rows:
        r["in_eval"] = r["utt_id"] in sampled

    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by = collections.Counter(r["split"] for r in rows)
    ev = collections.Counter(r["split"] for r in rows if r["in_eval"])
    durs = [r["dur"] for r in rows]
    print(f"utterances={len(rows)}  lectures={len(lectures)}  split={dict(by)}  skipped={dict(skipped)}")
    print(f"eval subset: {dict(ev)}  (dev lectures={len(dev)}, test lectures={len(lectures)-len(dev)})")
    print(f"total audio = {sum(durs)/3600:.2f} h   median dur = {statistics.median(durs):.1f} s")
    for split in ("dev", "test"):
        d = [r["dur"] for r in rows if r["in_eval"] and r["split"] == split]
        if d:
            print(f"  {split}: n={len(d)}  audio={sum(d)/60:.1f} min  mean={sum(d)/len(d):.1f} s")

    print("\nLecture IDs (use these to decide how many syllabi you need):")
    counts = collections.Counter(r["lecture_id"] for r in rows)
    for l in lectures[:30]:
        print(f"   {counts[l]:5d}  {l}")
    if len(lectures) > 30:
        print(f"   ... and {len(lectures)-30} more")


if __name__ == "__main__":
    main()
