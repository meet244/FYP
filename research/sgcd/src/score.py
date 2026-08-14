"""Score every cached hypothesis file.

Reports WER, CER, and — the diagnostic that matters — the K-WER / U-WER split:
error on syllabus-keyword reference words vs everything else. Aggregate WER alone
cannot tell you whether prompting is helping the terms and hurting the rest,
which is exactly the known failure mode this project is testing.
"""
import csv
import json
import pathlib

import jiwer

from config import HYP, OUT, SYL, CONDITION_DOC
from normalize import normalize, normalize_sa, script_of


def keyword_set():
    ks = set()
    for p in sorted(SYL.glob("*.json")):
        if p.stem in ("lecture_map", "lecture_titles"):
            continue
        for u in json.loads(p.read_text(encoding="utf-8"))["units"]:
            for k in u["keywords"]:
                nk = normalize(k)
                ks |= set(nk.split())  # multiword keywords count per token
    return ks


def score(path, kws):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return score_rows(rows, kws)


def score_rows(rows, kws):
    refs = [normalize(r["ref"]) for r in rows]
    hyps = [normalize(r["hyp"]) for r in rows]

    o = jiwer.process_words(refs, hyps)
    wer = o.wer
    cer = jiwer.process_characters(refs, hyps).cer

    # Secondary, transliteration-tolerant WER: credits an English term recognised
    # correctly but written in the other script. Lower bound on error.
    sa = jiwer.process_words([normalize_sa(r["ref"]) for r in rows],
                             [normalize_sa(r["hyp"]) for r in rows])
    wer_sa = sa.wer

    # K-WER / U-WER split and script fidelity, from the word alignments.
    # Insertions have no reference word, so they are counted separately and
    # distributed over neither bucket (standard practice for keyword WER).
    k_err = k_tot = u_err = u_tot = lat_ok = lat_tot = ins = 0
    per_utt = []
    for rw, hw, chunks in zip(o.references, o.hypotheses, o.alignments):
        u_e = 0
        for c in chunks:
            if c.type != "equal":
                u_e += max(c.ref_end_idx - c.ref_start_idx, c.hyp_end_idx - c.hyp_start_idx)
        per_utt.append(
            dict(errors=u_e, ref_len=len(rw), wer=(u_e / len(rw) if len(rw) else None))
        )
        for c in chunks:
            if c.type == "insert":
                ins += c.hyp_end_idx - c.hyp_start_idx
                continue
            for j in range(c.ref_end_idx - c.ref_start_idx):
                w = rw[c.ref_start_idx + j]
                err = c.type != "equal"
                if w in kws:
                    k_tot += 1
                    k_err += err
                else:
                    u_tot += 1
                    u_err += err
                if script_of(w) == "lat":
                    lat_tot += 1
                    lat_ok += c.type == "equal"

    for r, pu in zip(rows, per_utt):
        pu["utt_id"] = r["utt_id"]
    fb = [r.get("fallback") for r in rows if r.get("fallback") is not None]
    stats = dict(
        n=len(rows),
        wer=wer,
        wer_sa=wer_sa,
        cer=cer,
        k_wer=(k_err / k_tot if k_tot else None),
        k_n=k_tot,
        u_wer=(u_err / u_tot if u_tot else None),
        u_n=u_tot,
        ins_rate=(ins / max(1, k_tot + u_tot)),
        script_fidelity=(lat_ok / lat_tot if lat_tot else None),
        lat_n=lat_tot,
        mean_prompt_tokens=(
            sum(r.get("prompt_tokens") or 0 for r in rows) / len(rows) if rows else 0
        ),
        fallback_rate=(sum(fb) / len(fb) if fb else None),
    )
    return stats, per_utt


def main():
    kws = keyword_set()
    print(f"keyword vocabulary: {len(kws)} types\n")
    rows = []
    for p in sorted(HYP.glob("*.jsonl")):
        parts = p.stem.split("__")
        if len(parts) != 3:
            print(f"[skip] {p.name}: unexpected filename")
            continue
        m, cond, split = parts
        s, per = score(p, kws)
        rows.append(dict(model=m, cond=cond, split=split, **s))
        json.dump(per, (OUT / f"perutt__{p.stem}.json").open("w"))
        print(
            f"{m:6} {cond:3} {split:8} n={s['n']:4}  WER={s['wer']*100:6.2f}  "
            f"WER-sa={s['wer_sa']*100:6.2f}  "
            f"K-WER={100*(s['k_wer'] or 0):6.2f}  U-WER={100*(s['u_wer'] or 0):6.2f}  "
            f"CER={s['cer']*100:6.2f}  script={100*(s['script_fidelity'] or 0):5.1f}%  "
            f"ptok={s['mean_prompt_tokens']:5.0f}"
        )
    if not rows:
        print("no hypothesis files found — run decode.py first")
        return
    with (OUT / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'scores.csv'}")
    print("\nconditions: " + "; ".join(f"{k}={v}" for k, v in CONDITION_DOC.items()))


if __name__ == "__main__":
    main()
