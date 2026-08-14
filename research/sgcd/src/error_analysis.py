"""Qualitative error analysis: where SGCD wins and where it loses (plan step 12).

Dumps the N largest per-utterance improvements and regressions of a system
condition vs C0, with the retrieved syllabus units, so the paper's analysis
section is grounded in real examples rather than impressions.
"""
import argparse
import json

from config import HYP, OUT
from normalize import normalize


def load_hyp(model, cond, split):
    p = HYP / f"{model}__{cond}__{split}.jsonl"
    return {
        json.loads(l)["utt_id"]: json.loads(l)
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
    }


def load_per(model, cond, split):
    p = OUT / f"perutt__{model}__{cond}__{split}.json"
    return {x["utt_id"]: x for x in json.load(p.open())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cond", default="C4")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()

    base_h, sys_h = load_hyp(a.model, "C0", a.split), load_hyp(a.model, a.cond, a.split)
    base_p, sys_p = load_per(a.model, "C0", a.split), load_per(a.model, a.cond, a.split)

    deltas = []
    for u in sys_p:
        if u not in base_p or base_p[u]["ref_len"] == 0:
            continue
        d = (sys_p[u]["errors"] - base_p[u]["errors"]) / base_p[u]["ref_len"]
        deltas.append((d, u))
    deltas.sort()

    def dump(items, title):
        print(f"\n{'='*72}\n{title}\n{'='*72}")
        for d, u in items:
            s = sys_h[u]
            print(f"\n[{u}]  ΔWER = {d*100:+.1f}  ({base_p[u]['errors']} -> {sys_p[u]['errors']} errors, "
                  f"{base_p[u]['ref_len']} ref words)")
            print(f"  REF : {normalize(s['ref'])}")
            print(f"  C0  : {normalize(base_h[u]['hyp'])}")
            print(f"  {a.cond:<4}: {normalize(s['hyp'])}")
            if s.get("retrieved"):
                print(f"  retrieved: {s['retrieved']}  (course {s.get('prompt_course')})")
            if s.get("fallback"):
                print("  guard: FELL BACK to C0")

    dump(deltas[: a.n], f"TOP {a.n} WINS  ({a.cond} better than C0)")
    dump(deltas[-a.n:][::-1], f"TOP {a.n} LOSSES  ({a.cond} worse than C0)")

    n_better = sum(1 for d, _ in deltas if d < 0)
    n_worse = sum(1 for d, _ in deltas if d > 0)
    print(f"\nsummary: better={n_better}  worse={n_worse}  unchanged={len(deltas)-n_better-n_worse}"
          f"  (n={len(deltas)})")

    out = OUT / f"erroranalysis__{a.model}__{a.cond}__{a.split}.json"
    out.write_text(json.dumps([
        dict(utt_id=u, delta=d, ref=sys_h[u]["ref"], c0=base_h[u]["hyp"], sys=sys_h[u]["hyp"],
             retrieved=sys_h[u].get("retrieved"))
        for d, u in deltas[: a.n] + deltas[-a.n:]
    ], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
