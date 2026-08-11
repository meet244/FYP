"""Method B2 — LLM correction constrained by retrieved syllabus terms.

Hard-constrained on purpose: an unconstrained LLM paraphrases and destroys WER.
Any output that changes more than `max_change` of the tokens is discarded.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import score_file  # noqa: E402

SYSTEM = (
    "You correct ASR transcripts of Hindi-English code-switched technical lectures. "
    "Fix ONLY misrecognised technical terms, using the reference term list. "
    "Rules: preserve every other word exactly, including disfluencies and grammar. "
    "Preserve the original script of each word. Do not translate, punctuate, or reorder. "
    "Do not add or remove words. Output the corrected transcript and nothing else."
)


def build_user(hyp: str, terms: list[str]) -> str:
    return f"Reference terms: {', '.join(terms[:40])}\n\nTranscript: {hyp}"


def token_change_ratio(a: str, b: str) -> float:
    import jiwer
    if not a.strip():
        return 0.0 if not b.strip() else 1.0
    o = jiwer.process_words([a], [b])
    return (o.substitutions + o.insertions + o.deletions) / max(1, len(a.split()))


class GroqCorrector:
    def __init__(self, model="llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def __call__(self, hyp, terms):
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=512,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": build_user(hyp, terms)}])
        return r.choices[0].message.content.strip()


def main(a):
    from retrieve import SyllabusRetriever
    r = SyllabusRetriever()
    corrector = GroqCorrector(a.model)
    q = {j["utt_id"]: j["hyp"] for j in
         (json.loads(l) for l in open(a.pass1, encoding="utf-8"))}
    rows = [json.loads(l) for l in open(a.in_hyps, encoding="utf-8")]

    cache_dir = Path("cache/llm") / a.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    changed = discarded = 0
    for i, row in enumerate(rows, 1):
        cp = cache_dir / f"{row['utt_id']}.json"
        if cp.exists():
            new = json.loads(cp.read_text(encoding="utf-8"))["out"]
        else:
            terms = r.candidate_terms(q.get(row["utt_id"], row["hyp"]), k=a.k)
            new = corrector(row["hyp"], terms)
            cp.write_text(json.dumps({"in": row["hyp"], "out": new}, ensure_ascii=False),
                          encoding="utf-8")
        if token_change_ratio(row["hyp"], new) > a.max_change:
            discarded += 1
            new = row["hyp"]
        changed += new != row["hyp"]
        row["hyp_before_correction"] = row["hyp"]
        row["hyp"] = new
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} (changed={changed}, discarded={discarded})",
                  flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hyps.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    m = score_file(out_dir / "hyps.jsonl", a.terms)
    m.update({"utts_changed": changed, "corrections_discarded": discarded,
              "max_change": a.max_change, "llm": a.model})
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(json.dumps(m, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-hyps", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pass1", default="runs/S0_baseline/hyps.jsonl")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-change", type=float, default=0.20)
    ap.add_argument("--terms", default="syllabus/index/terms.txt")
    main(ap.parse_args())
