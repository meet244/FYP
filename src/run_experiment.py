"""Run one experiment condition end to end: decode -> (correct) -> score.

Conditions (see §10 of the plan):
  S0 none       large-v3, no prompt                      baseline
  S1 generic    generic "technical lecture" prompt       controls for any prompt
  S2 random     a random syllabus doc as prompt          controls for retrieval quality
  S3 retrieved  top-k retrieved syllabus prompt          Method A
  S6 oracle     the gold-topic syllabus doc as prompt    upper bound on retrieval
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backends import DecodeConfig, get_backend  # noqa: E402
from score import score_file  # noqa: E402
from transcribe import transcribe_manifest  # noqa: E402

GENERIC_PROMPT = ("विषय: technical lecture. यह एक technical lecture है "
                  "जिसमें Hindi और English दोनों का प्रयोग होता है.")


def build_prompt_fn(mode, args):
    if mode == "none":
        return lambda row: None
    if mode == "generic":
        return lambda row: GENERIC_PROMPT
    if mode == "random":
        from retrieve import SyllabusRetriever
        r = SyllabusRetriever()
        topics = sorted({d["topic"] for d in r.docs})
        rng = random.Random(1337)
        picks = {}

        def fn(row):
            if row["utt_id"] not in picks:
                picks[row["utt_id"]] = r.prompt_for_topic(rng.choice(topics))
            return picks[row["utt_id"]]
        return fn
    if mode == "retrieved":
        from retrieve import SyllabusRetriever
        r = SyllabusRetriever()
        pass1 = {j["utt_id"]: j["hyp"] for j in
                 (json.loads(l) for l in open(args.pass1, encoding="utf-8"))}
        cache = {}

        def fn(row):
            u = row["utt_id"]
            if u not in cache:
                cache[u] = r.prompt_for(pass1.get(u, ""), k=args.k,
                                        english_only=args.english_prompt)
            return cache[u]
        return fn
    if mode == "oracle":
        from retrieve import SyllabusRetriever
        r = SyllabusRetriever()

        def fn(row):
            return r.prompt_for_topic(r.oracle_topic(row))
        return fn
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", default="local")
    ap.add_argument("--manifest", default="data/manifests/subset.jsonl")
    ap.add_argument("--prompt-mode", default="none",
                    choices=["none", "generic", "random", "retrieved", "oracle"])
    ap.add_argument("--pass1", default="runs/S0_baseline/hyps.jsonl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--english-prompt", action="store_true")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--terms", default="syllabus/index/terms.txt")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()

    out_dir = Path("runs") / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    hyps = out_dir / "hyps.jsonl"

    if not a.score_only:
        prompt_fn = build_prompt_fn(a.prompt_mode, a)
        kw = {} if a.backend == "groq" else {
            "model_size": a.model, "compute_type": a.compute_type,
            "cpu_threads": a.cpu_threads}
        backend = get_backend(a.backend, **kw)
        lang = None if a.language in ("auto", "none") else a.language

        def cfg_fn(row):
            return DecodeConfig(language=lang, beam_size=a.beam,
                                prompt=prompt_fn(row))

        transcribe_manifest(backend, a.manifest, cfg_fn, hyps, limit=a.limit)

    m = score_file(hyps, a.terms)
    m["_config"] = vars(a)
    (out_dir / "metrics.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in m.items() if k != "_config"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
