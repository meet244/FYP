"""FYRP: ten independent WER-improvement methods on Qwen3-ASR-0.6B, SLR104 Hindi-English.

Every method is applied to the SAME baseline model on the SAME 50 test utterances,
never stacked. Hyper-parameters are tuned on a 30-utt DEV set (lecture-disjoint
from test) and frozen before the test number is read.

Stages (one process each, so fp16 / fp32 weights are never co-resident in 8 GB):
    python fyrp_exp.py prep
    python fyrp_exp.py decode                       # S0 baseline, S3, S5 beams, S6
    python fyrp_exp.py train --method emb|tts|partial|lora   # S7, S8, S9, S10
    python fyrp_exp.py report                       # S1, S2, S4, S5 rescoring + table
"""
import argparse
import gc
import json
import random
import subprocess
import time

import numpy as np
import soundfile as sf

from config import OUT, SEED
from decode import load_manifest, load_audio
from normalize import normalize
from score import keyword_set, score_rows

FY = OUT / "fyrp"
HY = FY / "hyps"
TTS = FY / "tts"
for _p in (FY, HY, TTS):
    _p.mkdir(parents=True, exist_ok=True)

N_TEST, N_TUNE, N_TRAIN, N_TTS = 50, 30, 100, 120
N_TTS_TRAIN = 80  # of the 120 generated; caps fine-tune time on 8 GB


# ----------------------------------------------------------------- data ---
def all_rows():
    return [json.loads(l) for l in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def splits():
    test = load_manifest("test")[::3][:N_TEST]          # stratified over lectures
    tune = load_manifest("dev")[::2][:N_TUNE]
    dev_all = [r for r in all_rows() if r["split"] == "dev"]
    tune_ids = {r["utt_id"] for r in load_manifest("dev")}
    pool = [r for r in dev_all if r["utt_id"] not in tune_ids and 2.0 <= r["dur"] <= 12.0]
    train = random.Random(SEED).sample(pool, N_TRAIN)
    return test, tune, train, dev_all


def save(name, recs):
    with (HY / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(name):
    p = HY / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def exists(name):
    return (HY / f"{name}.jsonl").exists()


def lexicon_assets():
    import fyrp_lexicon as L
    _, tune, _, dev_all = splits()
    tune_ids = {r["utt_id"] for r in tune}
    dev_txt = [r for r in dev_all if r["utt_id"] not in tune_ids]  # tune refs never seen
    gl = L.build_glossary([r["ref"] for r in dev_txt])
    corpus = L.build_corpus(dev_txt)
    return gl, corpus


def build_lm(tok, corpus):
    from fyrp_lm import TokenTrigramLM
    return TokenTrigramLM(tok, corpus, vocab_size=151936,
                          eos_id=tok.convert_tokens_to_ids("<|im_end|>"))


# ----------------------------------------------------------------- prep ---
def tts_sentences():
    import fyrp_lexicon as L
    sents = []
    for u in L.syllabus_units():
        for s in u["prose"].replace("।", ".").split("."):
            s = s.strip()
            if len(s.split()) >= 4:
                sents.append(s)
    carriers = ["अब हम {} पर click करेंगे", "यहाँ {} option दिखाई देता है",
                "इस tutorial में हम {} के बारे में सीखेंगे", "{} को select करें"]
    kws = sorted({k for u in L.syllabus_units() for k in u["keywords"]})
    rnd = random.Random(SEED)
    for i, k in enumerate(kws):
        sents.append(carriers[i % len(carriers)].format(k))
    rnd.shuffle(sents)
    return sents[:N_TTS]


def stage_prep():
    test, tune, train, _ = splits()
    gl, corpus = lexicon_assets()
    print(f"test={len(test)}  tune={len(tune)}  train(real)={len(train)}  "
          f"glossary={len(gl)}  corpus segments={len(corpus)}")
    man = []
    t0 = time.time()
    for i, s in enumerate(tts_sentences()):
        p = TTS / f"tts_{i:03d}.wav"
        if not p.exists():
            # macOS built-in Hindi voice: offline, ~0.5 s per sentence
            subprocess.run(["say", "-v", "Lekha", "-o", str(p), "--file-format=WAVE",
                            "--data-format=LEI16@16000", s], check=True)
        man.append(dict(utt_id=p.stem, wav=str(p), ref=normalize(s)))
    (FY / "tts_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"TTS: {len(man)} synthetic utterances in {time.time()-t0:.0f}s "
          f"({sum(sf.info(m['wav']).duration for m in man):.0f}s audio)")


# --------------------------------------------------------------- decode ---
def _run_greedy(name, rows, fn):
    if exists(name):
        print(f"[cached] {name}")
        return
    out, t0 = [], time.time()
    for r in rows:
        t1 = time.time()
        hyp = fn(load_audio(r))
        out.append(dict(utt_id=r["utt_id"], lecture_id=r["lecture_id"], ref=r["ref"], hyp=hyp,
                        dur=r["dur"], dec_s=round(time.time() - t1, 3)))
    save(name, out)
    print(f"[done] {name}: {len(rows)} utts in {time.time()-t0:.0f}s", flush=True)


def stage_decode():
    import fyrp_asr as A
    import fyrp_lexicon as L
    from fyrp_lm import FusionProcessor, lowercase_map, special_mask

    test, tune, _, _ = splits()
    proc, model = A.load()
    gl, corpus = lexicon_assets()

    # S0 baseline (greedy, no prompt)
    _run_greedy("S0_baseline__test", test, lambda a: A.transcribe(proc, model, a))
    _run_greedy("S0_baseline__tune", tune, lambda a: A.transcribe(proc, model, a))

    # S3 native context biasing: top-80 syllabus terms as system prompt
    hot = ", ".join(L.top_terms(gl, 80))
    (FY / "S3_prompt.txt").write_text(hot, encoding="utf-8")
    _run_greedy("S3_prompt__test", test, lambda a: A.transcribe(proc, model, a, prompt=hot))

    # S5 k-best lists (rescored offline in `report`)
    for split, rows in (("tune", tune), ("test", test)):
        name = f"S5_nbest__{split}"
        if exists(name):
            print(f"[cached] {name}")
            continue
        out, t0 = [], time.time()
        for r in rows:
            t1 = time.time()
            nb = A.nbest(proc, model, load_audio(r), k=5)
            out.append(dict(utt_id=r["utt_id"], lecture_id=r["lecture_id"], ref=r["ref"],
                            nbest=nb, dur=r["dur"], dec_s=round(time.time() - t1, 3)))
        save(name, out)
        print(f"[done] {name}: {time.time()-t0:.0f}s", flush=True)

    # S6 shallow fusion with the same LM; lambda tuned on DEV
    lm = build_lm(proc.tokenizer, corpus)
    V = model.config.text_config.vocab_size
    lo, sp = lowercase_map(proc.tokenizer, V), special_mask(proc.tokenizer, V)
    mk = lambda lam: (lambda a: A.transcribe(
        proc, model, a, logits_processor_fn=lambda p0: FusionProcessor(lm, p0, lam, lo, sp)))
    for lam in (0.1, 0.3):
        _run_greedy(f"S6_fusion_l{lam}__tune", tune, mk(lam))
    kws = keyword_set()
    grid = {0.0: score_rows(load("S0_baseline__tune"), kws)[0]["wer"]}
    for lam in (0.1, 0.3):
        grid[lam] = score_rows(load(f"S6_fusion_l{lam}__tune"), kws)[0]["wer"]
    best = min(grid, key=lambda k: (grid[k], k))
    print(f"S6 dev grid {grid} -> lambda={best}")
    (FY / "S6_lambda.json").write_text(json.dumps(dict(grid=grid, best=best)))
    if best == 0.0:  # fusion never helped on DEV: report the smallest nonzero weight honestly
        best = 0.1
    _run_greedy("S6_fusion__test", test, mk(best))


# ---------------------------------------------------------------- train ---
def stage_train(method):
    import torch
    import fyrp_asr as A
    from transformers import AddedToken

    name = {"emb": "S7_embtune", "tts": "S8_ttsaug", "partial": "S9_partialft",
            "lora": "S10_lora"}[method]
    if exists(f"{name}__test"):
        print(f"[cached] {name}")
        return
    test, _, train_rows, _ = splits()
    # bf16 frozen base (1.6 GB) + fp32 trainable params: fp32 base (3.1 GB) swapped to death on 8 GB
    proc, model = A.load(dtype=torch.bfloat16)
    hooks = []
    for p in model.parameters():
        p.requires_grad_(False)

    tts = json.loads((FY / "tts_manifest.json").read_text(encoding="utf-8"))
    tts_ex = lambda: [A.make_example(proc, sf.read(m["wav"], dtype="float32")[0], m["ref"]) for m in tts[:N_TTS_TRAIN]]
    real_ex = lambda: [A.make_example(proc, load_audio(r), normalize(r["ref"])) for r in train_rows]
    info = {}

    if method == "emb":  # S7: new syllabus tokens, train ONLY their embedding rows
        import fyrp_lexicon as L
        tok = proc.tokenizer
        terms = sorted({w for u in L.syllabus_units() for k in u["keywords"]
                        for w in normalize(k).split() if w.isascii() and len(w) >= 4})
        terms = [t for t in terms if len(tok.encode(" " + t, add_special_tokens=False)) >= 2]
        pieces = {t: tok.encode(" " + t, add_special_tokens=False) for t in terms}
        V = model.config.text_config.vocab_size
        room = V - len(tok)
        terms = terms[:room]
        tok.add_tokens([AddedToken(" " + t, normalized=False) for t in terms])
        new_ids = [tok.convert_tokens_to_ids(" " + t) for t in terms]
        assert max(new_ids) < V, "new ids must fit inside the padded vocab slots"
        W = model.get_input_embeddings().weight
        with torch.no_grad():
            for t, i in zip(terms, new_ids):  # init = mean of the subword pieces
                W[i] = W[pieces[t]].mean(0)
        W.requires_grad_(True)
        ex = real_ex()
        step = A.embedding_row_trainer(model, new_ids)
        secs = A.train(model, ex, [W], lr=5e-4, epochs=2, row_update=step)
        info = dict(new_tokens=len(new_ids), trained_params=len(new_ids) * W.shape[1],
                    examples=len(ex), epochs=2, lr=5e-4)
    else:
        if method == "lora":
            n = A.add_lora(model, r=8, alpha=16)
            params = [p for n_, p in model.named_parameters() if n_.endswith((".A", ".B"))]
            lr = 2e-4
            info["lora_modules"] = n
        else:  # tts / partial: last 4 decoder layers, encoder frozen
            lm = model.model.language_model
            for layer in lm.layers[-4:]:
                hooks += A.fp32_island(layer, model.dtype)
            params = [p for layer in lm.layers[-4:] for p in layer.parameters()]
            lr = 2e-5
        for p in params:
            p.requires_grad_(True)
        ex = tts_ex() if method == "tts" else real_ex() + tts_ex()
        secs = A.train(model, ex, params, lr=lr, epochs=1)
        info.update(trained_params=sum(p.numel() for p in params), examples=len(ex), epochs=1, lr=lr)

    info["train_s"] = round(secs, 1)
    print(f"[train] {name} {info}")
    (FY / f"{name}_train.json").write_text(json.dumps(info, indent=1))
    for h in hooks:
        h.remove()
    model.half()
    gc.collect()
    torch.mps.empty_cache()
    _run_greedy(f"{name}__test", test, lambda a: A.transcribe(proc, model, a))


# --------------------------------------------------------------- report ---
def _tune_posthoc(tag, base_tune, fn, grid, kws):
    res = {}
    for thr in grid:
        rows = [dict(r, hyp=fn(r["hyp"], thr)) for r in base_tune]
        res[thr] = score_rows(rows, kws)[0]["wer"]
    base = score_rows(base_tune, kws)[0]["wer"]
    best = min(res, key=lambda k: (res[k], -k))  # ties -> more conservative threshold
    print(f"{tag} dev: baseline {base:.4f}  grid {{{', '.join(f'{k}: {v:.4f}' for k, v in res.items())}}} -> {best}")
    return best


def _rescore(nb_rows, lam, beta, lm):
    out = []
    for r in nb_rows:
        best = max(r["nbest"], key=lambda h: h["am"] + lam * lm.seq_logprob(h["text"])
                   + beta * len(normalize(h["text"]).split()))
        out.append(dict(r, hyp=best["text"]))
    return out


def _bootstrap(base_pu, meth_pu, B=2000, seed=0):
    """Paired bootstrap over utterances: 95% CI of WER(method) - WER(baseline)."""
    be = np.array([p["errors"] for p in base_pu]); me = np.array([p["errors"] for p in meth_pu])
    n = np.array([p["ref_len"] for p in base_pu])
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(n), len(n))
        d.append((me[i].sum() - be[i].sum()) / n[i].sum())
    d = np.array(d)
    return np.percentile(d, 2.5), np.percentile(d, 97.5), float((d >= 0).mean())


def stage_report():
    import fyrp_lexicon as L
    from transformers import AutoTokenizer
    import fyrp_asr as A

    kws = keyword_set()
    gl, corpus = lexicon_assets()
    base_test, base_tune = load("S0_baseline__test"), load("S0_baseline__tune")
    meta = {}

    # S1 fuzzy snapping
    s1 = L.Snapper(gl, "fuzzy")
    t1 = _tune_posthoc("S1", base_tune, s1.correct, [75, 80, 85, 90, 95], kws)
    save("S1_fuzzy__test", [dict(r, hyp=s1.correct(r["hyp"], t1)) for r in base_test])
    meta["S1"] = f"rapidfuzz ratio >= {t1}"

    # S2 phonetic snapping
    s2 = L.Snapper(gl, "phonetic")
    t2 = _tune_posthoc("S2", base_tune, s2.correct, [80, 85, 90, 95, 101], kws)
    save("S2_phonetic__test", [dict(r, hyp=s2.correct(r["hyp"], t2)) for r in base_test])
    meta["S2"] = "exact phonetic key only" if t2 > 100 else f"phonetic-key ratio >= {t2}"

    # S4 BM25 retrieval-constrained correction
    s4 = L.RetrievalCorrector(corpus, gl, k=3)
    t4 = _tune_posthoc("S4", base_tune, s4.correct, [70, 80, 90, 101], kws)
    save("S4_bm25__test", [dict(r, hyp=s4.correct(r["hyp"], t4)) for r in base_test])
    meta["S4"] = f"BM25 top-3 of {len(corpus)} segs, thr {t4}"

    # S5 N-best rescoring with the token trigram LM
    tok = AutoTokenizer.from_pretrained(A.REPO)
    lm = build_lm(tok, corpus)
    nb_tune, nb_test = load("S5_nbest__tune"), load("S5_nbest__test")
    grid = {}
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        for beta in (0.0, 0.5, 1.0):
            grid[(lam, beta)] = score_rows(_rescore(nb_tune, lam, beta, lm), kws)[0]["wer"]
    (lam5, beta5) = min(grid, key=lambda k: (grid[k], k))
    print(f"S5 dev: top-beam {grid[(0.0, 0.0)]:.4f} -> best lam={lam5} beta={beta5} {grid[(lam5, beta5)]:.4f}")
    save("S5_nbest_rescore__test", _rescore(nb_test, lam5, beta5, lm))
    meta["S5"] = f"k=5 beams, lam={lam5}, len bonus={beta5}"
    meta["LM"] = dict(test_ppl=lm.perplexity([r["ref"] for r in base_test]))
    s6 = json.loads((FY / "S6_lambda.json").read_text())
    meta["S6"] = f"LM weight {s6['best'] if s6['best'] else 0.1} (dev grid {s6['grid']})"
    meta["S3"] = "top-80 syllabus terms as system prompt"
    for k, n in (("S7", "S7_embtune"), ("S8", "S8_ttsaug"), ("S9", "S9_partialft"), ("S10", "S10_lora")):
        p = FY / f"{n}_train.json"
        meta[k] = json.loads(p.read_text()) if p.exists() else None

    METHODS = [
        ("S0", "S0_baseline", "Baseline (Qwen3-ASR-0.6B, greedy, no context)"),
        ("S1", "S1_fuzzy", "Fuzzy / edit-distance lexicon correction"),
        ("S2", "S2_phonetic", "Phonetic matching for code-switched OOV"),
        ("S3", "S3_prompt", "Context biasing via system prompt"),
        ("S4", "S4_bm25", "BM25 retrieval-constrained span correction"),
        ("S5", "S5_nbest_rescore", "N-best rescoring with domain n-gram LM"),
        ("S6", "S6_fusion", "Shallow fusion (n-gram LM in decoding)"),
        ("S7", "S7_embtune", "Embedding-only tuning (tokenizer expansion)"),
        ("S8", "S8_ttsaug", "TTS synthetic data (partial FT on TTS only)"),
        ("S9", "S9_partialft", "Partial FT, last-4 decoder layers (real+TTS)"),
        ("S10", "S10_lora", "LoRA r=8 enc+dec attention (real+TTS)"),
    ]
    base_s, base_pu = score_rows(base_test, kws)
    table = []
    for sid, name, desc in METHODS:
        if not exists(f"{name}__test"):
            print(f"[missing] {name}")
            continue
        rows = load(f"{name}__test")
        s, pu = score_rows(rows, kws)
        lo, hi, p = _bootstrap(base_pu, pu) if sid != "S0" else (0, 0, None)
        dec = [r.get("dec_s") for r in rows if r.get("dec_s") is not None]
        rtf = sum(dec) / sum(r["dur"] for r in rows) if dec else None
        table.append(dict(sid=sid, desc=desc, **s, d=s["wer"] - base_s["wer"], lo=lo, hi=hi,
                          p_worse=p, rtf=rtf))
    table.sort(key=lambda x: (x["wer"], x["cer"]))
    (FY / "results.json").write_text(json.dumps(dict(table=table, meta=meta), indent=1, default=str))

    L_ = ["| Rank | Step | Method | WER % | Word Acc % | ΔWER vs base (95% CI) | CER % | WER-sa % | K-WER % | U-WER % | RTF |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, t in enumerate(table, 1):
        d = "—" if t["sid"] == "S0" else f"{100*t['d']:+.2f} [{100*t['lo']:+.1f}, {100*t['hi']:+.1f}]"
        rtf = f"{t['rtf']:.2f}" if t["rtf"] is not None else "—"
        L_.append(f"| {i} | {t['sid']} | {t['desc']} | {100*t['wer']:.2f} | {100*(1-t['wer']):.2f} | {d} | "
                  f"{100*t['cer']:.2f} | {100*t['wer_sa']:.2f} | {100*(t['k_wer'] or 0):.2f} | "
                  f"{100*(t['u_wer'] or 0):.2f} | {rtf} |")
    md = "\n".join(L_)
    (FY / "results.md").write_text(md + "\n\nSettings:\n" + json.dumps(meta, indent=1, default=str) + "\n",
                                   encoding="utf-8")
    print("\n" + md)
    print("\n" + json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "decode", "train", "report"])
    ap.add_argument("--method", choices=["emb", "tts", "partial", "lora"])
    a = ap.parse_args()
    {"prep": stage_prep, "decode": stage_decode, "report": stage_report,
     "train": lambda: stage_train(a.method)}[a.stage]()
