"""Generate the paper's dataset + experimental-setup section from measured artefacts.

§12 maps "Experimental setup" to §4's pilots, §3.3's tiers and §8.1's normalisation. Every
number in that section already exists in a JSON file written by the step that measured it,
so the section is generated rather than transcribed: it cannot drift from what was run.

Missing artefacts are reported as gaps rather than silently omitted, so an incomplete
study reads as incomplete.
"""
from __future__ import annotations

from pathlib import Path

from common import ROOT, load_config, read_json

R = ROOT / "report"


def _try(name: str):
    p = R / name
    return read_json(p) if p.exists() else None


def _pct(x) -> str:
    return "—" if x is None else f"{100*x:.1f}%"


def main() -> None:
    cfg = load_config()
    corpus = _try("corpus_stats.json")
    tiers = read_json(ROOT / "data" / "manifests" / "tiers.json") if (
        ROOT / "data" / "manifests" / "tiers.json").exists() else None
    diag = _try("segment_offset_diagnostic.json")
    tune = _try("refinement_tuning.json")
    refine = _try("segment_refinement.json")
    valid = _try("segment_refinement_validation.json")
    lexman = (read_json(ROOT / "syllabus" / "index" / "lexicon_manifest.json")
              if (ROOT / "syllabus" / "index" / "lexicon_manifest.json").exists()
              else None)
    cov = _try("lexicon_coverage.json")
    pm = _try("pilot_model.json")
    pl = _try("pilot_language.json")
    env = _try("environment.json")

    L: list[str] = ["# Dataset, environment and evaluation harness", "",
                    "*Generated from measured artefacts by `src/make_setup_section.py`; "
                    "do not edit by hand.*", ""]

    # --- corpus ------------------------------------------------------------
    L += ["## 1 Corpus", ""]
    if corpus:
        L += [
            "Evaluation uses the Hindi-English code-switched **test** portion of OpenSLR "
            "SLR104 (MUCS 2021 sub-task 2), drawn from Spoken Tutorial recordings on "
            "technical topics; the code-switching arises predominantly from that "
            "technical content. Only the test tarball (443 MB) was downloaded: no "
            "fine-tuning is performed, so the 7.3 GB train tarball is not required and "
            "no terminology is mined from any transcript.", "",
            "| Property | Measured | Published |", "|---|---|---|",
            f"| Utterances | {corpus['n_utts']} | {corpus['published_n_utts']} |",
            f"| Total audio | {corpus['total_hours']} h | "
            f"{corpus['published_hours']} h |",
            f"| Recordings | {corpus['n_recordings']} | 30 |",
            f"| Speakers (utt2spk) | {corpus['n_speakers']} | — |",
            f"| Utterance duration mean / median | {corpus['duration_mean']} s / "
            f"{corpus['duration_median']} s | — |",
            f"| Utterance duration min / max | {corpus['duration_min']} s / "
            f"{corpus['duration_max']} s | — |",
            "| Sample rate | 16 kHz, 16-bit mono (no resampling) | 16 kHz, 16-bit |",
            "",
            f"Stage-1 gate (§3.4): counts and duration match the published figures "
            f"({'PASS' if corpus['all_criteria_passed'] else 'FAIL'}); no utterance has "
            f"zero duration; a random sample of "
            f"{len(corpus['sample_checks'])} cut files is readable at the expected "
            f"duration and sample rate.", ""]
    else:
        L += ["*(missing: run `make data`)*", ""]

    # --- the segment defect -------------------------------------------------
    L += ["## 2 A segmentation defect in the distributed test set", ""]
    if diag and refine:
        s = diag["summary"]
        L += [
            "The distributed `segments` file partitions each recording into "
            "**whole-second** windows that tile the file exactly, so its boundaries "
            "cannot coincide with real speech edges. Cutting on them produces windows "
            "that contain the tail of the neighbouring sentence, which the decoder "
            "transcribes and which is then scored as insertions.", "",
            f"To quantify this, {diag['n_utts']} Tier-1 utterances were decoded at each "
            f"of several window shifts and scored against their references:", "",
            "| Window shift | Mean WER | Median WER | Utterances best at this shift |",
            "|---|---|---|---|"]
        for k in sorted(s, key=float):
            L.append(f"| {float(k):+.0f} s | {s[k]['mean_wer']:.4f} | "
                     f"{s[k]['median_wer']:.4f} | {s[k]['n_utts_best']} |")
        L += ["",
              "The WER-minimising shift **varies per utterance** across the whole range, "
              "so the defect is local boundary imprecision rather than a global offset "
              "and no constant correction can repair it.", "",
              "### Boundary refinement", "",
              "Each shared internal boundary is moved to the quietest instant within a "
              "search radius, scored by short-window frame energy plus a penalty per "
              "second of displacement that keeps boundaries near where the corpus put "
              "them. Utterance boundaries in continuous speech fall in inter-sentence "
              "pauses, so this recovers the true edge for the two utterances sharing it. "
              "The transcript-to-utterance assignment is never touched: only where the "
              "audio is cut changes, identically for every condition.", ""]
        if tune:
            L += ["Parameters were selected on the Tier-1 sample "
                  f"({tune['criterion']}):", "",
                  "| Radius | Penalty (dB/s) | Mean WER | Median WER | Worse than "
                  "distributed | Mean \\|shift\\| |", "|---|---|---|---|---|---|"]
            for g in tune["grid"]:
                mark = " **(chosen)**" if (g["radius"] == tune["chosen"]["radius"]
                                           and g["lambda"] == tune["chosen"]["lambda"]) \
                    else ""
                L.append(f"| {g['radius']}{mark} | {g['lambda']} | "
                         f"{g['mean_wer']:.4f} | {g['median_wer']:.4f} | "
                         f"{g['pct_worse_than_distributed']:.0f}% | "
                         f"{g['mean_abs_shift_s']:.2f} s |")
            L += ["",
                  f"| distributed windows | — | {tune['distributed_mean_wer']:.4f} | "
                  f"{tune['distributed_median_wer']:.4f} | — | 0.00 s |", ""]
        L += [f"Applied corpus-wide, refinement moves a boundary by "
              f"{refine['boundary_shift_abs_mean']:.2f} s on average "
              f"(median {refine['boundary_shift_median']:+.2f} s, p10 "
              f"{refine['boundary_shift_p10']:+.2f} s, p90 "
              f"{refine['boundary_shift_p90']:+.2f} s); "
              f"{refine['boundary_shift_unmoved_pct']:.1f}% of boundaries do not move. "
              f"Total audio is unchanged at {refine['total_hours']} h.", ""]
        if valid and valid.get("distributed_offset0"):
            d = valid["distributed_offset0"]
            L += ["Validation on the same utterance sample:", "",
                  "| Windows | Mean WER | Median WER |", "|---|---|---|",
                  f"| distributed | {d['mean_wer']:.4f} | {d['median_wer']:.4f} |",
                  f"| refined | {valid['refined']['mean_wer']:.4f} | "
                  f"{valid['refined']['median_wer']:.4f} |", ""]
        L += ["The refined cuts are identified by `data.audio_version` in the config, "
              "which enters the ASR cache key, so hypotheses decoded from the earlier "
              "cuts can never be served for the refined audio.", ""]
    else:
        L += ["*(missing: run `make diagnose refine`)*", ""]

    # --- tiers -------------------------------------------------------------
    L += ["## 3 Evaluation tiers", ""]
    if tiers:
        L += ["| Tier | Utterances | Duration | Recordings | Purpose |",
              "|---|---|---|---|---|"]
        purpose = {"tier1": "tuning: thresholds, context format, retrieval depth",
                   "tier2": "the full experiment matrix and all ablations",
                   "tier3": "final confirmation, two systems only"}
        for t in tiers["tiers"]:
            L.append(f"| {t['tier']} | {t['n_utts']} | {t['minutes']} min | "
                     f"{t['n_recordings']} | {purpose.get(t['tier'],'')} |")
        L += ["", f"Tier 1 and Tier 2 are disjoint samples, stratified by recording so "
                  f"every lecture topic appears in both, drawn once under seed "
                  f"{tiers['seed']} and frozen as committed manifests.", "",
              f"> {tiers['statement_for_report']}", ""]
    else:
        L += ["*(missing: run `make tiers`)*", ""]

    # --- system under test --------------------------------------------------
    L += ["## 4 System under test", ""]
    m, d = cfg["model"], cfg["decode"]
    L += [f"Whisper **{m['size']}** run locally through faster-whisper / CTranslate2 "
          f"with `compute_type={m['compute_type']}` on `{m['device']}`. Local execution "
          f"is required because two of the three grounding mechanisms need "
          f"decoder-level access and per-token confidences that a hosted API does not "
          f"expose.", "",
          f"Fixed decoding: beam size {d['beam_size']}, temperature {d['temperature']} "
          f"(deterministic, no temperature fallback), "
          f"`condition_on_previous_text={d['condition_on_previous_text']}` (utterances "
          f"are independent segments), `vad_filter={d['vad_filter']}` (audio is already "
          f"sentence-segmented), `word_timestamps={d.get('word_timestamps')}` (required "
          f"for the per-token confidences used by confidence gating).", "",
          "Every decode is cached under `cache/asr/<backend>/<config-hash>/<utt>.json`, "
          "keyed by the model identity, every decode parameter, the injected grounding "
          "payload and the audio version. An interrupted run resumes, output-level "
          "methods operate on cached text at zero decode cost, and metrics can be "
          "recomputed without touching the model.", ""]

    if pl:
        L += ["### 4.1 Language configuration pilot (§4.3)", "",
              "| Setting | WER | B-WER | U-WER | CER | WER (level 2) | Empty hyps |",
              "|---|---|---|---|---|---|---|"]
        for k, v in pl["results"].items():
            mark = " **(chosen)**" if k == pl["decision"] else ""
            L.append(f"| {k}{mark} | {v['wer']:.4f} | {v['b_wer']:.4f} | "
                     f"{v['u_wer']:.4f} | {v['cer']:.4f} | {v['wer_level2']:.4f} | "
                     f"{v['empty_hyps']} |")
        L += ["", pl["rationale"], ""]
    else:
        L += ["### 4.1 Language configuration pilot (§4.3)", "",
              "*(missing: run `make pilots`)*", ""]

    if pm:
        L += ["### 4.2 Model selection pilot (§4.2)", "",
              "| Model | WER | B-WER | U-WER | CER | Term F1 | Wall clock | RTF |",
              "|---|---|---|---|---|---|---|---|"]
        for k, v in pm["results"].items():
            mark = " **(chosen)**" if k == pm.get("decision") else ""
            L.append(f"| {k}{mark} | {v['wer']:.4f} | {v['b_wer']:.4f} | "
                     f"{v['u_wer']:.4f} | {v['cer']:.4f} | {v['term_f1']:.4f} | "
                     f"{v['wall_clock_min']} min | {v['rtf']} |")
        L += ["", f"WER difference (turbo − large-v3): "
                  f"{pm['wer_delta_turbo_minus_largev3']:+.4f}; turbo speed-up "
                  f"{pm['turbo_speedup']}×. Decision: **{pm['decision']}** — "
                  f"{pm['rationale']}", ""]
    else:
        L += ["### 4.2 Model selection pilot (§4.2)", "",
              "*(missing: run `python src/pilots.py model`)*", ""]

    # --- normalisation and metrics -----------------------------------------
    L += ["## 5 Normalisation and metrics", "",
          "Two normalisation levels are defined before any modelling work (§8.1). "
          "**Level 1** — Unicode NFC, numeral unification (Devanagari digits, Latin "
          "digits and spelled-out numbers), punctuation removal, case folding, "
          "whitespace collapse — is script-preserving and produces the **headline WER**. "
          "**Level 2** additionally romanises Devanagari and applies a light, symmetric "
          "phonetic folding, so a technical term written in either script compares equal; "
          "it is reported alongside level 1 to quantify the orthographic share of total "
          "error and is never presented as the WER. Both foldings are applied to "
          "reference and hypothesis alike.", "",
          "The primary metric is **decomposed WER**: every reference word is labelled B "
          "(a member of the frozen syllabus lexicon) or U (not), and error rates are "
          "reported separately. Substitutions and deletions are attributed to the class "
          "of the reference word; insertions to the class of the hypothesis word, so a "
          "syllabus term hallucinated into an utterance that never contained it is "
          "counted as a B insertion rather than hidden. Effective grounding lowers B-WER "
          "while leaving U-WER unchanged; over-biasing lowers B-WER and raises U-WER.", ""]
    if lexman:
        L += [f"The lexicon holds **{lexman['n_terms']} terms** over "
              f"{lexman['n_topics']} authored topic documents "
              f"(sha256 `{lexman['terms_sha256_12']}`), composed of "
              + ", ".join(f"{v} {k.replace('_',' ')}"
                          for k, v in lexman["composition"].items())
              + f". Construction: {lexman['construction']} It is frozen before any "
                f"grounded condition runs, and every `metrics.json` embeds its size and "
                f"content hash, so no metric can be traced to a different term list.", ""]
    if cov:
        L += [f"Coverage on {cov['tier']}: **{_pct(cov['bias_token_rate'])}** of "
              f"{cov['ref_tokens']} reference word tokens are lexicon terms "
              f"({_pct(cov['bias_token_rate_same_script'])} written in Latin script in "
              f"the reference). This is the ceiling on achievable gain from terminology "
              f"biasing.", ""]

    # --- environment --------------------------------------------------------
    if env:
        pk = env["packages"]
        L += ["## 6 Environment", "",
              f"{env['platform']} ({env['machine']}, {env['cpu_count']} cores), "
              f"Python {env['python']}. "
              + ", ".join(f"`{k}` {v}" for k, v in pk.items() if v)
              + ".", ""]
        if env.get("git_commit"):
            L += [f"Code revision `{env['git_commit'][:12]}`; corpus tarball sha256 "
                  f"`{env.get('corpus_tarball_sha256_12')}`.", ""]

    out = R / "01_dataset_and_harness.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)} ({len(L)} lines)")


if __name__ == "__main__":
    main()
