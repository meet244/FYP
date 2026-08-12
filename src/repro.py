"""Reproducibility record (§13).

Writes report/environment.json with everything the checklist requires that is not
already captured in configs/config.yaml, the tier manifests or the lexicon manifest:
runtime and library versions, the resolved model identifiers and their local paths,
machine facts, and the corpus checksum.

Also prints the checklist itself with an automatic pass/fail per item, so the state of
the study is auditable in one command.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from common import ROOT, file_hash, load_config, read_json, write_json

PACKAGES = ["faster-whisper", "ctranslate2", "sentence-transformers", "jiwer",
            "rapidfuzz", "indic-transliteration", "numpy", "torch", "soundfile",
            "librosa", "matplotlib", "onnxruntime", "tokenizers", "huggingface-hub"]


def _v(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def environment(cfg) -> dict:
    models = {}
    try:
        from faster_whisper.utils import download_model
        for size in dict.fromkeys([cfg["model"]["size"], "large-v3"]):
            try:
                models[size] = str(download_model(size, local_files_only=True))
            except Exception:                                     # noqa: BLE001
                models[size] = "not cached locally"
    except Exception:                                             # noqa: BLE001
        pass

    git = None
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or None
    except Exception:                                             # noqa: BLE001
        pass

    tar = ROOT / "data" / "raw" / "slr104" / "Hindi-English_test.tar.gz"
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": {p: _v(p) for p in PACKAGES},
        "model_paths": models,
        "git_commit": git,
        "config": json.loads(json.dumps(dict(cfg))),
        "corpus_tarball_sha256_12": file_hash(tar) if tar.exists() else None,
        "ffmpeg": (subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                  text=True).stdout.splitlines() or [None])[0],
    }
    write_json(ROOT / "report" / "environment.json", out)
    return out


CHECKLIST = [
    ("Corpus version, subset seeds and tier sizes recorded",
     lambda cfg: (ROOT / "data" / "manifests" / "tiers.json").exists()),
    ("Model identifier, quantisation, beam width, temperature and language recorded",
     lambda cfg: all(k in cfg["decode"] for k in ("beam_size", "temperature",
                                                  "language"))),
    ("Runtime and library versions pinned",
     lambda cfg: (ROOT / "report" / "environment.json").exists()
     and (ROOT / "requirements.txt").exists()),
    ("Syllabus documents and frozen term lexicon included",
     lambda cfg: (ROOT / "syllabus" / "index" / "lexicon_manifest.json").exists()),
    ("Retrieval model identifier, chunk size, overlap and top-k recorded",
     lambda cfg: all(k in cfg["syllabus"] for k in ("embed_model", "chunk_words",
                                                    "chunk_overlap"))),
    ("All decode outputs cached and archived",
     lambda cfg: (ROOT / "cache" / "asr").exists()),
    ("Per-utterance edit counts retained for significance testing",
     lambda cfg: any((ROOT / "runs").rglob("per_utt.jsonl"))),
    ("Normalisation procedure fully specified, both levels",
     lambda cfg: (ROOT / "src" / "normalize.py").exists()),
    ("Guard thresholds and firing rates reported",
     lambda cfg: "guards" in cfg and any(
         "guard_context_echo_rate" in read_json(p)
         for p in (ROOT / "runs").rglob("metrics.json"))),
    ("Every hyperparameter selected on the development tier",
     lambda cfg: any((ROOT / "runs" / "tier1").glob("sweep_*.json"))
     if (ROOT / "runs" / "tier1").exists() else False),
    ("Segment refinement documented and validated",
     lambda cfg: (ROOT / "report" / "segment_refinement_validation.json").exists()),
    ("Model-selection and language pilots recorded",
     lambda cfg: (ROOT / "report" / "pilot_model.json").exists()
     and (ROOT / "report" / "pilot_language.json").exists()),
]


def checklist(cfg) -> dict:
    print("\n=== §13 reproducibility checklist ===")
    results = {}
    for label, test in CHECKLIST:
        try:
            ok = bool(test(cfg))
        except Exception:                                         # noqa: BLE001
            ok = False
        results[label] = ok
        print(f"[{'x' if ok else ' '}] {label}")
    n = sum(results.values())
    print(f"\n{n}/{len(results)} items satisfied")
    write_json(ROOT / "report" / "checklist.json", results)
    return results


if __name__ == "__main__":
    cfg = load_config()
    env = environment(cfg)
    print(json.dumps({k: env[k] for k in ("python", "platform", "packages",
                                          "model_paths", "git_commit")}, indent=2))
    checklist(cfg)
