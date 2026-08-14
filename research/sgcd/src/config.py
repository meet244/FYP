"""Central paths and frozen constants for SGCD.

Everything is relative to SGCD_ROOT (defaults to the repo root, i.e. the parent
of src/). Override with the SGCD_ROOT environment variable if you move the data.
"""
import os
import pathlib

ROOT = pathlib.Path(os.environ.get("SGCD_ROOT", pathlib.Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
SYL = ROOT / "syllabi"
OUT = ROOT / "out"
HYP = OUT / "hyps"
TABLES = OUT / "tables"

for _p in (OUT, HYP, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# ---- frozen experiment constants (see PREREGISTRATION.md) ----
SEED = 1337
MIN_DUR, MAX_DUR, MIN_WORDS = 2.0, 28.0, 4
DEV_LECTURE_FRAC = 0.30
N_DEV, N_TEST = 60, 150

MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "small": "mlx-community/whisper-small-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}

# Conditions that are toplines / oracles: allowed to touch reference-derived
# information. Every other condition is asserted leakage-free at decode time.
ORACLE_CONDITIONS = {"C6", "C8"}

CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

CONDITION_DOC = {
    "C0": "no prompt (baseline)",
    "C1": "generic code-mixed sentence, no course content (style control)",
    "C2": "syllabus keywords, comma-separated (naive baseline)",
    "C3": "whole-syllabus prose",
    "C4": "retrieved k units, prose (SGCD)",
    "C5": "retrieved from a different course (content-specificity control)",
    "C6": "retrieved using the reference (oracle retrieval topline)",
    "C7": "C4 + confidence guard (full system)",
}
