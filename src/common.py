"""Shared plumbing: config loading, JSONL I/O, deterministic hashing, paths.

Kept deliberately small. Everything that decides *what the model computes* lives in
configs/config.yaml so that a single file defines the frozen experimental setup (§13).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


class Config(dict):
    """dict with dotted access: cfg['model']['size'] or cfg.get_path('model.size')."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


# --- JSONL -------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --- hashing -----------------------------------------------------------------

def stable_hash(obj: Any, n: int = 12) -> str:
    """Hash of a JSON-serialisable object, stable across processes and runs."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:n]


def file_hash(path: str | Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# --- run layout --------------------------------------------------------------

def run_dir(name: str, tier: str) -> Path:
    d = ROOT / "runs" / tier / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_for_tier(cfg: Config, tier: str) -> Path:
    p = cfg.get_path(f"data.tiers.{tier}")
    if p is None:
        raise KeyError(f"unknown tier {tier!r}; known: {list(cfg['data']['tiers'])}")
    return ROOT / p
