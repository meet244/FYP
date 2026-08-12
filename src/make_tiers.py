"""Freeze the three evaluation tiers (§3.3).

  Tier 1 (dev)        ~200 utterances   tuning: thresholds, prompt format, retrieval
                                        depth, debugging
  Tier 2 (eval subset) ~800 utterances  the full experiment matrix and all ablations
  Tier 3 (full test)   5.18 h           final confirmation, two systems only

Tier 1 and Tier 2 are drawn as **non-overlapping** samples from the test set, so every
threshold is tuned exclusively on Tier 1 and every reported result comes from Tier 2 or
Tier 3. Sampling happens once, under the seed recorded in configs/config.yaml, and the
resulting manifests are committed artefacts — later runs read the frozen files rather
than re-sampling.

Sampling is stratified by recording so that all lecture topics appear in both tiers in
roughly their corpus proportions; a tier that accidentally omitted a topic would make
retrieval accuracy meaningless.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from common import ROOT, load_config, read_jsonl, write_json, write_jsonl


def stratified_sample(rows: list[dict], n: int, rng: random.Random,
                      key: str = "rec") -> tuple[list[dict], list[dict]]:
    """Take `n` rows spread proportionally across `key` groups. Returns (taken, rest)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get(key) or "_"].append(r)
    for g in groups.values():
        rng.shuffle(g)

    order = sorted(groups, key=lambda g: (-len(groups[g]), g))
    quota = {g: max(1, round(n * len(groups[g]) / len(rows))) for g in order}
    taken: list[dict] = []
    for g in order:                       # round-robin honouring each group's quota
        take = groups[g][:quota[g]]
        taken.extend(take)
        groups[g] = groups[g][quota[g]:]
    rng.shuffle(taken)
    if len(taken) > n:
        returned = taken[n:]
        taken = taken[:n]
        for r in returned:
            groups[r.get(key) or "_"].append(r)
    else:                                  # top up from the largest remaining groups
        pool = [r for g in order for r in groups[g]]
        rng.shuffle(pool)
        need = n - len(taken)
        taken.extend(pool[:need])
        remaining_ids = {id(r) for r in pool[:need]}
        for g in order:
            groups[g] = [r for r in groups[g] if id(r) not in remaining_ids]
    rest = [r for g in order for r in groups[g]]
    return taken, rest


def _dump(rows: list[dict], path: Path, label: str) -> dict:
    rows = sorted(rows, key=lambda r: r["utt_id"])
    write_jsonl(path, rows)
    secs = sum(r["duration"] for r in rows)
    stat = {"tier": label, "manifest": str(path.relative_to(ROOT)),
            "n_utts": len(rows), "seconds": round(secs, 1),
            "minutes": round(secs / 60, 1), "hours": round(secs / 3600, 3),
            "n_recordings": len({r["rec"] for r in rows})}
    print(f"{label:6s} {len(rows):5d} utts  {secs/60:7.1f} min  "
          f"{stat['n_recordings']:3d} recordings -> {path.relative_to(ROOT)}")
    return stat


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-sample even if the tier manifests already exist")
    a = ap.parse_args()

    full = ROOT / cfg["data"]["manifest"]
    t1 = ROOT / cfg["data"]["tiers"]["tier1"]
    t2 = ROOT / cfg["data"]["tiers"]["tier2"]
    seed = cfg["data"]["seed"]

    if t1.exists() and t2.exists() and not a.force:
        print(f"tiers already frozen ({t1.name}, {t2.name}); use --force to re-sample")
        return

    rows = read_jsonl(full)
    rng = random.Random(seed)
    tier1, rest = stratified_sample(rows, cfg["data"]["tier1_size"], rng)
    tier2, _ = stratified_sample(rest, cfg["data"]["tier2_size"], rng)

    ids1, ids2 = {r["utt_id"] for r in tier1}, {r["utt_id"] for r in tier2}
    assert not (ids1 & ids2), "Tier 1 and Tier 2 must be disjoint (§3.3)"

    stats = {
        "seed": seed,
        "source_manifest": str(full.relative_to(ROOT)),
        "source_n_utts": len(rows),
        "disjoint": True,
        "tiers": [
            _dump(tier1, t1, "tier1"),
            _dump(tier2, t2, "tier2"),
            _dump(rows, ROOT / cfg["data"]["tiers"]["tier3"], "tier3"),
        ],
        "statement_for_report": (
            f"All development and ablation experiments use a fixed random subset of "
            f"{len(tier2)} utterances (seed {seed}); hyperparameters were selected on a "
            f"disjoint development slice of {len(tier1)} utterances; final systems are "
            f"additionally evaluated on the complete Hindi-English test set."),
    }
    write_json(ROOT / "data" / "manifests" / "tiers.json", stats)
    print("\n" + stats["statement_for_report"])


if __name__ == "__main__":
    main()
