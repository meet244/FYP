"""Step 0 — DISCOVERY. Report the real layout of the extracted SLR104 test dir.

Prints exactly what Part 1.3 of the plan asks you to verify, so the answers can
be pasted straight into RUNLOG.md. Read-only; changes nothing.
"""
import collections
import re
import sys

from config import DATA


def head(path, n=5):
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            return [next(f).rstrip("\n") for _ in range(n)]
    except StopIteration:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:  # noqa: BLE001
        return [f"<unreadable: {e}>"]


def main():
    if not DATA.exists():
        sys.exit(f"{DATA} does not exist — download and extract the tarball first")

    print("=== directories (depth<=4) ===")
    for d in sorted(p for p in DATA.rglob("*") if p.is_dir() and len(p.relative_to(DATA).parts) <= 4)[:50]:
        print("  ", d.relative_to(DATA))

    print("\n=== non-wav files (depth<=4) ===")
    for p in sorted(
        q for q in DATA.rglob("*")
        if q.is_file() and q.suffix not in (".wav", ".gz") and len(q.relative_to(DATA).parts) <= 4
    )[:50]:
        print(f"   {p.relative_to(DATA)}  ({p.stat().st_size} bytes)")

    for name in ("text", "segments", "wav.scp", "utt2spk", "spk2utt"):
        hits = list(DATA.rglob(name))
        print(f"\n=== {name} ({len(hits)} found) ===")
        for h in hits[:3]:
            n = sum(1 for _ in h.open(encoding="utf-8", errors="replace"))
            print(f"--- {h.relative_to(DATA)}  ({n} lines)")
            for line in head(h):
                print("   ", line)

    wavs = list(DATA.rglob("*.wav"))
    print(f"\n=== wav files: {len(wavs)} ===")
    for w in wavs[:5]:
        print("   ", w.relative_to(DATA))

    # Utterance-ID prefix structure: this is the candidate lecture grouping.
    texts = list(DATA.rglob("text"))
    if texts:
        utts = [ln.split(" ", 1)[0] for ln in texts[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
        print(f"\n=== utterance IDs: {len(utts)} ===")
        for u in utts[:10]:
            print("   ", u)
        for pat, label in ((r"^(.*)_[^_]+$", "strip last _field"), (r"^([^_]+)", "first _field")):
            groups = collections.Counter(re.match(pat, u).group(1) for u in utts if re.match(pat, u))
            print(f"\n  grouping by [{label}]: {len(groups)} groups; 10 largest:")
            for g, c in groups.most_common(10):
                print(f"     {c:5d}  {g}")

    print("\nPaste the above into RUNLOG.md before proceeding to build_manifest.py.")


if __name__ == "__main__":
    main()
