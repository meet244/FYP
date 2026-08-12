"""Write the §4.2 / §4.3 pilot decisions into configs/config.yaml.

"Record this decision explicitly in the paper's experimental setup section" (§4.2) and
"Fix the setting for all subsequent experiments and report which was chosen and why"
(§4.3). The config file *is* the frozen setup, so the decisions belong in it rather than
in a human's memory of what the pilot said.

Only two keys are touched — `model.size` and `decode.language` — by line-targeted edit,
so every comment in the file survives. A provenance comment is appended to each line.
Changing `model.size` invalidates every cached decode made with the other model, which is
correct: they are decodes of a different system.

Prints a diff and exits non-zero if a pilot has not been run, so an unattended chain
stops rather than silently proceeding on defaults.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from common import ROOT, load_config, read_json

CFG = ROOT / "configs" / "config.yaml"


def _set_scalar(text: str, section: str, key: str, value: str,
                note: str) -> tuple[str, str | None]:
    """Replace `key: value` inside `section:` keeping any trailing comment slot."""
    lines = text.splitlines()
    in_section = False
    for i, line in enumerate(lines):
        if re.match(rf"^{re.escape(section)}:\s*$", line):
            in_section = True
            continue
        if in_section:
            if line and not line.startswith((" ", "\t", "#")):
                break                      # left the section
            m = re.match(rf"^(\s+){re.escape(key)}:\s*([^#\n]*?)\s*(#.*)?$", line)
            if m:
                indent, old = m.group(1), m.group(2).strip()
                if old == value:
                    return text, None
                lines[i] = f"{indent}{key}: {value}   # {note}"
                return "\n".join(lines) + "\n", f"{section}.{key}: {old} -> {value}"
    raise KeyError(f"{section}.{key} not found in {CFG}")


def main() -> int:
    text = CFG.read_text(encoding="utf-8")
    changes: list[str] = []
    missing: list[str] = []

    pl = ROOT / "report" / "pilot_language.json"
    if pl.exists():
        d = read_json(pl)
        chosen = d["decision"]
        best = d["results"][chosen]
        text, ch = _set_scalar(
            text, "decode", "language", chosen,
            f"fixed by the §4.3 pilot on tier1: WER {best['wer']:.4f} "
            f"(lowest of {', '.join(d['results'])})")
        if ch:
            changes.append(ch)
    else:
        missing.append("report/pilot_language.json (run: python src/pilots.py language)")

    pm = ROOT / "report" / "pilot_model.json"
    if pm.exists():
        d = read_json(pm)
        decision = d["decision"]
        # "mixed" means: run the matrix on turbo and confirm on large-v3 (§4.2).
        size = "large-v3-turbo" if decision in ("turbo", "mixed") else "large-v3"
        note = (f"fixed by the §4.2 pilot: decision '{decision}', "
                f"ΔWER(turbo−large-v3) {d['wer_delta_turbo_minus_largev3']:+.4f}, "
                f"turbo {d['turbo_speedup']}x faster")
        text, ch = _set_scalar(text, "model", "size", size, note)
        if ch:
            changes.append(ch)
    else:
        missing.append("report/pilot_model.json (run: python src/pilots.py model)")

    if missing:
        print("cannot fix the configuration — these pilots have not been run:")
        for m in missing:
            print(f"  - {m}")
        return 1

    if changes:
        CFG.write_text(text, encoding="utf-8")
        print("configs/config.yaml updated from the pilots:")
        for c in changes:
            print(f"  {c}")
        print("\nNote: a change to model.size invalidates cached decodes made with the "
              "other model — they are decodes of a different system.")
    else:
        print("configuration already matches both pilot decisions; nothing to change.")

    cfg = load_config()
    print(f"\nfrozen setup: model={cfg['model']['size']} "
          f"({cfg['model']['compute_type']}), language={cfg['decode']['language']}, "
          f"beam={cfg['decode']['beam_size']}, "
          f"temperature={cfg['decode']['temperature']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
