"""Syllabus scaffolding and validation (plan §2.2-2.3).

    python src/make_syllabi.py --list                # courses still missing a syllabus
    python src/make_syllabi.py --template linux      # write a stub + the Tier-B prompt
    python src/make_syllabi.py --validate            # check every syllabus against §2.3

Tier A = real published outline (record source_url). Tier B = generated from the
course/lecture TITLE ONLY, never from audio or reference transcripts. Tier C =
oracle, allowed only for the clearly-labelled topline.
"""
import argparse
import json

from config import OUT, SYL
from courses import course_of
from normalize import script_mix
from prompts import n_tokens

TIER_B_PROMPT = """You are writing a realistic course syllabus for a spoken technical
tutorial series titled "{title}". You have NOT seen any audio or transcript — write
only from the title and general knowledge of the subject.

Produce 6-12 units. For each unit:
  - "title": short English unit title
  - "prose": 45-70 tokens of FLOWING NARRATION in Hindi-English code-mixed register,
    exactly as a tutorial narrator would speak it: Hindi function words in Devanagari,
    English technical terms in Latin script, second person, present/future tense
    ("इस tutorial में हम ... देखेंगे")। NOT bullets, NOT a keyword list.
    Put the densest, most distinctive technical terms at the END of the prose —
    later prompt tokens carry more influence on Whisper's decoder.
  - "keywords": 6-10 technical terms, in the script a narrator would actually use.

Return JSON: {{"course_id": "{cid}", "title": "{title}",
"provenance": "generated-from-title", "source_url": null, "units": [...]}}"""

STUB_UNIT = dict(
    unit_id="{cid}-u01",
    title="TODO unit title",
    prose="इस tutorial में हम TODO देखेंगे।",
    keywords=["TODO"],
)


def courses_needed():
    rows = [json.loads(l) for l in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted({course_of(r["lecture_id"]) for r in rows if r.get("in_eval")})


def validate_one(path):
    c = json.loads(path.read_text(encoding="utf-8"))
    issues = []
    if c.get("provenance") not in ("real", "generated-from-title", "oracle"):
        issues.append(f"bad provenance {c.get('provenance')!r}")
    if c.get("provenance") == "real" and not c.get("source_url"):
        issues.append("provenance='real' but no source_url")
    units = c.get("units", [])
    if not 6 <= len(units) <= 12:
        issues.append(f"{len(units)} units (want 6-12)")
    ids = set()
    for u in units:
        t = n_tokens(u["prose"])
        if not 60 <= t <= 105:
            issues.append(f"{u['unit_id']}: prose {t} Whisper tokens (want 60-105, so k=2 units fit the 200-token cap)")
        if not 5 <= len(u["keywords"]) <= 12:
            issues.append(f"{u['unit_id']}: {len(u['keywords'])} keywords (want 6-10)")
        if u["unit_id"] in ids:
            issues.append(f"duplicate unit_id {u['unit_id']}")
        ids.add(u["unit_id"])
        m = script_mix(u["prose"])
        if m["dev"] < 0.25 or m["lat"] < 0.15:
            issues.append(
                f"{u['unit_id']}: script mix dev={m['dev']*100:.0f}% lat={m['lat']*100:.0f}%"
                " — prose should be genuinely code-mixed"
            )
    return c, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--template", metavar="COURSE_ID")
    ap.add_argument("--title", default=None, help="human title for --template")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    if a.list:
        need = courses_needed()
        have = {p.stem for p in SYL.glob("*.json")} - {"lecture_map", "lecture_titles"}
        print(f"courses in eval set: {len(need)}")
        for c in need:
            print(f"   [{'ok ' if c in have else 'MISSING'}] syllabi/{c}.json")
        extra = have - set(need)
        if extra:
            print(f"unused syllabi (fine — they widen the mismatched pool): {sorted(extra)}")

    if a.template:
        cid = a.template
        title = a.title or cid.replace("-", " ").title()
        path = SYL / f"{cid}.json"
        if path.exists():
            print(f"{path} already exists — not overwriting")
        else:
            stub = dict(
                course_id=cid,
                title=title,
                provenance="generated-from-title",
                source_url=None,
                units=[{**STUB_UNIT, "unit_id": f"{cid}-u01"}],
            )
            path.write_text(json.dumps(stub, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"wrote stub {path}")
        print("\n--- Tier-B generation prompt ---\n")
        print(TIER_B_PROMPT.format(cid=cid, title=title))

    if a.validate:
        ok = True
        for p in sorted(SYL.glob("*.json")):
            if p.stem in ("lecture_map", "lecture_titles"):
                continue
            c, issues = validate_one(p)
            toks = [n_tokens(u["prose"]) for u in c["units"]]
            print(f"{p.name:28} {len(c['units']):3} units  prose tokens "
                  f"min/med/max={min(toks)}/{sorted(toks)[len(toks)//2]}/{max(toks)}  "
                  f"provenance={c['provenance']}")
            for i in issues:
                ok = False
                print(f"    !! {i}")
        print("\nall syllabi valid" if ok else "\nfix the issues above before decoding")


if __name__ == "__main__":
    main()
