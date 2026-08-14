"""Lecture -> course mapping, syllabus loading, and the leakage guard.

The mapping lives in syllabi/lecture_map.json (written by map_lectures.py after
Step 0), so no code has to be edited when the corpus ID format is discovered.
Format: {"<lecture_id>": "<course_id>", ...} plus optional "_default" rule.
"""
import functools
import hashlib
import json
import re

from config import SYL, ORACLE_CONDITIONS

MAP_PATH = SYL / "lecture_map.json"


@functools.lru_cache(maxsize=1)
def _lecture_map():
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text(encoding="utf-8"))
    return {}


def course_of(lecture_id: str) -> str:
    """Explicit map wins; otherwise fall back to the leading ID field."""
    m = _lecture_map()
    if lecture_id in m:
        return m[lecture_id]
    for pat, cid in m.get("_regex", {}).items():
        if re.search(pat, lecture_id):
            return cid
    return re.split(r"[_\-.]", lecture_id)[0].lower()


@functools.lru_cache(maxsize=None)
def _load(cid: str):
    path = SYL / f"{cid}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no syllabus for course '{cid}' ({path}). "
            "Run map_lectures.py and write the syllabus JSON first."
        )
    c = json.loads(path.read_text(encoding="utf-8"))
    for field in ("course_id", "title", "provenance", "units"):
        if field not in c:
            raise ValueError(f"{path} missing required field '{field}'")
    for u in c["units"]:
        for field in ("unit_id", "title", "prose", "keywords"):
            if field not in u:
                raise ValueError(f"{path} unit missing '{field}'")
    return c


@functools.lru_cache(maxsize=None)
def all_course_ids():
    return tuple(sorted(p.stem for p in SYL.glob("*.json") if p.stem not in ("lecture_map", "lecture_titles")))


def mismatched_id(cid: str) -> str:
    """Deterministic pairing with a DIFFERENT course (stable across processes:
    Python's hash() is salted per-run, md5 is not)."""
    others = [c for c in all_course_ids() if c != cid]
    if not others:
        raise ValueError("mismatched control needs >= 2 courses in syllabi/")
    h = int(hashlib.md5(cid.encode("utf-8")).hexdigest(), 16)
    return others[h % len(others)]


def get_course(cid: str, condition: str):
    """Load the syllabus this condition should see, then enforce the leakage rule."""
    course = _load(mismatched_id(cid) if condition == "C5" else cid)
    assert_leakage_free(course, condition)
    return course


def assert_leakage_free(course, condition):
    prov = course.get("provenance")
    if prov == "oracle" and condition not in ORACLE_CONDITIONS:
        raise AssertionError(
            f"LEAKAGE: course '{course['course_id']}' has provenance='oracle' but "
            f"condition {condition} is a scored (non-topline) condition"
        )
    if prov not in ("real", "generated-from-title", "oracle"):
        raise AssertionError(
            f"course '{course['course_id']}' has unrecognised provenance '{prov}'"
        )
