#!/usr/bin/env python3
"""Load the ITC604 fixtures in test/ into a running ClassScribe API.

    CLASSSCRIBE_API=http://127.0.0.1:8002 backend/.venv/bin/python test/load.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = os.environ.get("CLASSSCRIBE_API", "http://127.0.0.1:8002").rstrip("/")

MATERIALS = [
    ROOT / "slides" / "L01-Perceptrons-and-MLPs.pdf",
    ROOT / "readings" / "perceptron-learning-handout.pdf",
    ROOT / "readings" / "activation-functions.md",
    ROOT / "readings" / "xor-lab-notes.txt",
    ROOT / "media" / "perceptron-diagram.png",
    ROOT / "media" / "mlp-architecture.png",
    ROOT / "media" / "xor-decision-regions.png",
]

LECTURES = [
    (ROOT / "voice" / "L01-perceptrons-en.mp3", "Lecture 1 - Perceptrons and XOR"),
    (ROOT / "voice" / "L01-perceptrons-hi.mp3", "Lecture 1 (Hindi) - Perceptron"),
]

SYLLABUS = ROOT / "syllabus" / "ITC604-Neural-Networks-Syllabus.pdf"


def api(method: str, path: str, data: bytes | None = None, content_type: str | None = None):
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={"Accept": "application/json"},
    )
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            raw = res.read()
            return res.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"{method} {path} -> {exc.code}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot reach {API} ({exc.reason}). Start the backend first."
        ) from exc


def curl_upload(path: str, fields: list[tuple[str, str]], files: list[tuple[str, Path]]) -> dict:
    cmd = ["curl", "-sS", "-X", "POST", f"{API}{path}"]
    for name, value in fields:
        cmd.extend(["-F", f"{name}={value}"])
    for name, file in files:
        cmd.extend(["-F", f"{name}=@{file}"])
    import subprocess

    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def wait_job(job: dict, timeout_s: float = 600) -> dict:
    job_id = job["id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        _, body = api("GET", f"/jobs/{job_id}")
        status = body["status"]
        stage = body.get("stage") or ""
        msg = body.get("message") or ""
        print(f"  job {job_id[:8]}  {status:10} {stage} {msg}".rstrip())
        if status == "succeeded":
            return body
        if status == "failed":
            raise SystemExit(f"job failed: {body.get('error')}")
        time.sleep(1.5)
    raise SystemExit(f"job {job_id} timed out after {timeout_s:.0f}s")


def main() -> None:
    _, health = api("GET", "/health")
    if not health or health.get("status") != "ok":
        raise SystemExit(f"API unhealthy: {health}")
    print(f"API {API}  llm={health.get('llm_model')}  asr={health['asr']['backend']}")

    _, subjects = api("GET", "/subjects")
    subject = next((s for s in subjects if s.get("code") == "ITC604"), None)
    if subject is None:
        payload = json.dumps(
            {
                "name": "Neural Networks",
                "code": "ITC604",
                "description": "ClassScribe test subject — perceptrons, XOR, and multilayer nets.",
            }
        ).encode()
        _, subject = api("POST", "/subjects", data=payload, content_type="application/json")
        print(f"created subject {subject['id']}  {subject['name']}")
    else:
        print(f"using existing subject {subject['id']}  {subject['name']}")

    sid = subject["id"]

    if not subject.get("has_syllabus"):
        print("uploading syllabus")
        job = curl_upload(f"/subjects/{sid}/syllabus", [], [("file", SYLLABUS)])
        wait_job(job, timeout_s=180)
    else:
        print("syllabus already present")

    _, existing_mats = api("GET", f"/subjects/{sid}/materials")
    have = {m.get("original_filename") for m in existing_mats}
    missing = [p for p in MATERIALS if p.name not in have]
    if missing:
        print(f"uploading {len(missing)} materials")
        files = [("files", p) for p in missing]
        jobs = curl_upload(f"/subjects/{sid}/materials", [], files)
        if isinstance(jobs, dict):
            jobs = [jobs]
        for job in jobs:
            wait_job(job, timeout_s=180)
    else:
        print("materials already present")

    _, existing_lecs = api("GET", f"/subjects/{sid}/lectures")
    lec_titles = {l.get("title") for l in existing_lecs}
    for path, title in LECTURES:
        if title in lec_titles:
            print(f"lecture already present: {title}")
            continue
        print(f"uploading lecture {title}")
        job = curl_upload(
            f"/subjects/{sid}/lectures",
            [("title", title)],
            [("file", path)],
        )
        wait_job(job, timeout_s=900)

    _, subject = api("GET", f"/subjects/{sid}")
    print()
    print(f"done  {subject['name']} ({subject['code']})")
    print(f"  lectures {subject['lecture_count']}  files {subject['material_count']}")
    print(f"  open  http://localhost:3000/subjects/{sid}")


if __name__ == "__main__":
    sys.exit(main())
