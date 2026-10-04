"""Bounded streaming upload writes and cleanup of partial files."""
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import settings


def save_upload(file: UploadFile, dest: Path) -> None:
    total = 0
    try:
        with dest.open("wb") as out:
            while chunk := file.file.read(1 << 20):
                total += len(chunk)
                if total > settings.max_upload_mb * (1 << 20):
                    raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB upload limit")
                out.write(chunk)
        if not total:
            raise HTTPException(400, "Uploaded file is empty")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
