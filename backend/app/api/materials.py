from __future__ import annotations

import pathlib
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.ingest import materials as materials_ingest
from app.jobs import queue
from app.ingest.uploads import save_upload
from app.models import Material, Subject
from app.rag import store
from app.schemas import JobOut, MaterialOut, MaterialPreviewOut

router = APIRouter(tags=["materials"])


def _get_subject(db: Session, subject_id: str) -> Subject:
    subject = db.get(Subject, subject_id)
    if subject is None:
        raise HTTPException(404, "subject not found")
    return subject


@router.post("/subjects/{subject_id}/materials", response_model=list[JobOut], status_code=202)
def upload_materials(
    subject_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> list[JobOut]:
    """Upload PDFs, images, or text docs into the subject's local RAG index."""
    subject = _get_subject(db, subject_id)
    created: list[tuple[str, str]] = []
    dest_dir = settings.materials_dir / subject.id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Validate the entire batch before writing any file or row.
    for file in files:
        if materials_ingest.classify(file.filename or "upload") is None:
            raise HTTPException(400, f"unsupported material {file.filename!r}; expected PDF, image, TXT, MD, CSV, DOCX or PPTX")

    paths: list[pathlib.Path] = []
    try:
        for file in files:
            filename = file.filename or "upload"
            kind = materials_ingest.classify(filename)
            suffix = pathlib.Path(filename).suffix.lower()
            dest = dest_dir / f"{uuid.uuid4().hex}{suffix}"
            paths.append(dest)
            save_upload(file, dest)
            mime = materials_ingest.mime_for(filename, file.content_type)
            material = Material(
                subject_id=subject.id, kind=kind, title=pathlib.Path(filename).stem,
                original_filename=filename, path=str(dest), mime=mime, status="uploaded",
            )
            db.add(material)
            db.flush()
            created.append((material.id, str(dest)))
        db.commit()
    except BaseException:
        db.rollback()
        for path in paths:
            path.unlink(missing_ok=True)
        raise

    jobs: list[JobOut] = []
    for material_id, dest in created:
        job = queue.enqueue(
            "ingest_material", subject_id=subject.id,
            payload={"material_id": material_id, "source_path": dest},
        )
        jobs.append(JobOut.of(job))
    return jobs


@router.get("/subjects/{subject_id}/materials", response_model=list[MaterialOut])
def list_materials(subject_id: str, db: Session = Depends(get_db)) -> list[MaterialOut]:
    _get_subject(db, subject_id)
    rows = db.scalars(
        select(Material)
        .where(Material.subject_id == subject_id)
        .order_by(Material.created_at.desc())
    ).all()
    return [MaterialOut.model_validate(r) for r in rows]


@router.get("/materials/{material_id}", response_model=MaterialPreviewOut)
def get_material_preview(material_id: str, db: Session = Depends(get_db)) -> MaterialPreviewOut:
    material = db.get(Material, material_id)
    if material is None:
        raise HTTPException(404, "material not found")
    return MaterialPreviewOut.model_validate(material)


@router.get("/materials/{material_id}/file")
def get_material_file(material_id: str, db: Session = Depends(get_db)) -> FileResponse:
    material = db.get(Material, material_id)
    if material is None:
        raise HTTPException(404, "material not found")
    path = pathlib.Path(material.path)
    if not path.exists():
        raise HTTPException(404, "file missing on disk")
    return FileResponse(
        path,
        media_type=material.mime or "application/octet-stream",
        filename=material.original_filename or path.name,
        content_disposition_type="inline",
    )


@router.delete("/materials/{material_id}", status_code=204, response_model=None)
def delete_material(material_id: str, db: Session = Depends(get_db)) -> None:
    material = db.get(Material, material_id)
    if material is None:
        raise HTTPException(404, "material not found")
    if queue.active_for(db, material_id=material_id):
        raise HTTPException(409, "cancel or finish active jobs before deleting this material")
    store.delete_material(material.id)
    pathlib.Path(material.path).unlink(missing_ok=True)
    db.delete(material)
    db.commit()
