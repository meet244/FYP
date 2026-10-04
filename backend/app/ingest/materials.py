"""Extract text from PDFs, images, and documents for the local multimodal index.

Voice is handled by the lecture pipeline. This module is the other three
modalities: PDF page text, image OCR/caption, and plain documents. Chunks go
into Chroma; the original file stays on disk so the UI can preview it.
"""
from __future__ import annotations

import logging
import pathlib
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from pypdf import PdfReader

from app.llm.client import describe_image

log = logging.getLogger(__name__)

KIND_BY_SUFFIX = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".txt": "doc",
    ".md": "doc",
    ".csv": "doc",
    ".docx": "doc",
    ".pptx": "doc",
}

MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


def classify(filename: str) -> str | None:
    return KIND_BY_SUFFIX.get(pathlib.Path(filename).suffix.lower())


def mime_for(filename: str, fallback: str | None = None) -> str:
    return MIME_BY_SUFFIX.get(pathlib.Path(filename).suffix.lower()) or fallback or "application/octet-stream"


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split extracted text into overlapping retrieval windows."""
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + size)
        if end < n:
            snap = text.rfind(" ", i + size // 2, end)
            if snap != -1:
                end = snap
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return chunks


def extract_pdf(path: pathlib.Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = (page.extract_text() or "").strip()
        if raw:
            pages.append(f"[page {i}]\n{raw}")
    return "\n\n".join(pages)


def extract_doc(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".docx":
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return "\n\n".join(
            "".join(element.text or "" for element in paragraph.findall(".//w:t", ns))
            for paragraph in root.findall(".//w:p", ns)
        )
    if path.suffix.lower() == ".pptx":
        # Follow the presentation's relationship order, not slide filenames.
        with ZipFile(path) as archive:
            presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
            relationships = ET.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))
            targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships
                       if rel.attrib.get("TargetMode") != "External"}
            p = "http://schemas.openxmlformats.org/presentationml/2006/main"
            r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            a = "http://schemas.openxmlformats.org/drawingml/2006/main"
            slides = []
            for number, slide in enumerate(presentation.findall(f".//{{{p}}}sldId"), 1):
                target = targets[slide.attrib[f"{{{r}}}id"]]
                part = target.lstrip("/") if target.startswith("/") else str(pathlib.PurePosixPath("ppt") / target)
                root = ET.fromstring(archive.read(part))
                text = "\n".join(
                    "".join(node.text or "" for node in paragraph.findall(f".//{{{a}}}t"))
                    for paragraph in root.findall(f".//{{{a}}}p")
                ).strip()
                if text:
                    slides.append(f"[slide {number}]\n{text}")
            return "\n\n".join(slides)
    return path.read_text(encoding="utf-8", errors="replace")


def extract_image(path: pathlib.Path, mime: str) -> str:
    # An unavailable OCR service is retryable; a filename is not extracted content.
    return describe_image(path.read_bytes(), mime)


def extract(kind: str, path: pathlib.Path, mime: str) -> str:
    if kind == "pdf":
        return extract_pdf(path)
    if kind == "doc":
        return extract_doc(path)
    if kind == "image":
        return extract_image(path, mime)
    raise ValueError(f"unknown material kind: {kind!r}")
