"""
ask_z/ingestion/doc_ingest.py
──────────────────────────────
Ingestion pipeline for external documents: PDF, PPTX, DOCX, and plain text.

Use this to ingest onboarding guides, access-request procedures, pod setup
instructions, architecture decks, and any other material that lives *outside*
the git repository.

Supported formats
──────────────────
  .pdf   — extracted with PyMuPDF (fitz); page-aware chunking
  .pptx  — per-slide text extraction via python-pptx
  .docx  — paragraph extraction via python-docx
  .txt / .md / .rst — plain text via SentenceSplitter

Metadata injected per chunk
────────────────────────────
  file_path       — path as supplied (relative or absolute)
  source_url      — same as file_path (override with --source-url)
  doc_type        — always "external_doc"
  component_tag   — inferred from filename or --tag argument
  last_updated    — file modification date (ISO-8601)
  content_hash    — MD5 of chunk text (delta key)
  chunk_index     — position within the source file
  version         — "external" (no git SHA)
  staleness_ttl_flag — False at ingest time

Usage
──────
  # Ingest a whole folder:
  python -m ask_z.scripts.ingest_docs --dir /path/to/docs

  # Ingest one file with a custom tag and source URL:
  python -m ask_z.scripts.ingest_docs \\
      --file /path/to/pod-setup.pdf \\
      --tag pod_setup \\
      --source-url https://ibm.box.com/s/abc123
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator

log = logging.getLogger("ask_z.ingestion.doc_ingest")

# Chunk size (characters) for page/slide text before sentence-splitting.
_CHUNK_CHARS = 1200
_OVERLAP_CHARS = 150


# ── Helpers ───────────────────────────────────────────────────────────────────


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _mtime_iso(path: Path) -> str:
    """Return ISO-8601 date of the file's last modification."""
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except OSError:
        return "unknown"


def _infer_tag(path: Path, override: str | None) -> str:
    """Infer a component tag from the filename stem or use the override."""
    if override:
        return override
    stem = path.stem.lower()
    for noise in (
        "guide",
        "doc",
        "document",
        "presentation",
        "slides",
        "v1",
        "v2",
        "final",
    ):
        stem = stem.replace(noise, "")
    stem = stem.strip("_- ").replace(" ", "_").replace("-", "_")
    return stem or path.stem


def _sliding_window(
    text: str, chunk_size: int = _CHUNK_CHARS, overlap: int = _OVERLAP_CHARS
) -> list[str]:
    """Split *text* into overlapping character-window chunks."""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            for sep in (". ", "? ", "! ", "\n\n", "\n"):
                pos = text.rfind(sep, start + chunk_size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = end - overlap

    return chunks


def _build_chunk(
    text: str,
    idx: int,
    *,
    file_path: Path,
    source_url: str,
    component_tag: str,
    last_updated: str,
) -> dict:
    """Build a metadata dict matching the Elasticsearch schema."""
    return {
        "text": text,
        "metadata": {
            "file_path": str(file_path),
            "source_url": source_url or str(file_path),
            "doc_type": "external_doc",
            "component_tag": component_tag,
            "last_updated": last_updated,
            "git_blame_author": "",
            "version": "external",
            "chunk_index": idx,
            "content_hash": _md5(text),
            "staleness_ttl_flag": False,
        },
    }


# ── Format extractors ─────────────────────────────────────────────────────────


def _extract_pdf(path: Path) -> list[str]:
    """Extract text from a PDF using PyMuPDF. Returns one string per page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.error("PyMuPDF not installed. Run: pip install pymupdf")
        return []

    pages: list[str] = []
    try:
        doc = fitz.open(str(path))
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                pages.append(text)
        doc.close()
    except Exception as exc:
        log.warning("Failed to extract PDF %s: %s", path, exc)
    return pages


def _extract_pptx(path: Path) -> list[str]:
    """Extract text from a PowerPoint file. Returns one string per slide."""
    try:
        from pptx import Presentation
    except ImportError:
        log.error("python-pptx not installed. Run: pip install python-pptx")
        return []

    slides: list[str] = []
    try:
        prs = Presentation(str(path))
        for i, slide in enumerate(prs.slides, start=1):
            parts: list[str] = []
            if slide.shapes.title and slide.shapes.title.text.strip():
                parts.append(f"[Slide {i}] {slide.shapes.title.text.strip()}")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = para.text.strip()
                        if line:
                            parts.append(line)
            text = "\n".join(parts).strip()
            if text:
                slides.append(text)
    except Exception as exc:
        log.warning("Failed to extract PPTX %s: %s", path, exc)
    return slides


def _extract_docx(path: Path) -> list[str]:
    """Extract text from a Word document. Returns one string per section."""
    try:
        import docx
    except ImportError:
        log.error("python-docx not installed. Run: pip install python-docx")
        return []

    sections: list[str] = []
    try:
        doc = docx.Document(str(path))
        current: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if para.style.name.startswith("Heading") and current:
                sections.append("\n".join(current))
                current = [text]
            else:
                current.append(text)
        if current:
            sections.append("\n".join(current))
    except Exception as exc:
        log.warning("Failed to extract DOCX %s: %s", path, exc)
    return sections


def _extract_text(path: Path) -> list[str]:
    """Read a plain text / Markdown / RST file as a single string."""
    try:
        return [path.read_text(encoding="utf-8", errors="replace")]
    except OSError as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return []


# ── Public API ─────────────────────────────────────────────────────────────────


def chunk_document(
    file_path: Path | str,
    *,
    component_tag: str | None = None,
    source_url: str = "",
) -> list[dict]:
    """
    Chunk a single external document into a list of dicts ready for embedding.

    Each dict has keys ``text`` and ``metadata`` matching the Elasticsearch schema.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        log.warning("File not found: %s", path)
        return []

    suffix = path.suffix.lower()
    tag = _infer_tag(path, component_tag)
    last_updated = _mtime_iso(path)
    src_url = source_url or str(path)

    if suffix == ".pdf":
        blocks = _extract_pdf(path)
    elif suffix == ".pptx":
        blocks = _extract_pptx(path)
    elif suffix in (".docx", ".doc"):
        blocks = _extract_docx(path)
    elif suffix in (".txt", ".md", ".rst"):
        blocks = _extract_text(path)
    else:
        log.warning("Unsupported format %s — skipping %s", suffix, path)
        return []

    if not blocks:
        log.warning("No text extracted from %s", path)
        return []

    chunks: list[dict] = []
    idx = 0
    for block in blocks:
        for chunk_text in _sliding_window(block):
            chunks.append(
                _build_chunk(
                    chunk_text,
                    idx,
                    file_path=path,
                    source_url=src_url,
                    component_tag=tag,
                    last_updated=last_updated,
                )
            )
            idx += 1

    log.info("Chunked %s → %d chunks (tag=%s)", path.name, len(chunks), tag)
    return chunks


def chunk_directory(
    directory: Path | str,
    *,
    component_tag: str | None = None,
    source_url_prefix: str = "",
    extensions: set[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> Iterator[dict]:
    """Walk *directory* recursively and yield chunk dicts for every supported file."""
    if extensions is None:
        extensions = {".pdf", ".pptx", ".docx", ".txt", ".md", ".rst"}
    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", ".git", ".venv", "node_modules"]

    directory = Path(directory).resolve()

    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        dirs[:] = [
            d
            for d in dirs
            if not any(pat in str(root_path / d) for pat in exclude_patterns)
        ]
        for fname in sorted(files):
            fpath = root_path / fname
            if fpath.suffix.lower() not in extensions:
                continue
            if any(pat in str(fpath) for pat in exclude_patterns):
                continue
            rel = fpath.relative_to(directory)
            url = f"{source_url_prefix}/{rel}" if source_url_prefix else str(fpath)
            yield from chunk_document(
                fpath, component_tag=component_tag, source_url=url
            )
