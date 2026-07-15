"""
ask_z/ingestion/chunker.py
───────────────────────────
LlamaIndex document & code ingestion pipeline for the ask-z-knowledge index.

Chunking strategy
──────────────────
• Python source files (.py)
    → AST-aware splitting strictly at function/class boundaries.
    → Falls back to CodeSplitter (token-based with code awareness) if AST
      parsing fails (e.g. syntax errors in legacy files).

• Markdown documentation files (.md)
    → SentenceSplitter: 512-token chunks, 64-token overlap.
    → Code fences inside .md are extracted first and chunked as code nodes,
      the surrounding prose is chunked as doc nodes.

• Plain text / other (.txt, .rst, etc.)
    → SentenceSplitter: 512-token chunks, 64-token overlap.

Metadata injected per chunk
────────────────────────────
    file_path       — repo-relative path of the source file
    doc_type        — "code" | "doc" | "ADR" | "test"
    component_tag   — inferred from the file path (e.g. "spyre_attention")
    last_commit_date — ISO-8601 date from `git log`
    git_blame_author — author of the most recent commit touching this file
    content_hash    — MD5 of the raw chunk text (deduplication key)
    chunk_index     — position of this chunk within the source file
    source_url      — same as file_path (populated later with GitHub URL)
    version         — git short SHA of HEAD at ingestion time
    staleness_ttl_flag — always False at ingestion time; set by a separate job

Delta ingestion
────────────────
The `content_hash` field (MD5 of raw text) is computed before embedding.
The caller compares this hash against the value stored in Elasticsearch.
If the hash is unchanged the chunk is skipped; if new or changed it is
re-embedded and upserted.  This module only produces the chunk nodes —
the upsert logic lives in ask_z/ingestion/upsert.py (Phase 3).
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import regex as re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter

log = logging.getLogger("ask_z.ingestion.chunker")

# ── Constants ─────────────────────────────────────────────────────────────────

# Matches the EmbeddingSettings.chunk_size / chunk_overlap defaults.
SENTENCE_CHUNK_SIZE = 512
SENTENCE_CHUNK_OVERLAP = 64

# File extensions treated as Python source code.
CODE_EXTENSIONS = {".py"}

# File extensions treated as documentation.
DOC_EXTENSIONS = {".md", ".rst", ".txt"}

# Regex that identifies Architecture Decision Records by filename.
_ADR_RE = re.compile(r"(adr|decision|architecture)", re.IGNORECASE)

# Regex to extract fenced code blocks from Markdown.
_FENCE_RE = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)


# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass
class ChunkNode:
    """
    A fully populated chunk ready for embedding and upsert.

    `embedding` is None until the caller fills it by calling the
    watsonx.ai embeddings endpoint.  The upsert layer (Phase 3) skips
    chunks whose `content_hash` already exists in Elasticsearch with the
    same value (delta ingestion).
    """

    text: str
    metadata: dict
    embedding: list[float] | None = field(default=None, repr=False)

    @property
    def content_hash(self) -> str:
        return self.metadata["content_hash"]

    def to_text_node(self) -> TextNode:
        """Convert to a LlamaIndex TextNode for downstream compatibility."""
        return TextNode(text=self.text, metadata=self.metadata)


# ── Git helpers ────────────────────────────────────────────────────────────────


def _git_log(file_path: Path, repo_root: Path) -> tuple[str, str]:
    """
    Return (last_commit_date, git_blame_author) for *file_path*.
    Falls back to ("unknown", "unknown") when git is unavailable or the
    file has no commits (e.g. untracked new file).
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ai|%an", "--", str(file_path)],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=10,
        )
        output = result.stdout.strip()
        if "|" in output:
            date_str, author = output.split("|", 1)
            # Normalise to ISO-8601 date only (drop time + tz offset)
            date_only = date_str.strip()[:10]
            return date_only, author.strip()
    except Exception:
        pass
    return "unknown", "unknown"


def _git_head_sha(repo_root: Path) -> str:
    """Return the short SHA of HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ── Component tag inference ────────────────────────────────────────────────────


def _infer_component_tag(rel_path: Path) -> str:
    """
    Infer a logical component name from the file path.

    Examples
    ────────
    torch_spyre/_inductor/spyre_attention.py  → "spyre_attention"
    docs/source/user_guide/profiling/foo.md   → "profiling"
    tests/inductor/test_ops.py                → "tests"
    """
    parts = rel_path.parts

    # Use the immediate parent directory name as the component tag,
    # unless it's a top-level file — then use the stem.
    if len(parts) >= 2:
        tag = parts[-2]
    else:
        tag = rel_path.stem

    # Clean up common noise tokens.
    for noise in ("source", "src", "lib", "."):
        tag = tag.replace(noise, "").strip("_- ")

    return tag or rel_path.stem


# ── Doc type inference ─────────────────────────────────────────────────────────


def _infer_doc_type(rel_path: Path) -> str:
    """
    Return one of: "code" | "doc" | "ADR" | "test"
    """
    name = rel_path.name.lower()
    path_str = str(rel_path).lower()

    if _ADR_RE.search(name):
        return "ADR"
    if "test" in path_str or name.startswith("test_"):
        return "test"
    if rel_path.suffix in CODE_EXTENSIONS:
        return "code"
    return "doc"


# ── MD5 content hash ───────────────────────────────────────────────────────────


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Metadata builder ──────────────────────────────────────────────────────────


def _build_metadata(
    *,
    rel_path: Path,
    doc_type: str,
    chunk_text: str,
    chunk_index: int,
    last_commit_date: str,
    git_blame_author: str,
    head_sha: str,
) -> dict:
    return {
        "file_path": str(rel_path),
        "source_url": str(rel_path),  # replaced with GitHub URL in upsert
        "doc_type": doc_type,
        "component_tag": _infer_component_tag(rel_path),
        "last_updated": last_commit_date,
        "git_blame_author": git_blame_author,
        "version": head_sha,
        "chunk_index": chunk_index,
        "content_hash": _md5(chunk_text),
        "staleness_ttl_flag": False,  # set by the staleness checker job
    }


# ── AST-aware Python splitter ──────────────────────────────────────────────────


def _split_python_by_ast(source: str) -> list[str]:
    """
    Split a Python source string strictly at top-level function and class
    boundaries using the AST.

    Returns a list of source-code strings, one per top-level definition.
    Module-level code between definitions is collected as a "module_header"
    chunk.  If AST parsing fails the entire file is returned as one chunk.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        log.debug("AST parse failed — returning file as single chunk.")
        return [source]

    lines = source.splitlines(keepends=True)

    # Collect (start_line, end_line, node) for every top-level def/class.
    # ast line numbers are 1-based; end_lineno requires Python ≥ 3.8.
    boundaries: list[tuple[int, int]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1  # convert to 0-based index
            end = node.end_lineno  # exclusive upper bound (already 1-based)
            boundaries.append((start, end))

    if not boundaries:
        return [source]

    chunks: list[str] = []

    # Module header: everything before the first definition.
    first_start = boundaries[0][0]
    if first_start > 0:
        header = "".join(lines[:first_start]).strip()
        if header:
            chunks.append(header)

    for idx, (start, end) in enumerate(boundaries):
        chunk = "".join(lines[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        # Gap between this definition and the next (e.g. module-level code).
        if idx + 1 < len(boundaries):
            next_start = boundaries[idx + 1][0]
            gap = "".join(lines[end:next_start]).strip()
            if gap:
                chunks.append(gap)

    # Anything after the last definition.
    last_end = boundaries[-1][1]
    trailer = "".join(lines[last_end:]).strip()
    if trailer:
        chunks.append(trailer)

    return [c for c in chunks if c.strip()]


# ── Sentence splitter (docs + plain text) ────────────────────────────────────


_sentence_splitter = SentenceSplitter(
    chunk_size=SENTENCE_CHUNK_SIZE,
    chunk_overlap=SENTENCE_CHUNK_OVERLAP,
)


def _split_doc(text: str) -> list[str]:
    """
    Split documentation text using LlamaIndex SentenceSplitter.
    Returns a list of chunk strings.
    """
    nodes = _sentence_splitter.split_text(text)
    return [n for n in nodes if n.strip()]


# ── Markdown: extract code fences then split prose ────────────────────────────


def _split_markdown(source: str) -> list[tuple[str, str]]:
    """
    Returns a list of (chunk_text, chunk_doc_type) tuples.

    Code fences are extracted and tagged "code".
    Remaining prose is split with SentenceSplitter and tagged "doc".
    """
    results: list[tuple[str, str]] = []

    # Extract and remove code fences.
    code_blocks = _FENCE_RE.findall(source)
    prose = _FENCE_RE.sub("", source)

    for block in code_blocks:
        block = block.strip()
        if block:
            results.append((block, "code"))

    for chunk in _split_doc(prose):
        if chunk.strip():
            results.append((chunk, "doc"))

    return results


# ── Public API ─────────────────────────────────────────────────────────────────


def chunk_file(file_path: Path, repo_root: Path) -> list[ChunkNode]:
    """
    Chunk a single file into a list of :class:`ChunkNode` objects.

    Each node carries its text, full metadata dict, and a None embedding
    (filled by the embedding step).

    Parameters
    ----------
    file_path:
        Absolute or relative path to the file being processed.
    repo_root:
        Root of the git repository (used for git log and relative paths).

    Returns
    -------
    list[ChunkNode]
        Fully populated chunk nodes ready for embedding.
    """
    file_path = Path(file_path).resolve()
    repo_root = Path(repo_root).resolve()
    rel_path = file_path.relative_to(repo_root)
    suffix = file_path.suffix.lower()
    base_doc_type = _infer_doc_type(rel_path)

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("Cannot read %s: %s", file_path, exc)
        return []

    if not source.strip():
        return []

    last_commit_date, git_blame_author = _git_log(file_path, repo_root)
    head_sha = _git_head_sha(repo_root)

    raw_chunks: list[tuple[str, str]] = []  # (text, doc_type)

    if suffix in CODE_EXTENSIONS:
        # ── Python: AST boundary splitting ────────────────────────────────
        for chunk_text in _split_python_by_ast(source):
            raw_chunks.append((chunk_text, base_doc_type))

    elif suffix == ".md":
        # ── Markdown: prose via SentenceSplitter + code fences as code ────
        raw_chunks = _split_markdown(source)
        # Override doc_type with file-level inference (e.g. ADR)
        raw_chunks = [
            (text, base_doc_type if base_doc_type == "ADR" else dtype)
            for text, dtype in raw_chunks
        ]

    else:
        # ── Plain text / RST / other ───────────────────────────────────────
        for chunk_text in _split_doc(source):
            raw_chunks.append((chunk_text, base_doc_type))

    nodes: list[ChunkNode] = []
    for idx, (chunk_text, doc_type) in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        metadata = _build_metadata(
            rel_path=rel_path,
            doc_type=doc_type,
            chunk_text=chunk_text,
            chunk_index=idx,
            last_commit_date=last_commit_date,
            git_blame_author=git_blame_author,
            head_sha=head_sha,
        )
        nodes.append(ChunkNode(text=chunk_text, metadata=metadata))

    log.debug(
        "chunked %s → %d nodes (doc_type=%s)", rel_path, len(nodes), base_doc_type
    )
    return nodes


def chunk_directory(
    directory: Path,
    repo_root: Path,
    extensions: set[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> Iterator[ChunkNode]:
    """
    Walk *directory* recursively and yield :class:`ChunkNode` objects for
    every file matching *extensions*.

    Parameters
    ----------
    directory:
        Root directory to walk (e.g. ``torch_spyre/`` or ``docs/``).
    repo_root:
        Root of the git repository.
    extensions:
        Set of file extensions to include, e.g. ``{".py", ".md"}``.
        Defaults to CODE_EXTENSIONS | DOC_EXTENSIONS.
    exclude_patterns:
        List of path substrings to skip (e.g. ``["__pycache__", ".venv"]``).
    """
    if extensions is None:
        extensions = CODE_EXTENSIONS | DOC_EXTENSIONS
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__",
            ".venv",
            ".git",
            "node_modules",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            "build",
        ]

    directory = Path(directory).resolve()

    for root, dirs, files in os.walk(directory):
        root_path = Path(root)

        # Prune excluded directories in-place (prevents os.walk descending).
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

            nodes = chunk_file(fpath, repo_root)
            yield from nodes
