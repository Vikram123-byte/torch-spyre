"""
ask_z/scripts/ingest.py
────────────────────────
Ingestion pipeline: chunk → embed → upsert into the ask-z-knowledge index.

Usage
──────
    # Full repo ingest (code + docs):
    python -m ask_z.scripts.ingest

    # Single file:
    python -m ask_z.scripts.ingest --file torch_spyre/ops/eager.py

    # Specific directories:
    python -m ask_z.scripts.ingest --dirs torch_spyre docs

Delta ingestion
────────────────
Only new or changed chunks are re-embedded and upserted.  A chunk is
considered unchanged when its content_hash (MD5 of text) already exists
in Elasticsearch with the same value.  Unchanged chunks are skipped,
making repeated runs cheap.

Environment variables (read from ask_z/.env or shell):
    ELASTIC_HOST, ELASTIC_API_KEY, ELASTIC_INDEX
    WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_API_URL, WATSONX_API_VERSION
    EMBEDDING_MODEL
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError as ESConnectionError

from ask_z.ingestion.chunker import ChunkNode, chunk_directory, chunk_file
from ask_z.ingestion.embedder import embed_chunks

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ask_z.ingest")

# ── Defaults ──────────────────────────────────────────────────────────────────

# Directories ingested when no --dirs argument is given.
_DEFAULT_DIRS = ["torch_spyre", "docs", "ask_z"]

# File extensions to walk.
_EXTENSIONS = {".py", ".md", ".rst", ".txt"}

# Patterns excluded from walking (never useful to index).
_EXCLUDE = [
    "__pycache__",
    ".venv",
    ".git",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".eggs",
]

# ES bulk upsert batch size (number of docs per _bulk request).
_BULK_BATCH = 100


# ── Elasticsearch helpers ─────────────────────────────────────────────────────


def _build_es_client() -> Elasticsearch:
    host = os.environ.get("ELASTIC_HOST", "http://localhost:9200")
    api_key = os.environ.get("ELASTIC_API_KEY")
    user = os.environ.get("ELASTIC_USER")
    password = os.environ.get("ELASTIC_PASSWORD")
    ca_cert = os.environ.get("ELASTIC_CA_CERT")

    kwargs: dict[str, Any] = {"hosts": [host]}
    if ca_cert:
        kwargs["ca_certs"] = ca_cert
    else:
        kwargs["verify_certs"] = (
            os.environ.get("ELASTIC_VERIFY_CERTS", "false").lower() == "true"
        )
    if api_key:
        kwargs["api_key"] = api_key
    elif user and password:
        kwargs["basic_auth"] = (user, password)
    return Elasticsearch(**kwargs)


def _fetch_seen_hashes(es: Elasticsearch, index: str) -> set[str]:
    """Return the set of content_hash values already in the index (for delta)."""
    try:
        resp = es.search(
            index=index,
            body={
                "size": 10_000,
                "_source": ["content_hash"],
                "query": {"match_all": {}},
            },
            scroll="1m",
        )
        hashes: set[str] = set()
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]
        while hits:
            for h in hits:
                ch = h["_source"].get("content_hash")
                if ch:
                    hashes.add(ch)
            resp = es.scroll(scroll_id=scroll_id, scroll="1m")
            hits = resp["hits"]["hits"]
        es.clear_scroll(scroll_id=scroll_id)
        log.info("Fetched %d existing content hashes from '%s'.", len(hashes), index)
        return hashes
    except Exception as exc:
        log.warning("Could not fetch existing hashes (assuming empty index): %s", exc)
        return set()


def _build_actions(nodes: list[ChunkNode], index: str):
    """Yield ES bulk action dicts for each embedded chunk."""
    for node in nodes:
        if node.embedding is None:
            continue  # unchanged (delta-skipped) — already in ES
        meta = {k: v for k, v in node.metadata.items() if k != "content_hash"}
        # ES date field rejects "unknown" — omit last_updated when not a real date.
        if meta.get("last_updated") == "unknown":
            meta.pop("last_updated")
        doc: dict[str, Any] = {
            "text": node.text,
            "vector": node.embedding,
            **meta,
            "content_hash": node.content_hash,
        }
        yield {
            "_op_type": "index",
            "_index": index,
            "_id": node.content_hash,  # deterministic ID → idempotent upserts
            "_source": doc,
        }


# ── Main ingestion logic ──────────────────────────────────────────────────────


def ingest(
    paths: list[Path],
    repo_root: Path,
    es: Elasticsearch,
    index: str,
) -> None:
    """Chunk, embed, and upsert *paths* into *index*."""

    # 1. Chunk all files/directories.
    log.info("Chunking %d path(s) …", len(paths))
    all_nodes: list[ChunkNode] = []
    for p in paths:
        if p.is_dir():
            all_nodes.extend(
                chunk_directory(
                    p, repo_root, extensions=_EXTENSIONS, exclude_patterns=_EXCLUDE
                )
            )
        elif p.is_file():
            all_nodes.extend(chunk_file(p, repo_root))
        else:
            log.warning("Path not found, skipping: %s", p)
    log.info("Total chunks produced: %d", len(all_nodes))

    if not all_nodes:
        log.warning("No chunks produced — nothing to ingest.")
        return

    # 2. Fetch existing hashes for delta ingestion.
    seen = _fetch_seen_hashes(es, index)

    # 3. Embed only new/changed chunks.
    embedded = embed_chunks(
        all_nodes,
        api_key=os.environ["WATSONX_API_KEY"],
        project_id=os.environ["WATSONX_PROJECT_ID"],
        base_url=os.environ.get(
            "WATSONX_API_URL", "https://us-south.ml.cloud.ibm.com/ml/v1"
        ),
        api_version=os.environ.get("WATSONX_API_VERSION", "2024-09-01"),
        model_id=os.environ.get(
            "EMBEDDING_MODEL", "ibm/granite-embedding-278m-multilingual"
        ),
        seen_hashes=seen,
    )

    # 4. Bulk upsert into Elasticsearch.
    actions = list(_build_actions(embedded, index))
    if not actions:
        log.info("No new or changed chunks — index is up to date.")
        return

    log.info("Upserting %d chunks into '%s' …", len(actions), index)
    success, errors = helpers.bulk(es, actions, raise_on_error=False, stats_only=False)
    failed = [e for e in (errors or []) if e]
    log.info("Bulk upsert complete: %d succeeded, %d failed.", success, len(failed))
    if failed:
        for err in failed[:5]:
            log.error("Bulk error: %s", err)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest source files into ask-z-knowledge."
    )
    parser.add_argument("--file", metavar="PATH", help="Ingest a single file.")
    parser.add_argument(
        "--dirs", nargs="+", metavar="DIR", help="Directories to ingest."
    )
    parser.add_argument(
        "--repo-root", default=".", metavar="DIR", help="Repo root (default: cwd)."
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    index = os.environ.get("ELASTIC_INDEX", "ask-z-knowledge")

    if args.file:
        paths = [Path(args.file)]
    elif args.dirs:
        paths = [repo_root / d for d in args.dirs]
    else:
        paths = [repo_root / d for d in _DEFAULT_DIRS if (repo_root / d).exists()]

    log.info("Repo root   : %s", repo_root)
    log.info("Index       : %s", index)
    log.info("Paths       : %s", [str(p) for p in paths])

    es = _build_es_client()
    try:
        info = es.info()
        log.info(
            "Elasticsearch %s @ %s",
            info["version"]["number"],
            os.environ.get("ELASTIC_HOST"),
        )
    except ESConnectionError as exc:
        log.error("Cannot reach Elasticsearch: %s", exc)
        sys.exit(1)

    ingest(paths, repo_root, es, index)
    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
