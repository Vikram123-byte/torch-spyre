"""
ask_z/scripts/ingest_docs.py
─────────────────────────────
Ingest external documents (PDF, PPTX, DOCX, TXT, MD) into the
ask-z-knowledge Elasticsearch index.

Usage
──────
  python -m ask_z.scripts.ingest_docs --dir /path/to/my-docs
  python -m ask_z.scripts.ingest_docs --file pod-setup.pdf --tag pod_setup --source-url "https://ibm.box.com/s/abc123"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError as ESConnectionError

from ask_z.ingestion.doc_ingest import chunk_directory, chunk_document
from ask_z.ingestion.embedder import _embed_batch, _truncate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ask_z.ingest_docs")


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
        log.info("Fetched %d existing hashes from '%s'.", len(hashes), index)
        return hashes
    except Exception as exc:
        log.warning("Could not fetch existing hashes: %s", exc)
        return set()


def ingest_docs(chunks: list[dict], es: Elasticsearch, index: str) -> None:
    if not chunks:
        log.warning("No chunks to ingest.")
        return

    api_key = os.environ.get("WATSONX_API_KEY", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")
    base_url = os.environ.get(
        "WATSONX_API_URL", "https://us-south.ml.cloud.ibm.com/ml/v1"
    )
    api_version = os.environ.get("WATSONX_API_VERSION", "2024-09-01")
    model_id = os.environ.get(
        "EMBEDDING_MODEL", "ibm/granite-embedding-278m-multilingual"
    )

    if not api_key or not project_id:
        log.error(
            "WATSONX_API_KEY and WATSONX_PROJECT_ID must be set. "
            "Copy ask_z/.env.example to ask_z/.env and fill in the values."
        )
        sys.exit(1)

    seen = _fetch_seen_hashes(es, index)
    to_embed = [c for c in chunks if c["metadata"]["content_hash"] not in seen]
    skipped = len(chunks) - len(to_embed)
    log.info(
        "Total chunks: %d | new/changed: %d | delta-skipped: %d",
        len(chunks),
        len(to_embed),
        skipped,
    )

    if not to_embed:
        log.info("Index is already up to date.")
        return

    # ── Try embedding; fall back to BM25-only upsert on quota exhaustion ─────
    # A list of (chunk, vector_or_None) — None means no vector available yet.
    results: list[tuple[dict, list[float] | None]] = []
    quota_exhausted = False
    batch_size = 10

    for i in range(0, len(to_embed), batch_size):
        if quota_exhausted:
            # No point calling the API again — mark remaining without vectors.
            for chunk in to_embed[i : i + batch_size]:
                results.append((chunk, None))
            continue

        batch = to_embed[i : i + batch_size]
        texts = [_truncate(c["text"]) for c in batch]
        log.info("Embedding batch %d–%d / %d …", i + 1, i + len(batch), len(to_embed))
        try:
            vectors = _embed_batch(
                texts,
                api_key=api_key,
                project_id=project_id,
                base_url=base_url,
                api_version=api_version,
                model_id=model_id,
            )
            for chunk, vec in zip(batch, vectors):
                results.append((chunk, vec))
        except Exception as exc:
            if "403" in str(exc) or "quota" in str(exc).lower():
                log.warning(
                    "watsonx.ai quota exhausted — switching to BM25-only mode. "
                    "Chunks will be stored as text-only (no vector). "
                    "Re-run ingest_docs after quota resets to add embeddings."
                )
                quota_exhausted = True
                for chunk in batch:
                    results.append((chunk, None))
            else:
                log.error("Embedding batch %d failed: %s — skipping batch.", i, exc)

    # Split into embedded vs text-only.
    embedded = [(c, v) for c, v in results if v is not None]
    text_only = [c for c, v in results if v is None]

    log.info(
        "Embedded: %d | BM25-only (no vector): %d",
        len(embedded),
        len(text_only),
    )

    def _make_action(chunk: dict, vector: list[float] | None) -> dict:
        meta = chunk["metadata"]
        if meta.get("last_updated") == "unknown":
            meta.pop("last_updated", None)
        source: dict = {
            "text": chunk["text"],
            **{k: v for k, v in meta.items() if k != "content_hash"},
            "content_hash": meta["content_hash"],
        }
        # Only include the vector field when we actually have one —
        # ES will accept the doc without it for BM25-only search.
        if vector is not None:
            source["vector"] = vector
        return {
            "_op_type": "index",
            "_index": index,
            "_id": meta["content_hash"],
            "_source": source,
        }

    actions = [_make_action(c, v) for c, v in embedded] + [
        _make_action(c, None) for c in text_only
    ]

    log.info("Upserting %d chunks into '%s' …", len(actions), index)
    success, errors = helpers.bulk(es, actions, raise_on_error=False, stats_only=False)
    failed = [e for e in (errors or []) if e]
    log.info(
        "Upsert complete: %d succeeded, %d failed.%s",
        success,
        len(failed),
        " (BM25-only — re-run after quota resets to add vectors)"
        if quota_exhausted
        else "",
    )
    for err in failed[:3]:
        log.error("Bulk error: %s", err)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest external documents (PDF/PPTX/DOCX/TXT/MD) into ask-z-knowledge.",
    )
    parser.add_argument("--file", metavar="PATH", help="Ingest a single document.")
    parser.add_argument(
        "--dir", metavar="DIR", help="Ingest all documents in a directory (recursive)."
    )
    parser.add_argument(
        "--tag", metavar="TAG", help="Component tag (e.g. pod_setup, onboarding)."
    )
    parser.add_argument(
        "--source-url",
        metavar="URL",
        default="",
        help="Source URL stored in each chunk.",
    )
    parser.add_argument(
        "--source-url-prefix",
        metavar="PREFIX",
        default="",
        help="URL prefix for --dir mode.",
    )
    parser.add_argument(
        "--extensions",
        metavar="EXT",
        nargs="+",
        default=[".pdf", ".pptx", ".docx", ".txt", ".md", ".rst"],
        help="File extensions to ingest.",
    )
    args = parser.parse_args()

    if not args.file and not args.dir:
        parser.error("Provide --file or --dir.")

    index = os.environ.get("ELASTIC_INDEX", "ask-z-knowledge")
    log.info("Index: %s", index)

    es = _build_es_client()
    try:
        info = es.info()
        log.info(
            "Elasticsearch %s @ %s",
            info["version"]["number"],
            os.environ.get("ELASTIC_HOST", "http://localhost:9200"),
        )
    except ESConnectionError as exc:
        log.error("Cannot reach Elasticsearch: %s", exc)
        sys.exit(1)

    ext_set = {e if e.startswith(".") else f".{e}" for e in args.extensions}

    if args.file:
        chunks = chunk_document(
            args.file, component_tag=args.tag, source_url=args.source_url
        )
    else:
        chunks = list(
            chunk_directory(
                args.dir,
                component_tag=args.tag,
                source_url_prefix=args.source_url_prefix,
                extensions=ext_set,
            )
        )

    log.info("Chunks produced: %d", len(chunks))
    ingest_docs(chunks, es, index)
    log.info("Done.")


if __name__ == "__main__":
    main()
