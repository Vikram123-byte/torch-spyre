"""
ask_z/scripts/init_index.py
────────────────────────────
Idempotent initialisation of the "ask-z-knowledge" Elasticsearch 8.x index.

Hybrid search design
─────────────────────
• BM25 keyword matching  → 'text' field  (standard analyser, inverted index)
• Dense vector (HNSW)    → 'vector' field (1024-dim, cosine, IBM Slate / BGE-large)

Run directly:
    python -m ask_z.scripts.init_index

Or via the helper entry-point (see ask_z/config/settings.py):
    ELASTIC_HOST=https://localhost:9200 python -m ask_z.scripts.init_index
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.exceptions import ConnectionError as ESConnectionError

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ask_z.init_index")

# ── Constants ─────────────────────────────────────────────────────────────────

INDEX_NAME = "ask-z-knowledge"

# 768 dimensions — confirmed from IBM watsonx.ai model specs API:
#   ibm/granite-embedding-278m-multilingual  → 768 dims  (primary, actively maintained)
#   ibm/slate-125m-english-rtrvr-v2          → 768 dims  (deprecated 2026-05-05)
# Source: GET https://us-south.ml.cloud.ibm.com/ml/v1/foundation_model_specs
VECTOR_DIMS = 768

# ── Index mapping ─────────────────────────────────────────────────────────────

# fmt: off
INDEX_MAPPINGS: dict[str, Any] = {
    "mappings": {
        "properties": {
            # ── Core content ──────────────────────────────────────────────────
            # BM25 full-text search over the chunk's raw text.
            "text": {
                "type": "text",
                "analyzer": "standard",
            },
            # HNSW dense-vector field for semantic / embedding-based search.
            # cosine similarity matches the training objective of Granite/Slate:
            # "maximize cosine similarity between query and passage embeddings."
            # Dims=768 confirmed from ibm/granite-embedding-278m-multilingual spec.
            "vector": {
                "type": "dense_vector",
                "dims": VECTOR_DIMS,
                "index": True,
                "similarity": "cosine",
                "index_options": {
                    "type": "hnsw",
                    # m=16: good recall/memory balance for O(10k–1M) doc corpora.
                    "m": 16,
                    # ef_construction=100: conservative production-safe default.
                    # Raise to 200 for corpora > 1M chunks.
                    "ef_construction": 100,
                },
            },

            # ── Structured metadata ───────────────────────────────────────────
            # Full URL or relative path to the originating document / file.
            "source_url": {
                "type": "keyword",
            },
            # One of: code | doc | ADR | test
            "doc_type": {
                "type": "keyword",
            },
            # ISO-8601 timestamp of the source document's last modification.
            "last_updated": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },
            # Logical component this chunk belongs to (e.g. "spyre_attention").
            "component_tag": {
                "type": "keyword",
            },
            # Semantic version string tied to the source artefact (e.g. "2.11.0").
            "version": {
                "type": "keyword",
            },
            # True when the source document has not been updated within the
            # configured staleness TTL (default: 90 days).  Set by the ingestion
            # pipeline; used by the retrieval layer to annotate cited chunks.
            "staleness_ttl_flag": {
                "type": "boolean",
            },
        }
    },
    # ── Index settings ────────────────────────────────────────────────────────
    "settings": {
        "number_of_shards": 1,       # single-node / dev default; tune for prod
        "number_of_replicas": 1,
        # Increase the BM25 window for larger corpora.
        "index": {
            "max_result_window": 10_000,
        },
    },
}
# fmt: on


# ── Client factory ────────────────────────────────────────────────────────────


def build_client() -> Elasticsearch:
    """
    Build an Elasticsearch client from environment variables.

    Required env vars:
        ELASTIC_HOST   — e.g. "https://localhost:9200"
        ELASTIC_API_KEY — API key (preferred over basic auth in ES 8.x)

    Optional env vars (basic auth fallback):
        ELASTIC_USER
        ELASTIC_PASSWORD
        ELASTIC_CA_CERT — path to a custom CA certificate file
    """
    host = os.environ.get("ELASTIC_HOST", "https://localhost:9200")
    api_key = os.environ.get("ELASTIC_API_KEY")
    user = os.environ.get("ELASTIC_USER")
    password = os.environ.get("ELASTIC_PASSWORD")
    ca_cert = os.environ.get("ELASTIC_CA_CERT")

    kwargs: dict[str, Any] = {"hosts": [host]}

    if ca_cert:
        kwargs["ca_certs"] = ca_cert
    else:
        # Accept self-signed certs in local dev; override to True in prod.
        kwargs["verify_certs"] = (
            os.environ.get("ELASTIC_VERIFY_CERTS", "false").lower() == "true"
        )

    if api_key:
        kwargs["api_key"] = api_key
        log.debug("Authenticating with API key.")
    elif user and password:
        kwargs["basic_auth"] = (user, password)
        log.debug("Authenticating with basic auth.")
    else:
        log.warning(
            "No authentication credentials found (ELASTIC_API_KEY / "
            "ELASTIC_USER + ELASTIC_PASSWORD). Proceeding without auth — "
            "this will fail against a secured cluster."
        )

    return Elasticsearch(**kwargs)


# ── Index lifecycle helpers ───────────────────────────────────────────────────


def index_exists(client: Elasticsearch, index_name: str) -> bool:
    """Return True if the index already exists."""
    try:
        return bool(client.indices.exists(index=index_name))
    except NotFoundError:
        return False


def create_index(client: Elasticsearch, index_name: str) -> None:
    """Create the index with the canonical mapping.  Raises on error."""
    client.indices.create(index=index_name, body=INDEX_MAPPINGS)
    log.info("Index '%s' created successfully.", index_name)


def validate_mapping(client: Elasticsearch, index_name: str) -> bool:
    """
    Retrieve the live mapping and confirm the three critical fields are present:
    'text' (text), 'vector' (dense_vector), and 'staleness_ttl_flag' (boolean).
    Returns True when all checks pass.
    """
    mapping = client.indices.get_mapping(index=index_name)
    props: dict[str, Any] = mapping[index_name]["mappings"].get("properties", {})

    checks = {
        "text field (BM25)": props.get("text", {}).get("type") == "text",
        "vector field (HNSW)": props.get("vector", {}).get("type") == "dense_vector",
        f"vector dims == {VECTOR_DIMS}": props.get("vector", {}).get("dims")
        == VECTOR_DIMS,
        "staleness_ttl_flag (boolean)": props.get("staleness_ttl_flag", {}).get("type")
        == "boolean",
    }

    all_ok = True
    for label, passed in checks.items():
        status = "OK" if passed else "FAIL"
        log.info("  Mapping check %-35s [%s]", label, status)
        if not passed:
            all_ok = False

    return all_ok


# ── Main entry-point ──────────────────────────────────────────────────────────


def initialise(index_name: str = INDEX_NAME) -> None:
    """
    Idempotent index initialisation.

    - If the index does not exist  → create it and validate the mapping.
    - If the index already exists  → skip creation, validate the live mapping.
    - Exits with code 1 on any unrecoverable error.
    """
    log.info("Connecting to Elasticsearch …")
    client = build_client()

    try:
        info = client.info()
        es_version = info["version"]["number"]
        log.info("Connected. Elasticsearch version: %s", es_version)
    except ESConnectionError as exc:
        log.error("Cannot reach Elasticsearch: %s", exc)
        sys.exit(1)

    if index_exists(client, index_name):
        log.info(
            "Index '%s' already exists — skipping creation (idempotent).",
            index_name,
        )
    else:
        log.info("Index '%s' not found — creating …", index_name)
        try:
            create_index(client, index_name)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to create index '%s': %s", index_name, exc)
            sys.exit(1)

    log.info("Validating mapping for '%s' …", index_name)
    if validate_mapping(client, index_name):
        log.info(
            "All mapping checks passed. Index '%s' is ready for hybrid search.",
            index_name,
        )
    else:
        log.error(
            "One or more mapping checks failed for index '%s'. "
            "Inspect the output above and re-run after correcting the mapping.",
            index_name,
        )
        sys.exit(1)


if __name__ == "__main__":
    initialise()
