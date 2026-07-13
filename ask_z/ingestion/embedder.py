"""
ask_z/ingestion/embedder.py
────────────────────────────
Embedding layer: calls the watsonx.ai embeddings endpoint with IBM Granite
to populate ChunkNode.embedding fields.

Design decisions
─────────────────
• Batched calls (up to 10 texts per request — watsonx.ai API limit).
• Delta-aware: chunks whose content_hash already exists in the caller's
  seen_hashes set are skipped before any API call is made.
• IAM token is cached for its lifetime (3600s) and auto-refreshed.
• Retries with exponential back-off on transient HTTP errors (429, 5xx).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

from ask_z.ingestion.chunker import ChunkNode

log = logging.getLogger("ask_z.ingestion.embedder")

# watsonx.ai allows up to 10 inputs per embeddings request.
_BATCH_SIZE = 10

# Granite-embedding-278m-multilingual max context length is 512 BPE tokens.
# Code identifiers tokenize as many sub-words per "word", so we truncate by
# character count instead.  Dense Python with underscores/identifiers can hit
# ~2 BPE tokens per char; 900 chars * 2 = 900 < 512 still too many. Use 800
# chars which gives ≈ 400 BPE tokens even for the worst-case code.
_MAX_CHARS = 800


def _truncate(text: str, max_chars: int = _MAX_CHARS) -> str:
    """Truncate *text* to at most *max_chars* characters."""
    if len(text) <= max_chars:
        return text
    # Break on a word boundary so we don't cut mid-token.
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated


# ── IAM token cache ───────────────────────────────────────────────────────────


class _TokenCache:
    """Thread-unsafe but sufficient for single-process ingestion jobs."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at: float = 0.0

    def get(self, api_key: str) -> str:
        now = time.time()
        # Refresh 60 s before expiry to avoid mid-batch 401s.
        if self._token and now < self._expires_at - 60:
            return self._token

        log.debug("Refreshing IAM Bearer token …")
        data = urllib.parse.urlencode(
            {
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": api_key,
            }
        ).encode()
        req = urllib.request.Request(
            "https://iam.cloud.ibm.com/identity/token",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.load(resp)

        self._token = payload["access_token"]
        self._expires_at = now + payload.get("expires_in", 3600)
        log.debug("IAM token refreshed (expires in %ss).", payload.get("expires_in"))
        return self._token


_token_cache = _TokenCache()


# ── Core embedding call ───────────────────────────────────────────────────────


# Dimensionality of the granite-embedding-278m-multilingual model.
_EMBED_DIMS = 768


def _call_embed_api(
    texts: list[str],
    *,
    bearer: str,
    url: str,
    model_id: str,
    project_id: str,
) -> list[list[float]]:
    """Single HTTP call to the watsonx.ai embeddings endpoint."""
    payload = json.dumps(
        {
            "model_id": model_id,
            "project_id": project_id,
            "inputs": texts,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return [r["embedding"] for r in data["results"]]


def _embed_batch(
    texts: list[str],
    *,
    api_key: str,
    project_id: str,
    base_url: str,
    api_version: str,
    model_id: str,
    max_retries: int = 3,
) -> list[list[float]]:
    """
    Call watsonx.ai /text/embeddings for a batch of texts.
    Returns a list of float vectors in the same order as *texts*.

    On HTTP 400 (token-too-long) the batch is recursively halved until
    individual chunks succeed.  A single chunk that still fails at length 1
    is replaced with a zero vector and a warning is logged.
    """
    bearer = _token_cache.get(api_key)
    url = f"{base_url}/text/embeddings?version={api_version}"

    def _embed_recursive(batch: list[str]) -> list[list[float]]:
        nonlocal bearer
        for attempt in range(1, max_retries + 1):
            try:
                return _call_embed_api(
                    batch,
                    bearer=bearer,
                    url=url,
                    model_id=model_id,
                    project_id=project_id,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode()
                if exc.code == 401:
                    log.warning("401 received — refreshing token and retrying.")
                    bearer = _token_cache.get(api_key)
                    continue
                if exc.code in (429, 500, 502, 503) and attempt < max_retries:
                    wait = 2**attempt
                    log.warning(
                        "HTTP %s on attempt %d/%d — retrying in %ds.",
                        exc.code,
                        attempt,
                        max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                if exc.code == 400 and len(batch) > 1:
                    # Split and recurse — one item in the batch is too long.
                    mid = len(batch) // 2
                    log.debug(
                        "HTTP 400 on batch of %d — splitting into %d + %d.",
                        len(batch),
                        mid,
                        len(batch) - mid,
                    )
                    return _embed_recursive(batch[:mid]) + _embed_recursive(batch[mid:])
                if exc.code == 400 and len(batch) == 1:
                    # Single chunk too long even after truncation — skip with zeros.
                    log.warning(
                        "HTTP 400 on single chunk (still too long after truncation) "
                        "— replacing with zero vector. Text (first 120 chars): %r",
                        batch[0][:120],
                    )
                    return [[0.0] * _EMBED_DIMS]
                log.error("Embedding API error %s: %s", exc.code, body[:300])
                raise
        raise RuntimeError(f"Embedding batch failed after {max_retries} attempts.")

    return _embed_recursive(texts)


# ── Public API ─────────────────────────────────────────────────────────────────


def embed_chunks(
    nodes: Iterable[ChunkNode],
    *,
    api_key: str,
    project_id: str,
    base_url: str = "https://us-south.ml.cloud.ibm.com/ml/v1",
    api_version: str = "2024-09-01",
    model_id: str = "ibm/granite-embedding-278m-multilingual",
    seen_hashes: set[str] | None = None,
) -> list[ChunkNode]:
    """
    Embed a collection of :class:`ChunkNode` objects using watsonx.ai.

    Delta ingestion
    ────────────────
    Pass *seen_hashes* (a set of content_hash strings already stored in
    Elasticsearch) to skip chunks whose content has not changed since the
    last ingestion run.  Skipped chunks are returned with ``embedding=None``
    so the caller can distinguish new/changed vs. unchanged.

    Parameters
    ----------
    nodes:
        Iterable of ChunkNode objects produced by :mod:`ask_z.ingestion.chunker`.
    api_key:
        IBM Cloud IAM API key (``WATSONX_API_KEY``).
    project_id:
        watsonx.ai project ID (``WATSONX_PROJECT_ID``).
    base_url:
        watsonx.ai API base URL.
    api_version:
        API version date string.
    model_id:
        Embedding model ID.
    seen_hashes:
        Set of MD5 hashes already indexed.  Chunks in this set are skipped.

    Returns
    -------
    list[ChunkNode]
        All nodes, with ``.embedding`` populated for new/changed chunks
        and ``None`` for unchanged (delta-skipped) chunks.
    """
    if seen_hashes is None:
        seen_hashes = set()

    all_nodes = list(nodes)
    to_embed: list[tuple[int, ChunkNode]] = []  # (original index, node)
    skipped = 0

    for idx, node in enumerate(all_nodes):
        if node.content_hash in seen_hashes:
            skipped += 1
        else:
            to_embed.append((idx, node))

    log.info(
        "Embedding: %d new/changed chunks, %d unchanged (delta-skipped).",
        len(to_embed),
        skipped,
    )

    # Batch the API calls.
    for batch_start in range(0, len(to_embed), _BATCH_SIZE):
        batch = to_embed[batch_start : batch_start + _BATCH_SIZE]
        texts = [_truncate(node.text) for _, node in batch]

        log.debug(
            "Calling watsonx.ai embeddings: batch %d–%d of %d …",
            batch_start + 1,
            batch_start + len(batch),
            len(to_embed),
        )

        vectors = _embed_batch(
            texts,
            api_key=api_key,
            project_id=project_id,
            base_url=base_url,
            api_version=api_version,
            model_id=model_id,
        )

        for (orig_idx, node), vector in zip(batch, vectors):
            all_nodes[orig_idx].embedding = vector

    embedded_count = sum(1 for n in all_nodes if n.embedding is not None)
    log.info(
        "Embedding complete: %d embedded, %d skipped.",
        embedded_count,
        skipped,
    )
    return all_nodes
