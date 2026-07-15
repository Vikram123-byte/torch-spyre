"""
ask_z/api/pipeline.py
──────────────────────
Core RAG retrieval pipeline executed by the FastAPI endpoint.

Pipeline stages (in order)
───────────────────────────
1. HyDE  — generate a synthetic answer with Granite, embed it with Granite
           Embedding to produce the dense query vector.
2. Hybrid search — BM25 (raw query) + KNN (HyDE vector) in Elasticsearch,
                   top-20 candidates each.
3. RRF   — Reciprocal Rank Fusion: 60% dense / 40% BM25 weight.
4. Rerank — BGE-reranker-large cross-encoder scores each of the top-20 fused
            candidates against the *original* query text.
5. Filter — drop scores below threshold, return top-k structured ContextChunk
            objects with full metadata.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from elasticsearch import AsyncElasticsearch

from ask_z.api.schemas import ContextChunk, PipelineDiagnostics
from ask_z.config.settings import settings

log = logging.getLogger("ask_z.api.pipeline")

# ── Reranker (loaded once at process start, not per request) ───────────────────

_reranker = None  # lazy-loaded on first use
_reranker_available: bool | None = None  # None = not yet probed


def _get_reranker():
    """
    Lazy-load the BGE cross-encoder reranker.
    Uses sentence-transformers CrossEncoder.
    Model: BAAI/bge-reranker-large  (~1.3 GB, loaded once).

    Returns None if sentence-transformers is not installed — the pipeline
    will fall back to RRF-only ordering in that case.
    """
    global _reranker, _reranker_available
    if _reranker_available is False:
        return None
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            model_name = "BAAI/bge-reranker-large"
            log.info("Loading cross-encoder reranker: %s …", model_name)
            _reranker = CrossEncoder(model_name, max_length=512)
            _reranker_available = True
            log.info("Reranker loaded.")
        except ModuleNotFoundError:
            log.warning(
                "sentence-transformers not installed — reranker disabled. "
                "Results will be ordered by RRF score only. "
                "Install sentence-transformers to enable cross-encoder reranking."
            )
            _reranker_available = False
            return None
    return _reranker


# ── IAM token cache (same pattern as embedder.py) ─────────────────────────────

_iam_token: str | None = None
_iam_expires_at: float = 0.0


async def _get_iam_token(client: httpx.AsyncClient) -> str:
    global _iam_token, _iam_expires_at
    now = time.time()
    if _iam_token and now < _iam_expires_at - 60:
        return _iam_token

    log.debug("Refreshing IAM token …")
    resp = await client.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": settings.watsonx.api_key,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    _iam_token = payload["access_token"]
    _iam_expires_at = now + payload.get("expires_in", 3600)
    return _iam_token


# ── Stage 1: HyDE ─────────────────────────────────────────────────────────────

_HYDE_SYSTEM = (
    "You are an IBM Z and Spyre systems expert. "
    "When given a question, write a concise 2-3 sentence technical answer "
    "as if it were an excerpt from internal engineering documentation. "
    "Be specific and use exact technical terminology. "
    "Output ONLY the answer text — no preamble, no labels."
)


async def run_hyde(
    query: str,
    http_client: httpx.AsyncClient,
) -> tuple[str, list[float]]:
    """
    Stage 1 — HyDE (Hypothetical Document Embedding).

    1. Generate a synthetic technical answer using Granite text generation.
    2. Embed the synthetic answer using Granite Embedding.

    Returns (hypothesis_text, embedding_vector).
    """
    bearer = await _get_iam_token(http_client)
    base = settings.watsonx.api_base_url
    version = settings.watsonx.api_version
    project_id = settings.watsonx.project_id

    # ── 1a: Generate synthetic answer ──────────────────────────────────────
    generation_model = settings.watsonx.generation_model
    gen_payload = {
        "model_id": generation_model,
        "project_id": project_id,
        "input": f"{_HYDE_SYSTEM}\n\nQuestion: {query}\n\nAnswer:",
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 150,
            "min_new_tokens": 20,
            "stop_sequences": ["\n\n"],
        },
    }
    gen_resp = await http_client.post(
        f"{base}/text/generation?version={version}",
        json=gen_payload,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    gen_resp.raise_for_status()
    hypothesis = gen_resp.json()["results"][0]["generated_text"].strip()
    log.debug("HyDE hypothesis: %s", hypothesis[:120])

    # ── 1b: Embed the hypothesis ────────────────────────────────────────────
    # Embedding model read from settings (EMBEDDING_MODEL env var).
    embed_payload = {
        "model_id": settings.embedding.model_name,
        "project_id": project_id,
        "inputs": [hypothesis],
    }
    embed_resp = await http_client.post(
        f"{base}/text/embeddings?version={version}",
        json=embed_payload,
        headers={
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        },
        timeout=20,
    )
    embed_resp.raise_for_status()
    vector: list[float] = embed_resp.json()["results"][0]["embedding"]

    return hypothesis, vector


# ── Stage 2: Parallel hybrid search ──────────────────────────────────────────


async def run_bm25_search(
    query: str,
    es: AsyncElasticsearch,
    index: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """BM25 full-text search over the 'text' field."""
    resp = await es.search(
        index=index,
        body={
            "size": top_k,
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "operator": "or",
                    }
                }
            },
            "_source": True,
        },
    )
    return resp["hits"]["hits"]


async def run_knn_search(
    vector: list[float],
    es: AsyncElasticsearch,
    index: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """HNSW KNN dense-vector search over the 'vector' field."""
    resp = await es.search(
        index=index,
        body={
            "size": top_k,
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": top_k,
                "num_candidates": top_k * 5,
            },
            "_source": True,
        },
    )
    return resp["hits"]["hits"]


# ── Stage 3: Reciprocal Rank Fusion ──────────────────────────────────────────

# RRF smoothing constant — standard value from the original paper.
_RRF_K = 60


def run_rrf(
    bm25_hits: list[dict[str, Any]],
    knn_hits: list[dict[str, Any]],
    *,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[tuple[str, float, dict[str, Any]]]:
    """
    Reciprocal Rank Fusion with configurable per-list weights.

    Each hit is scored as:
        rrf_score = dense_weight * (1 / (k + dense_rank))
                  + bm25_weight  * (1 / (k + bm25_rank))

    Returns a list of (doc_id, rrf_score, source_dict) sorted descending
    by rrf_score.
    """
    # Build rank maps: doc_id → rank (1-based)
    bm25_ranks: dict[str, int] = {
        hit["_id"]: rank + 1 for rank, hit in enumerate(bm25_hits)
    }
    knn_ranks: dict[str, int] = {
        hit["_id"]: rank + 1 for rank, hit in enumerate(knn_hits)
    }

    # Collect all unique doc IDs across both result sets.
    all_ids: dict[str, dict[str, Any]] = {}
    for hit in bm25_hits + knn_hits:
        all_ids[hit["_id"]] = hit["_source"]

    scored: list[tuple[str, float, dict[str, Any]]] = []
    for doc_id, source in all_ids.items():
        dense_r = knn_ranks.get(doc_id, len(knn_hits) + _RRF_K)
        bm25_r = bm25_ranks.get(doc_id, len(bm25_hits) + _RRF_K)

        rrf_score = dense_weight * (1.0 / (_RRF_K + dense_r)) + bm25_weight * (
            1.0 / (_RRF_K + bm25_r)
        )
        scored.append((doc_id, rrf_score, source))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Stage 4: Cross-encoder reranking ─────────────────────────────────────────


def run_rerank(
    query: str,
    candidates: list[tuple[str, float, dict[str, Any]]],
) -> list[tuple[float, float, dict[str, Any]]]:
    """
    Score each candidate with the BGE-reranker-large cross-encoder.

    Returns list of (cross_encoder_score, rrf_score, source) sorted by
    cross_encoder_score descending.  Runs in a thread via asyncio.to_thread
    (called from the async route handler).

    If sentence-transformers is not installed the reranker is skipped and
    candidates are returned in their existing RRF order (rrf_score is used
    as the cross_encoder_score placeholder so downstream filtering still works).
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    if reranker is None:
        # No cross-encoder — preserve RRF order, use rrf_score as proxy score.
        return [(rrf_score, rrf_score, source) for _, rrf_score, source in candidates]

    pairs = [(query, cand[2].get("text", "")) for cand in candidates]
    scores: list[float] = reranker.predict(pairs).tolist()

    results = [
        (score, rrf_score, source)
        for score, (_, rrf_score, source) in zip(scores, candidates)
    ]
    results.sort(key=lambda x: x[0], reverse=True)
    return results


# ── Stage 5: Build response chunks ───────────────────────────────────────────


def _build_context_chunk(
    rank: int,
    ce_score: float,
    rrf_score: float,
    source: dict[str, Any],
) -> ContextChunk:
    """Map an Elasticsearch _source document to a ContextChunk."""
    return ContextChunk(
        text=source.get("text", ""),
        score=round(ce_score, 6),
        rrf_score=round(rrf_score, 6),
        rank=rank,
        file_path=source.get("file_path", ""),
        source_url=source.get("source_url", ""),
        doc_type=source.get("doc_type", ""),
        component_tag=source.get("component_tag", ""),
        last_updated=source.get("last_updated", ""),
        git_blame_author=source.get("git_blame_author", ""),
        version=source.get("version", ""),
        chunk_index=source.get("chunk_index", 0),
        content_hash=source.get("content_hash", ""),
        staleness_ttl_flag=bool(source.get("staleness_ttl_flag", False)),
    )


# ── Quota helpers ─────────────────────────────────────────────────────────────


def _is_quota_error(exc: Exception) -> bool:
    """Return True when *exc* is a watsonx.ai 403 quota or 429 rate-limit error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (403, 429)
    return False


# ── Orchestrator ──────────────────────────────────────────────────────────────


async def execute_pipeline(
    query: str,
    top_k: int,
    rerank_threshold: float,
    es: AsyncElasticsearch,
    http_client: httpx.AsyncClient,
) -> tuple[str, str, list[ContextChunk], PipelineDiagnostics]:
    """
    Execute the full Ask-Z pipeline.

    If the query is asking about a GitHub PR (e.g. "summarise PR 4345"),
    the pipeline short-circuits: fetches the PR from GitHub API and sends
    it directly to the generator — no RAG retrieval needed.

    Otherwise runs the full RAG pipeline:
      HyDE → hybrid BM25+KNN → RRF → rerank → generate.
    """
    from ask_z.api.github_tools import detect_pr_query, fetch_pr_context
    from ask_z.api.generator import (
        generate_grounded_answer,
        generate_pr_review,
        generate_pr_summary,
    )

    t_start = time.perf_counter()

    # ── PR short-circuit ──────────────────────────────────────────────────────
    pr_result = detect_pr_query(query)
    if pr_result is not None:
        pr_number, pr_intent = pr_result
        log.info("PR query detected — PR #%d | intent=%s", pr_number, pr_intent)
        pr_data = await fetch_pr_context(
            pr_number,
            http_client,
            include_diff=(pr_intent == "review"),
        )

        if pr_data is None:
            answer = (
                f"⚠ Could not fetch PR #{pr_number} from GitHub. "
                "Make sure GITHUB_TOKEN is set in ask_z/.env and the PR number is correct."
            )
            empty_diag = PipelineDiagnostics(
                hyde_ms=0,
                bm25_hits=0,
                knn_hits=0,
                rrf_candidates=0,
                reranked_count=0,
                dropped_below_threshold=0,
                total_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )
            return "", answer, [], empty_diag

        # Build a synthetic ContextChunk from the PR data so the generator
        # can cite it like any other source.
        import hashlib

        pr_chunk = ContextChunk(
            rank=1,
            score=1.0,
            rrf_score=1.0,
            text=pr_data["context_text"],
            file_path=f"github.com/{pr_data['org']}/{pr_data['repo']}/pull/{pr_number}",
            source_url=pr_data["url"],
            doc_type="external_doc",
            component_tag="github_pr",
            chunk_index=0,
            git_blame_author=pr_data["author"],
            last_updated="",
            version=pr_data["state"],
            staleness_ttl_flag=False,
            content_hash=hashlib.md5(pr_data["context_text"].encode()).hexdigest(),
        )

        if pr_intent == "review":
            answer = await generate_pr_review(
                pr_data["context_text"], query, http_client
            )
            action = "reviewed"
        else:
            answer = await generate_pr_summary(
                pr_data["context_text"], query, http_client
            )
            action = "summarised"

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        pr_diag = PipelineDiagnostics(
            hyde_ms=0,
            bm25_hits=0,
            knn_hits=0,
            rrf_candidates=1,
            reranked_count=1,
            dropped_below_threshold=0,
            total_ms=total_ms,
        )
        log.info(
            "PR #%d %s: %d chars | %.0f ms",
            pr_number,
            action,
            len(answer),
            total_ms,
        )
        return "", answer, [pr_chunk], pr_diag

    # ── Normal RAG pipeline ───────────────────────────────────────────────────
    index = settings.elasticsearch.index_name
    # ── Stage 1: HyDE (with quota fallback) ──────────────────────────────────
    t_hyde_start = time.perf_counter()
    hypothesis: str = ""
    hyde_vector: list[float] | None = None
    try:
        hypothesis, hyde_vector = await run_hyde(query, http_client)
    except Exception as exc:
        if _is_quota_error(exc):
            log.warning(
                "watsonx.ai quota/rate-limit — falling back to BM25-only search. "
                "Groq will be used for answer generation."
            )
            hypothesis = "[HyDE skipped — watsonx.ai quota/rate-limit]"
        else:
            raise
    hyde_ms = (time.perf_counter() - t_hyde_start) * 1000

    # ── Stage 2: Search (hybrid when vector available, BM25-only fallback) ────
    bm25_hits = await run_bm25_search(query, es, index, top_k=20)
    knn_hits: list[dict[str, Any]] = []
    if hyde_vector is not None:
        knn_hits = await run_knn_search(hyde_vector, es, index, top_k=20)

    log.debug("BM25: %d hits | KNN: %d hits", len(bm25_hits), len(knn_hits))

    # ── Stage 3: RRF ─────────────────────────────────────────────────────────
    fused = run_rrf(
        bm25_hits,
        knn_hits,
        dense_weight=settings.retrieval.rrf_dense_weight,
        bm25_weight=settings.retrieval.rrf_bm25_weight,
    )
    fused_top20 = fused[:20]

    # ── Stage 4: Cross-encoder rerank (CPU-bound → thread pool) ──────────────
    reranked = await asyncio.to_thread(run_rerank, query, fused_top20)

    # ── Stage 5: Filter + slice ───────────────────────────────────────────────
    above_threshold = [
        (ce, rrf, src) for ce, rrf, src in reranked if ce >= rerank_threshold
    ]
    dropped = len(reranked) - len(above_threshold)
    final = above_threshold[:top_k]

    chunks = [
        _build_context_chunk(rank=i + 1, ce_score=ce, rrf_score=rrf, source=src)
        for i, (ce, rrf, src) in enumerate(final)
    ]

    total_ms = (time.perf_counter() - t_start) * 1000

    diagnostics = PipelineDiagnostics(
        hyde_ms=round(hyde_ms, 1),
        bm25_hits=len(bm25_hits),
        knn_hits=len(knn_hits),
        rrf_candidates=len(fused),
        reranked_count=len(reranked),
        dropped_below_threshold=dropped,
        total_ms=round(total_ms, 1),
    )

    # ── Stage 6: Answer generation ────────────────────────────────────────────
    # generate_grounded_answer handles the watsonx→Groq fallback internally.
    if not chunks:
        answer = (
            "⚠ No relevant context found after reranking. "
            "The knowledge base may not contain information about this topic yet."
        )
    else:
        answer = await generate_grounded_answer(query, chunks, http_client)

    return hypothesis, answer, chunks, diagnostics
