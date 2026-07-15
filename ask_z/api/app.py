"""
ask_z/api/app.py
─────────────────
FastAPI application — POST /api/v1/query

Lifespan
─────────
On startup:
  • Initialises the async Elasticsearch client
  • Initialises the shared async httpx client (watsonx.ai + IAM)
  • Warms the cross-encoder reranker (loads weights into memory once)

On shutdown:
  • Closes both clients gracefully

Error boundaries
─────────────────
• 422 — request validation failure (Pydantic, handled by FastAPI)
• 503 — Elasticsearch or watsonx.ai unreachable
• 504 — upstream timeout
• 500 — unexpected internal error (full traceback logged, sanitised message returned)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import httpx
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ask_z.api.pipeline import _get_reranker, execute_pipeline
from ask_z.api.schemas import (
    ErrorDetail,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
)
from ask_z.config.settings import settings

log = logging.getLogger("ask_z.api.app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ── Shared client singletons ──────────────────────────────────────────────────
# Populated during lifespan startup; consumed by route handlers.

_es_client: AsyncElasticsearch | None = None
_http_client: httpx.AsyncClient | None = None


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _es_client, _http_client

    log.info("Ask-Z API starting up …")

    # ── Elasticsearch async client ─────────────────────────────────────────
    es_kwargs: dict = {"hosts": [settings.elasticsearch.host]}
    if settings.elasticsearch.api_key:
        es_kwargs["api_key"] = settings.elasticsearch.api_key
    elif settings.elasticsearch.user and settings.elasticsearch.password:
        es_kwargs["basic_auth"] = (
            settings.elasticsearch.user,
            settings.elasticsearch.password,
        )
    if settings.elasticsearch.ca_cert:
        es_kwargs["ca_certs"] = settings.elasticsearch.ca_cert
    else:
        es_kwargs["verify_certs"] = settings.elasticsearch.verify_certs

    _es_client = AsyncElasticsearch(**es_kwargs)
    log.info("Elasticsearch client initialised → %s", settings.elasticsearch.host)

    # ── httpx async client (watsonx.ai + IAM) ────────────────────────────
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=float(settings.bob.timeout_seconds),
            write=10.0,
            pool=5.0,
        ),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    log.info("HTTP client initialised.")

    # ── Warm the reranker if sentence-transformers is available ───────────
    import asyncio

    reranker = await asyncio.to_thread(_get_reranker)
    if reranker is not None:
        log.info("Cross-encoder reranker warmed.")
    else:
        log.warning("Cross-encoder reranker unavailable — RRF-only mode active.")

    log.info("Ask-Z API ready.")
    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("Ask-Z API shutting down …")
    await _es_client.close()
    await _http_client.aclose()
    log.info("Clients closed.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ask-Z Knowledge API",
    description=(
        "RAG-powered IBM Z & Spyre engineering knowledge assistant. "
        "Combines HyDE, hybrid BM25+KNN search, RRF fusion, and "
        "BGE cross-encoder reranking against the ask-z-knowledge index."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Static UI ─────────────────────────────────────────────────────────────────
_UI_DIR = Path(__file__).parent.parent / "ui"
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect / → /ui so opening http://localhost:8000 loads the UI."""
    return RedirectResponse(url="/ui/index.html")


# ── Global exception handlers ─────────────────────────────────────────────────


@app.exception_handler(httpx.TimeoutException)
async def timeout_handler(
    request: Request, exc: httpx.TimeoutException
) -> JSONResponse:
    log.warning("Upstream timeout: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content=ErrorResponse(
            error=ErrorDetail(
                code="upstream_timeout",
                message="A watsonx.ai or Elasticsearch call timed out. Retry in a moment.",
            )
        ).model_dump(),
    )


@app.exception_handler(httpx.HTTPStatusError)
async def upstream_http_handler(
    request: Request, exc: httpx.HTTPStatusError
) -> JSONResponse:
    log.error(
        "Upstream HTTP error %s: %s", exc.response.status_code, exc.response.text[:200]
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error=ErrorDetail(
                code="upstream_error",
                message=f"Upstream service returned HTTP {exc.response.status_code}.",
                detail=exc.response.text[:200],
            )
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception in request pipeline.")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=ErrorDetail(
                code="internal_error",
                message="An unexpected error occurred. Check server logs.",
            )
        ).model_dump(),
    )


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"], summary="Liveness probe")
async def health() -> dict:
    """Returns 200 when the service is alive."""
    return {"status": "ok", "service": "ask-z"}


@app.get("/health/ready", tags=["ops"], summary="Readiness probe")
async def readiness() -> dict:
    """
    Checks that Elasticsearch is reachable.
    Returns 503 if not ready — suitable for Kubernetes readinessProbe.
    """
    if _es_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Elasticsearch client not initialised.",
        )
    try:
        info = await _es_client.info()
        return {
            "status": "ready",
            "elasticsearch": info["version"]["number"],
            "index": settings.elasticsearch.index_name,
        }
    except Exception as exc:
        log.warning("Readiness check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Elasticsearch unreachable: {exc}",
        )


# ── Main query endpoint ───────────────────────────────────────────────────────


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["retrieval"],
    summary="RAG retrieval — HyDE + hybrid search + RRF + rerank",
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Upstream service unavailable"},
        504: {"model": ErrorResponse, "description": "Upstream timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def query_endpoint(body: QueryRequest) -> QueryResponse:
    """
    Execute the full Ask-Z RAG retrieval pipeline.

    **Pipeline stages:**

    1. **HyDE** — Granite generates a synthetic answer; Granite Embedding
       encodes it into a dense query vector.
    2. **Parallel hybrid search** — BM25 (raw query) and KNN (HyDE vector)
       run concurrently against the `ask-z-knowledge` Elasticsearch index.
       Top-20 candidates from each.
    3. **RRF fusion** — 60% dense / 40% BM25 weighted Reciprocal Rank Fusion
       merges both ranked lists.
    4. **Cross-encoder reranking** — `BAAI/bge-reranker-large` scores all
       fused candidates against the original query text.
    5. **Filter + slice** — drops chunks below `rerank_threshold`, returns
       the top `top_k` chunks with full metadata.
    """
    if _es_client is None or _http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not fully initialised. Retry in a moment.",
        )

    hypothesis, answer, chunks, diagnostics = await execute_pipeline(
        query=body.query,
        top_k=body.top_k,
        rerank_threshold=body.rerank_threshold,
        es=_es_client,
        http_client=_http_client,
    )

    log.info(
        "query=%r | answer=%d chars | chunks=%d | total_ms=%.0f",
        body.query[:60],
        len(answer),
        len(chunks),
        diagnostics.total_ms,
    )

    return QueryResponse(
        query=body.query,
        hyde_hypothesis=hypothesis,
        answer=answer,
        chunks=chunks,
        diagnostics=diagnostics,
    )
