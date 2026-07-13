"""
ask_z/tests/test_pipeline.py
──────────────────────────────
Unit tests for ask_z/api/pipeline.py and ask_z/api/schemas.py.

All external calls (Elasticsearch, watsonx.ai, cross-encoder) are mocked.
No live services required.

Run with:  pytest ask_z/tests/test_pipeline.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from ask_z.api.schemas import (
    ContextChunk,
    PipelineDiagnostics,
    QueryRequest,
)
from ask_z.api.pipeline import (
    _build_context_chunk,
    run_rrf,
)


# ── QueryRequest validation ────────────────────────────────────────────────────


class TestQueryRequest:
    def test_valid_query(self) -> None:
        req = QueryRequest(query="How does Spyre handle attention?")
        assert req.query == "How does Spyre handle attention?"

    def test_default_top_k(self) -> None:
        req = QueryRequest(query="test query here")
        assert req.top_k == 5

    def test_custom_top_k(self) -> None:
        req = QueryRequest(query="test query here", top_k=10)
        assert req.top_k == 10

    def test_top_k_max_enforced(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test query here", top_k=21)

    def test_top_k_min_enforced(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test query here", top_k=0)

    def test_query_too_short_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="ab")

    def test_query_whitespace_stripped(self) -> None:
        req = QueryRequest(query="  spyre ops  ")
        assert req.query == "spyre ops"

    def test_rerank_threshold_default(self) -> None:
        req = QueryRequest(query="test query here")
        assert req.rerank_threshold == 0.0

    def test_rerank_threshold_bounds(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test query here", rerank_threshold=2.0)


# ── ContextChunk ──────────────────────────────────────────────────────────────


class TestContextChunk:
    def _make_chunk(self, **kwargs) -> ContextChunk:
        defaults = dict(
            text="Some chunk text about Spyre.",
            score=0.85,
            rrf_score=0.012,
            rank=1,
            file_path="torch_spyre/model_utils.py",
            source_url="torch_spyre/model_utils.py",
            doc_type="code",
            component_tag="model_utils",
            last_updated="2025-06-01",
            git_blame_author="author@ibm.com",
            version="abc1234",
            chunk_index=0,
            content_hash="d41d8cd98f00b204e9800998ecf8427e",
            staleness_ttl_flag=False,
        )
        defaults.update(kwargs)
        return ContextChunk(**defaults)

    def test_construction(self) -> None:
        chunk = self._make_chunk()
        assert chunk.rank == 1
        assert chunk.doc_type == "code"

    def test_staleness_flag_false(self) -> None:
        chunk = self._make_chunk(staleness_ttl_flag=False)
        assert chunk.staleness_ttl_flag is False


# ── RRF ───────────────────────────────────────────────────────────────────────


def _make_hit(doc_id: str, text: str = "text") -> dict[str, Any]:
    return {
        "_id": doc_id,
        "_score": 1.0,
        "_source": {
            "text": text,
            "file_path": f"{doc_id}.py",
            "source_url": f"{doc_id}.py",
            "doc_type": "code",
            "component_tag": "test",
            "last_updated": "2025-01-01",
            "git_blame_author": "author",
            "version": "abc",
            "chunk_index": 0,
            "content_hash": "abc123",
            "staleness_ttl_flag": False,
        },
    }


class TestRRF:
    def test_top_result_appears_in_both_lists(self) -> None:
        """A doc ranked #1 in both lists should have the highest RRF score."""
        bm25 = [_make_hit("doc_a"), _make_hit("doc_b"), _make_hit("doc_c")]
        knn = [_make_hit("doc_a"), _make_hit("doc_d"), _make_hit("doc_e")]
        results = run_rrf(bm25, knn)
        assert results[0][0] == "doc_a"

    def test_rrf_scores_descending(self) -> None:
        bm25 = [_make_hit(f"doc_{i}") for i in range(5)]
        knn = [_make_hit(f"doc_{i}") for i in range(5)]
        results = run_rrf(bm25, knn)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_unique_docs_from_both_lists(self) -> None:
        bm25 = [_make_hit("a"), _make_hit("b")]
        knn = [_make_hit("c"), _make_hit("d")]
        results = run_rrf(bm25, knn)
        ids = [r[0] for r in results]
        assert set(ids) == {"a", "b", "c", "d"}

    def test_no_duplicate_ids(self) -> None:
        bm25 = [_make_hit("x"), _make_hit("y")]
        knn = [_make_hit("x"), _make_hit("z")]
        results = run_rrf(bm25, knn)
        ids = [r[0] for r in results]
        assert len(ids) == len(set(ids))

    def test_empty_bm25_returns_knn_results(self) -> None:
        knn = [_make_hit("a"), _make_hit("b")]
        results = run_rrf([], knn)
        assert len(results) == 2

    def test_empty_both_returns_empty(self) -> None:
        assert run_rrf([], []) == []

    def test_dense_weight_dominance(self) -> None:
        """Doc ranked #1 only in dense list should outscore #1 only in BM25
        when dense_weight > bm25_weight."""
        # doc_dense: rank 1 in knn, absent from bm25
        # doc_bm25:  rank 1 in bm25, absent from knn
        bm25 = [_make_hit("doc_bm25")] + [_make_hit(f"filler_{i}") for i in range(10)]
        knn = [_make_hit("doc_dense")] + [_make_hit(f"filler_{i}") for i in range(10)]
        results = run_rrf(bm25, knn, dense_weight=0.6, bm25_weight=0.4)
        ids = [r[0] for r in results]
        assert ids.index("doc_dense") < ids.index("doc_bm25")

    def test_source_dict_preserved(self) -> None:
        hit = _make_hit("doc_a")
        hit["_source"]["component_tag"] = "spyre_attention"
        results = run_rrf([hit], [])
        assert results[0][2]["component_tag"] == "spyre_attention"


# ── _build_context_chunk ──────────────────────────────────────────────────────


class TestBuildContextChunk:
    def _source(self) -> dict[str, Any]:
        return {
            "text": "def forward(x): return x",
            "file_path": "torch_spyre/ops.py",
            "source_url": "torch_spyre/ops.py",
            "doc_type": "code",
            "component_tag": "ops",
            "last_updated": "2025-06-15",
            "git_blame_author": "dev@ibm.com",
            "version": "997bafc",
            "chunk_index": 2,
            "content_hash": "deadbeef",
            "staleness_ttl_flag": False,
        }

    def test_rank_assigned(self) -> None:
        chunk = _build_context_chunk(1, 0.9, 0.015, self._source())
        assert chunk.rank == 1

    def test_scores_rounded(self) -> None:
        chunk = _build_context_chunk(1, 0.123456789, 0.0123456789, self._source())
        assert chunk.score == round(0.123456789, 6)
        assert chunk.rrf_score == round(0.0123456789, 6)

    def test_all_metadata_fields_populated(self) -> None:
        chunk = _build_context_chunk(1, 0.9, 0.01, self._source())
        assert chunk.file_path == "torch_spyre/ops.py"
        assert chunk.doc_type == "code"
        assert chunk.component_tag == "ops"
        assert chunk.version == "997bafc"
        assert chunk.chunk_index == 2
        assert chunk.content_hash == "deadbeef"
        assert chunk.staleness_ttl_flag is False

    def test_missing_source_fields_default(self) -> None:
        """Gracefully handles a partial source dict (e.g. legacy documents)."""
        chunk = _build_context_chunk(1, 0.5, 0.01, {})
        assert chunk.text == ""
        assert chunk.file_path == ""
        assert chunk.staleness_ttl_flag is False


# ── PipelineDiagnostics ───────────────────────────────────────────────────────


class TestPipelineDiagnostics:
    def test_construction(self) -> None:
        d = PipelineDiagnostics(
            hyde_ms=120.5,
            bm25_hits=18,
            knn_hits=20,
            rrf_candidates=32,
            reranked_count=20,
            dropped_below_threshold=5,
            total_ms=350.2,
        )
        assert d.total_ms == 350.2
        assert d.dropped_below_threshold == 5


# ── FastAPI endpoint integration (httpx TestClient) ───────────────────────────


class TestQueryEndpoint:
    """
    Tests the /api/v1/query endpoint with all external dependencies mocked.
    The pipeline.execute_pipeline coroutine is patched directly so we test
    request/response serialisation and error handling independently.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from ask_z.api import app as app_module

        # Inject mock clients so lifespan checks pass
        app_module._es_client = MagicMock()
        app_module._http_client = MagicMock()
        with TestClient(app_module.app, raise_server_exceptions=False) as c:
            yield c
        app_module._es_client = None
        app_module._http_client = None

    def _mock_pipeline_result(self):
        """Return value for a patched execute_pipeline call.
        Signature: (hyde_hypothesis, answer, chunks, diagnostics)
        """
        chunk = _build_context_chunk(
            rank=1,
            ce_score=0.95,
            rrf_score=0.015,
            source={
                "text": "Spyre attention uses tiled matmul.",
                "file_path": "torch_spyre/ops.py",
                "source_url": "torch_spyre/ops.py",
                "doc_type": "code",
                "component_tag": "ops",
                "last_updated": "2025-06-01",
                "git_blame_author": "dev@ibm.com",
                "version": "abc1234",
                "chunk_index": 0,
                "content_hash": "abc123",
                "staleness_ttl_flag": False,
            },
        )
        diag = PipelineDiagnostics(
            hyde_ms=110.0,
            bm25_hits=15,
            knn_hits=18,
            rrf_candidates=28,
            reranked_count=20,
            dropped_below_threshold=0,
            total_ms=300.0,
        )
        answer = (
            "Spyre attention uses tiled matmul "
            "[Source: torch_spyre/ops.py, Component: ops, "
            "Last Updated: 2025-06-01, Version: abc1234]."
        )
        return "Spyre uses tiled matmul for attention.", answer, [chunk], diag

    def test_valid_query_returns_200(self, client) -> None:
        with patch(
            "ask_z.api.app.execute_pipeline",
            new=AsyncMock(return_value=self._mock_pipeline_result()),
        ):
            resp = client.post(
                "/api/v1/query",
                json={"query": "How does Spyre attention work?"},
            )
        assert resp.status_code == 200

    def test_response_schema_valid(self, client) -> None:
        with patch(
            "ask_z.api.app.execute_pipeline",
            new=AsyncMock(return_value=self._mock_pipeline_result()),
        ):
            resp = client.post(
                "/api/v1/query",
                json={"query": "How does Spyre attention work?"},
            )
        body = resp.json()
        assert "query" in body
        assert "hyde_hypothesis" in body
        assert "answer" in body
        assert "chunks" in body
        assert "diagnostics" in body
        assert len(body["chunks"]) == 1
        assert body["chunks"][0]["rank"] == 1
        assert "[Source:" in body["answer"]

    def test_short_query_returns_422(self, client) -> None:
        resp = client.post("/api/v1/query", json={"query": "hi"})
        assert resp.status_code == 422

    def test_missing_query_returns_422(self, client) -> None:
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_health_endpoint(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_uninitialised_client_returns_503(self) -> None:
        from fastapi.testclient import TestClient
        from ask_z.api import app as app_module

        # Don't inject clients — simulate not-yet-initialised state
        app_module._es_client = None
        app_module._http_client = None
        with TestClient(app_module.app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/api/v1/query",
                json={"query": "How does Spyre attention work?"},
            )
        assert resp.status_code == 503
