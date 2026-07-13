"""
ask_z/tests/test_generator.py
──────────────────────────────
Unit tests for ask_z/api/generator.py — context assembly and citation enforcement.
All HTTP calls are mocked; no live watsonx.ai calls required.

Run with:  pytest ask_z/tests/test_generator.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ask_z.api.generator import _assemble_context, generate_grounded_answer
from ask_z.api.schemas import ContextChunk


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(
    text: str = "Spyre uses tiled matmul for attention.",
    file_path: str = "torch_spyre/ops.py",
    component_tag: str = "ops",
    last_updated: str = "2025-06-01",
    version: str = "997bafc",
    rank: int = 1,
) -> ContextChunk:
    return ContextChunk(
        text=text,
        score=0.9,
        rrf_score=0.015,
        rank=rank,
        file_path=file_path,
        source_url=file_path,
        doc_type="code",
        component_tag=component_tag,
        last_updated=last_updated,
        git_blame_author="dev@ibm.com",
        version=version,
        chunk_index=0,
        content_hash="abc123",
        staleness_ttl_flag=False,
    )


# ── _assemble_context ─────────────────────────────────────────────────────────


class TestAssembleContext:
    def test_empty_chunks_returns_placeholder(self) -> None:
        result = _assemble_context([])
        assert "(No context provided.)" in result

    def test_single_chunk_contains_source_metadata(self) -> None:
        chunk = _make_chunk()
        result = _assemble_context([chunk])
        assert "[Source: torch_spyre/ops.py" in result
        assert "Component: ops" in result
        assert "Last Updated: 2025-06-01" in result
        assert "Version: 997bafc" in result

    def test_chunk_text_included(self) -> None:
        chunk = _make_chunk(text="Spyre uses tiled matmul for attention.")
        result = _assemble_context([chunk])
        assert "Spyre uses tiled matmul for attention." in result

    def test_multiple_chunks_numbered(self) -> None:
        chunks = [
            _make_chunk(text="First chunk.", rank=1),
            _make_chunk(
                text="Second chunk.", file_path="torch_spyre/model_utils.py", rank=2
            ),
        ]
        result = _assemble_context(chunks)
        assert "--- Chunk 1 ---" in result
        assert "--- Chunk 2 ---" in result
        assert "First chunk." in result
        assert "Second chunk." in result

    def test_multiple_chunks_have_separate_metadata(self) -> None:
        chunks = [
            _make_chunk(file_path="a.py", component_tag="comp_a", rank=1),
            _make_chunk(file_path="b.py", component_tag="comp_b", rank=2),
        ]
        result = _assemble_context(chunks)
        assert "Source: a.py" in result
        assert "Source: b.py" in result
        assert "Component: comp_a" in result
        assert "Component: comp_b" in result

    def test_five_chunks_all_present(self) -> None:
        chunks = [_make_chunk(text=f"chunk {i}", rank=i) for i in range(1, 6)]
        result = _assemble_context(chunks)
        for i in range(1, 6):
            assert f"--- Chunk {i} ---" in result
            assert f"chunk {i}" in result

    def test_metadata_format_exact(self) -> None:
        chunk = _make_chunk(
            file_path="torch_spyre/attention.py",
            component_tag="attention",
            last_updated="2025-07-01",
            version="abc1234",
        )
        result = _assemble_context([chunk])
        expected_meta = (
            "[Source: torch_spyre/attention.py, "
            "Component: attention, "
            "Last Updated: 2025-07-01, "
            "Version: abc1234]"
        )
        assert expected_meta in result


# ── generate_grounded_answer ──────────────────────────────────────────────────


def _make_mock_http_client(
    answer_text: str = "Granite answer with [Source: ops.py].",
) -> MagicMock:
    """Build a mock httpx.AsyncClient that returns a valid watsonx.ai text/chat response."""
    chat_response = {
        "choices": [{"message": {"content": answer_text}}],
        "model_id": "ibm/granite-4-h-small",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }
    iam_response = {
        "access_token": "mock-bearer-token",
        "expires_in": 3600,
    }

    mock_iam_resp = MagicMock()
    mock_iam_resp.json.return_value = iam_response
    mock_iam_resp.raise_for_status = MagicMock()

    mock_chat_resp = MagicMock()
    mock_chat_resp.json.return_value = chat_response
    mock_chat_resp.raise_for_status = MagicMock()

    client = MagicMock()
    # post() is called twice: IAM token, then chat endpoint
    client.post = AsyncMock(side_effect=[mock_iam_resp, mock_chat_resp])
    return client


def _reset_token_cache() -> None:
    """Reset the module-level IAM token cache so each test starts clean."""
    import ask_z.api.generator as gen_module

    gen_module._iam_token = None
    gen_module._iam_expires_at = 0.0


def _mock_settings():
    """Return a MagicMock that mimics the settings object used by generator.py."""
    s = MagicMock()
    s.watsonx.api_base_url = "https://us-south.ml.cloud.ibm.com/ml/v1"
    s.watsonx.api_version = "2024-09-01"
    s.watsonx.api_key = "test-api-key"
    s.watsonx.project_id = "test-project-id"
    s.watsonx.generation_model = "ibm/granite-4-h-small"
    s.bob.timeout_seconds = 60
    return s


class TestGenerateGroundedAnswer:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset IAM cache and patch settings before every test."""
        _reset_token_cache()
        with patch("ask_z.api.generator.settings", _mock_settings()):
            yield
        _reset_token_cache()

    @pytest.mark.asyncio
    async def test_returns_string(self) -> None:
        chunks = [_make_chunk()]
        client = _make_mock_http_client("Answer with [Source: torch_spyre/ops.py].")
        result = await generate_grounded_answer("test query", chunks, client)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_raises_on_empty_chunks(self) -> None:
        client = _make_mock_http_client()
        with pytest.raises(ValueError, match="empty context"):
            await generate_grounded_answer("test query", [], client)

    @pytest.mark.asyncio
    async def test_answer_contains_source_citation(self) -> None:
        chunks = [_make_chunk()]
        answer_text = "Spyre uses tiled matmul [Source: torch_spyre/ops.py]."
        client = _make_mock_http_client(answer_text)
        result = await generate_grounded_answer(
            "How does Spyre handle attention?", chunks, client
        )
        assert "[Source:" in result

    @pytest.mark.asyncio
    async def test_iam_token_called_first(self) -> None:
        chunks = [_make_chunk()]
        client = _make_mock_http_client()
        await generate_grounded_answer("query", chunks, client)
        first_call_url = client.post.call_args_list[0][0][0]
        assert "iam.cloud.ibm.com" in first_call_url

    @pytest.mark.asyncio
    async def test_chat_endpoint_called_second(self) -> None:
        chunks = [_make_chunk()]
        client = _make_mock_http_client()
        await generate_grounded_answer("query", chunks, client)
        second_call_url = client.post.call_args_list[1][0][0]
        assert "text/chat" in second_call_url

    @pytest.mark.asyncio
    async def test_payload_contains_system_prompt(self) -> None:
        chunks = [_make_chunk()]
        client = _make_mock_http_client()
        await generate_grounded_answer("query", chunks, client)
        _, kwargs = client.post.call_args_list[1]
        payload = kwargs.get("json") or client.post.call_args_list[1][1].get("json", {})
        messages = payload.get("messages", [])
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_payload_contains_context_block(self) -> None:
        chunk = _make_chunk(
            text="Spyre tiled matmul.",
            file_path="torch_spyre/ops.py",
        )
        client = _make_mock_http_client()
        await generate_grounded_answer("query", [chunk], client)
        _, kwargs = client.post.call_args_list[1]
        payload = kwargs.get("json") or client.post.call_args_list[1][1].get("json", {})
        user_content = next(
            m["content"] for m in payload["messages"] if m["role"] == "user"
        )
        assert "torch_spyre/ops.py" in user_content
        assert "Spyre tiled matmul." in user_content

    @pytest.mark.asyncio
    async def test_timeout_propagates(self) -> None:
        import httpx

        chunks = [_make_chunk()]
        client = MagicMock()
        mock_iam = MagicMock()
        mock_iam.json.return_value = {"access_token": "tok", "expires_in": 3600}
        mock_iam.raise_for_status = MagicMock()
        client.post = AsyncMock(
            side_effect=[mock_iam, httpx.TimeoutException("timed out")]
        )
        with pytest.raises(httpx.TimeoutException):
            await generate_grounded_answer("query", chunks, client)
