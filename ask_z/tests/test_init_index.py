"""
ask_z/tests/test_init_index.py
───────────────────────────────
Unit tests for ask_z/scripts/init_index.py.

Uses unittest.mock to isolate all Elasticsearch calls — no live cluster needed.
Run with:  pytest ask_z/tests/test_init_index.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ask_z.scripts.init_index import (
    INDEX_MAPPINGS,
    INDEX_NAME,
    VECTOR_DIMS,
    create_index,
    index_exists,
    initialise,
    validate_mapping,
)


# ── Mapping schema tests ───────────────────────────────────────────────────────


class TestIndexMappingSchema:
    """Validate the static INDEX_MAPPINGS constant matches all requirements."""

    @property
    def props(self) -> dict[str, Any]:
        return INDEX_MAPPINGS["mappings"]["properties"]

    def test_text_field_is_bm25(self) -> None:
        assert self.props["text"]["type"] == "text"
        assert self.props["text"]["analyzer"] == "standard"

    def test_vector_field_type(self) -> None:
        assert self.props["vector"]["type"] == "dense_vector"

    def test_vector_dims_are_768(self) -> None:
        # 768 confirmed from IBM watsonx.ai model specs:
        #   ibm/granite-embedding-278m-multilingual → embedding_dimension: 768
        #   ibm/slate-125m-english-rtrvr-v2         → embedding_dimension: 768
        assert self.props["vector"]["dims"] == 768
        assert VECTOR_DIMS == 768

    def test_vector_similarity_is_cosine(self) -> None:
        assert self.props["vector"]["similarity"] == "cosine"

    def test_vector_index_type_is_hnsw(self) -> None:
        assert self.props["vector"]["index_options"]["type"] == "hnsw"

    def test_vector_index_is_enabled(self) -> None:
        assert self.props["vector"]["index"] is True

    def test_source_url_is_keyword(self) -> None:
        assert self.props["source_url"]["type"] == "keyword"

    def test_doc_type_is_keyword(self) -> None:
        assert self.props["doc_type"]["type"] == "keyword"

    def test_last_updated_is_date(self) -> None:
        assert self.props["last_updated"]["type"] == "date"

    def test_component_tag_is_keyword(self) -> None:
        assert self.props["component_tag"]["type"] == "keyword"

    def test_version_is_keyword(self) -> None:
        assert self.props["version"]["type"] == "keyword"

    def test_staleness_ttl_flag_is_boolean(self) -> None:
        assert self.props["staleness_ttl_flag"]["type"] == "boolean"

    def test_all_required_fields_present(self) -> None:
        required = {
            "text",
            "vector",
            "source_url",
            "doc_type",
            "last_updated",
            "component_tag",
            "version",
            "staleness_ttl_flag",
        }
        assert required.issubset(self.props.keys())

    def test_index_settings_present(self) -> None:
        assert "settings" in INDEX_MAPPINGS
        assert INDEX_MAPPINGS["settings"]["number_of_shards"] >= 1


# ── index_exists() tests ──────────────────────────────────────────────────────


class TestIndexExists:
    def test_returns_true_when_index_found(self) -> None:
        client = MagicMock()
        client.indices.exists.return_value = True
        assert index_exists(client, INDEX_NAME) is True
        client.indices.exists.assert_called_once_with(index=INDEX_NAME)

    def test_returns_false_when_index_not_found(self) -> None:
        from elasticsearch import NotFoundError

        client = MagicMock()
        client.indices.exists.side_effect = NotFoundError(
            message="not found", meta=MagicMock(), body={}
        )
        assert index_exists(client, INDEX_NAME) is False


# ── create_index() tests ──────────────────────────────────────────────────────


class TestCreateIndex:
    def test_calls_es_create_with_correct_body(self) -> None:
        client = MagicMock()
        create_index(client, INDEX_NAME)
        client.indices.create.assert_called_once_with(
            index=INDEX_NAME, body=INDEX_MAPPINGS
        )

    def test_propagates_exception_on_failure(self) -> None:
        from elasticsearch import BadRequestError

        client = MagicMock()
        client.indices.create.side_effect = BadRequestError(
            message="resource_already_exists_exception",
            meta=MagicMock(),
            body={},
        )
        with pytest.raises(BadRequestError):
            create_index(client, INDEX_NAME)


# ── validate_mapping() tests ──────────────────────────────────────────────────


def _make_mapping_response(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a minimal mock get_mapping response.
    Uses VECTOR_DIMS (768) — confirmed from IBM watsonx.ai model spec API.
    """
    props: dict[str, Any] = {
        "text": {"type": "text"},
        "vector": {"type": "dense_vector", "dims": VECTOR_DIMS},  # 768
        "staleness_ttl_flag": {"type": "boolean"},
    }
    if overrides:
        props.update(overrides)
    return {INDEX_NAME: {"mappings": {"properties": props}}}


class TestValidateMapping:
    def test_returns_true_when_all_checks_pass(self) -> None:
        client = MagicMock()
        client.indices.get_mapping.return_value = _make_mapping_response()
        assert validate_mapping(client, INDEX_NAME) is True

    def test_returns_false_when_vector_type_wrong(self) -> None:
        client = MagicMock()
        client.indices.get_mapping.return_value = _make_mapping_response(
            {"vector": {"type": "keyword", "dims": VECTOR_DIMS}}
        )
        assert validate_mapping(client, INDEX_NAME) is False

    def test_returns_false_when_dims_wrong(self) -> None:
        client = MagicMock()
        client.indices.get_mapping.return_value = _make_mapping_response(
            {"vector": {"type": "dense_vector", "dims": 1024}}  # wrong dims (old value)
        )
        assert validate_mapping(client, INDEX_NAME) is False

    def test_returns_false_when_staleness_flag_missing(self) -> None:
        client = MagicMock()
        response = _make_mapping_response()
        del response[INDEX_NAME]["mappings"]["properties"]["staleness_ttl_flag"]
        client.indices.get_mapping.return_value = response
        assert validate_mapping(client, INDEX_NAME) is False


# ── initialise() integration tests ───────────────────────────────────────────


class TestInitialise:
    """
    Patch build_client() to inject a mock, then test the full orchestration
    path of initialise() without touching a real cluster.
    """

    def _make_mock_client(
        self,
        *,
        index_already_exists: bool = False,
        mapping_valid: bool = True,
    ) -> MagicMock:
        client = MagicMock()
        client.info.return_value = {"version": {"number": "8.14.0"}}
        client.indices.exists.return_value = index_already_exists
        # Provide a valid mapping response for validate_mapping().
        client.indices.get_mapping.return_value = _make_mapping_response()
        if not mapping_valid:
            client.indices.get_mapping.return_value = _make_mapping_response(
                {"vector": {"type": "keyword", "dims": 0}}
            )
        return client

    def test_creates_index_when_absent(self) -> None:
        client = self._make_mock_client(index_already_exists=False)
        with patch("ask_z.scripts.init_index.build_client", return_value=client):
            initialise()
        client.indices.create.assert_called_once()

    def test_skips_creation_when_index_exists(self) -> None:
        client = self._make_mock_client(index_already_exists=True)
        with patch("ask_z.scripts.init_index.build_client", return_value=client):
            initialise()
        client.indices.create.assert_not_called()

    def test_exits_1_on_bad_mapping(self) -> None:
        client = self._make_mock_client(index_already_exists=True, mapping_valid=False)
        with patch("ask_z.scripts.init_index.build_client", return_value=client):
            with pytest.raises(SystemExit) as exc_info:
                initialise()
        assert exc_info.value.code == 1

    def test_exits_1_on_connection_error(self) -> None:
        from elasticsearch.exceptions import ConnectionError as ESConnectionError

        client = MagicMock()
        client.info.side_effect = ESConnectionError("refused")
        with patch("ask_z.scripts.init_index.build_client", return_value=client):
            with pytest.raises(SystemExit) as exc_info:
                initialise()
        assert exc_info.value.code == 1
