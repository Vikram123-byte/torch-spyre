"""
ask_z/tests/test_chunker.py
────────────────────────────
Unit tests for ask_z/ingestion/chunker.py.

All tests are offline — no git, no filesystem writes, no watsonx.ai calls.
Real files from the torch_spyre repo are used to validate splitting behaviour.

Run with:  pytest ask_z/tests/test_chunker.py -v
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from ask_z.ingestion.chunker import (
    ChunkNode,
    _infer_component_tag,
    _infer_doc_type,
    _md5,
    _split_doc,
    _split_markdown,
    _split_python_by_ast,
    chunk_file,
)

# ── Repo root used for relative path calculations ──────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent  # torch-spyre/


# ── _md5 ──────────────────────────────────────────────────────────────────────


class TestMd5:
    def test_deterministic(self) -> None:
        assert _md5("hello") == _md5("hello")

    def test_different_text_different_hash(self) -> None:
        assert _md5("hello") != _md5("world")

    def test_matches_stdlib(self) -> None:
        text = "IBM Z Spyre attention operator"
        expected = hashlib.md5(text.encode()).hexdigest()
        assert _md5(text) == expected


# ── _infer_doc_type ────────────────────────────────────────────────────────────


class TestInferDocType:
    def test_python_source_is_code(self) -> None:
        assert _infer_doc_type(Path("torch_spyre/model_utils.py")) == "code"

    def test_test_file_is_test(self) -> None:
        assert _infer_doc_type(Path("tests/inductor/test_ops.py")) == "test"

    def test_test_file_by_prefix(self) -> None:
        assert _infer_doc_type(Path("torch_spyre/test_foo.py")) == "test"

    def test_markdown_is_doc(self) -> None:
        assert _infer_doc_type(Path("docs/source/index.md")) == "doc"

    def test_adr_is_adr(self) -> None:
        assert _infer_doc_type(Path("docs/adr/001-architecture-decision.md")) == "ADR"

    def test_architecture_in_name_is_adr(self) -> None:
        assert _infer_doc_type(Path("docs/architecture_overview.md")) == "ADR"

    def test_rst_is_doc(self) -> None:
        assert _infer_doc_type(Path("docs/source/contributing.rst")) == "doc"


# ── _infer_component_tag ──────────────────────────────────────────────────────


class TestInferComponentTag:
    def test_uses_parent_directory(self) -> None:
        # Leading underscores are stripped → "_inductor" becomes "inductor"
        # for a cleaner component tag label.
        tag = _infer_component_tag(Path("torch_spyre/_inductor/spyre_attention.py"))
        assert tag == "inductor"

    def test_profiling_dir(self) -> None:
        tag = _infer_component_tag(Path("docs/source/user_guide/profiling/foo.md"))
        assert tag == "profiling"

    def test_top_level_file_uses_stem(self) -> None:
        tag = _infer_component_tag(Path("README.md"))
        assert tag == "README"


# ── _split_python_by_ast ──────────────────────────────────────────────────────


class TestSplitPythonByAst:
    def test_splits_two_functions(self) -> None:
        source = textwrap.dedent("""\
            def foo():
                return 1

            def bar():
                return 2
        """)
        chunks = _split_python_by_ast(source)
        assert len(chunks) == 2
        assert "def foo" in chunks[0]
        assert "def bar" in chunks[1]

    def test_splits_class_and_method(self) -> None:
        source = textwrap.dedent("""\
            class MyModel:
                def forward(self, x):
                    return x

            def helper():
                pass
        """)
        chunks = _split_python_by_ast(source)
        assert any("class MyModel" in c for c in chunks)
        assert any("def helper" in c for c in chunks)

    def test_module_header_preserved(self) -> None:
        source = textwrap.dedent("""\
            \"\"\"Module docstring.\"\"\"
            import os

            def main():
                pass
        """)
        chunks = _split_python_by_ast(source)
        # Header chunk should contain the import
        assert any("import os" in c for c in chunks)

    def test_syntax_error_returns_whole_file(self) -> None:
        bad_source = "def foo(\n    # unterminated"
        chunks = _split_python_by_ast(bad_source)
        assert len(chunks) == 1
        assert chunks[0] == bad_source

    def test_empty_file_returns_empty(self) -> None:
        chunks = _split_python_by_ast("")
        # Either returns [] or a single empty-ish chunk — both are acceptable
        non_empty = [c for c in chunks if c.strip()]
        assert non_empty == []

    def test_no_functions_returns_whole_file(self) -> None:
        source = "x = 1\ny = 2\n"
        chunks = _split_python_by_ast(source)
        assert len(chunks) == 1

    def test_async_function_split(self) -> None:
        source = textwrap.dedent("""\
            async def fetch():
                pass

            async def post():
                pass
        """)
        chunks = _split_python_by_ast(source)
        assert len(chunks) == 2


# ── _split_doc ────────────────────────────────────────────────────────────────


class TestSplitDoc:
    def test_short_text_single_chunk(self) -> None:
        text = "This is a short document."
        chunks = _split_doc(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_long_text_multiple_chunks(self) -> None:
        # Generate text clearly longer than 512 tokens
        text = "IBM Z Spyre is a high-performance AI inference accelerator. " * 200
        chunks = _split_doc(text)
        assert len(chunks) > 1

    def test_no_empty_chunks(self) -> None:
        text = "Hello world.\n\n\nAnother paragraph."
        chunks = _split_doc(text)
        assert all(c.strip() for c in chunks)


# ── _split_markdown ───────────────────────────────────────────────────────────


class TestSplitMarkdown:
    def test_code_fence_extracted_as_code(self) -> None:
        md = textwrap.dedent("""\
            # Guide

            Some prose here.

            ```python
            def spyre_op():
                pass
            ```

            More prose after.
        """)
        results = _split_markdown(md)
        types = [dtype for _, dtype in results]
        assert "code" in types
        assert "doc" in types

    def test_code_fence_text_correct(self) -> None:
        md = "Prose.\n```python\nx = 1\n```\nMore prose."
        results = _split_markdown(md)
        code_chunks = [t for t, dt in results if dt == "code"]
        assert any("x = 1" in c for c in code_chunks)

    def test_no_fence_all_doc(self) -> None:
        md = "# Title\n\nJust documentation text with no code."
        results = _split_markdown(md)
        assert all(dt == "doc" for _, dt in results)


# ── ChunkNode ─────────────────────────────────────────────────────────────────


class TestChunkNode:
    def _make_node(self, text: str = "hello world") -> ChunkNode:
        return ChunkNode(
            text=text,
            metadata={
                "file_path": "torch_spyre/foo.py",
                "source_url": "torch_spyre/foo.py",
                "doc_type": "code",
                "component_tag": "foo",
                "last_updated": "2025-01-01",
                "git_blame_author": "test",
                "version": "abc1234",
                "chunk_index": 0,
                "content_hash": _md5(text),
                "staleness_ttl_flag": False,
            },
        )

    def test_content_hash_property(self) -> None:
        node = self._make_node("test text")
        assert node.content_hash == _md5("test text")

    def test_to_text_node_roundtrip(self) -> None:
        from llama_index.core.schema import TextNode

        node = self._make_node("roundtrip test")
        text_node = node.to_text_node()
        assert isinstance(text_node, TextNode)
        assert text_node.text == "roundtrip test"
        assert text_node.metadata["doc_type"] == "code"

    def test_embedding_starts_none(self) -> None:
        node = self._make_node()
        assert node.embedding is None


# ── chunk_file integration — real torch_spyre files ───────────────────────────


class TestChunkFileIntegration:
    """
    Uses real files from the torch_spyre repo.
    Git calls are mocked to return deterministic values.
    """

    @pytest.fixture(autouse=True)
    def mock_git(self):
        with (
            patch(
                "ask_z.ingestion.chunker._git_log",
                return_value=("2025-06-01", "test-author"),
            ),
            patch("ask_z.ingestion.chunker._git_head_sha", return_value="abc1234"),
        ):
            yield

    def test_python_file_produces_chunks(self) -> None:
        py_file = REPO_ROOT / "torch_spyre" / "model_utils.py"
        if not py_file.exists():
            pytest.skip("torch_spyre/model_utils.py not found")

        nodes = chunk_file(py_file, REPO_ROOT)
        assert len(nodes) > 0

    def test_python_chunks_have_required_metadata_keys(self) -> None:
        py_file = REPO_ROOT / "torch_spyre" / "model_utils.py"
        if not py_file.exists():
            pytest.skip("torch_spyre/model_utils.py not found")

        nodes = chunk_file(py_file, REPO_ROOT)
        required_keys = {
            "file_path",
            "source_url",
            "doc_type",
            "component_tag",
            "last_updated",
            "git_blame_author",
            "version",
            "chunk_index",
            "content_hash",
            "staleness_ttl_flag",
        }
        for node in nodes:
            assert required_keys.issubset(node.metadata.keys()), (
                f"Missing keys: {required_keys - node.metadata.keys()}"
            )

    def test_python_chunks_doc_type_is_code_or_test(self) -> None:
        py_file = REPO_ROOT / "torch_spyre" / "model_utils.py"
        if not py_file.exists():
            pytest.skip("torch_spyre/model_utils.py not found")

        nodes = chunk_file(py_file, REPO_ROOT)
        for node in nodes:
            assert node.metadata["doc_type"] in ("code", "test", "ADR")

    def test_python_chunks_have_unique_hashes(self) -> None:
        py_file = REPO_ROOT / "torch_spyre" / "model_utils.py"
        if not py_file.exists():
            pytest.skip("torch_spyre/model_utils.py not found")

        nodes = chunk_file(py_file, REPO_ROOT)
        hashes = [n.content_hash for n in nodes]
        assert len(hashes) == len(set(hashes)), "Duplicate content hashes found"

    def test_python_chunks_staleness_flag_false(self) -> None:
        py_file = REPO_ROOT / "torch_spyre" / "model_utils.py"
        if not py_file.exists():
            pytest.skip("torch_spyre/model_utils.py not found")

        nodes = chunk_file(py_file, REPO_ROOT)
        assert all(node.metadata["staleness_ttl_flag"] is False for node in nodes)

    def test_markdown_file_produces_chunks(self) -> None:
        md_file = (
            REPO_ROOT
            / "docs"
            / "source"
            / "user_guide"
            / "profiling"
            / "end_to_end_example.md"
        )
        if not md_file.exists():
            pytest.skip("profiling end_to_end_example.md not found")

        nodes = chunk_file(md_file, REPO_ROOT)
        assert len(nodes) > 0

    def test_markdown_chunks_have_doc_or_code_type(self) -> None:
        md_file = (
            REPO_ROOT
            / "docs"
            / "source"
            / "user_guide"
            / "profiling"
            / "end_to_end_example.md"
        )
        if not md_file.exists():
            pytest.skip("profiling end_to_end_example.md not found")

        nodes = chunk_file(md_file, REPO_ROOT)
        for node in nodes:
            assert node.metadata["doc_type"] in ("doc", "code", "ADR")

    def test_chunk_indices_are_sequential(self) -> None:
        py_file = REPO_ROOT / "torch_spyre" / "model_utils.py"
        if not py_file.exists():
            pytest.skip("torch_spyre/model_utils.py not found")

        nodes = chunk_file(py_file, REPO_ROOT)
        indices = [n.metadata["chunk_index"] for n in nodes]
        assert indices == list(range(len(indices)))

    def test_content_hash_matches_text(self) -> None:
        py_file = REPO_ROOT / "torch_spyre" / "model_utils.py"
        if not py_file.exists():
            pytest.skip("torch_spyre/model_utils.py not found")

        nodes = chunk_file(py_file, REPO_ROOT)
        for node in nodes:
            assert node.content_hash == _md5(node.text)

    def test_empty_file_returns_no_chunks(self, tmp_path) -> None:
        empty = tmp_path / "empty.py"
        empty.write_text("")
        nodes = chunk_file(empty, tmp_path)
        assert nodes == []
