"""
ask_z/tests/test_living_arch_documenter.py
───────────────────────────────────────────
Unit tests for ask_z/scripts/living_arch_documenter.py.

All git, filesystem, and HTTP calls are mocked.
No live services, no actual git history required.

Run with:  pytest ask_z/tests/test_living_arch_documenter.py -v
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch


import ask_z.scripts.living_arch_documenter as doc


# ── _component_tag ────────────────────────────────────────────────────────────


class TestComponentTag:
    def test_python_module(self, tmp_path) -> None:
        # Leading underscores are stripped → "_inductor" becomes "inductor"
        with patch.object(doc, "REPO_ROOT", tmp_path):
            f = tmp_path / "torch_spyre" / "_inductor" / "spyre_attention.py"
            tag = doc._component_tag(f)
        assert tag == "inductor"

    def test_special_chars_replaced(self, tmp_path) -> None:
        with patch.object(doc, "REPO_ROOT", tmp_path):
            f = tmp_path / "My Module" / "foo.py"
            tag = doc._component_tag(f)
        assert " " not in tag

    def test_top_level_uses_stem(self, tmp_path) -> None:
        with patch.object(doc, "REPO_ROOT", tmp_path):
            f = tmp_path / "README.md"
            tag = doc._component_tag(f)
        assert tag == "readme"


# ── _adr_filename ─────────────────────────────────────────────────────────────


class TestAdrFilename:
    def test_returns_correct_path(self, tmp_path) -> None:
        with patch.object(doc, "ADR_OUTPUT_DIR", tmp_path / "adr"):
            result = doc._adr_filename("spyre_attention")
        assert result == tmp_path / "adr" / "adr-spyre_attention.md"


# ── load_existing_adr ─────────────────────────────────────────────────────────


class TestLoadExistingAdr:
    def test_returns_content_when_file_exists(self, tmp_path) -> None:
        adr_dir = tmp_path / "adr"
        adr_dir.mkdir()
        adr_file = adr_dir / "adr-ops.md"
        adr_file.write_text("# ADR: ops\n\nExisting content.")
        with patch.object(doc, "ADR_OUTPUT_DIR", adr_dir):
            content = doc.load_existing_adr("ops")
        assert "Existing content." in content

    def test_returns_empty_when_no_file(self, tmp_path) -> None:
        with patch.object(doc, "ADR_OUTPUT_DIR", tmp_path / "adr"):
            content = doc.load_existing_adr("nonexistent")
        assert content == ""


# ── write_adr ─────────────────────────────────────────────────────────────────


class TestWriteAdr:
    def test_creates_file(self, tmp_path) -> None:
        adr_dir = tmp_path / "adr"
        with patch.object(doc, "ADR_OUTPUT_DIR", adr_dir):
            path = doc.write_adr("model_utils", "# ADR: model_utils\n\nContent.")
        assert path.exists()
        assert path.read_text() == "# ADR: model_utils\n\nContent."

    def test_creates_parent_dirs(self, tmp_path) -> None:
        adr_dir = tmp_path / "deep" / "nested" / "adr"
        with patch.object(doc, "ADR_OUTPUT_DIR", adr_dir):
            doc.write_adr("comp", "content")
        assert adr_dir.exists()

    def test_overwrites_existing(self, tmp_path) -> None:
        adr_dir = tmp_path / "adr"
        adr_dir.mkdir()
        with patch.object(doc, "ADR_OUTPUT_DIR", adr_dir):
            doc.write_adr("comp", "v1")
            doc.write_adr("comp", "v2")
            assert (adr_dir / "adr-comp.md").read_text() == "v2"


# ── update_adr_index ──────────────────────────────────────────────────────────


class TestUpdateAdrIndex:
    def test_generates_index_with_all_adrs(self, tmp_path) -> None:
        adr_dir = tmp_path / "adr"
        adr_dir.mkdir()
        (adr_dir / "adr-ops.md").write_text("# ops")
        (adr_dir / "adr-attention.md").write_text("# attention")
        with patch.object(doc, "ADR_OUTPUT_DIR", adr_dir):
            doc.update_adr_index()
        index = (adr_dir / "index.md").read_text()
        assert "adr-ops.md" in index
        assert "adr-attention.md" in index

    def test_index_contains_table_header(self, tmp_path) -> None:
        adr_dir = tmp_path / "adr"
        adr_dir.mkdir()
        (adr_dir / "adr-x.md").write_text("")
        with patch.object(doc, "ADR_OUTPUT_DIR", adr_dir):
            doc.update_adr_index()
        index = (adr_dir / "index.md").read_text()
        assert "| Component |" in index

    def test_no_index_when_no_adrs(self, tmp_path) -> None:
        adr_dir = tmp_path / "adr"
        adr_dir.mkdir()
        with patch.object(doc, "ADR_OUTPUT_DIR", adr_dir):
            doc.update_adr_index()
        assert not (adr_dir / "index.md").exists()


# ── get_changed_files ─────────────────────────────────────────────────────────


class TestGetChangedFiles:
    def test_filters_watched_extensions(self, tmp_path) -> None:
        # Create real files so path.exists() passes
        (tmp_path / "torch_spyre").mkdir()
        py_file = tmp_path / "torch_spyre" / "ops.py"
        py_file.write_text("x=1")
        txt_file = tmp_path / "torch_spyre" / "data.txt"
        txt_file.write_text("data")

        diff_output = "torch_spyre/ops.py\ntorch_spyre/data.txt\n"
        mock_result = MagicMock(returncode=0, stdout=diff_output)

        with (
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch("subprocess.run", return_value=mock_result),
        ):
            files = doc.get_changed_files()

        assert any("ops.py" in str(f) for f in files)
        assert not any("data.txt" in str(f) for f in files)

    def test_excludes_ask_z_own_files(self, tmp_path) -> None:
        ask_z_dir = tmp_path / "ask_z" / "api"
        ask_z_dir.mkdir(parents=True)
        ask_z_file = ask_z_dir / "app.py"
        ask_z_file.write_text("x=1")

        diff_output = "ask_z/api/app.py\n"
        mock_result = MagicMock(returncode=0, stdout=diff_output)

        with (
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch("subprocess.run", return_value=mock_result),
        ):
            files = doc.get_changed_files()

        assert files == []

    def test_returns_empty_on_no_diff(self, tmp_path) -> None:
        mock_result = MagicMock(returncode=0, stdout="")
        # Second call for --cached also returns nothing
        with (
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch("subprocess.run", return_value=mock_result),
        ):
            files = doc.get_changed_files()
        assert files == []


# ── get_file_diff ─────────────────────────────────────────────────────────────


class TestGetFileDiff:
    def test_returns_diff_output(self, tmp_path) -> None:
        f = tmp_path / "foo.py"
        f.write_text("x=1")
        mock_result = MagicMock(returncode=0, stdout="+x=1\n-x=0\n")
        with (
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch("subprocess.run", return_value=mock_result),
        ):
            diff = doc.get_file_diff(f)
        assert "+x=1" in diff

    def test_falls_back_to_file_content_when_no_diff(self, tmp_path) -> None:
        f = tmp_path / "new_file.py"
        f.write_text("def new_function(): pass")
        mock_result = MagicMock(returncode=0, stdout="")
        with (
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch("subprocess.run", return_value=mock_result),
        ):
            diff = doc.get_file_diff(f)
        assert "[NEW FILE]" in diff
        assert "new_function" in diff

    def test_truncates_to_max_chars(self, tmp_path) -> None:
        f = tmp_path / "big.py"
        f.write_text("x" * 10_000)
        big_diff = "+" + "x" * 10_000
        mock_result = MagicMock(returncode=0, stdout=big_diff)
        with (
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch("subprocess.run", return_value=mock_result),
        ):
            diff = doc.get_file_diff(f)
        assert len(diff) <= doc.MAX_DIFF_CHARS


# ── notify_ingestion_webhook ──────────────────────────────────────────────────


class TestNotifyIngestionWebhook:
    def test_skips_when_webhook_not_configured(self, tmp_path) -> None:
        with (
            patch.object(doc, "ASKZ_WEBHOOK", ""),
            patch("urllib.request.urlopen") as mock_open,
        ):
            doc.notify_ingestion_webhook([tmp_path / "foo.md"])
        mock_open.assert_not_called()

    def test_posts_to_webhook_when_configured(self, tmp_path) -> None:
        f = tmp_path / "foo.md"
        f.write_text("")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(doc, "ASKZ_WEBHOOK", "https://example.com/webhook"),
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch.object(doc, "ADR_OUTPUT_DIR", tmp_path / "adr"),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            doc.notify_ingestion_webhook([f])

        mock_resp.__enter__.assert_called_once()

    def test_non_fatal_on_webhook_failure(self, tmp_path) -> None:
        f = tmp_path / "foo.md"
        f.write_text("")
        with (
            patch.object(doc, "ASKZ_WEBHOOK", "https://bad.host/webhook"),
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch.object(doc, "ADR_OUTPUT_DIR", tmp_path / "adr"),
            patch("urllib.request.urlopen", side_effect=Exception("conn refused")),
        ):
            # Should not raise — webhook failure is non-fatal
            doc.notify_ingestion_webhook([f])


# ── main() integration ────────────────────────────────────────────────────────


class TestMain:
    def test_exits_1_when_no_credentials(self) -> None:
        with (
            patch.object(doc, "WATSONX_API_KEY", ""),
            patch.object(doc, "WATSONX_PROJECT_ID", ""),
        ):
            rc = doc.main()
        assert rc == 1

    def test_exits_0_when_no_changed_files(self) -> None:
        with (
            patch.object(doc, "WATSONX_API_KEY", "key"),
            patch.object(doc, "WATSONX_PROJECT_ID", "proj"),
            patch.object(doc, "get_changed_files", return_value=[]),
        ):
            rc = doc.main()
        assert rc == 0

    def test_exits_0_and_writes_adr_on_success(self, tmp_path) -> None:
        adr_dir = tmp_path / "adr"
        # File in torch_spyre/ → component tag = "torch_spyre" (parent dir name)
        py_file = tmp_path / "torch_spyre" / "ops.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("def forward(): pass")

        adr_content = textwrap.dedent("""\
            # ADR: torch_spyre
            **Date:** 2025-07-13
            ## Context
            Ops module.
            ## Decision
            Use tiled matmul.
            ## Consequences
            Faster inference.
            ## Implementation Notes
            See ops.py.
        """)

        written: list[Path] = []

        def fake_write_adr(component_tag: str, adr_markdown: str) -> Path:
            adr_dir.mkdir(parents=True, exist_ok=True)
            out = adr_dir / f"adr-{component_tag}.md"
            out.write_text(adr_markdown, encoding="utf-8")
            written.append(out)
            return out

        with (
            patch.object(doc, "WATSONX_API_KEY", "key"),
            patch.object(doc, "WATSONX_PROJECT_ID", "proj"),
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch.object(doc, "ADR_OUTPUT_DIR", adr_dir),
            patch.object(doc, "get_changed_files", return_value=[py_file]),
            patch.object(
                doc, "get_file_diff", return_value="+" + "def forward(): pass\n" * 10
            ),
            patch.object(
                doc,
                "generate_docs",
                return_value=("Ops module handles matmul.", adr_content),
            ),
            patch.object(doc, "write_adr", side_effect=fake_write_adr),
            patch.object(doc, "update_adr_index"),
            patch.object(doc, "notify_ingestion_webhook"),
        ):
            rc = doc.main()

        assert rc == 0
        assert len(written) == 1
        assert "tiled matmul" in written[0].read_text()

    def test_continues_on_generation_failure(self, tmp_path) -> None:
        """A generation failure for one file should not abort the whole run."""
        py_file = tmp_path / "torch_spyre" / "ops.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("def forward(): pass")

        with (
            patch.object(doc, "WATSONX_API_KEY", "key"),
            patch.object(doc, "WATSONX_PROJECT_ID", "proj"),
            patch.object(doc, "REPO_ROOT", tmp_path),
            patch.object(doc, "ADR_OUTPUT_DIR", tmp_path / "adr"),
            patch.object(doc, "get_changed_files", return_value=[py_file]),
            patch.object(doc, "get_file_diff", return_value="+def forward(): pass"),
            patch.object(doc, "generate_docs", side_effect=RuntimeError("API error")),
        ):
            rc = doc.main()

        assert rc == 0  # non-fatal
