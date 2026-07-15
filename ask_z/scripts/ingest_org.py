"""
ask_z/scripts/ingest_org.py
────────────────────────────
Ingest ALL repositories from a GitHub organisation into ask-z-knowledge.

Each repo is cloned to a temp directory, chunked with the standard pipeline,
and upserted into Elasticsearch.  Already-indexed chunks (MD5 delta) are
skipped so re-runs are fast.

Usage
──────
  # Ingest the full torch-spyre org:
  python -m ask_z.scripts.ingest_org --org torch-spyre

  # Skip specific repos (e.g. forks, archived, or irrelevant):
  python -m ask_z.scripts.ingest_org --org torch-spyre \\
      --skip triton aiu-bench

  # Only specific repos:
  python -m ask_z.scripts.ingest_org --org torch-spyre \\
      --only RFCs hf-adapters spyre-inference

  # Dry run — list repos that would be ingested, don't clone or index:
  python -m ask_z.scripts.ingest_org --org torch-spyre --dry-run

Required env (read from ask_z/.env):
  GITHUB_TOKEN       GitHub personal access token (for private repos + higher rate limit)
  ELASTIC_HOST, ELASTIC_INDEX
  WATSONX_API_KEY, WATSONX_PROJECT_ID  (BM25-only fallback if quota exhausted)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError

from ask_z.ingestion.chunker import chunk_directory
from ask_z.scripts.ingest_docs import _build_es_client, ingest_docs as _ingest_docs_bulk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ask_z.ingest_org")

# File extensions to index from each repo.
_EXTENSIONS = {".py", ".md", ".rst", ".txt"}

# Always skip these directories inside any repo.
_EXCLUDE = [
    "__pycache__",
    ".venv",
    ".git",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".eggs",
    "third_party",
]

# Repos to skip by default (forks, benchmarks, low signal).
_DEFAULT_SKIP = {"triton"}


# ── GitHub API helpers ────────────────────────────────────────────────────────


def _gh_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _list_org_repos(org: str) -> list[dict[str, Any]]:
    """Return all repos in *org* via GitHub API (handles pagination)."""
    import urllib.request
    import json

    repos: list[dict] = []
    url = f"https://api.github.com/orgs/{org}/repos?per_page=100&type=all"
    headers = _gh_headers()

    while url:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                page = json.loads(resp.read())
                repos.extend(page)
                # Follow Link: header for pagination
                link = resp.headers.get("Link", "")
                url = None
                for part in link.split(","):
                    part = part.strip()
                    if 'rel="next"' in part:
                        url = part.split(";")[0].strip().strip("<>")
        except Exception as exc:
            log.error("GitHub API error listing %s repos: %s", org, exc)
            break

    return repos


# ── Clone helpers ─────────────────────────────────────────────────────────────


def _clone_repo(clone_url: str, dest: Path, token: str | None) -> bool:
    """Shallow-clone *clone_url* into *dest*. Returns True on success."""
    # Inject token into HTTPS URL for private repos.
    if token and clone_url.startswith("https://"):
        clone_url = clone_url.replace("https://", f"https://x-access-token:{token}@", 1)

    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", clone_url, str(dest)],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log.error("git clone failed for %s: %s", clone_url, exc.stderr.decode()[:200])
        return False
    except subprocess.TimeoutExpired:
        log.error("git clone timed out for %s", clone_url)
        return False


# ── Per-repo ingestion ────────────────────────────────────────────────────────


def _ingest_repo(
    repo: dict[str, Any],
    *,
    es: Elasticsearch,
    index: str,
    token: str | None,
) -> tuple[int, int]:
    """
    Clone *repo*, chunk its files, upsert into ES.
    Returns (chunks_produced, chunks_upserted).
    """
    name = repo["name"]
    clone_url = repo["clone_url"]
    default_branch = repo.get("default_branch", "main")
    tag = name.replace("-", "_").lower()

    log.info("━━━ %s (branch=%s) ━━━", name, default_branch)

    with tempfile.TemporaryDirectory(prefix=f"ask_z_{name}_") as tmp:
        tmp_path = Path(tmp)
        if not _clone_repo(clone_url, tmp_path, token):
            log.warning("Skipping %s — clone failed.", name)
            return 0, 0

        # Walk the cloned repo for supported files.
        nodes = list(
            chunk_directory(
                tmp_path,
                tmp_path,
                extensions=_EXTENSIONS,
                exclude_patterns=_EXCLUDE,
            )
        )
        log.info("%s: %d chunks produced", name, len(nodes))

        if not nodes:
            return 0, 0

        # Convert ChunkNode objects to the dict format expected by ingest_docs.
        chunks: list[dict] = []
        for node in nodes:
            # Make file_path relative to repo root, prefixed with repo name
            # so citations show e.g. "hf-adapters/src/model.py"
            try:
                rel = Path(node.metadata["file_path"]).relative_to(tmp_path)
                display_path = f"{name}/{rel}"
            except ValueError:
                display_path = node.metadata["file_path"]

            chunks.append(
                {
                    "text": node.text,
                    "metadata": {
                        **node.metadata,
                        "file_path": display_path,
                        "source_url": f"https://github.com/torch-spyre/{name}/blob/{default_branch}/{rel}",
                        "component_tag": tag,
                        "doc_type": node.metadata.get("doc_type", "code"),
                        "version": node.metadata.get("version", "unknown"),
                        "content_hash": node.content_hash,
                        "chunk_index": node.metadata.get("chunk_index", 0),
                    },
                }
            )

        _ingest_docs_bulk(chunks, es, index)
        return len(nodes), len(chunks)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest all repos from a GitHub org into ask-z-knowledge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--org",
        default="torch-spyre",
        metavar="ORG",
        help="GitHub org name (default: torch-spyre).",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        metavar="REPO",
        default=[],
        help="Repo names to skip (in addition to defaults: triton).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="REPO",
        default=[],
        help="If set, only ingest these repo names (ignores --skip).",
    )
    parser.add_argument(
        "--include-archived",
        action="store_true",
        default=False,
        help="Also ingest archived repos (skipped by default).",
    )
    parser.add_argument(
        "--include-forks",
        action="store_true",
        default=False,
        help="Also ingest forked repos (skipped by default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List repos that would be ingested without cloning or indexing.",
    )
    args = parser.parse_args()

    index = os.environ.get("ELASTIC_INDEX", "ask-z-knowledge")
    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        log.warning(
            "GITHUB_TOKEN not set — private repos will fail to clone "
            "and rate limits will be low (60 req/hr). "
            "Add GITHUB_TOKEN to ask_z/.env for best results."
        )

    # List all repos.
    log.info("Fetching repo list for org: %s", args.org)
    all_repos = _list_org_repos(args.org)
    if not all_repos:
        log.error("No repos found (or API error). Check GITHUB_TOKEN and org name.")
        sys.exit(1)

    log.info("Found %d repos in org '%s'.", len(all_repos), args.org)

    # Filter.
    skip_set = _DEFAULT_SKIP | set(args.skip)
    only_set = set(args.only)

    repos_to_ingest = []
    for repo in sorted(all_repos, key=lambda r: r["name"]):
        name = repo["name"]
        reasons: list[str] = []
        if only_set and name not in only_set:
            reasons.append("not in --only list")
        elif name in skip_set:
            reasons.append("in skip list")
        elif repo.get("archived") and not args.include_archived:
            reasons.append("archived")
        elif repo.get("fork") and not args.include_forks:
            reasons.append("fork")
        if reasons:
            log.info("  SKIP  %s  (%s)", name, ", ".join(reasons))
        else:
            repos_to_ingest.append(repo)
            log.info(
                "  INGEST  %s  (%s)",
                name,
                "private" if repo.get("private") else "public",
            )

    log.info("\n%d repo(s) selected for ingestion.", len(repos_to_ingest))

    if args.dry_run:
        log.info("Dry run — exiting without ingesting.")
        return

    if not repos_to_ingest:
        log.info("Nothing to ingest.")
        return

    # Connect to ES.
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

    # Ingest each repo.
    total_chunks = 0
    total_repos_ok = 0
    for repo in repos_to_ingest:
        produced, upserted = _ingest_repo(repo, es=es, index=index, token=token)
        total_chunks += produced
        if produced > 0:
            total_repos_ok += 1

    log.info(
        "\n✓ Done. %d/%d repos ingested, %d total chunks produced.",
        total_repos_ok,
        len(repos_to_ingest),
        total_chunks,
    )


if __name__ == "__main__":
    main()
