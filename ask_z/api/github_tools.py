"""
ask_z/api/github_tools.py
──────────────────────────
GitHub PR fetcher for the Ask-Z pipeline.

Detects when a query is asking about a GitHub PR and fetches live data
from the GitHub REST API — title, description, diff stats, file list,
comments, and review status — then returns it as a ContextChunk so the
generator can produce a grounded summary.

Supported query patterns (case-insensitive):
  "summarise PR 4345"
  "summarize PR #3189"
  "what is in PR 3200?"
  "can you explain pull request 4345"
  "PR #4345 summary"
  "review PR 4000"
  "tell me about pr 3500"

Environment variable:
  GITHUB_TOKEN — personal access token for private repos + higher rate limit.
                 Without it, public repos work but rate limit is 60 req/hr.
"""

from __future__ import annotations

import logging
import os
import regex as re
from typing import Any

import httpx

log = logging.getLogger("ask_z.api.github_tools")

# Default org — the one Ask-Z is built for.
_DEFAULT_ORG = "torch-spyre"
_DEFAULT_REPO = "torch-spyre"

# Regex to extract PR number from a natural-language query.
_PR_PATTERN = re.compile(
    r"""
    (?:
        (?:summarise|summarize|summary|explain|describe|review|show|tell\s+me\s+about|what(?:'s|\s+is)?\s+in)
        \s+
        (?:pr|pull\s*request|pull-request)
        \s*[#]?(\d+)
    |
        (?:pr|pull\s*request|pull-request)
        \s*[#]?(\d+)
        (?:\s+(?:summary|summarise|summarize|review|details?))?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_pr_query(query: str) -> int | None:
    """
    Return the PR number if the query is asking about a specific PR,
    otherwise return None.
    """
    m = _PR_PATTERN.search(query)
    if m:
        num = m.group(1) or m.group(2)
        return int(num)
    return None


def _gh_headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def fetch_pr_context(
    pr_number: int,
    http_client: httpx.AsyncClient,
    *,
    org: str = _DEFAULT_ORG,
    repo: str = _DEFAULT_REPO,
) -> dict[str, Any] | None:
    """
    Fetch a GitHub PR and return a structured context dict ready for the
    generator.  Returns None if the PR cannot be fetched.

    Fetches:
      - PR metadata (title, author, state, labels, base/head branch)
      - PR body (description)
      - Changed files list (filename + additions + deletions)
      - Review comments summary (latest 10)
      - Review status (approved/changes-requested/pending)
    """
    base = f"https://api.github.com/repos/{org}/{repo}"
    headers = _gh_headers()

    try:
        # ── 1. PR metadata + body ──────────────────────────────────────────
        pr_resp = await http_client.get(
            f"{base}/pulls/{pr_number}", headers=headers, timeout=10
        )
        if pr_resp.status_code == 404:
            log.warning("PR #%d not found in %s/%s.", pr_number, org, repo)
            return None
        pr_resp.raise_for_status()
        pr = pr_resp.json()

        # ── 2. Changed files ───────────────────────────────────────────────
        files_resp = await http_client.get(
            f"{base}/pulls/{pr_number}/files",
            headers=headers,
            params={"per_page": 50},
            timeout=10,
        )
        files_resp.raise_for_status()
        files = files_resp.json()

        # ── 3. Reviews ────────────────────────────────────────────────────
        reviews_resp = await http_client.get(
            f"{base}/pulls/{pr_number}/reviews",
            headers=headers,
            params={"per_page": 20},
            timeout=10,
        )
        reviews_resp.raise_for_status()
        reviews = reviews_resp.json()

        # ── 4. Comments (issue-level) ──────────────────────────────────────
        comments_resp = await http_client.get(
            f"{base}/issues/{pr_number}/comments",
            headers=headers,
            params={"per_page": 10},
            timeout=10,
        )
        comments_resp.raise_for_status()
        comments = comments_resp.json()

    except httpx.HTTPStatusError as exc:
        log.error(
            "GitHub API error fetching PR #%d: HTTP %s",
            pr_number,
            exc.response.status_code,
        )
        return None
    except Exception as exc:
        log.error("GitHub API error fetching PR #%d: %s", pr_number, exc)
        return None

    # ── Build structured context ───────────────────────────────────────────
    file_lines: list[str] = []
    total_additions = 0
    total_deletions = 0
    for f in files[:30]:  # cap at 30 files to stay within context window
        fname = f.get("filename", "")
        add = f.get("additions", 0)
        dele = f.get("deletions", 0)
        status_str = f.get("status", "")
        total_additions += add
        total_deletions += dele
        file_lines.append(f"  {status_str:10s}  +{add:4d} / -{dele:4d}  {fname}")
    if len(files) > 30:
        file_lines.append(f"  ... and {len(files) - 30} more files")

    # Review summary
    review_states: dict[str, list[str]] = {}
    for r in reviews:
        state = r.get("state", "COMMENTED")
        user = r.get("user", {}).get("login", "unknown")
        review_states.setdefault(state, []).append(user)

    review_lines: list[str] = []
    for state, users in review_states.items():
        review_lines.append(f"  {state}: {', '.join(users)}")
    if not review_lines:
        review_lines = ["  No reviews yet."]

    # Comment excerpts
    comment_lines: list[str] = []
    for c in comments[:5]:
        user = c.get("user", {}).get("login", "?")
        body = (c.get("body") or "").strip()[:200]
        if body:
            comment_lines.append(f"  @{user}: {body}")

    # Labels
    labels = [lbl.get("name", "") for lbl in pr.get("labels", [])]

    # Assemble context text
    pr_url = pr.get("html_url", f"https://github.com/{org}/{repo}/pull/{pr_number}")
    body_text = (pr.get("body") or "(No description provided.)").strip()[:1500]

    context_text = f"""GitHub Pull Request #{pr_number} — {org}/{repo}

Title:   {pr.get("title", "(no title)")}
Author:  {pr.get("user", {}).get("login", "unknown")}
State:   {pr.get("state", "unknown").upper()}
Draft:   {"Yes" if pr.get("draft") else "No"}
Branch:  {pr.get("head", {}).get("ref", "?")} → {pr.get("base", {}).get("ref", "?")}
Labels:  {", ".join(labels) if labels else "none"}
URL:     {pr_url}

Description:
{body_text}

Changed Files ({len(files)} total · +{total_additions} / -{total_deletions} lines):
{chr(10).join(file_lines) if file_lines else "  (no files)"}

Review Status:
{chr(10).join(review_lines)}

Comments ({len(comments)} total):
{chr(10).join(comment_lines) if comment_lines else "  (no comments)"}
"""

    return {
        "pr_number": pr_number,
        "title": pr.get("title", ""),
        "author": pr.get("user", {}).get("login", ""),
        "state": pr.get("state", ""),
        "url": pr_url,
        "org": org,
        "repo": repo,
        "context_text": context_text,
        "files_changed": len(files),
        "additions": total_additions,
        "deletions": total_deletions,
        "labels": labels,
        "reviews": review_states,
    }
