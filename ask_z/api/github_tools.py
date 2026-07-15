"""
ask_z/api/github_tools.py
──────────────────────────
GitHub PR fetcher for the Ask-Z pipeline.

Detects when a query is asking about a GitHub PR and fetches live data
from the GitHub REST API — title, description, diff stats, file list,
comments, review status, and optionally the actual patch diff for code
review mode.

Supported query patterns (case-insensitive):
  "summarise PR 4345"        → intent="summary"
  "summarize PR #3189"       → intent="summary"
  "what is in PR 3200?"      → intent="summary"
  "can you explain PR 4345"  → intent="summary"
  "PR #4345 summary"         → intent="summary"
  "review PR 4000"           → intent="review"
  "do a code review of PR 3500"  → intent="review"
  "check PR 3178"            → intent="review"

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

# ── Intent-aware PR detection ─────────────────────────────────────────────────

# Patterns that signal a code-review intent (vs a plain summary).
_REVIEW_VERBS = re.compile(
    r"\b(review|code[\s-]review|check|inspect|audit|critique|assess)\b",
    re.IGNORECASE,
)

# Patterns that extract a PR number from the query.
_PR_NUMBER_PATTERN = re.compile(
    r"""
    (?:
        (?:summarise|summarize|summary|explain|describe|review|show
          |check|inspect|audit|critique|assess
          |tell\s+me\s+about|what(?:'s|\s+is)?\s+in
          |code[\s-]review\s+(?:of\s+)?|do\s+a\s+(?:code[\s-])?review\s+(?:of\s+)?)
        \s+
        (?:pr|pull\s*request|pull-request)
        \s*[#]?(\d+)
    |
        (?:pr|pull\s*request|pull-request)
        \s*[#]?(\d+)
        (?:\s+(?:summary|summarise|summarize|review|check|details?))?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_pr_query(query: str) -> tuple[int, str] | None:
    """
    Return ``(pr_number, intent)`` if the query is asking about a specific PR,
    otherwise return ``None``.

    ``intent`` is either ``"review"`` (code-review request) or ``"summary"``.
    """
    m = _PR_NUMBER_PATTERN.search(query)
    if not m:
        return None
    num = int(m.group(1) or m.group(2))
    intent = "review" if _REVIEW_VERBS.search(query) else "summary"
    return num, intent


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
    include_diff: bool = False,
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

    When ``include_diff=True`` the actual patch hunks from each changed file
    are fetched and appended so the LLM can do a real code review.
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

        # ── 2. Changed files (includes patch hunk when include_diff=True) ──
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

        # ── 5. Review comments (inline, line-level) ───────────────────────
        review_comments_resp = await http_client.get(
            f"{base}/pulls/{pr_number}/comments",
            headers=headers,
            params={"per_page": 20},
            timeout=10,
        )
        review_comments_resp.raise_for_status()
        review_comments = review_comments_resp.json()

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

    # Comment excerpts (issue-level)
    comment_lines: list[str] = []
    for c in comments[:5]:
        user = c.get("user", {}).get("login", "?")
        body = (c.get("body") or "").strip()[:200]
        if body:
            comment_lines.append(f"  @{user}: {body}")

    # Inline review comments (line-level, most useful for code review)
    inline_comment_lines: list[str] = []
    for c in review_comments[:15]:
        user = c.get("user", {}).get("login", "?")
        path = c.get("path", "")
        line = c.get("line") or c.get("original_line") or "?"
        body = (c.get("body") or "").strip()[:300]
        if body:
            inline_comment_lines.append(f"  @{user} on {path}:{line} — {body}")

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

Discussion Comments ({len(comments)} total):
{chr(10).join(comment_lines) if comment_lines else "  (no comments)"}

Inline Review Comments ({len(review_comments)} total):
{chr(10).join(inline_comment_lines) if inline_comment_lines else "  (no inline comments)"}
"""

    # ── Append diff patch for review mode ─────────────────────────────────
    if include_diff:
        diff_sections: list[str] = []
        # Budget: ~6000 chars of diff to stay comfortably within context window.
        budget = 6000
        for f in files[:20]:
            if budget <= 0:
                break
            fname = f.get("filename", "")
            patch = (f.get("patch") or "").strip()
            if not patch:
                continue
            # Truncate per-file patch if very long.
            if len(patch) > 1500:
                patch = patch[:1500] + "\n... (truncated)"
            section = f"--- {fname} ---\n{patch}"
            diff_sections.append(section)
            budget -= len(section)
        if diff_sections:
            context_text += "\nCode Diff:\n" + "\n\n".join(diff_sections) + "\n"
        else:
            context_text += "\nCode Diff:\n  (no patch data available)\n"

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
