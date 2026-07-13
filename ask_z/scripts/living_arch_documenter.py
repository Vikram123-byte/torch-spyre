#!/usr/bin/env python3
"""
ask_z/scripts/living_arch_documenter.py
─────────────────────────────────────────
Living Architecture Documenter — runner script for the GitHub Actions workflow.

Execution steps
────────────────
1. DIFF     — Isolate changed Python/Markdown files between HEAD and HEAD~1.
2. CONTEXT  — For each changed file, load the matching existing ADR or README
              block if one exists.
3. GENERATE — Send diff + existing context to Granite via watsonx.ai text/chat.
              Granite returns: (a) updated component summary, (b) pre-filled ADR.
4. WRITE    — Write/update the ADR file under docs/source/architecture/adr/.
5. WEBHOOK  — POST a lightweight notification to the Ask-Z ingestion endpoint
              so the new ADR is immediately re-embedded into the vector store.

Anti-loop protection
─────────────────────
The GitHub Actions workflow skips runs triggered by commits whose message
contains "[skip-doc-bot]". This script never writes that token itself —
that token is added by the workflow's "Commit and push" step.

Run locally
────────────
    WATSONX_API_KEY=... WATSONX_PROJECT_ID=... \
    python ask_z/scripts/living_arch_documenter.py
"""

from __future__ import annotations

import json
import logging
import os
import regex as re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ask_z.living_arch")

# ── Constants ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
ADR_OUTPUT_DIR = REPO_ROOT / "docs" / "source" / "architecture" / "adr"

# File extensions that trigger documentation generation.
WATCHED_EXTENSIONS = {".py", ".md", ".rst"}

# Maximum diff size sent to Granite (chars). Prevents token overflow.
MAX_DIFF_CHARS = 6_000

# Minimum diff length — diffs shorter than this are trivial (whitespace etc.)
# and don't warrant a new ADR entry.
MIN_DIFF_CHARS = 80

# ── Environment ────────────────────────────────────────────────────────────────

WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_API_URL = os.environ.get(
    "WATSONX_API_URL", "https://us-south.ml.cloud.ibm.com/ml/v1"
)
WATSONX_API_VER = os.environ.get("WATSONX_API_VERSION", "2024-09-01")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "ibm/granite-4-h-small")
ASKZ_WEBHOOK = os.environ.get("ASKZ_INGEST_WEBHOOK", "")
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "")
GITHUB_SHA = os.environ.get("GITHUB_SHA", "")[:8] or "local"


# ── IAM token ─────────────────────────────────────────────────────────────────

_token: str | None = None
_token_expires: float = 0.0


def _get_token() -> str:
    global _token, _token_expires
    if _token and time.time() < _token_expires - 60:
        return _token
    if not WATSONX_API_KEY:
        raise EnvironmentError("WATSONX_API_KEY is not set.")
    data = urllib.parse.urlencode(
        {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": WATSONX_API_KEY,
        }
    ).encode()
    req = urllib.request.Request(
        "https://iam.cloud.ibm.com/identity/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        payload = json.load(r)
    _token = payload["access_token"]
    _token_expires = time.time() + payload.get("expires_in", 3600)
    log.info("IAM token refreshed.")
    return _token


# ── Step 1: Identify changed files ────────────────────────────────────────────


def get_changed_files() -> list[Path]:
    """
    Return repo-relative paths of files changed between HEAD and HEAD~1
    that match WATCHED_EXTENSIONS.

    Falls back to the full list of tracked source files when there is no
    parent commit (first commit) or when running outside of CI.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise ValueError("git diff returned no output.")
        changed = [
            REPO_ROOT / p.strip() for p in result.stdout.splitlines() if p.strip()
        ]
    except Exception as exc:
        log.warning("Could not diff HEAD~1 vs HEAD (%s). Using staged files.", exc)
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=30,
        )
        changed = [
            REPO_ROOT / p.strip() for p in result.stdout.splitlines() if p.strip()
        ]

    filtered = [
        p
        for p in changed
        if p.suffix.lower() in WATCHED_EXTENSIONS
        and p.exists()
        and "ask_z/" not in str(p.relative_to(REPO_ROOT))  # skip our own files
    ]
    log.info("Changed files matching watched extensions: %d", len(filtered))
    for p in filtered:
        log.info("  → %s", p.relative_to(REPO_ROOT))
    return filtered


def get_file_diff(file_path: Path) -> str:
    """Return the git diff for a single file, truncated to MAX_DIFF_CHARS."""
    result = subprocess.run(
        ["git", "diff", "HEAD~1", "HEAD", "--", str(file_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=30,
    )
    diff = result.stdout.strip()
    if not diff:
        # File is new — show the full content instead.
        try:
            diff = (
                f"[NEW FILE]\n{file_path.read_text(encoding='utf-8', errors='replace')}"
            )
        except OSError:
            diff = "[NEW FILE — content unreadable]"

    return diff[:MAX_DIFF_CHARS]


# ── Step 2: Load existing ADR context ─────────────────────────────────────────


def _component_tag(file_path: Path) -> str:
    """Derive a stable component tag from the file path."""
    rel = file_path.relative_to(REPO_ROOT)
    parts = rel.parts
    tag = parts[-2] if len(parts) >= 2 else rel.stem
    return re.sub(r"[^a-z0-9_-]", "_", tag.lower()).strip("_-")


def _adr_filename(component_tag: str) -> Path:
    return ADR_OUTPUT_DIR / f"adr-{component_tag}.md"


def load_existing_adr(component_tag: str) -> str:
    """Return the content of an existing ADR, or an empty string."""
    adr_file = _adr_filename(component_tag)
    if adr_file.exists():
        content = adr_file.read_text(encoding="utf-8")
        log.info("Loaded existing ADR: %s (%d chars)", adr_file.name, len(content))
        return content
    return ""


# ── Step 3: Granite generation ────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert software architect documenting an IBM Z and Spyre AI inference \
codebase. Given a git diff and (optionally) an existing Architecture Decision Record, \
your job is to produce two things:

1. COMPONENT_SUMMARY: A concise 2-3 sentence plain-English description of what \
   the changed component does and why the change was made. Max 80 words.

2. ADR: A complete, pre-filled Architecture Decision Record in Markdown using \
   the template below. Update existing sections if an existing ADR is provided; \
   create fresh content if none exists.

ADR template:
---
# ADR: {component_name}

**Date:** {today}
**Status:** Active
**Author:** ask-z-doc-bot
**Commit:** {sha}

## Context
<!-- What is this component and why does it exist? -->

## Decision
<!-- What specific architectural decision was made or changed? -->

## Consequences
<!-- What are the expected outcomes — positive and negative? -->

## Implementation Notes
<!-- Key code patterns, gotchas, or pointers for engineers. -->
---

Output format — respond with ONLY valid JSON, no markdown fences:
{
  "component_summary": "...",
  "adr_markdown": "..."
}
"""


def generate_docs(
    file_path: Path,
    diff: str,
    existing_adr: str,
) -> tuple[str, str]:
    """
    Call Granite to generate a component summary + ADR for the changed file.

    Returns (component_summary, adr_markdown).
    Raises on HTTP errors after retries.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    component = _component_tag(file_path)
    rel_path = str(file_path.relative_to(REPO_ROOT))
    source_url = (
        f"https://github.com/{GITHUB_REPO}/blob/{GITHUB_SHA}/{rel_path}"
        if GITHUB_REPO
        else rel_path
    )

    # Build user message.
    user_parts = [
        f"**File changed:** `{rel_path}` — {source_url}",
        f"**Component:** `{component}`",
        "",
        "**Git diff:**",
        "```diff",
        diff,
        "```",
    ]
    if existing_adr:
        user_parts += ["", "**Existing ADR (update this):**", existing_adr]
    else:
        user_parts.append("\n(No existing ADR — generate a new one.)")

    user_content = "\n".join(user_parts)

    system_filled = (
        _SYSTEM_PROMPT.replace("{component_name}", component)
        .replace("{today}", today)
        .replace("{sha}", GITHUB_SHA)
    )

    payload = json.dumps(
        {
            "model_id": GENERATION_MODEL,
            "project_id": WATSONX_PROJECT_ID,
            "messages": [
                {"role": "system", "content": system_filled},
                {"role": "user", "content": user_content},
            ],
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 800,
                "repetition_penalty": 1.05,
                "temperature": 0.0,
            },
        }
    ).encode()

    url = f"{WATSONX_API_URL}/text/chat?version={WATSONX_API_VER}"
    token = _get_token()

    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            raw = data["choices"][0]["message"]["content"].strip()
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                wait = 2**attempt
                log.warning("HTTP %s — retrying in %ds …", exc.code, wait)
                time.sleep(wait)
                token = _get_token()  # refresh token on 401
                continue
            log.error("Granite API error %s: %s", exc.code, body[:300])
            raise
    else:
        raise RuntimeError("Granite API failed after 3 attempts.")

    # ── Parse JSON response ───────────────────────────────────────────────
    # Strip markdown fences if the model wrapped the JSON anyway.
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        summary = parsed.get("component_summary", "").strip()
        adr = parsed.get("adr_markdown", "").strip()
    except json.JSONDecodeError:
        log.warning("JSON parse failed — extracting ADR from raw text.")
        # Best-effort: treat everything after "adr_markdown" as the ADR.
        summary = ""
        adr = raw

    return summary, adr


# ── Step 4: Write ADR to disk ─────────────────────────────────────────────────


def write_adr(component_tag: str, adr_markdown: str) -> Path:
    """Write the ADR markdown file and return its path."""
    ADR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adr_file = _adr_filename(component_tag)
    adr_file.write_text(adr_markdown, encoding="utf-8")
    try:
        display = adr_file.relative_to(REPO_ROOT)
    except ValueError:
        display = adr_file
    log.info("Wrote ADR: %s", display)
    return adr_file


def update_adr_index() -> None:
    """
    Regenerate docs/source/architecture/adr/index.md — a table of all ADRs.
    This gives engineers a single browsable list of every component decision.
    """
    adr_files = sorted(ADR_OUTPUT_DIR.glob("adr-*.md"))
    if not adr_files:
        return

    lines = [
        "# Architecture Decision Records",
        "",
        "Auto-generated by Living Architecture Documenter. "
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| Component | ADR File | Last Updated |",
        "|-----------|----------|--------------|",
    ]
    for f in adr_files:
        component = f.stem.replace("adr-", "")
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )
        lines.append(f"| `{component}` | [{f.name}](./{f.name}) | {mtime} |")

    index_file = ADR_OUTPUT_DIR / "index.md"
    index_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        display = index_file.relative_to(REPO_ROOT)
    except ValueError:
        display = index_file
    log.info("Updated ADR index: %s", display)


# ── Step 5: Notify ingestion webhook ──────────────────────────────────────────


def notify_ingestion_webhook(changed_paths: list[Path]) -> None:
    """
    POST a lightweight notification to the Ask-Z ingestion webhook.
    The ingestion service will re-embed the updated docs and upsert them
    into the vector store.

    Silently skips if ASKZ_INGEST_WEBHOOK is not configured.
    """
    if not ASKZ_WEBHOOK:
        log.info("ASKZ_INGEST_WEBHOOK not configured — skipping webhook call.")
        return

    payload = json.dumps(
        {
            "event": "docs_updated",
            "commit": GITHUB_SHA,
            "repository": GITHUB_REPO,
            "changed_files": [str(p.relative_to(REPO_ROOT)) for p in changed_paths],
            "adr_dir": str(
                ADR_OUTPUT_DIR.relative_to(REPO_ROOT)
                if ADR_OUTPUT_DIR.is_relative_to(REPO_ROOT)
                else ADR_OUTPUT_DIR
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ).encode()

    req = urllib.request.Request(
        ASKZ_WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            log.info("Ingestion webhook notified → HTTP %s", r.status)
    except Exception as exc:
        # Non-fatal: the ingestion job also runs on a schedule as a fallback.
        log.warning("Ingestion webhook call failed (non-fatal): %s", exc)


# ── Main orchestrator ─────────────────────────────────────────────────────────


def main() -> int:
    """
    Run the full Living Architecture Documenter pipeline.
    Returns 0 on success, 1 on fatal error.
    """
    log.info("=" * 60)
    log.info("Living Architecture Documenter — commit %s", GITHUB_SHA)
    log.info("=" * 60)

    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        log.error(
            "WATSONX_API_KEY and WATSONX_PROJECT_ID must be set. "
            "Add them as GitHub Actions secrets."
        )
        return 1

    # Step 1: Find changed files.
    changed_files = get_changed_files()
    if not changed_files:
        log.info("No watched files changed. Nothing to document.")
        return 0

    written_adrs: list[Path] = []

    for file_path in changed_files:
        rel = file_path.relative_to(REPO_ROOT)
        log.info("─── Processing: %s", rel)

        # Step 1b: Get the diff for this file.
        diff = get_file_diff(file_path)
        if len(diff) < MIN_DIFF_CHARS:
            log.info("  Diff too small (%d chars) — skipping.", len(diff))
            continue

        component = _component_tag(file_path)

        # Step 2: Load any existing ADR.
        existing_adr = load_existing_adr(component)

        # Step 3: Generate docs via Granite.
        log.info("  Generating documentation via Granite …")
        try:
            summary, adr_markdown = generate_docs(file_path, diff, existing_adr)
        except Exception as exc:
            log.error("  Generation failed for %s: %s", rel, exc)
            continue  # Don't fail the whole run for one file.

        if not adr_markdown.strip():
            log.warning("  Granite returned empty ADR — skipping writeback.")
            continue

        log.info("  Component summary: %s", summary[:100])

        # Step 4: Write ADR to disk.
        adr_path = write_adr(component, adr_markdown)
        written_adrs.append(adr_path)

    if written_adrs:
        # Regenerate the index after all ADRs are written.
        update_adr_index()
        log.info("Written %d ADR(s) + index.", len(written_adrs))

        # Step 5: Notify the ingestion webhook.
        notify_ingestion_webhook(written_adrs)
    else:
        log.info("No ADRs written (all diffs below threshold or generation failed).")

    log.info("Living Architecture Documenter complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
