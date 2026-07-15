"""
ask_z/api/generator.py
───────────────────────
Answer generation with automatic fallback chain:
  1. IBM Granite via watsonx.ai  (primary — best quality, grounded citations)
  2. Groq Llama-3 via Groq API   (fallback — free, fast, when watsonx quota exhausted)

The same system prompt and context format is used for both backends so the
answer quality and citation style are consistent.
"""

from __future__ import annotations

import logging
import time

import httpx

from ask_z.api.schemas import ContextChunk
from ask_z.config.settings import settings

log = logging.getLogger("ask_z.api.generator")

# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an IBM Z and Spyre engineering documentation assistant. Your role is to \
provide accurate, grounded answers to technical questions using only the context \
provided in each request.

**Strict rules:**
1. Answer ONLY using facts explicitly stated in the context block below.
2. Cite sources inline whenever making a factual statement. Use the exact \
   [Source: ...] reference format shown in the context.
3. If the context does not contain enough information to answer fully, say: \
   "⚠ Low confidence: the provided context does not fully cover [specific gap]."
4. If the question asks about implementation (e.g., "how do I...", "write code to..."), \
   provide a minimal working code stub using the patterns shown in the context.
5. Be concise. Avoid filler phrases like "Based on the context" or "According to the document."

**Output format:**
• Plain-English explanation with inline [Source: ...] citations.
• If code is relevant, include a fenced ```python block.
• Flag uncertainty explicitly at the end if confidence is low.
"""

_PR_SYSTEM_PROMPT = """\
You are a helpful code review assistant. You will be given the full details of a \
GitHub Pull Request — title, description, changed files, review status, and comments.

Your job is to produce a clear, structured summary of the PR. Always use the \
actual data provided. Never say the information is missing or unavailable.

**Output format (use these exact sections):**

## PR Summary
One or two sentences describing what this PR does.

## Changes
Bullet list of the key changes — what files/areas were modified and why.

## Review Status
Who has reviewed it and what their verdict is (Approved / Changes Requested / Pending).

## Notable Comments
Any important discussion points from the comments (skip if none).

## Quick Take
One sentence: is this ready to merge, needs work, or is it a draft?

Be direct and factual. Do not add disclaimers or say "I cannot find" anything — \
all the information you need is in the PR data block provided.
"""

_PR_REVIEW_SYSTEM_PROMPT = """\
You are a senior software engineer performing a thorough code review of a GitHub \
Pull Request. You will be given the PR metadata, description, changed files, diff \
patches, existing review comments, and discussion threads.

Your job is to produce an actionable, structured code review. Be specific — \
reference file names and, where possible, the relevant diff lines. \
Do not add disclaimers or say "I cannot assess" anything — \
all the information you need is in the PR data block provided.

**Output format (use these exact sections):**

## Overview
Two to three sentences: what this PR changes and why.

## Code Quality
Point-by-point observations about correctness, logic, naming, style, and \
maintainability. Flag any bugs or logic errors explicitly.

## Test Coverage
Are the changes covered by tests? Are edge cases handled? \
Call out anything that should be tested but isn't.

## Security & Safety
Any security concerns (injection, auth bypass, unsafe data handling, etc.). \
Write "None identified" if there are no concerns.

## Existing Review Comments
Summarise what reviewers have already flagged and whether those issues appear \
to be addressed in the diff.

## Merge Readiness
One of: ✅ Ready to merge | ⚠ Needs minor changes | ❌ Needs significant work — \
followed by a single sentence explaining why.
"""

# ── Context assembly ──────────────────────────────────────────────────────────


def _assemble_context(chunks: list[ContextChunk]) -> str:
    """
    Convert retrieved chunks into a unified context block for the LLM.

    Each chunk is formatted as:

        [Source: {file_path}, Component: {component_tag}, Last Updated: {last_updated}, Version: {version}]
        {text}

    The structured metadata enables the model to cite sources accurately.
    """
    if not chunks:
        return "(No context provided.)"

    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        # Use just the filename for external docs; keep full path for repo files.
        source_label = (
            chunk.source_url
            if (
                chunk.doc_type == "external_doc"
                and chunk.source_url
                and chunk.source_url != chunk.file_path
            )
            else chunk.file_path.split("/")[-1]
            if chunk.doc_type == "external_doc"
            else chunk.file_path
        )
        meta = (
            f"[Source: {source_label}, "
            f"Component: {chunk.component_tag}, "
            f"Slide/Chunk: {chunk.chunk_index + 1}]"
        )
        lines.append(f"--- Chunk {i} ---")
        lines.append(meta)
        lines.append(chunk.text)
        lines.append("")

    return "\n".join(lines).strip()


# ── IAM token cache (reused from pipeline.py) ────────────────────────────────

_iam_token: str | None = None
_iam_expires_at: float = 0.0


async def _get_iam_token(client: httpx.AsyncClient) -> str:
    global _iam_token, _iam_expires_at
    now = time.time()
    if _iam_token and now < _iam_expires_at - 60:
        return _iam_token

    log.debug("Refreshing IAM token for generation endpoint …")
    resp = await client.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": settings.watsonx.api_key,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    _iam_token = payload["access_token"]
    _iam_expires_at = now + payload.get("expires_in", 3600)
    return _iam_token


# ── Generation ────────────────────────────────────────────────────────────────


async def _call_llm(
    system_prompt: str,
    user_message: str,
    http_client: httpx.AsyncClient,
    *,
    max_tokens: int = 1024,
) -> str:
    """
    Call watsonx.ai (IBM Granite) with the given system + user message.
    Falls back to Groq on HTTP 403/429 quota errors.

    This is the single LLM entry point — both RAG answers and PR summaries
    go through here, each with their own system prompt.
    """
    # ── watsonx.ai (primary) ──────────────────────────────────────────────
    try:
        bearer = await _get_iam_token(http_client)
        payload = {
            "model_id": settings.watsonx.generation_model,
            "project_id": settings.watsonx.project_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_tokens,
                "repetition_penalty": 1.1,
                "stop_sequences": [],
                "temperature": 0.0,
            },
        }
        url = f"{settings.watsonx.api_base_url}/text/chat?version={settings.watsonx.api_version}"
        resp = await http_client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {bearer}",
                "Content-Type": "application/json",
            },
            timeout=float(settings.bob.timeout_seconds),
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        log.info(
            "Answer via watsonx.ai: %d chars | model=%s",
            len(answer),
            settings.watsonx.generation_model,
        )
        return answer
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (403, 429):
            raise
        log.warning(
            "watsonx.ai quota/rate-limit (HTTP %s) — falling back to Groq.",
            exc.response.status_code,
        )

    # ── Groq fallback ─────────────────────────────────────────────────────
    if not settings.groq.api_key:
        raise RuntimeError(
            "watsonx.ai quota exhausted and GROQ_API_KEY is not set. "
            "Add GROQ_API_KEY to ask_z/.env to enable the Groq fallback."
        )

    import asyncio
    from groq import Groq

    client = Groq(api_key=settings.groq.api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    def _groq_call():
        completion = client.chat.completions.create(
            model=settings.groq.model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()

    answer = await asyncio.to_thread(_groq_call)
    log.info("Answer via Groq (%s): %d chars", settings.groq.model, len(answer))
    return answer


# ── Public API ────────────────────────────────────────────────────────────────


async def generate_grounded_answer(
    user_query: str,
    top_chunks: list[ContextChunk],
    http_client: httpx.AsyncClient,
) -> str:
    """Generate a grounded RAG answer using the retrieved context chunks."""
    context_block = _assemble_context(top_chunks)
    user_message = (
        f"**Context:**\n{context_block}\n\n"
        f"**Question:** {user_query}\n\n"
        "Provide a grounded answer with inline source citations."
    )
    return await _call_llm(_SYSTEM_PROMPT, user_message, http_client)


async def generate_pr_summary(
    pr_data_text: str,
    user_query: str,
    http_client: httpx.AsyncClient,
) -> str:
    """Generate a structured PR summary using the raw PR data text."""
    user_message = f"**PR Data:**\n{pr_data_text}\n\n**Request:** {user_query}"
    return await _call_llm(
        _PR_SYSTEM_PROMPT, user_message, http_client, max_tokens=1024
    )


async def generate_pr_review(
    pr_data_text: str,
    user_query: str,
    http_client: httpx.AsyncClient,
) -> str:
    """
    Perform a structured code review of a PR.

    Uses ``_PR_REVIEW_SYSTEM_PROMPT`` which instructs the LLM to act as a
    senior engineer — covering code quality, test coverage, security, and
    merge readiness.  The diff patch is expected to be embedded in
    ``pr_data_text`` by ``fetch_pr_context(include_diff=True)``.
    """
    user_message = (
        f"**PR Data (with diff):**\n{pr_data_text}\n\n**Request:** {user_query}"
    )
    return await _call_llm(
        _PR_REVIEW_SYSTEM_PROMPT, user_message, http_client, max_tokens=1500
    )
