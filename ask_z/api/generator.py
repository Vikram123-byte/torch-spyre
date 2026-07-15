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

# ── System prompt ─────────────────────────────────────────────────────────────

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


async def _generate_via_watsonx(
    user_query: str,
    context_block: str,
    http_client: httpx.AsyncClient,
) -> str:
    """Call watsonx.ai text/chat endpoint."""
    bearer = await _get_iam_token(http_client)
    payload = {
        "model_id": settings.watsonx.generation_model,
        "project_id": settings.watsonx.project_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"**Context:**\n{context_block}\n\n"
                    f"**Question:** {user_query}\n\n"
                    "Provide a grounded answer with inline source citations."
                ),
            },
        ],
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 512,
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
    return resp.json()["choices"][0]["message"]["content"].strip()


async def _generate_via_groq(user_query: str, context_block: str) -> str:
    """Call Groq API using the groq SDK (async). Used as fallback."""
    import asyncio
    from groq import Groq

    client = Groq(api_key=settings.groq.api_key)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"**Context:**\n{context_block}\n\n"
                f"**Question:** {user_query}\n\n"
                "Provide a grounded answer with inline source citations."
            ),
        },
    ]

    # Groq SDK is sync — run in thread pool to avoid blocking the event loop.
    def _call():
        completion = client.chat.completions.create(
            model=settings.groq.model,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        return completion.choices[0].message.content.strip()

    return await asyncio.to_thread(_call)


async def generate_grounded_answer(
    user_query: str,
    top_chunks: list[ContextChunk],
    http_client: httpx.AsyncClient,
) -> str:
    """
    Generate a grounded answer from retrieved context chunks.

    Tries watsonx.ai (IBM Granite) first. On HTTP 403/429 quota errors,
    automatically falls back to Groq (Llama-3) if GROQ_API_KEY is set.
    """
    if not top_chunks:
        raise ValueError("Cannot generate answer with empty context.")

    context_block = _assemble_context(top_chunks)
    log.debug("Context: %d chars from %d chunks.", len(context_block), len(top_chunks))

    # ── Try watsonx.ai first ───────────────────────────────────────────────
    try:
        answer = await _generate_via_watsonx(user_query, context_block, http_client)
        log.info(
            "Answer via watsonx.ai: %d chars | model=%s | chunks=%d",
            len(answer),
            settings.watsonx.generation_model,
            len(top_chunks),
        )
        return answer
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (403, 429):
            raise
        log.warning(
            "watsonx.ai quota/rate-limit (HTTP %s) — falling back to Groq.",
            exc.response.status_code,
        )

    # ── Fallback: Groq ─────────────────────────────────────────────────────
    if not settings.groq.api_key:
        raise RuntimeError(
            "watsonx.ai quota exhausted and GROQ_API_KEY is not set. "
            "Add GROQ_API_KEY to ask_z/.env to enable the Groq fallback."
        )

    answer = await _generate_via_groq(user_query, context_block)
    log.info(
        "Answer via Groq (%s): %d chars | chunks=%d",
        settings.groq.model,
        len(answer),
        len(top_chunks),
    )
    return answer
