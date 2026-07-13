"""
ask_z/api/generator.py
───────────────────────
IBM Granite generation layer with structured context assembly and strict citation enforcement.

This module wraps the watsonx.ai text/chat endpoint (not a separate "Bob API" —
Bob is the internal IBM name for Granite-based assistants accessible via watsonx.ai).

Design
───────
• Each retrieved chunk's metadata is prepended to its text as a structured citation block.
• The system prompt instructs the model to:
    - Only answer using facts from the provided context
    - Cite sources inline using the [Source: ...] references
    - Flag low confidence or missing information explicitly
    - Generate code stubs when the question is implementation-focused
• Returns the generated answer as plain text with inline citations.
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
        meta = (
            f"[Source: {chunk.file_path}, "
            f"Component: {chunk.component_tag}, "
            f"Last Updated: {chunk.last_updated}, "
            f"Version: {chunk.version}]"
        )
        lines.append(f"--- Chunk {i} ---")
        lines.append(meta)
        lines.append(chunk.text)
        lines.append("")  # blank line between chunks

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


async def generate_grounded_answer(
    user_query: str,
    top_chunks: list[ContextChunk],
    http_client: httpx.AsyncClient,
) -> str:
    """
    Generate a grounded answer using IBM Granite via watsonx.ai text/chat.

    Parameters
    ----------
    user_query:
        The original user question (after HyDE, retrieval, and reranking).
    top_chunks:
        The top-k reranked context chunks with full metadata.
    http_client:
        Shared async HTTP client (passed from the FastAPI app lifespan).

    Returns
    -------
    str
        The generated answer text with inline [Source: ...] citations.

    Raises
    ------
    httpx.HTTPStatusError:
        If the watsonx.ai API returns an error status.
    httpx.TimeoutException:
        If the generation call exceeds the configured timeout.
    ValueError:
        If top_chunks is empty (the caller should handle this before calling).
    """
    if not top_chunks:
        raise ValueError(
            "Cannot generate answer with empty context. "
            "Ensure at least one chunk passed the rerank threshold."
        )

    context_block = _assemble_context(top_chunks)
    log.debug(
        "Assembled context: %d chars from %d chunks.",
        len(context_block),
        len(top_chunks),
    )

    # ── Prepare the payload ────────────────────────────────────────────────
    bearer = await _get_iam_token(http_client)
    base_url = settings.watsonx.api_base_url
    api_version = settings.watsonx.api_version
    project_id = settings.watsonx.project_id

    generation_model = settings.watsonx.generation_model

    payload = {
        "model_id": generation_model,
        "project_id": project_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"**Context:**\n{context_block}\n\n"
                    f"**Question:** {user_query}\n\n"
                    f"Provide a grounded answer with inline source citations."
                ),
            },
        ],
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 512,
            "repetition_penalty": 1.1,
            "stop_sequences": [],
            "temperature": 0.0,  # deterministic for repeatability
        },
    }

    # ── Call watsonx.ai ─────────────────────────────────────────────────────
    url = f"{base_url}/text/chat?version={api_version}"
    log.debug("Calling %s with model %s …", url, generation_model)

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

    data = resp.json()
    answer_text: str = data["choices"][0]["message"]["content"].strip()

    log.info(
        "Generated answer: %d chars | model=%s | chunks=%d",
        len(answer_text),
        generation_model,
        len(top_chunks),
    )

    return answer_text
