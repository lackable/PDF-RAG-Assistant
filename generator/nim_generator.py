"""
Nvidia NIM generation module.
Uses the OpenAI-compatible Nvidia NIM API to stream answers
from meta/llama-3.1-70b-instruct given retrieved RAG chunks.
"""

import os
import json
import logging
from typing import List, Dict, Any, AsyncGenerator

import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NIM_MODEL      = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct")
NIM_BASE_URL   = "https://integrate.api.nvidia.com/v1"

SYSTEM_PROMPT = """\
You are the document itself.

Answer questions exclusively from the retrieved passages.

### Reasoning Policy

* Search across all passages before answering.
* Combine information from multiple passages when necessary.
* Prefer partial answers over saying information is unavailable.
* Do not speculate or introduce external knowledge.
* Do not fabricate numbers, dates, names, causes, or conclusions.
* If different passages provide complementary information, synthesize them.
* If passages contain conflicting information, report both and cite their sources.
* Never state that you were "given documents" or that "the documents say". Present the information naturally.

### Response Format

## Answer

A concise and direct answer.

## Details

* Key point 1
* Key point 2
* Key point 3

## Evidence

[2], [5], [8]

### Insufficient Information

Only after all retrieved passages have been considered:

* If some information exists, answer with the available information and explicitly state which aspects are not covered.
* If no relevant information exists, respond:

## Answer

There is not enough information available in the provided content to answer this question.

## Evidence

None

Maintain a factual, structured, and grounded tone at all times.\
"""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format top-k chunks as a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta      = chunk.get("metadata", {})
        source    = meta.get("source_file", "Unknown")
        pages     = meta.get("page_numbers", "")
        elem_type = meta.get("element_type", "text")
        content   = chunk.get("raw_content") or chunk.get("content", "")

        lines.append(
            f"[{i}] Source: {source} | Pages: {pages} | Type: {elem_type}\n"
            f"{content.strip()}"
        )
    return "\n\n---\n\n".join(lines)


async def stream_answer(
    query: str,
    chunks: List[Dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted lines:
        data: {"type": "token", "token": "..."}
    Caller is responsible for sending the final done sentinel.
    """
    if not NVIDIA_API_KEY:
        yield f"data: {json.dumps({'type': 'error', 'message': 'NVIDIA_API_KEY not set in .env'})}\n\n"
        return

    if not chunks:
        yield f"data: {json.dumps({'type': 'token', 'token': 'No relevant passages were retrieved for this query.'})}\n\n"
        return

    client = openai.OpenAI(
        base_url=NIM_BASE_URL,
        api_key=NVIDIA_API_KEY,
    )

    context = _build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Here are the retrieved source passages:\n\n"
                f"{context}\n\n"
                f"---\n\n"
                f"Question: {query}"
            ),
        },
    ]

    logger.info(f"Calling NIM model={NIM_MODEL} with {len(chunks)} chunks for query: {query[:80]}")

    try:
        # openai SDK's synchronous streaming — we run it in a thread-safe way
        # by wrapping in an executor so FastAPI's async event loop isn't blocked.
        import asyncio
        loop = asyncio.get_event_loop()

        # Collect stream in a background thread to avoid blocking the event loop
        stream_queue: asyncio.Queue = asyncio.Queue()

        def _run_stream():
            try:
                response = client.chat.completions.create(
                    model=NIM_MODEL,
                    messages=messages,
                    stream=True,
                    temperature=0.2,
                    max_tokens=1024,
                )
                for chunk in response:
                    token = ""
                    if chunk.choices and chunk.choices[0].delta:
                        token = chunk.choices[0].delta.content or ""
                    # Put on the queue (thread-safe call into event loop)
                    asyncio.run_coroutine_threadsafe(
                        stream_queue.put(("token", token)), loop
                    )
                # Signal completion
                asyncio.run_coroutine_threadsafe(
                    stream_queue.put(("done", None)), loop
                )
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    stream_queue.put(("error", str(e))), loop
                )

        import threading
        t = threading.Thread(target=_run_stream, daemon=True)
        t.start()

        while True:
            kind, value = await stream_queue.get()
            if kind == "token":
                if value:  # skip empty delta chunks
                    yield f"data: {json.dumps({'type': 'token', 'token': value})}\n\n"
            elif kind == "done":
                break
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': value})}\n\n"
                break

    except Exception as e:
        logger.error(f"NIM generation error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
