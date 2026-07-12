"""
Gemini 2.5 Flash generation module with conversational support.
Uses LangChain's ChatGoogleGenerativeAI with astream() for true async
token-by-token streaming.
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List

import tiktoken

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# --- Prompts ---

GREETING_SYSTEM_PROMPT = """\
<system_role>
You are a helpful PDF document assistant. You answer questions from documents
that have been ingested into a retrieval system. When no question is being asked,
respond naturally and conversationally as a knowledgeable assistant.
</system_role>"""

QUERY_SYSTEM_PROMPT = """\
<system_role>
You are an expert document analyst. You answer questions exclusively from
the retrieved passages provided below. You have no knowledge outside of
these passages.
</system_role>

<reasoning_policy>
- Read ALL passages carefully before beginning your answer.
- The most relevant passages appear at the beginning AND end of the list.
  Pay close attention to both ends.
- Synthesize information across multiple passages when needed.
- If passages contain conflicting information, report all views and cite sources.
- Prefer a partial but accurate answer over claiming information is unavailable.
- Never fabricate numbers, dates, names, or conclusions.
- Do not say "the documents say" or "according to the provided passages".
  Present information naturally and directly.
</reasoning_policy>

<citation_policy>
After every factual claim, cite the passage number(s) in brackets: [1], [2], [1][3].
</citation_policy>

<response_format>
Use clear markdown formatting. Be concise and well-structured.
If no relevant information exists in any passage, respond with exactly:

## Answer
There is not enough information in the provided content to answer this question.
</response_format>"""


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """Format top-k chunks as a numbered context block with Lost-in-the-Middle mitigation."""
    if not chunks:
        return ""
        
    chunk_with_idx = [(i+1, c) for i, c in enumerate(chunks)]
    
    if len(chunk_with_idx) <= 3:
        display_chunks_with_idx = chunk_with_idx
    else:
        # top-1 first, top-2 last, rest in middle
        display_chunks_with_idx = [chunk_with_idx[0]] + chunk_with_idx[2:] + [chunk_with_idx[1]]

    lines = []
    for original_idx, chunk in display_chunks_with_idx:
        meta      = chunk.get("metadata", {})
        source    = meta.get("source_file", "Unknown")
        pages     = meta.get("page_numbers", "")
        elem_type = meta.get("element_type", "text")
        content   = chunk.get("raw_content") or chunk.get("content", "")

        lines.append(
            f"[{original_idx}] Source: {source} | Pages: {pages} | Type: {elem_type}\n"
            f"{content.strip()}"
        )
    return "\n\n---\n\n".join(lines)


# Tiktoken encoding
_TIKTOKEN_ENC: tiktoken.Encoding | None = None

def _get_encoding() -> tiktoken.Encoding:
    """Lazily load tiktoken encoding."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENC


def _count_tokens(text: str) -> int:
    """Return approximate token count."""
    try:
        return len(_get_encoding().encode(text))
    except Exception:
        return max(1, len(text.split()))


async def _stream_llm(messages: list, input_token_count: int, temperature: float = 0.2) -> AsyncGenerator[str, None]:
    """Helper to stream from LLM and emit SSE format."""
    if not GOOGLE_API_KEY:
        yield f"data: {json.dumps({'type': 'error', 'message': 'GOOGLE_API_KEY not set in .env'})}\n\n"
        return

    output_token_count = 0
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            streaming=True,
        )

        async for chunk in llm.astream(messages):
            token = chunk.content or ""
            if token:
                output_token_count += _count_tokens(token)
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'usage', 'input_tokens': input_token_count, 'output_tokens': output_token_count})}\n\n"
        logger.info(f"Token usage — input: {input_token_count}, output: {output_token_count}")

    except Exception as e:
        logger.error(f"Gemini generation error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


async def stream_greeting(user_input: str) -> AsyncGenerator[str, None]:
    """Streams a greeting response."""
    user_content = f"<user_message>\n{user_input}\n</user_message>"
    messages = [
        SystemMessage(content=GREETING_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    input_tokens = _count_tokens(GREETING_SYSTEM_PROMPT) + _count_tokens(user_content)
    logger.info(f"Calling Gemini for greeting (~{input_tokens} tokens)")
    
    async for sse in _stream_llm(messages, input_tokens, temperature=0.7):
        yield sse


async def stream_query_answer(query: str, chunks: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """Streams an answer for a standard query."""
    if not chunks:
        yield f"data: {json.dumps({'type': 'token', 'token': 'No relevant passages were retrieved for this query.'})}\n\n"
        yield f"data: {json.dumps({'type': 'usage', 'input_tokens': 0, 'output_tokens': 0})}\n\n"
        return

    context_block = _build_context(chunks)
    user_content = (
        f"<retrieved_passages>\n{context_block}\n</retrieved_passages>\n\n"
        f"<question>\n{query}\n</question>"
    )

    messages = [
        SystemMessage(content=QUERY_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    input_tokens = _count_tokens(QUERY_SYSTEM_PROMPT) + _count_tokens(user_content)
    logger.info(f"Calling Gemini for query (~{input_tokens} tokens)")
    
    async for sse in _stream_llm(messages, input_tokens, temperature=0.2):
        yield sse


async def stream_followup_answer(current_input: str, chunks: List[Dict[str, Any]], history_context: str) -> AsyncGenerator[str, None]:
    """Streams an answer for a follow-up query, including conversation history."""
    if not chunks:
        yield f"data: {json.dumps({'type': 'token', 'token': 'No relevant passages were retrieved for this query.'})}\n\n"
        yield f"data: {json.dumps({'type': 'usage', 'input_tokens': 0, 'output_tokens': 0})}\n\n"
        return

    context_block = _build_context(chunks)
    user_content = (
        f"<conversation_history>\n{history_context}\n</conversation_history>\n\n"
        f"<retrieved_passages>\n{context_block}\n</retrieved_passages>\n\n"
        f"<current_question>\n{current_input}\n</current_question>\n\n"
        f"<instruction>\nAnswer the current question using the retrieved passages.\n"
        f"The conversation history is provided so you understand what has already been\n"
        f"discussed. Do not repeat information already covered in the history unless\n"
        f"directly relevant. Build on the prior conversation naturally.\n</instruction>"
    )

    messages = [
        SystemMessage(content=QUERY_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    input_tokens = _count_tokens(QUERY_SYSTEM_PROMPT) + _count_tokens(user_content)
    logger.info(f"Calling Gemini for follow-up (~{input_tokens} tokens)")
    
    async for sse in _stream_llm(messages, input_tokens, temperature=0.2):
        yield sse
