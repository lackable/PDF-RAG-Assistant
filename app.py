"""
FastAPI server for pdf_rag_pipeline_web.
Features stateful multi-turn conversational chatbot support.
"""

import os
import sys
import json
import logging
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.retriever import search
from generator.gemini_generator import stream_greeting, stream_query_answer, stream_followup_answer
from generator.state_classifier import classify_state
from generator.query_builder import build_master_query
from generator.history_builder import build_history_context
import session_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF RAG Assistant", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response models ──────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_pdf_stems: Optional[list[str]] = None

class GenerateRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_pdf_stems: Optional[list[str]] = None

class ChatRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = 5

class CreateSessionRequest(BaseModel):
    name: Optional[str] = None

class RenameSessionRequest(BaseModel):
    name: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


# --- Session Management ---

@app.get("/api/sessions")
async def api_list_sessions():
    return session_store.list_sessions()

@app.post("/api/sessions")
async def api_create_session(req: CreateSessionRequest):
    return session_store.create_session(name=req.name)

@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    session_store.delete_session(session_id)
    return {"status": "deleted"}

@app.get("/api/sessions/{session_id}/history")
async def api_get_session_history(session_id: str):
    session = session_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.get("messages", [])

@app.patch("/api/sessions/{session_id}/rename")
async def api_rename_session(session_id: str, req: RenameSessionRequest):
    session_store.rename_session(session_id, req.name)
    return {"status": "renamed"}


# --- Main Chat Endpoint ---

def extract_token(sse_line: str) -> str:
    """Extracts text token from SSE line for saving to history."""
    try:
        if sse_line.startswith("data: "):
            data = json.loads(sse_line[6:])
            if data.get("type") == "token":
                return data.get("token", "")
    except Exception:
        pass
    return ""


def extract_usage(sse_line: str) -> dict | None:
    """Extracts input/output token counts from a 'usage' SSE event."""
    try:
        if sse_line.startswith("data: "):
            data = json.loads(sse_line[6:])
            if data.get("type") == "usage":
                return {
                    "input_tokens": data.get("input_tokens"),
                    "output_tokens": data.get("output_tokens")
                }
    except Exception:
        pass
    return None


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    history = session_store.get_history(req.session_id)
    if history is None and not session_store.get_session(req.session_id):
        # The frontend expects a valid session
        raise HTTPException(status_code=404, detail="Session not found")

    recent_window = history[-6:] if len(history) > 6 else history

    async def event_generator():
        start_time = time.time()
        # Classify state
        agent_state = await classify_state(req.query, recent_window)
        yield f"data: {json.dumps({'type': 'state', 'state': agent_state})}\n\n"

        full_output = ""
        usage_data = {}
        sources = []

        try:
            if agent_state == "greeting":
                async for sse in stream_greeting(req.query):
                    full_output += extract_token(sse)
                    usage = extract_usage(sse)
                    if usage:
                        usage_data = usage
                    yield sse
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            elif agent_state == "query":
                chunks = await asyncio.to_thread(search, query=req.query, top_k=req.top_k)
                sources = [
                    {
                        "pdf_name": c.get("metadata", {}).get("source_file", "Unknown"),
                        "page_numbers": c.get("metadata", {}).get("page_numbers", ""),
                        "element_type": c.get("metadata", {}).get("element_type", "text")
                    }
                    for c in chunks
                ]
                yield f"data: {json.dumps({'type': 'chunks', 'chunks': chunks})}\n\n"
                
                async for sse in stream_query_answer(req.query, chunks):
                    full_output += extract_token(sse)
                    usage = extract_usage(sse)
                    if usage:
                        usage_data = usage
                    yield sse
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            elif agent_state == "follow_up":
                has_prior_query = any(m.get("agent_state") == "query" for m in history)
                if not has_prior_query:
                    # Edge case: follow up without a prior query
                    agent_state = "query" # Update state locally for storing
                    master_query = req.query
                else:
                    master_query = build_master_query(history, req.query)

                yield f"data: {json.dumps({'type': 'master_query', 'master_query': master_query})}\n\n"

                chunks = await asyncio.to_thread(search, query=master_query, top_k=req.top_k)
                sources = [
                    {
                        "pdf_name": c.get("metadata", {}).get("source_file", "Unknown"),
                        "page_numbers": c.get("metadata", {}).get("page_numbers", ""),
                        "element_type": c.get("metadata", {}).get("element_type", "text")
                    }
                    for c in chunks
                ]
                yield f"data: {json.dumps({'type': 'chunks', 'chunks': chunks})}\n\n"

                history_context = build_history_context(history)
                async for sse in stream_followup_answer(req.query, chunks, history_context):
                    full_output += extract_token(sse)
                    usage = extract_usage(sse)
                    if usage:
                        usage_data = usage
                    yield sse
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            end_time = time.time()
            time_taken_s = round(end_time - start_time, 2)

            # Persist message
            session_store.append_message(
                session_id=req.session_id,
                agent_state=agent_state,
                input_text=req.query,
                output_text=full_output,
                input_tokens=usage_data.get("input_tokens"),
                output_tokens=usage_data.get("output_tokens"),
                time_taken_s=time_taken_s,
                sources=sources
            )

            # Auto-name if first message
            history_after = session_store.get_history(req.session_id)
            if len(history_after) == 1:
                auto_name = session_store.auto_name_from_message(req.query)
                session_store.rename_session(req.session_id, auto_name)

        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --- Legacy endpoints ---

@app.post("/api/search")
async def api_search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        results = await asyncio.to_thread(
            search,
            query=req.query,
            top_k=req.top_k,
            filter_pdf_stems=req.filter_pdf_stems,
        )
        return {"query": req.query, "results": results}
    except Exception as e:
        logger.error(f"/api/search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """Legacy endpoint for backward compatibility."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    async def event_generator():
        try:
            chunks = await asyncio.to_thread(
                search,
                query=req.query,
                top_k=req.top_k,
                filter_pdf_stems=req.filter_pdf_stems,
            )
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'chunks', 'chunks': chunks})}\n\n"

        try:
            # We use stream_query_answer directly for legacy support
            async for sse_line in stream_query_answer(req.query, chunks):
                yield sse_line
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
