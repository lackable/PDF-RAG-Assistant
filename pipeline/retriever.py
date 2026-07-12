import logging
from typing import List, Dict, Any

from . import embedder
from . import chroma_store
from . import bm25_store
from . import docstore
from . import reranker

logger = logging.getLogger(__name__)

# Candidates pulled from each retrieval source before reranking.
# Both pools are merged and de-duplicated, so the reranker sees at most
# VECTOR_CANDIDATES + BM25_CANDIDATES unique chunks.
VECTOR_CANDIDATES = 20
BM25_CANDIDATES   = 20


def search(query: str, top_k: int = 5, filter_pdf_stems: List[str] = None) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval pipeline:
      1. Dense ANN search   → ChromaDB          (VECTOR_CANDIDATES results)
      2. BM25 keyword search → persistent index  (BM25_CANDIDATES results)
      3. De-duplicate by chunk_id (dense-first; dense metadata wins on collision)
      4. Enrich each candidate with full raw_content from docstore
      5. Qwen3-Reranker-0.6B scores all candidates → return top_k

    The reranker acts as the fusion mechanism — no RRF or score blending needed.
    """
    # ── Stage 1: Dense recall ──────────────────────────────────────────────── #
    q_emb = embedder.embed_query(query)
    if not q_emb:
        logger.error("Query embedding failed — returning empty results.")
        return []

    dense_results = chroma_store.query(q_emb, VECTOR_CANDIDATES, filter_pdf_stems)

    # ── Stage 2: BM25 keyword recall ──────────────────────────────────────── #
    bm25_results = bm25_store.query(query, BM25_CANDIDATES, filter_pdf_stems)

    # ── Stage 3: Merge & de-duplicate, tagging retrieval_source ─────────── #
    dense_ids = {r["id"] for r in dense_results}
    bm25_ids  = {r["id"] for r in bm25_results}

    seen_ids: set = set()
    merged: List[Dict[str, Any]] = []
    for r in dense_results + bm25_results:
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            in_dense = r["id"] in dense_ids
            in_bm25  = r["id"] in bm25_ids
            r["retrieval_source"] = "both" if (in_dense and in_bm25) else ("dense" if in_dense else "bm25")
            merged.append(r)

    if not merged:
        return []

    # ── Stage 4: Enrich with full raw_content from docstore ──────────────── #
    for r in merged:
        chunk_data = docstore.get_chunk(r["id"])
        if chunk_data:
            r["raw_content"] = chunk_data.get("raw_content", r["content"])
        else:
            logger.warning(f"Chunk ID {r['id']} not found in docstore.")
            r["raw_content"] = r["content"]

    # ── Stage 5: Reranker scores full merged pool → top_k ────────────────── #
    n_dense = len(dense_results)
    n_bm25  = len(bm25_results)
    n_total = len(merged)
    logger.info(
        f"Reranking {n_total} candidates "
        f"(dense={n_dense}, bm25={n_bm25}, unique={n_total}) → top {top_k}"
    )
    return reranker.rerank(query, merged, top_n=top_k)
