import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

import config
from .models import ChunkRecord

logger = logging.getLogger(__name__)

# In-memory cache — avoids re-reading the entire JSON on every chunk lookup
_store_cache: Dict[str, Any] = None

def _invalidate_cache():
    global _store_cache
    _store_cache = None

def _load_store() -> Dict[str, Any]:
    global _store_cache
    if _store_cache is not None:
        return _store_cache

    if not os.path.exists(config.DOCSTORE_FILE):
        _store_cache = {"pdfs": {}, "chunk_index": {}, "chunks": []}
        return _store_cache
        
    with open(config.DOCSTORE_FILE, "r", encoding="utf-8") as f:
        try:
            _store_cache = json.load(f)
            return _store_cache
        except Exception as e:
            logger.error(f"Failed to parse docstore JSON: {e}")
            _store_cache = {"pdfs": {}, "chunk_index": {}, "chunks": []}
            return _store_cache

def _save_store(store: Dict[str, Any]):
    _invalidate_cache()
    tmp_file = f"{config.DOCSTORE_FILE}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    try:
        os.replace(tmp_file, config.DOCSTORE_FILE)
    except OSError:
        import shutil
        if os.path.exists(config.DOCSTORE_FILE):
            try:
                os.remove(config.DOCSTORE_FILE)
            except OSError:
                pass
        shutil.move(tmp_file, config.DOCSTORE_FILE)

def pdf_exists(pdf_stem: str) -> bool:
    store = _load_store()
    return pdf_stem in store.get("pdfs", {})

def list_pdfs() -> Dict[str, Any]:
    store = _load_store()
    return store.get("pdfs", {})

def append_pdf(pdf_stem: str, source_file: str, total_pages: int, chunks: List[ChunkRecord]):
    store = _load_store()
    
    timestamp = datetime.utcnow().isoformat()
    
    # 1. Update PDFs metadata
    store["pdfs"][pdf_stem] = {
        "source_file": source_file,
        "total_pages": total_pages,
        "chunk_count": len(chunks),
        "ingested_at": timestamp
    }
    
    # 2. Append chunks
    start_idx = len(store["chunks"])
    for i, c in enumerate(chunks):
        c.ingested_at = timestamp
        # Dump chunk without embedding
        chunk_dict = c.model_dump(exclude={"embedding"})
        store["chunks"].append(chunk_dict)
        
        # 3. Update index map
        store["chunk_index"][c.chunk_id] = start_idx + i
        
    _save_store(store)
    logger.info(f"Appended {len(chunks)} chunks for {pdf_stem} to docstore.")

def delete_pdf(pdf_stem: str):
    store = _load_store()
    
    if pdf_stem not in store.get("pdfs", {}):
        return
        
    # Remove from pdfs
    del store["pdfs"][pdf_stem]
    
    # Filter chunks
    new_chunks = [c for c in store["chunks"] if c.get("pdf_stem") != pdf_stem]
    store["chunks"] = new_chunks
    
    # Rebuild index
    new_index = {}
    for i, c in enumerate(new_chunks):
        new_index[c["chunk_id"]] = i
    store["chunk_index"] = new_index
    
    _save_store(store)
    logger.info(f"Deleted {pdf_stem} from docstore.")

def get_chunk(chunk_id: str) -> Dict[str, Any]:
    store = _load_store()
    idx = store.get("chunk_index", {}).get(chunk_id)
    if idx is not None and idx < len(store.get("chunks", [])):
        return store["chunks"][idx]
    return None
