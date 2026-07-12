import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any

import config
from .models import ChunkRecord

logger = logging.getLogger(__name__)

# Initialize client and collection
client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name=config.CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

def upsert(chunks: List[ChunkRecord]) -> None:
    valid_chunks = [c for c in chunks if c.embedding is not None]
    if not valid_chunks:
        return
        
    ids = [c.chunk_id for c in valid_chunks]
    embeddings = [c.embedding for c in valid_chunks]
    documents = [c.text_for_embedding for c in valid_chunks]
    
    metadatas = []
    for c in valid_chunks:
        metadatas.append({
            "source_file": c.source_file,
            "pdf_stem": c.pdf_stem,
            "page_numbers": ",".join(map(str, c.page_numbers)),
            "element_type": c.element_type,
            "chunk_index": c.chunk_index,
            "ingested_at": c.ingested_at
        })

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    logger.info(f"Upserted {len(valid_chunks)} vectors to ChromaDB")

def delete_by_pdf(pdf_stem: str):
    try:
        results = collection.get(where={"pdf_stem": pdf_stem})
        if results and results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} vectors for {pdf_stem} from ChromaDB")
    except Exception as e:
        logger.error(f"Failed to delete {pdf_stem} from ChromaDB: {e}")

def query(query_embedding: List[float], top_k: int = 5, filter_pdf_stems: List[str] = None) -> List[Dict[str, Any]]:
    where = None
    if filter_pdf_stems:
        if len(filter_pdf_stems) == 1:
            where = {"pdf_stem": filter_pdf_stems[0]}
        else:
            where = {"pdf_stem": {"$in": filter_pdf_stems}}
            
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if not results['ids'] or not results['ids'][0]:
            return []
            
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
            
        return formatted_results
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        return []
