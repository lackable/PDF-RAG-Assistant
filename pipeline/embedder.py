import logging
import torch
import gc
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from .models import ChunkRecord

logger = logging.getLogger(__name__)

_model = None

def _load():
    global _model
    if _model is not None:
        return
        
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL} on {config.CUDA_DEVICE}")
    
    # --- MONKEY PATCH FOR JINA EMBEDDINGS V3 / TRANSFORMERS 4.45+ COMPATIBILITY ---
    try:
        import transformers
        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
            def _get_tied(self):
                val = getattr(self, "_tied_weights_keys", {})
                return val if val is not None else {}
            def _set_tied(self, value):
                self._tied_weights_keys = value if value is not None else {}
            transformers.PreTrainedModel.all_tied_weights_keys = property(_get_tied, _set_tied)
    except Exception as e:
        logger.warning(f"Could not apply monkey patch for transformers: {e}")
    # -------------------------------------------------------------------------------
    
    try:
        _model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            trust_remote_code=True,
            device=config.CUDA_DEVICE
        )
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM loading embedder. Falling back to CPU.")
        torch.cuda.empty_cache()
        gc.collect()
        _model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            trust_remote_code=True,
            device="cpu"
        )
    except Exception as e:
        logger.error(f"Failed to load embedder: {e}")
        raise e

def unload():
    global _model
    if _model is not None:
        del _model
        _model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Unloaded embedding model to free VRAM")

def embed_chunks(chunks: List[ChunkRecord]) -> None:
    """Embeds a list of ChunkRecords in-place using retrieval.passage task."""
    if not chunks:
        return
        
    if _model is None:
        _load()
        
    texts = [c.text_for_embedding for c in chunks]
    
    logger.info(f"Embedding {len(texts)} chunks...")
    
    try:
        embeddings = _model.encode(
            texts,
            task="retrieval.passage",
            batch_size=config.EMBEDDING_BATCH,
            show_progress_bar=True
        )
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
            
        print(f"\n[Embedder]: Successfully generated {len(texts)} embeddings.\n")
        logger.info(f"Successfully generated {len(texts)} embeddings.")
            
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM during embedding. Retrying on CPU.")
        torch.cuda.empty_cache()
        gc.collect()
        
        _model.to("cpu")
        embeddings = _model.encode(
            texts,
            task="retrieval.passage",
            batch_size=config.EMBEDDING_BATCH,
            show_progress_bar=True
        )
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
            
        # Try to restore
        if "cuda" in config.CUDA_DEVICE:
            try:
                _model.to(config.CUDA_DEVICE)
            except:
                pass

def embed_query(query: str) -> List[float]:
    """Embeds a query string using retrieval.query task."""
    if _model is None:
        _load()
        
    try:
        emb = _model.encode(
            query,
            task="retrieval.query",
            show_progress_bar=False
        )
        return emb.tolist()
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        return []
