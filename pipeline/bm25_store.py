"""
Persistent BM25 keyword index.

The index is pickled to BM25_STORE_PATH (./docstore/bm25_store.pkl) so it
survives process restarts without a rebuild.

Lifecycle
---------
* First query after a fresh install: no pickle exists → build_index() is called
  automatically, index is saved to disk, then the query proceeds.
* Normal startup (pickle exists): _load_from_disk() restores the index in <1 s
  for typical corpus sizes.
* After each ingest / delete: ingest.py calls build_index() explicitly, which
  rebuilds from the updated docstore and overwrites the pickle.

Thread-safety
-------------
build_index() writes the pickle atomically (write tmp → os.replace) so a
concurrent query that reads from the old in-memory index is safe.
"""

import gc
import logging
import os
import pickle
import re
import tempfile
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

import config
from . import docstore

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# In-memory singletons                                                         #
# --------------------------------------------------------------------------- #

_bm25: Optional[BM25Okapi] = None
_corpus_chunks: List[Dict[str, Any]] = []          # parallel list → BM25 rows


# --------------------------------------------------------------------------- #
# Stopwords                                                                    #
# --------------------------------------------------------------------------- #

# Comprehensive English stopword set (no external dependency required).
# Kept as a frozenset for O(1) lookup during tokenisation.
_STOPWORDS: frozenset = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "get", "got", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've",
    "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just",
    "let's", "like", "ll", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "now", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "t", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "us", "ve", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "will", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
})


# --------------------------------------------------------------------------- #
# Tokenisation                                                                 #
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> List[str]:
    """
    Lowercase word-level tokenisation with stopword removal.

    * Keeps digits — model numbers, years, and statistics remain searchable.
    * Strips common English stopwords so BM25's IDF weighting focuses on
      rare, distinctive terms rather than high-frequency function words.
    """
    return [
        tok for tok in re.findall(r"\w+", text.lower())
        if tok not in _STOPWORDS
    ]


# --------------------------------------------------------------------------- #
# Persistence                                                                  #
# --------------------------------------------------------------------------- #

def _save_to_disk() -> None:
    """Atomically write the current index to BM25_STORE_PATH."""
    payload = {"bm25": _bm25, "chunks": _corpus_chunks}
    tmp_fd, tmp_path = tempfile.mkstemp(dir=config.DOCSTORE_DIR, suffix=".pkl.tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, config.BM25_STORE_PATH)
        logger.info(f"BM25 store saved → {config.BM25_STORE_PATH}  ({len(_corpus_chunks)} chunks)")
    except Exception:
        # Clean up temp file on failure
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _load_from_disk() -> bool:
    """
    Load index from BM25_STORE_PATH into the in-memory singletons.
    Returns True on success, False if the file is missing or corrupt.
    """
    global _bm25, _corpus_chunks
    try:
        with open(config.BM25_STORE_PATH, "rb") as f:
            payload = pickle.load(f)
        _bm25 = payload["bm25"]
        _corpus_chunks = payload["chunks"]
        logger.info(f"BM25 store loaded from disk  ({len(_corpus_chunks)} chunks)")
        return True
    except FileNotFoundError:
        logger.info("BM25 store not found on disk — will build from docstore.")
        return False
    except Exception as e:
        logger.warning(f"BM25 store corrupt ({e}) — will rebuild.")
        _bm25 = None
        _corpus_chunks = []
        return False


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def build_index() -> None:
    """
    (Re)build the BM25 index from the current docstore and persist it.

    Called:
      • By ingest.py after every successful PDF ingest or deletion.
      • Lazily on the first query when no pickle exists.
    """
    global _bm25, _corpus_chunks

    store = docstore._load_store()
    chunks = store.get("chunks", [])

    tokenized = [
        _tokenize(c.get("raw_content") or c.get("text_for_embedding", ""))
        for c in chunks
    ]

    _corpus_chunks = chunks
    _bm25 = BM25Okapi(tokenized)

    _save_to_disk()

    # Free the tokenized list — BM25Okapi keeps its own internal copy
    del tokenized
    gc.collect()

    logger.info(f"BM25 index built: {len(chunks)} chunks")


def _ensure_index() -> None:
    """Guarantee the in-memory index is ready, loading or building as needed."""
    if _bm25 is None:
        if not _load_from_disk():
            build_index()


def query(
    query_text: str,
    top_k: int = 20,
    filter_pdf_stems: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Score all chunks with BM25 and return the top_k results.

    Parameters
    ----------
    query_text       : raw query string (tokenised internally)
    top_k            : number of results to return
    filter_pdf_stems : if provided, only chunks from these PDFs are considered

    Returns
    -------
    List of dicts with keys: id, content, metadata, bm25_score
    The shape intentionally mirrors chroma_store.query() output so the
    retriever can treat both sources uniformly.
    """
    _ensure_index()

    if not _corpus_chunks:
        logger.warning("BM25 index is empty — no chunks in docstore.")
        return []

    tokens = _tokenize(query_text)
    scores = _bm25.get_scores(tokens)           # ndarray, length = len(_corpus_chunks)

    # Apply pdf_stem filter *before* sorting to avoid ranking filtered-out chunks
    indexed: List[tuple] = list(enumerate(scores))
    if filter_pdf_stems:
        pdf_set = set(filter_pdf_stems)
        indexed = [
            (i, s) for i, s in indexed
            if _corpus_chunks[i].get("pdf_stem") in pdf_set
        ]

    indexed.sort(key=lambda x: x[1], reverse=True)
    top = indexed[:top_k]

    results = []
    for idx, score in top:
        c = _corpus_chunks[idx]
        results.append({
            "id": c["chunk_id"],
            "content": c.get("text_for_embedding", ""),
            "metadata": {
                "source_file": c.get("source_file", ""),
                "pdf_stem":    c.get("pdf_stem", ""),
                "page_numbers": ",".join(map(str, c.get("page_numbers", []))),
                "element_type": c.get("element_type", ""),
                "chunk_index":  c.get("chunk_index", 0),
                "ingested_at":  c.get("ingested_at", ""),
            },
            "bm25_score": float(score),
        })

    return results
