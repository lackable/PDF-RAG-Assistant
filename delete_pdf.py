"""
delete_pdf.py
-------------
Removes a named PDF's data from:
  1. docstore/docstore.json   (JSON flat-file store)
  2. chroma_db                (ChromaDB vector store)
  3. docstore/bm25_store.pkl  (BM25 keyword index — rebuilt from updated docstore)

Usage (run from the project root):
    python delete_pdf.py HZL_Small
"""

import sys
import os
import logging

# ── Make project-root importable ────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("delete_pdf")

# ── Resolve pdf_stem from CLI or default ────────────────────────────────────
pdf_stem = sys.argv[1] if len(sys.argv) > 1 else "HZL_Small"
logger.info(f"Target PDF stem: {pdf_stem}")

# ── 1. Docstore ──────────────────────────────────────────────────────────────
logger.info("=== Step 1 — Docstore ===")
from pipeline import docstore as ds

store_before = ds._load_store()
pdfs_before  = list(store_before.get("pdfs", {}).keys())
chunks_before = len(store_before.get("chunks", []))
logger.info(f"PDFs in store before: {pdfs_before}")
logger.info(f"Chunks in store before: {chunks_before}")

if pdf_stem in store_before.get("pdfs", {}):
    ds.delete_pdf(pdf_stem)
    store_after   = ds._load_store()
    chunks_after  = len(store_after.get("chunks", []))
    logger.info(f"Deleted '{pdf_stem}' from docstore.")
    logger.info(f"Chunks remaining: {chunks_after}  (removed {chunks_before - chunks_after})")
else:
    logger.warning(f"'{pdf_stem}' not found in docstore — skipping.")

# ── 2. ChromaDB ──────────────────────────────────────────────────────────────
logger.info("=== Step 2 — ChromaDB ===")
from pipeline import chroma_store as cs

before_count = cs.collection.count()
logger.info(f"ChromaDB vector count before: {before_count}")

cs.delete_by_pdf(pdf_stem)

after_count = cs.collection.count()
logger.info(f"ChromaDB vector count after:  {after_count}  (removed {before_count - after_count})")

# ── 3. BM25 store (rebuild from updated docstore) ────────────────────────────
logger.info("=== Step 3 — BM25 index rebuild ===")
from pipeline import bm25_store as bm

bm.build_index()
logger.info("BM25 index rebuilt and saved.")

logger.info("=== Done — all three stores updated. ===")
