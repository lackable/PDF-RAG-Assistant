import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pypdf import PdfReader
from tqdm import tqdm
import multiprocessing

# Eagerly import heavy data science libraries that use lazy loading.
# This prevents them from being imported during runtime when log files are open,
# which avoids the OneDrive [WinError 6714] directory transaction lock error.
import pyarrow
import pandas
import sklearn
import sentence_transformers
import transformers

import config
from pipeline.logger import setup_logging
import logging
from pipeline import docstore, chroma_store, bm25_store, parser, image_vlm, md_builder, chunker, embedder

def run_pipeline(pdf_path: Path, force_reingest: bool):
    logger = logging.getLogger("ingest")
    
    pdf_stem = pdf_path.stem
    source_file = pdf_path.name
    
    if docstore.pdf_exists(pdf_stem) and not force_reingest:
        logger.info(f"Skipping {source_file} — already ingested")
        return
        
    if force_reingest:
        logger.info(f"Force reingest for {source_file}: Deleting old data")
        docstore.delete_pdf(pdf_stem)
        chroma_store.delete_by_pdf(pdf_stem)
        
    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF {source_file}: {e}")
        return
        
    logger.info(f"Starting pipeline for {source_file} ({total_pages} pages)")
    
    # Phase 1: Parallel parsing
    logger.info("Phase 1: Parallel Document Parsing")
    page_results = {}
    
    with ProcessPoolExecutor(max_workers=config.PARSE_WORKERS) as pool:
        batch_size = 5
        batches = [list(range(i, min(i + batch_size, total_pages + 1))) for i in range(1, total_pages + 1, batch_size)]
        futures = {pool.submit(parser.parse_page_batch, (str(pdf_path), batch)): batch for batch in batches}
        
        from concurrent.futures import as_completed
        pbar = tqdm(total=total_pages, desc="Parsing pages")
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                for page_no, doc_dict in batch_results.items():
                    page_results[page_no] = doc_dict
                pbar.update(len(batch_results))
            except Exception as e:
                logger.error(f"Failed to parse a batch: {e}")
                pbar.update(len(futures[future]))
        pbar.close()
            
    # Phase 2: Sequential VLM and Markdown generation
    logger.info("Phase 2: VLM Image Processing & Markdown Assembly")
    page_markdowns = {}
    
    for page_no in tqdm(sorted(page_results.keys()), desc="Processing VLM"):
        doc_dict = page_results[page_no]
        page_md = md_builder.build_page_markdown(page_no, doc_dict)
        page_markdowns[page_no] = page_md
        
    # Phase 3: Ordered Concatenation
    logger.info("Phase 3: Concatenating Markdown")
    # Join with separator, no extra markers as the builder prepends them
    full_md = "\n\n---\n".join(page_markdowns[i] for i in sorted(page_markdowns.keys()))
    
    # Phase 4: Chunking
    logger.info("Phase 4: Smart Chunking")
    chunks = chunker.chunk(full_md, pdf_stem, source_file)
    if not chunks:
        logger.warning(f"No chunks produced for {source_file}")
        return
        
    # Phase 5: Batch Embedding
    logger.info("Phase 5: Embedding")
    
    # Explicitly unload the VLM here to free up VRAM so the embedder doesn't OOM
    # image_vlm.unload()
    
    embedder.embed_chunks(chunks)
    
    # Unload the embedder so the next PDF's VLM load has full VRAM
    embedder.unload()
    
    # Phase 6: Parallel Writes
    logger.info("Phase 6: Saving to Stores")
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(docstore.append_pdf, pdf_stem, source_file, total_pages, chunks)
        f2 = executor.submit(chroma_store.upsert, chunks)

        f1.result()
        f2.result()

    # Phase 7: Rebuild persistent BM25 index
    # Must happen after docstore write (f1) so the new chunks are included.
    logger.info("Phase 7: Rebuilding BM25 index")
    bm25_store.build_index()

    logger.info(f"Successfully completed {source_file}")


def main():
    setup_logging()
    logger = logging.getLogger("ingest")
    
    parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline")
    parser.add_argument("--content-dir", type=str, default=config.CONTENT_DIR, help="Input PDF directory")
    parser.add_argument("--force-reingest", action="store_true", help="Re-process all PDFs")
    parser.add_argument("--pdf", type=str, help="Process a single PDF file")
    args = parser.parse_args()
    
    content_dir = Path(args.content_dir)
    pdf_files = []
    
    if args.pdf:
        pdf_path = Path(args.pdf)
        if pdf_path.exists():
            pdf_files.append(pdf_path)
        else:
            logger.error(f"File not found: {args.pdf}")
            return
    else:
        pdf_files = sorted(content_dir.glob("*.pdf"))
        
    if not pdf_files:
        logger.warning(f"No PDFs found to process.")
        return
        
    logger.info(f"Found {len(pdf_files)} PDFs to process.")
    
    # Load singletons
    from pipeline import image_vlm, embedder
    logger.info("Pre-loading models...")
    # image_vlm._load()
    
    
    for pdf_path in pdf_files:
        run_pipeline(pdf_path, args.force_reingest)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
