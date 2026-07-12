import logging
import hashlib
import re
from typing import List, Tuple
import tiktoken

import config
from .models import ChunkRecord

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

def _hash_id(source_file: str, index: int) -> str:
    return hashlib.sha256(f"{source_file}:{index}".encode("utf-8")).hexdigest()

def _get_pages_for_span(start_char: int, end_char: int, page_map: List[Tuple[int, int]]) -> List[int]:
    """Determine which pages a character span covers based on the page map."""
    pages = set()
    
    # If no map, default to page 1
    if not page_map:
        return [1]
        
    for i, (offset, page_no) in enumerate(page_map):
        next_offset = page_map[i+1][0] if i + 1 < len(page_map) else float('inf')
        
        # Check if the span overlaps with this page's range
        if start_char < next_offset and end_char > offset:
            pages.add(page_no)
            
    return sorted(list(pages)) if pages else [1]

def chunk(full_md: str, pdf_stem: str, source_file: str) -> List[ChunkRecord]:
    """Smart chunking algorithm over full markdown stream."""
    
    # Step A: Build page map
    page_map = []
    # Pattern to match <!-- page:N -->
    page_pattern = re.compile(r'<!-- page:(\d+) -->')
    for match in page_pattern.finditer(full_md):
        page_no = int(match.group(1))
        page_map.append((match.start(), page_no))
        
    # Sort map just in case
    page_map.sort(key=lambda x: x[0])
    
    # Step B: Segment detection (text vs table vs image)
    # We will iterate line by line to build segments
    segments = []
    current_segment = {"type": "text", "start": 0, "lines": [], "char_offset": 0}
    
    lines = full_md.split('\n')
    current_char = 0
    
    for line in lines:
        line_len = len(line) + 1 # +1 for \n
        
        # Detect table
        if line.strip().startswith('|'):
            if current_segment["type"] != "table":
                # Close previous segment
                if current_segment["lines"]:
                    current_segment["end"] = current_char
                    segments.append(current_segment)
                # Start table segment
                current_segment = {"type": "table", "start": current_char, "lines": [line], "char_offset": current_char}
            else:
                current_segment["lines"].append(line)
                
        # Detect Image summary
        elif line.startswith("> **[Image Summary, Page"):
            # Close previous
            if current_segment["lines"]:
                current_segment["end"] = current_char
                segments.append(current_segment)
            # Start and immediately close image segment
            segments.append({
                "type": "image",
                "start": current_char,
                "end": current_char + line_len,
                "lines": [line],
                "char_offset": current_char
            })
            current_segment = {"type": "text", "start": current_char + line_len, "lines": [], "char_offset": current_char + line_len}
            
        # Detect text
        else:
            if current_segment["type"] != "text":
                # Close previous
                if current_segment["lines"]:
                    current_segment["end"] = current_char
                    segments.append(current_segment)
                # Start text segment
                current_segment = {"type": "text", "start": current_char, "lines": [line], "char_offset": current_char}
            else:
                current_segment["lines"].append(line)
                
        current_char += line_len
        
    if current_segment["lines"]:
        current_segment["end"] = current_char
        segments.append(current_segment)
        
    records = []
    global_chunk_idx = 0
    
    # Step C, D, E: Process segments
    for i, segment in enumerate(segments):
        raw_text = '\n'.join(segment["lines"])
        start_offset = segment["start"]
        end_offset = segment["end"]
        
        if segment["type"] == "text":
            # Strip page markers from content
            clean_text = page_pattern.sub('', raw_text).strip()
            if not clean_text:
                continue
                
            tokens = _enc.encode(clean_text)
            
            # Avoid small standalone text chunks by merging/prepending them
            if len(tokens) < 150:
                pages = _get_pages_for_span(start_offset, end_offset, page_map)
                
                # 1. Try to append to the last text record
                if records and records[-1].element_type == "text":
                    records[-1].text_for_embedding += "\n\n" + clean_text
                    records[-1].raw_content += "\n\n" + clean_text
                    # Merge page numbers
                    records[-1].page_numbers = sorted(list(set(records[-1].page_numbers + pages)))
                    continue
                    
                # 2. Try to prepend to the next text segment
                next_text_seg = None
                for j in range(i + 1, len(segments)):
                    if segments[j]["type"] == "text":
                        next_text_seg = segments[j]
                        break
                if next_text_seg:
                    next_text_seg["lines"].insert(0, clean_text)
                    continue
            
            start_tok = 0
            
            while start_tok < len(tokens):
                end_tok = min(start_tok + config.CHUNK_SIZE, len(tokens))
                chunk_tokens = tokens[start_tok:end_tok]
                chunk_text = _enc.decode(chunk_tokens)
                
                # Approximate char offset mapping for page numbers
                # A bit tricky since we stripped markers, but we can use segment offsets
                tok_ratio_start = start_tok / len(tokens)
                tok_ratio_end = end_tok / len(tokens)
                chunk_char_start = start_offset + int(tok_ratio_start * (end_offset - start_offset))
                chunk_char_end = start_offset + int(tok_ratio_end * (end_offset - start_offset))
                
                pages = _get_pages_for_span(chunk_char_start, chunk_char_end, page_map)
                
                records.append(ChunkRecord(
                    chunk_id=_hash_id(source_file, global_chunk_idx),
                    source_file=source_file,
                    pdf_stem=pdf_stem,
                    page_numbers=pages,
                    element_type="text",
                    chunk_index=global_chunk_idx,
                    text_for_embedding=chunk_text,
                    raw_content=chunk_text
                ))
                global_chunk_idx += 1
                
                # Advance window
                step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
                start_tok += step
                if start_tok >= len(tokens):
                    break
                    
        elif segment["type"] == "image":
            # Extract page number and summary
            line = segment["lines"][0]
            # match '> **[Image Summary, Page 3]** A bar chart...'
            match = re.match(r'> \*\*\[Image Summary, Page (\d+)\]\*\*\s+(.*)', line)
            if match:
                pages = [int(match.group(1))]
                summary = match.group(2).strip()
            else:
                pages = _get_pages_for_span(start_offset, end_offset, page_map)
                summary = line.replace("> **[Image Summary", "").strip()
                
            records.append(ChunkRecord(
                chunk_id=_hash_id(source_file, global_chunk_idx),
                source_file=source_file,
                pdf_stem=pdf_stem,
                page_numbers=pages,
                element_type="image",
                chunk_index=global_chunk_idx,
                text_for_embedding=summary,
                raw_content=line
            ))
            global_chunk_idx += 1
            
        elif segment["type"] == "table":
            # Extract context from preceding text segment if available
            context_header = ""
            if i > 0 and segments[i-1]["type"] == "text":
                prev_text = '\n'.join(segments[i-1]["lines"])
                clean_prev = page_pattern.sub('', prev_text).strip()
                # Split by sentences approximately (punctuation followed by space)
                sentences = re.split(r'(?<=[.!?])\s+', clean_prev)
                context_sentences = sentences[-config.TABLE_CONTEXT_SENTENCES:] if len(sentences) >= config.TABLE_CONTEXT_SENTENCES else sentences
                context_header = ' '.join(context_sentences)
                
            chunk_content = raw_text
            if context_header:
                chunk_content = f"{context_header}\n\n{raw_text}"
                
            pages = _get_pages_for_span(start_offset, end_offset, page_map)
            
            records.append(ChunkRecord(
                chunk_id=_hash_id(source_file, global_chunk_idx),
                source_file=source_file,
                pdf_stem=pdf_stem,
                page_numbers=pages,
                element_type="table",
                chunk_index=global_chunk_idx,
                text_for_embedding=chunk_content,
                raw_content=chunk_content
            ))
            global_chunk_idx += 1

    logger.info(f"[{pdf_stem}] Produced {len(records)} chunks")
    return records
