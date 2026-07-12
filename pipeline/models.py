from pydantic import BaseModel
from typing import List, Optional, Literal

class ChunkRecord(BaseModel):
    chunk_id: str
    source_file: str
    pdf_stem: str
    page_numbers: List[int]
    element_type: Literal["text", "table", "image"]
    chunk_index: int
    text_for_embedding: str
    raw_content: str
    embedding: Optional[List[float]] = None
    ingested_at: str = ""
