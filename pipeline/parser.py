import logging
from typing import Tuple, Dict, Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import torch

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

def _build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    # Always ON based on user requirement
    opts.do_ocr = True               
    opts.do_table_structure = True
    # opts.images_scale = 2.0
    # opts.generate_picture_images = True
    # opts.generate_table_images = True
    # opts.do_picture_classification = True
    
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
        }
    )

_converter = None

def parse_page_batch(args: Tuple[str, list[int]]) -> Dict[int, Dict[str, Any]]:
    """Worker: parse a batch of pages of a PDF and return dict of serialisable dicts."""
    global _converter
    pdf_path, pages = args
    
    # Initialize once per worker process to save memory and CPU
    if _converter is None:
        _converter = _build_converter()
    
    results = {}
    for page_no in pages:
        try:
            result = _converter.convert(
                source=pdf_path,
                page_range=(page_no, page_no),
                raises_on_error=True,
            )
            results[page_no] = result.document.export_to_dict()
        except Exception as e:
            logger.error(f"Error parsing page {page_no} of {pdf_path}: {e}")
            raise e
            
    return results
