import logging
from typing import Dict, Any
from docling_core.types.doc.document import DoclingDocument, PictureItem
# from . import image_vlm

logger = logging.getLogger(__name__)

def build_page_markdown(page_no: int, doc_dict: Dict[str, Any]) -> str:
    """Builds markdown for a single page, embedding VLM image summaries."""
    
    doc = DoclingDocument.model_validate(doc_dict)
    
    # 1. Collect image summaries in document encounter order
    summaries = []
    
    # Chart/graph class names that Docling's classifier produces
    CHART_CLASSES = {
        "bar_chart", "line_chart", "scatter_plot", "pie_chart",
        "flow_chart", "box_plot", "engineering_drawing",
        "geographical_map", "topographical_map",
        "screenshot_from_manual", "screenshot_from_computer",
    }

    for item, _ in doc.iterate_items():
        if isinstance(item, PictureItem):
            # Image processing and VLM summarization commented out
            # is_target = False
            # 
            # try:
            #     item_dict = item.model_dump() if hasattr(item, "model_dump") else item.dict()
            #     
            #     for ann in item_dict.get("annotations", []):
            #         if not isinstance(ann, dict):
            #             continue
            #         if ann.get("kind") != "classification":
            #             continue
            #         
            #         # predicted_classes is sorted by confidence descending — top prediction is [0]
            #         predicted_classes = ann.get("predicted_classes", [])
            #         if predicted_classes:
            #             top_class = predicted_classes[0].get("class_name", "").lower()
            #             top_conf = predicted_classes[0].get("confidence", 0.0)
            #             logger.debug(f"Page {page_no} image top class: {top_class} ({top_conf:.2%})")
            #             
            #             if top_class in CHART_CLASSES and top_conf >= 0.3:
            #                 is_target = True
            #         break  # only one classification annotation expected
            # except Exception as e:
            #     logger.warning(f"Failed to read image classification on page {page_no}: {e}")
            #     # If classification fails, default to skipping to avoid wasting VLM time
            #     is_target = False
            # 
            # if is_target:
            #     pil_img = item.get_image(doc)
            #     if pil_img:
            #         logger.debug(f"Summarizing chart on page {page_no}")
            #         summary = image_vlm.summarize_image(pil_img)
            #     else:
            #         summary = "[image could not be extracted]"
            # else:
            #     summary = "[SKIP]"
            
            summaries.append("[SKIP]")
            
    # 2. Export raw markdown (contains <!-- image --> placeholders)
    raw_md = doc.export_to_markdown()
    
    # 3. Replace placeholders with summaries
    enriched_md = raw_md
    for summary in summaries:
        if summary == "[SKIP]":
            replacement = ""  # Hide non-chart images entirely
        else:
            replacement = f"\n> **[Image Summary, Page {page_no}]** {summary}\n"
            
        enriched_md = enriched_md.replace("<!-- image -->", replacement, 1)
        
    # 4. Prepend page marker
    final_md = f"<!-- page:{page_no} -->\n\n{enriched_md}"
    
    return final_md
