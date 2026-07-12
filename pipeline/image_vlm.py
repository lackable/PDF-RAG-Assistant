import logging
import threading
from PIL import Image
import torch
import gc

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import config

logger = logging.getLogger(__name__)

_model = None
_processor = None
_lock = threading.Lock()

def _load():
    global _model, _processor
    if _model is not None and _processor is not None:
        return

    logger.info(f"Loading image VLM: {config.IMAGE_VLM_MODEL} (4-bit={config.LOAD_VLM_4BIT})")
    
    try:
        if config.LOAD_VLM_4BIT and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            _model = AutoModelForImageTextToText.from_pretrained(
                config.IMAGE_VLM_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="sdpa"
            )
        else:
            _model = AutoModelForImageTextToText.from_pretrained(
                config.IMAGE_VLM_MODEL,
                device_map="auto",
                attn_implementation="sdpa"
            )
            
        _processor = AutoProcessor.from_pretrained(config.IMAGE_VLM_MODEL)
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA OOM during VLM load. Falling back to CPU.")
        torch.cuda.empty_cache()
        gc.collect()
        
        _model = AutoModelForImageTextToText.from_pretrained(
            config.IMAGE_VLM_MODEL,
            device_map="cpu"
        )
        _processor = AutoProcessor.from_pretrained(config.IMAGE_VLM_MODEL)
    except Exception as e:
        logger.error(f"Failed to load VLM: {e}")
        raise e

def unload():
    global _model, _processor
    with _lock:
        if _model is not None:
            del _model
            _model = None
        if _processor is not None:
            del _processor
            _processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Unloaded image VLM to free VRAM")

def summarize_image(pil_img: Image.Image) -> str:
    if pil_img is None:
        return "[image could not be extracted]"
        
    # Resize image to prevent massive VRAM spikes and CUDA asserts on very large images
    try:
        pil_img = pil_img.copy()
        pil_img.thumbnail((1024, 1024))
        
        # Qwen2-VL crashes (device-side assert in RoPE) if image patches are too small or 0
        if pil_img.width < 28 or pil_img.height < 28:
            # Pad the image to at least 28x28
            new_img = Image.new("RGB", (max(28, pil_img.width), max(28, pil_img.height)), (255, 255, 255))
            new_img.paste(pil_img, (0, 0))
            pil_img = new_img
    except Exception as e:
        logger.warning(f"Image resize failed: {e}")
        pass
        
    with _lock:
        if _model is None or _processor is None:
            _load()
            
        prompt = "Describe this image for a document retrieval system. Include all visible data, labels, values, trends, and key insights. Be concise but complete.Focus on capturing every detail of the values and what they corespond to."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            text = _processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = _processor(text=text, images=[pil_img], padding=True, return_tensors="pt")
            
            # Move inputs to same device as model
            inputs = inputs.to(_model.device)
            
            generated_ids = _model.generate(**inputs, max_new_tokens=300)
            
            # Remove prompt from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = _processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result_str = output_text[0].strip()
            print(f"\n[VLM Response]: {result_str}\n")
            logger.info(f"VLM Response: {result_str}")
            return result_str
            
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM during image summarization. Retrying on CPU.")
            torch.cuda.empty_cache()
            gc.collect()
            
            # Temporarily move model to CPU
            _model.to("cpu")
            inputs = inputs.to("cpu")
            
            generated_ids = _model.generate(**inputs, max_new_tokens=300)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = _processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            result_str = output_text[0].strip()
            print(f"\n[VLM Response (Fallback)]: {result_str}\n")
            logger.info(f"VLM Response (Fallback): {result_str}")
            
            # Move back to GPU if possible
            if config.LOAD_VLM_4BIT and "cuda" in config.CUDA_DEVICE:
                 # 4-bit models cannot be moved easily once loaded, so if we OOMed on 4-bit, 
                 # we just leave it or let accelerate handle it. We will just trust device_map.
                 pass
            else:
                 _model.to(config.CUDA_DEVICE)
                 
            return output_text[0].strip()
        except Exception as e:
            logger.error(f"Error during image summarization: {e}")
            return f"[Error summarizing image: {e}]"
