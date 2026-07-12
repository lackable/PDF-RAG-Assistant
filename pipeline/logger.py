import logging
import os
import sys
import json
from rich.logging import RichHandler
import config

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging():
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Rich Console Handler for human-readable output
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True
    )
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)
    
    # JSON File Handler for machine-readable audit
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    
    # Suppress verbose third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers_modules").setLevel(logging.ERROR)
    logging.getLogger("docling_core").setLevel(logging.WARNING)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation").setLevel(logging.WARNING)
