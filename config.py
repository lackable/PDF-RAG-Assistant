import os
from dotenv import load_dotenv

load_dotenv()

# Disable PyTorch compile on Windows to prevent missing Triton crashes
os.environ["TORCH_COMPILE_DISABLE"] = "1"

HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

# Models
IMAGE_VLM_MODEL        = "Qwen/Qwen2.5-VL-3B-Instruct"
EMBEDDING_MODEL        = "jinaai/jina-embeddings-v3"
LOAD_VLM_4BIT          = True

# Device
CUDA_DEVICE            = "cuda:0"
EMBEDDING_BATCH        = 16

# Chunking
CHUNK_SIZE             = 512
CHUNK_OVERLAP          = 64
TABLE_CONTEXT_SENTENCES = 2

# Parallelism
PARSE_WORKERS          = 2

# Docling
DO_OCR                 = True

# Paths
CONTENT_DIR            = "./content"
DOCSTORE_DIR           = "./docstore"
DOCSTORE_FILE          = "./docstore/docstore.json"
BM25_STORE_PATH        = "./docstore/bm25_store.pkl"
CHROMA_DB_PATH         = "./chroma_db"
LOG_DIR                = "./logs"
LOG_FILE               = "./logs/ingest.log"
CHROMA_COLLECTION      = "pdf_chunks"

# Ensure directories exist
os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(DOCSTORE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
