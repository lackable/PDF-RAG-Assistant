# PDF RAG Assistant (Stateful & Multi-Turn Chatbot)

A powerful, high-performance **Retrieval-Augmented Generation (RAG) Assistant** designed for stateful, multi-turn conversations over uploaded PDF documents. The application uses a hybrid search pipeline combining **BM25 keyword matching** and **ChromaDB vector search** with **jina-embeddings-v3**, delivering responses via **Gemini 2.0/2.5 Flash** with token-by-token streaming.

---

## 🚀 Key Features

* **Advanced PDF Ingestion**: Parallel processing using `docling` and `pypdf` with OCR support.
* **Hybrid Search Retrieval**: Combines semantic retrieval (Dense ChromaDB with `jina-embeddings-v3`) and lexical retrieval (Sparse BM25 via `rank-bm25`) for precision retrieval.
* **Stateful Conversations**: Multi-turn chat session management (persisted locally under the `sessions/` directory).
* **Real-time Streaming**: True asynchronous streaming responses using FastAPI and LangChain's Gemini integration.
* **Interactive Frontend UI**: Sleek, modern, and responsive chat web interface served directly from the FastAPI static folder.
* **Document Management**: Dedicated script `delete_pdf.py` to seamlessly wipe PDF representations across docstore, ChromaDB, and BM25 index.

---

## 📁 Repository Structure

```text
├── content/               # Place your PDF documents here for ingestion
├── static/                # Frontend web application (index.html, styles, etc.)
├── pipeline/              # Core RAG components (logger, parser, chunker, embedder, docstore, etc.)
├── generator/             # Chat response generation models & prompts (Gemini setup)
├── sessions/              # Persisted chat session transcripts (JSON) [Git Ignored]
├── chroma_db/             # Chroma database files [Git Ignored]
├── docstore/              # Processed chunks & BM25 index database [Git Ignored]
├── logs/                  # System logs [Git Ignored]
├── app.py                 # FastAPI Web Application entrypoint
├── ingest.py              # PDF Document Ingestion entrypoint
├── delete_pdf.py          # Script to remove PDF documents from indices
├── config.py              # Configuration manager for paths, models, chunk sizes, and batching
├── requirements.txt       # Python dependencies list
└── README.md              # Project Documentation
```

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd "Final PDF Rag"
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)
Create a `.env` file in the root of the project:

```env
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Custom Gemini Model (Defaults to gemini-2.0-flash)
GEMINI_MODEL=gemini-2.0-flash

# Optional: HuggingFace Hub Token (for downloading HuggingFace models/embeddings if needed)
HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here
```

---

## 📂 Usage

### Step 1: Ingesting PDF Documents
Place the PDF documents you want to chat with inside the `content/` folder. Then, execute the ingestion pipeline:

```bash
# Ingest all PDFs in the content directory
python ingest.py

# Ingest a specific PDF file
python ingest.py --pdf content/your_document.pdf

# Force re-ingest all PDFs (wipes existing database for those files first)
python ingest.py --force-reingest
```

### Step 2: Running the Web App
Start the FastAPI backend server:

```bash
uvicorn app:app --reload
```
Once the server is running, open your web browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to launch the chat dashboard.

### Step 3: Deleting an Ingested PDF
If you wish to remove a document and all its embeddings/chunks from the system database:

```bash
# Run delete script with the PDF stem name (filename without extension)
python delete_pdf.py name_of_pdf_without_extension
```

---

## ⚠️ Git & Security Guidelines

This project contains a pre-configured `.gitignore` file to ensure confidential information and heavy files are never uploaded to GitHub.
The following items are excluded from version control:
* **`.env` files** containing sensitive API keys.
* **`chroma_db/`** (local database store).
* **`sessions/`** (private conversation logs).
* **`docstore/`** (processed text chunks).
* **`logs/`** (server & parsing logs).
* **`content/*.pdf`** (large document files).
