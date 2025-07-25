# RAG LLM - Intelligent Document Search & Chat System

This project implements a Retrieval-Augmented Generation (RAG) system using local LLMs for document search and contextual Q&A.

## Features

- Document ingestion (PDF, DOCX, JSON, HTML, images, CSV)
- Local LLM integration (Llama-2)
- Vector storage using FAISS
- HuggingFace embeddings
- Interactive chat interface

## Project Structure

```
rag_llm/
├── data/                  # Store your documents here
├── src/
│   ├── __init__.py
│   ├── document_loader.py # Document processing
│   ├── embeddings.py      # Embedding generation
│   ├── llm_interface.py   # LLM initialization
│   └── vector_store.py    # FAISS vector store
├── config.py             # Configuration settings
├── main.py              # Main application
└── requirements.txt     # Dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag_llm.git
cd rag_llm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your documents in the `data/` directory

5. Run the application:
```bash
python main.py
```

## Usage

1. The system will first process all documents in the `data/` directory
2. A vector store will be created (or loaded if it exists)
3. The LLM will be initialized
4. You can then interact with the