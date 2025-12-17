# Recommendation Checker RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed to validate and check recommendations using AI-powered knowledge retrieval and generation.

## Overview

This project implements a complete RAG pipeline that combines vector search with large language models to provide intelligent recommendation checking and validation. The system uses FAISS for efficient vector similarity search and integrates with modern LLM APIs for natural language understanding and generation.

## Features

- **Knowledge Base Management**: Build and maintain a searchable knowledge base from various document formats
- **Vector Store**: Efficient similarity search using FAISS indexing
- **RAG Pipeline**: Combines retrieval and generation for context-aware responses
- **Session History**: Track and manage conversation history
- **Feedback System**: Collect and store user feedback for continuous improvement
- **Encryption**: Secure handling of sensitive data
- **File Processing**: Support for multiple document formats (DOCX, JSON, etc.)
- **Gradio Interface**: User-friendly web interface for interaction

## Architecture

```
┌─────────────────┐
│  User Interface │ (Gradio/Web)
└────────┬────────┘
         │
┌────────▼────────┐
│  Server Layer   │ (server.py)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│ RAG  │  │ Query │
│ Gen  │  │  KB   │
└───┬──┘  └──┬────┘
    │        │
┌───▼────────▼───┐
│ Knowledge Base │ (FAISS + Metadata)
└────────────────┘
```

## Project Structure

```
rec_checker-rag/
├── server.py                          # Main application server
├── rag_generate.py                    # RAG generation logic
├── query_knowledge_base.py            # Knowledge base querying
├── build_knowledge_base.py            # KB construction pipeline
├── build_vector_store.py              # Vector store builder
├── llm_client.py                      # LLM API integration
├── embedding_client.py                # Embedding model client
├── file_processor.py                  # Document processing utilities
├── db_utils.py                        # Database operations
├── history_manager.py                 # Session history management
├── feedback_utils.py                  # Feedback collection
├── encryption_utils.py                # Data encryption/decryption
├── config_manager.py                  # Configuration handling
├── patterns.py                        # Pattern matching utilities
├── utils.py                           # General utilities
├── clean.py                           # Cleanup scripts
├── data/                              # Data directory
├── knowledge_base_flat.index          # FAISS vector index
├── knowledge_base_metadata.json       # KB metadata storage
├── knowledge_chunks.json              # Chunked knowledge data
├── feedback.db                        # Feedback database
├── session_history.db                 # Session history database
├── .env.exemple                       # Environment variables template
├── requirements.txt                   # Python dependencies
└── vertex_service_key.json           # GCP Vertex AI credentials
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/arouedkhelifi/rec_checker-rag.git
cd rec_checker-rag
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.exemple .env
# Edit .env with your API keys and configuration
```

5. **Set up GCP credentials** (if using Vertex AI)
```bash
# Place your service account key in vertex_service_key.json
# Or set GOOGLE_APPLICATION_CREDENTIALS environment variable
```

## Configuration

Create a `.env` file with the following variables:

```env
# LLM Configuration
LLM_API_KEY=your_api_key_here
LLM_MODEL=your_model_name
LLM_ENDPOINT=your_api_endpoint

# Embedding Configuration
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_MODEL=your_embedding_model

# Database Configuration
DB_PATH=./session_history.db
FEEDBACK_DB_PATH=./feedback.db

# Vector Store Configuration
VECTOR_STORE_PATH=./knowledge_base_flat.index
METADATA_PATH=./knowledge_base_metadata.json

# Encryption
ENCRYPTION_KEY=your_encryption_key

# Server Configuration
HOST=0.0.0.0
PORT=7860
```

## Usage

### Building the Knowledge Base

1. **Prepare your documents**
   - Place documents in the `data/` directory
   - Supported formats: DOCX, JSON, TXT, PDF

2. **Build the knowledge base**
```bash
python build_knowledge_base.py
```

3. **Build the vector store**
```bash
python build_vector_store.py
```

### Running the Server

```bash
python server.py
```

Access the interface at `http://localhost:7860`

### Querying the Knowledge Base

```python
from query_knowledge_base import query_kb

results = query_kb("What are the recommendations for X?", top_k=5)
print(results)
```

### Using the RAG Pipeline

```python
from rag_generate import generate_response

response = generate_response(
    query="Check this recommendation: ...",
    context=retrieved_context
)
print(response)
```

## API Reference

### Knowledge Base Operations

```python
# Build knowledge base
from build_knowledge_base import build_kb
build_kb(data_dir="./data", output_path="./knowledge_chunks.json")

# Query knowledge base
from query_knowledge_base import search_similar
results = search_similar(query_vector, top_k=5)
```

### RAG Generation

```python
from rag_generate import RAGGenerator

rag = RAGGenerator()
response = rag.generate(
    query="Your query here",
    num_results=5,
    temperature=0.7
)
```

### Feedback Management

```python
from feedback_utils import save_feedback, get_feedback

# Save feedback
save_feedback(
    session_id="123",
    query="user query",
    response="system response",
    rating=5
)

# Retrieve feedback
feedback = get_feedback(session_id="123")
```

## Features in Detail

### Vector Search
- Uses FAISS for efficient similarity search
- Supports multiple index types (Flat, IVF, HNSW)
- Configurable search parameters

### Session Management
- Persistent conversation history
- Session-based context retention
- Multi-user support

### Security
- Encrypted sensitive data storage
- Secure API key management
- Data privacy compliance

### Monitoring
- Query logging
- Performance metrics
- Feedback analytics

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

### Cleaning Up
```bash
python clean.py
```


## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- FAISS by Facebook AI Research
- Gradio for the web interface
- OpenAI/Anthropic/Google for LLM APIs

**Note**: This is an active project under development. Features and APIs may change.

