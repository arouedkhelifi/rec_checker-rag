"""
Vector Store Builder for RAG Pipelines

This script takes pre-processed knowledge chunks (usually from a RAG chunking step)
and embeds them using a SentenceTransformer model. The resulting embeddings are
indexed using FAISS to create a searchable vector store.

Main Features:
- Batch embedding with optional caching
- Multiple FAISS index types: Flat, IVF, HNSW
- Automatic language detection for each chunk
- Progress-aware processing with logging
- Saves both index and metadata to disk
"""

import json
import logging
import argparse
import os
import time
import hashlib
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
CACHE_DIR = ".embedding_cache"

def get_text_hash(text: str) -> str:
    """Generate a hash for caching embeddings of a text string."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_embeddings_batch(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_cache: bool = False
) -> np.ndarray:
    """
    Generate sentence embeddings for a list of texts, optionally using a cache.

    Args:
        texts: The list of text strings to embed.
        model: The sentence transformer model.
        batch_size: Number of samples per embedding batch.
        use_cache: Whether to cache/reuse embeddings.

    Returns:
        A numpy array of embeddings.
    """
    embeddings, texts_to_embed, cache_keys, cache_indices = [], [], [], []

    # Prepare cache directory if needed
    if use_cache and not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Try to load cached embeddings
    if use_cache:
        for i, text in enumerate(texts):
            text_hash = get_text_hash(text)
            cache_path = os.path.join(CACHE_DIR, f"{text_hash}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    embeddings.append((i, pickle.load(f)))
            else:
                texts_to_embed.append(text)
                cache_keys.append(text_hash)
                cache_indices.append(i)
    else:
        texts_to_embed = texts
        cache_indices = list(range(len(texts)))

    if not texts_to_embed:
        # All texts were cached — return sorted
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])

    # Compute new embeddings in batches
    batch_embeddings = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch = model.encode(texts_to_embed[i:i+batch_size], show_progress_bar=False)
        batch_embeddings.extend(batch)

        # Save to cache if enabled
        if use_cache:
            for j, embedding in enumerate(batch):
                idx = i + j
                if idx < len(cache_keys):
                    cache_path = os.path.join(CACHE_DIR, f"{cache_keys[idx]}.pkl")
                    with open(cache_path, 'wb') as f:
                        pickle.dump(embedding, f)

    # Recombine with cached embeddings
    if use_cache:
        for i, embedding in zip(cache_indices, batch_embeddings):
            embeddings.append((i, embedding))
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    else:
        return np.array(batch_embeddings)

def create_faiss_index(vectors: np.ndarray, index_type: str = "flat") -> faiss.Index:
    """
    Create a FAISS index from a matrix of vectors.

    Args:
        vectors: The numpy matrix of shape (n_samples, dim).
        index_type: One of ["flat", "ivf", "hnsw"].

    Returns:
        A FAISS index object.
    """
    dimension = vectors.shape[1]

    if index_type == "flat":
        idx = faiss.IndexFlatL2(dimension)
    elif index_type == "ivf":
        nlist = max(1, min(2048, int(np.sqrt(vectors.shape[0]))))  # # of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        idx = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        idx.train(vectors)
        idx.nprobe = min(20, nlist)
    elif index_type == "hnsw":
        idx = faiss.IndexHNSWFlat(dimension, 16)
        idx.hnsw.efConstruction = 200
        idx.hnsw.efSearch = 128
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    idx.add(vectors)
    return idx

def detect_language(text: str) -> str:
    """
    Very basic heuristic language detector based on code keywords.
    Returns a string like 'python', 'java', 'cpp', or 'general'.
    """
    txt = text.lower()
    if any(kw in txt for kw in ["def ", "import ", "self", "python"]): return "python"
    if any(kw in txt for kw in ["public static void", "class ", "java"]): return "java"
    if any(kw in txt for kw in ["function ", "console.log", "javascript", "var ", "let ", "const "]): return "javascript"
    if any(kw in txt for kw in ["#include", "std::", "c++", "cpp"]): return "cpp"
    if any(kw in txt for kw in ["using namespace", "c#"]): return "csharp"
    if any(kw in txt for kw in ["fmt.", "package main", "go "]): return "go"
    if any(kw in txt for kw in ["fn main()", "rust"]): return "rust"
    return "general"

def build_vector_store(
    knowledge_chunks: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    index_type: str = "flat",
    batch_size: int = DEFAULT_BATCH_SIZE,
    use_cache: bool = False,
    save_files: bool = False,
    index_path: Optional[str] = None,
    metadata_path: Optional[str] = None
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Main vector store builder: embeds knowledge chunks and creates a FAISS index.

    Args:
        knowledge_chunks: List or dict of text chunks with metadata.
        model_name: SentenceTransformer model name.
        index_type: FAISS index type to use.
        batch_size: Embedding batch size.
        use_cache: Enable embedding caching.
        save_files: Whether to save the index and metadata to disk.
        index_path: Filepath to save FAISS index.
        metadata_path: Filepath to save metadata JSON.

    Returns:
        A tuple: (FAISS index, list of enriched metadata)
    """
    # Handle input formats
    if "chunks" in knowledge_chunks:
        chunks = knowledge_chunks["chunks"]
    elif isinstance(knowledge_chunks, list):
        chunks = knowledge_chunks
    else:
        raise ValueError("Invalid knowledge chunks format")

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract texts and enrich metadata
    texts, metadatas = [], []
    for chunk in chunks:
        if "text" in chunk and chunk["text"].strip():
            texts.append(chunk["text"])
            chunk["language"] = detect_language(chunk["text"])
            metadatas.append(chunk)

    vectors = get_embeddings_batch(texts, model, batch_size, use_cache)
    index = create_faiss_index(vectors, index_type)

    # Save to disk if needed
    if save_files:
        index_path = index_path or f"knowledge_base_{index_type}.index"
        metadata_path = metadata_path or "knowledge_base_metadata.json"

        faiss.write_index(index, index_path)

        metadata_with_info = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "model": model_name,
                "index_type": index_type,
                "vector_count": len(vectors),
                "vector_dimension": vectors.shape[1]
            },
            "chunks": metadatas
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_with_info, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved FAISS index to {index_path} and metadata to {metadata_path}")

    return index, metadatas

def main():
    """Command-line interface for building a vector store from a knowledge file."""
    parser = argparse.ArgumentParser(description='Build vector store for RAG')
    parser.add_argument('--input', '-i', default="knowledge_chunks.json")
    parser.add_argument('--index-output')
    parser.add_argument('--metadata-output')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL)
    parser.add_argument('--index-type', '-t', choices=['flat', 'ivf', 'hnsw'], default='flat')
    parser.add_argument('--batch-size', '-b', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--cache-embeddings', '-c', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    with open(args.input, "r", encoding="utf-8") as f:
        knowledge_chunks = json.load(f)

    build_vector_store(
        knowledge_chunks,
        model_name=args.model,
        index_type=args.index_type,
        batch_size=args.batch_size,
        use_cache=args.cache_embeddings,
        save_files=True,
        index_path=args.index_output,
        metadata_path=args.metadata_output
    )

if __name__ == "__main__":
    main()
