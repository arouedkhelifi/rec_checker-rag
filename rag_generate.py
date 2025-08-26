"""
Recommendation Checker Chatbot - RAG Generation Interface

Updated to use environment variables and company LLM proxy support.

This file is the improved, cleaned version:
 - Removed gradio_wrapper (frontend-specific)
 - Added provenance metadata when retrieving KB chunks (id, section, score)
 - generate_node prompts the LLM to reference KB items by ID/section
 - Appends a machine-readable Sources summary to the output
"""
import sqlite3
from db_utils import get_past_feedback_for_file
from utils import prepare_state_from_input
from config_manager import config
from llm_client import llm_client, call_llm, call_llm_stream
from feedback_utils import analyze_feedback, submit_user_feedback
from patterns import LANGUAGE_PATTERNS, COMMENT_PATTERNS, COMPLEXITY_PATTERNS
import os
import json
import sys
import argparse
import time
import hashlib
import re
import logging
import concurrent.futures
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential
import tempfile
from fpdf import FPDF
from reportlab.pdfgen import canvas
import litellm

# Initialize litellm with your configuration
config.setup_litellm()

# Use the DEFAULT_LLM property from config
DEFAULT_LLM = config.DEFAULT_LLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for caching and resources
RESPONSE_CACHE: Dict[str, Any] = {}
index = None
metadatas = None
embedding_model = None
agent = None


# TypedDict for agent state
class AgentState(TypedDict):
    code: str
    code_language: str
    code_filename: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    metrics: Optional[Dict[str, Any]]
    dependencies: Optional[Dict[str, Any]]
    error: Optional[str]
    target_language: Optional[str]


def load_resources(index_path: str = None, metadata_path: str = None, model_name: str = None):
    """
    Load the FAISS index, metadata JSON, and the embedding model using config.
    Enhanced with forced fallback mechanism.
    """
    index_path = index_path or config.VECTOR_INDEX_PATH
    metadata_path = metadata_path or config.VECTOR_METADATA_PATH
    model_name = model_name or config.EMBEDDING_MODEL

    DEFAULT_FALLBACK_MODEL = "all-MiniLM-L6-v2"

    logger.info(f"📦 Loading FAISS index from {index_path}")
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
        # metadatas expected to be list of dicts; some JSONs wrap chunks under "chunks"
        metadatas = data["chunks"] if isinstance(data, dict) and "chunks" in data else data

        # Always load fallback embedding model first as safety net
        logger.info(f"🔄 Loading fallback model first: {DEFAULT_FALLBACK_MODEL}")
        fallback_model = SentenceTransformer(DEFAULT_FALLBACK_MODEL)
        _ = fallback_model.encode("test")
        logger.info(f"✅ Fallback embedding model loaded successfully: {DEFAULT_FALLBACK_MODEL}")

        # Try to load the specified embedding model (proxy clients handled externally)
        primary_model = None

        if model_name and model_name.startswith("vertex_ai/"):
            logger.info(f"🔄 Attempting to load Vertex AI embedding model: {model_name}")
            try:
                from embedding_client import ProxyEmbeddingModel
                primary_model = ProxyEmbeddingModel(model_name)
                _ = primary_model.encode("test")
                logger.info(f"✅ Vertex AI embedding model loaded successfully: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Vertex AI embedding model: {e}")
                logger.info(f"🔄 Falling back to default model: {DEFAULT_FALLBACK_MODEL}")
                primary_model = None

        embedding_model = primary_model if primary_model is not None else fallback_model

        # FORCE FALLBACK for known threading problems with external models
        if model_name and model_name.startswith("vertex_ai/"):
            logger.warning("🔄 FORCING fallback model due to known Vertex AI threading issues")
            embedding_model = fallback_model

        logger.info(f"✅ Resources loaded: {index.ntotal} vectors")
        logger.info(f"📋 Final embedding model: {type(embedding_model).__name__}")

        return index, metadatas, embedding_model

    except Exception as e:
        logger.error(f"❌ Failed to load resources: {e}")
        raise


def detect_language(code: str, filename: str = "") -> str:
    """Detect the programming language of the given code."""
    if filename:
        for lang, patterns in LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if pattern.startswith(r'\.') and re.search(pattern, filename, re.IGNORECASE):
                    return lang

    scores = {
        lang: sum(
            bool(re.findall(p, code, re.IGNORECASE | re.MULTILINE))
            for p in patterns if not p.startswith(r'\.'))
        for lang, patterns in LANGUAGE_PATTERNS.items()
    }

    best_lang, score = max(scores.items(), key=lambda x: x[1], default=("unknown", 0))
    return best_lang if score > 0 else "unknown"


def get_cache_key(code: str) -> str:
    """Generate a unique MD5 hash key for caching based on code content."""
    return hashlib.md5(code.encode('utf-8')).hexdigest()


def mmr_search(
    query: str,
    code_language: str,
    index,
    metadatas,
    embedding_model,
    top_k=10,
    lambda_param=0.5
):
    """
    Perform Maximum Marginal Relevance search with section info.
    Returns items with text, section, id (if present), keywords, score.
    """
    language_query = f"recommendations for {code_language} code best practices"
    combined_query = f"{query} {language_query}"
    query_vec = embedding_model.encode(combined_query).astype("float32")
    query_vec = query_vec.reshape(1, -1)

    # We search a larger pool and then MMR-select top_k from candidates
    distances, indices = index.search(query_vec, top_k * 3)
    distances = distances.flatten()
    indices = indices.flatten()

    candidate_ids = [idx for idx in indices if 0 <= idx < len(metadatas)]
    if not candidate_ids:
        return []

    candidate_texts = [metadatas[idx].get("text", "") for idx in candidate_ids]
    # Encode candidate texts
    candidate_embeddings = np.vstack([
        embedding_model.encode(text).astype("float32") for text in candidate_texts
    ])

    query_vec_flat = query_vec.flatten()

    selected_indices = []
    remaining_indices = list(range(len(candidate_ids)))

    for _ in range(min(top_k, len(candidate_ids))):
        if not remaining_indices:
            break

        mmr_scores = []
        for i in remaining_indices:
            # Relevance = cosine(query, candidate)
            cand_emb = candidate_embeddings[i]
            rel = np.dot(cand_emb, query_vec_flat) / (
                (np.linalg.norm(cand_emb) * np.linalg.norm(query_vec_flat)) + 1e-12
            )

            # Diversity relative to already selected
            diversity = 1.0
            if selected_indices:
                sims = [
                    np.dot(candidate_embeddings[i], candidate_embeddings[j]) /
                    ((np.linalg.norm(candidate_embeddings[i]) * np.linalg.norm(candidate_embeddings[j])) + 1e-12)
                    for j in selected_indices
                ]
                diversity = 1 - max(sims) if sims else 1.0

            mmr_score = lambda_param * rel + (1 - lambda_param) * diversity
            mmr_scores.append((i, mmr_score))

        next_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)

    results = []
    for idx in selected_indices:
        original_idx = candidate_ids[idx]
        chunk = metadatas[original_idx]
        # find distance location for this original_idx in indices
        distance_idx = np.where(indices == original_idx)[0]
        raw_distance = float(distances[distance_idx[0]]) if len(distance_idx) > 0 else 1.0
        score = 1.0 / (1.0 + raw_distance) if raw_distance >= 0 else 0.0

        results.append({
            "id": chunk.get("id", str(original_idx)),
            "text": chunk.get("text", ""),
            "section": chunk.get("section", "Unknown"),
            "score": score,
            "keywords": chunk.get("keywords", []),
            # include original metadata for traceability
            "meta_index": original_idx
        })

    return results


def analyze_code_metrics(code: str, language: str) -> Dict[str, Any]:
    """Calculate basic code quality metrics."""
    metrics = {
        "line_count": len(code.splitlines()),
        "char_count": len(code),
        "avg_line_length": round(len(code) / max(1, len(code.splitlines())), 2),
        "comment_ratio": 0,
        "complexity_estimate": 0
    }

    if language in COMMENT_PATTERNS:
        comments = re.findall(COMMENT_PATTERNS[language], code)
        comment_chars = sum(len(c) for c in comments)
        metrics["comment_ratio"] = round(comment_chars / max(1, len(code)), 3)

    if language in COMPLEXITY_PATTERNS:
        control_structures = re.findall(COMPLEXITY_PATTERNS[language], code)
        metrics["complexity_estimate"] = len(control_structures)

    if language in ["python", "javascript", "java", "c#", "c++", "c"]:
        decision_points = 0
        decision_patterns = [
            r'\bif\b', r'\belse\s+if\b', r'\bswitch\b', r'\bcase\b',
            r'\bfor\b', r'\bwhile\b', r'\bdo\b', r'\bcatch\b'
        ]
        if language == "python":
            decision_patterns.extend([r'\belif\b', r'\bexcept\b'])
        elif language in ["javascript", "typescript"]:
            decision_patterns.extend([r'\b\?\b', r'\?\.\b', r'\?\?'])
        elif language == "java":
            decision_patterns.extend([r'\b\?\b', r'&&', r'\|\|'])

        for pattern in decision_patterns:
            decision_points += len(re.findall(pattern, code))

        metrics["cyclomatic_complexity"] = decision_points + 1

    return metrics


def analyze_dependencies(code: str, language: str) -> Dict[str, Any]:
    """Extract dependencies/imports from the code."""
    dependencies = []

    if language == "python":
        import_patterns = [
            r'import\s+(\w+)',
            r'from\s+(\w+)\s+import',
            r'import\s+(\w+)\s+as\s+\w+'
        ]
        for pattern in import_patterns:
            dependencies.extend(re.findall(pattern, code))

    elif language in ["javascript", "typescript"]:
        import_patterns = [
            r'import.*from\s+[\'"](.+?)[\'"]',
            r'require\([\'"](.+?)[\'"]\)'
        ]
        for pattern in import_patterns:
            dependencies.extend(re.findall(pattern, code))

    elif language == "java":
        dependencies.extend(re.findall(r'import\s+([\w\.]+);', code))

    elif language == "go":
        imports = re.findall(r'import\s+\(\s*(.*?)\s*\)', code, re.DOTALL)
        if imports:
            for imp_block in imports:
                dependencies.extend(re.findall(r'[\'"](.+?)[\'"]', imp_block))
        dependencies.extend(re.findall(r'import\s+[\'"](.+?)[\'"]', code))

    return {
        "dependencies": list(set(dependencies)),
        "count": len(set(dependencies))
    }


def translate_recommendations(recommendations: str, target_language: str) -> str:
    """Translate recommendations to target language."""
    if target_language.lower() == "english":
        return recommendations

    prompt = f"""Translate the following code recommendations to {target_language}. 
Keep code snippets and technical terms in English.

Original recommendations:
{recommendations}

{target_language} translation:"""

    try:
        result = call_llm(prompt, temperature=0.1)
        return result.strip()
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"⚠️ Translation failed due to an error: {e}"


def generate_node(state: AgentState) -> AgentState:
    """
    Generate recommendations, including section info in the context.
    Adds explicit Sources summary after LLM output with KB item ids and sections.
    """
    if state.get("error"):
        state["answer"] = f"An error occurred: {state['error']}"
        return state

    # If no retrieved chunks, still perform code-only analysis
    retrieved = state.get("retrieved_chunks", []) or []

    # Build context enumerating KB items with IDs so the LLM can reference them like [KB:Architecture_14_0]
    context_lines = []
    for i, c in enumerate(retrieved):
        # ensure id exists
        kb_id = c.get("id", f"kb_{i}")
        section = c.get("section", "Unknown")
        text = c.get("text", "")
        context_lines.append(f"[{i+1}] [KB:{kb_id}] [{section}] {text}")

    context = "\n\n".join(context_lines)

    language = state.get("code_language", "unknown")
    filename = state.get("code_filename", "unknown")

    metrics_info = ""
    if state.get("metrics"):
        metrics_info = f"""
Code metrics:
- Lines of code: {state['metrics'].get('line_count', 'N/A')}
- Comment ratio: {state['metrics'].get('comment_ratio', 'N/A')}
- Complexity estimate: {state['metrics'].get('complexity_estimate', 'N/A')}
"""
        if "cyclomatic_complexity" in state["metrics"]:
            metrics_info += f"- Cyclomatic complexity: {state['metrics']['cyclomatic_complexity']}\n"

    dependencies_info = ""
    if state.get("dependencies") and state["dependencies"].get("count", 0) > 0:
        deps = state["dependencies"].get("dependencies", [])[:10]
        dependencies_info = f"""
Dependencies ({state['dependencies']['count']}):
{', '.join(deps)}
"""
        if len(state['dependencies'].get('dependencies', [])) > 10:
            dependencies_info += f"... and {len(state['dependencies']['dependencies']) - 10} more"

    # Instruct the LLM to reference KB items using [KB:<id>] tokens, and to mark file-origin items with [FILE]
    prompt = (
    f"## 🧠 ROLE: You are an expert code reviewer specializing in {language}.\n"
    f"You are tasked with a deep, professional review of the provided {language} codebase.\n\n"
    
    f"### 📌 Evaluation Must Cover:\n"
    f"1. Performance – runtime efficiency and computational cost.\n"
    f"2. Efficiency – logic simplification, memory/resource optimization.\n"
    f"3. Environmental Impact (Green IT) – energy and resource footprint.\n"
    f"4. Software Design Principles – SOLID, DRY, maintainability.\n"
    f"5. Design Patterns – usage and opportunities for improvement.\n\n"
    
    f"Do NOT recommend changing the programming language.\n"
    f"Keep all guidance descriptive and actionable, but do NOT provide code implementations.\n"
    f"---\n"
    
    f"## 📘 Best Practice Context (Knowledge Base Items):\n{context if context else 'No KB items matched.'}\n"
    f"---\n"
    
    f"## 📊 Code Metrics:\n{metrics_info}\n"
    f"## 📦 Dependencies:\n{dependencies_info}\n"
    f"## 📁 Folder Structure:\n{state.get('folder_structure', 'N/A')}\n\n"
    
    f"## 💻 Provided Code Snippet (Language: {language}):\n```{language}\n{state['code']}\n```\n"
    f"---\n"
    
    f"## TASK INSTRUCTIONS:\n"
    f"1. Check each best practice recommendation against the code and project structure.\n"
    f"2. For each issue or inefficiency, provide:\n"
    f"   - ⚠️ **Issue**: [Clear description of the problem]\n"
    f"   - ✅ **Recommendation**: [Descriptive, actionable guidance WITHOUT code]\n"
    f"   - 📖 **Source**: [KB:<id>] (with section) if from knowledge base OR [FILE:{state.get('code_filename','unknown')}] if derived from the file.\n"
    f"3. Analyze architecture on two levels:\n"
    f"   - File-level: cohesion, responsibilities, maintainability.\n"
    f"   - Project-level: folder layout, modularity, coupling, interdependencies.\n"
    f"4. Evaluate Green IT impact descriptively.\n"
    f"---\n"
    
    f"## OUTPUT FORMAT:\n"
    f"Start the report with the title '🔍 Code Review'.\n\n"
    f"For each file with issues:\n"
    f"📄 **filename.py**\n\n"
    f"For each category with issues (skip empty categories):\n"
    f"🔶 **Category Name**\n\n"
    f"⚠️ **Issue**: [Describe clearly]\n"
    f"✅ **Recommendation**: [Actionable guidance]\n"
    f"📖 **Source**: [KB:<id>] or [FILE:<filename>]\n\n"
    
    f"Project-level architecture issues:\n"
    f"🏢 **Project Architecture**\n\n"
    f"⚠️ **Issue**: [Describe clearly]\n"
    f"✅ **Recommendation**: [Descriptive guidance, no code]\n"
    f"📖 **Source**: [KB:<id>] if applicable\n\n"
    
    f"Rules:\n"
    f"1. Only mention categories where issues exist.\n"
    f"2. Skip empty sections.\n"
    f"3. Use emojis and bold for hierarchy.\n"
    f"4. Separate sections clearly for readability.\n"
    f"5. If no issues found, output exactly:\n"
    f"🔍 Code Review\n\nNo issues found in the codebase.\n"


    )

    # Append past feedback if available
    past_feedbacks = get_past_feedback_for_file(filename)
    if past_feedbacks:
        prompt += "\n\n## Past user feedback (consider these when writing recommendations):\n"
        for fb in past_feedbacks:
            prompt += f"- {fb}\n"

    try:
        # Streaming accumulation
        state["answer"] = ""
        for chunk in litellm.completion(
            model=DEFAULT_LLM,
            messages=[
                {"role": "system", "content": f"Expert {language} reviewer focusing on best practices"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=min(4096, config.LLM_MAX_TOKENS),
            temperature=0.2,
            top_p=0.9,
            stream=True
        ):
            logger.debug(f"Chunk received: {chunk}")
            # Extract streamed text content if available (litellm/completion streaming may yield dict-like objects)
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = getattr(chunk.choices[0], 'delta', None)
                if delta and getattr(delta, 'content', None):
                    content = delta.content
                    if isinstance(content, str) and content.strip() != "":
                        state["answer"] += content
            elif isinstance(chunk, dict) and "choices" in chunk:
                # Compatible with llm_client._convert_openai_stream output
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content:
                    state["answer"] += content

        # Add machine-readable sources summary so callers can reliably read provenance
        sources_summary_lines = []
        if retrieved:
            sources_summary_lines.append("\n\n---\n\nSources (knowledge base items used):\n")
            for c in retrieved:
                sources_summary_lines.append(f"- KB ID: {c.get('id')} | section: {c.get('section')} | score: {c.get('score'):.4f}")
        else:
            sources_summary_lines.append("\n\n---\n\nSources: No knowledge-base items were matched; analysis was based on the provided file.\n")

        # Attach a short provenance footer
        provenance_footer = "\n".join(sources_summary_lines)
        state["answer"] = state["answer"].strip() + provenance_footer

        # Translate answer if requested
        if state.get("target_language") and state["target_language"].lower() != "english":
            state["answer"] = translate_recommendations(state["answer"], state["target_language"])

    except Exception as e:
        state["error"] = f"Error during generation: {str(e)}"
        state["answer"] = f"Failed to generate recommendations: {str(e)}"
        logger.error(f"Generation error: {e}")

    return state


def process_chunks_parallel(chunks: List[Dict], target_language: str, index_ref, metadatas_ref, embedding_model_ref, agent_ref) -> List[str]:
    """Process chunks in parallel (limited concurrency) with resource verification."""
    import concurrent.futures

    # Verify resources before starting parallel processing
    if index_ref is None or metadatas_ref is None:
        logger.error("Missing required resources for parallel processing")
        return [f"Error: Missing vector index or metadata resources" for _ in chunks]

    # Create fallback embedding model if needed
    if embedding_model_ref is None:
        logger.warning("Creating fallback embedding model for parallel processing")
        from sentence_transformers import SentenceTransformer
        embedding_model_ref = SentenceTransformer("all-MiniLM-L6-v2")

    def process_single_chunk(chunk_data):
        chunk, idx = chunk_data
        try:
            # Create fresh embedding model for each thread to avoid sharing issues
            from sentence_transformers import SentenceTransformer
            thread_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            initial_state: AgentState = {
                "code": chunk["content"],
                "code_language": chunk["language"],
                "code_filename": f"{chunk['filename']} (chunk {chunk['chunk_index']+1})",
                "retrieved_chunks": [],
                "answer": "",
                "metrics": None,
                "dependencies": None,
                "error": None,
                "target_language": target_language
            }

            # Use thread-safe processing with fresh resources
            state_after_retrieve = custom_retrieve_node_safe(
                initial_state,
                index_ref,
                metadatas_ref,
                thread_embedding_model
            )
            final_state = generate_node(state_after_retrieve)
            return idx, final_state["answer"]

        except Exception as e:
            logger.error(f"Error processing chunk {idx}: {e}")
            return idx, f"Error processing chunk {idx}: {str(e)}"

    results = [""] * len(chunks)

    # Use ThreadPoolExecutor with limited workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]
        future_to_index = {
            executor.submit(process_single_chunk, data): data[1]
            for data in chunk_data
        }

        for future in concurrent.futures.as_completed(future_to_index):
            try:
                i, result = future.result()
                results[i] = result
            except Exception as e:
                idx = future_to_index[future]
                results[idx] = f"Failed to process chunk {idx}: {str(e)}"

    return results


def custom_retrieve_node_safe(state: AgentState, index_ref, metadatas_ref, embedding_model_ref) -> AgentState:
    """
    Enhanced retrieve node that returns KB items with provenance metadata.
    state["retrieved_chunks"] will be a list of dicts:
      { id, text, section, score, keywords, meta_index, source }
    """
    code_sample = state["code"][:1000]
    language = state["code_language"]
    # query tuned for code recommendations
    query = f"recommendations for {language} code best practices regarding performance, efficiency, and environmental impact"

    try:
        # Verify resources
        if index_ref is None:
            logger.error("FAISS index is None")
            state["error"] = "Error during retrieval: FAISS index not available"
            return state

        if metadatas_ref is None:
            logger.error("Metadatas is None")
            state["error"] = "Error during retrieval: Metadata not available"
            return state

        if embedding_model_ref is None:
            logger.warning("Embedding model is None, creating fallback")
            from sentence_transformers import SentenceTransformer
            embedding_model_ref = SentenceTransformer("all-MiniLM-L6-v2")

        # Use mmr_search to get top kb items with ids
        retrieved = mmr_search(query, language, index_ref, metadatas_ref, embedding_model_ref, top_k=7)

        # Filter by score threshold and attach source tag
        filtered = []
        for c in retrieved:
            if c.get("score", 0) > 0.25:
                c["source"] = "knowledge_base"
                filtered.append(c)

        # attach retrieval to state
        state["retrieved_chunks"] = filtered[:5]
        state["metrics"] = analyze_code_metrics(state["code"], language)
        state["dependencies"] = analyze_dependencies(state["code"], language)

        if not filtered:
            logger.warning(f"No relevant KB chunks found for {language}")

    except Exception as e:
        state["error"] = f"Error during retrieval: {str(e)}"
        logger.error(f"Retrieval error: {e}")

    return state


def process_chunks_sequential_safe(chunks: List[Dict], target_language: str, index_ref, metadatas_ref, embedding_model_ref) -> List[str]:
    """Safe sequential processing with verified resources."""
    results = []

    # Verify resources
    if index_ref is None or metadatas_ref is None:
        logger.error("Missing required resources in sequential processing")
        return [f"Error: Missing vector index or metadata resources" for _ in chunks]

    # Create fallback embedding model if needed
    if embedding_model_ref is None:
        logger.warning("Creating fallback embedding model for sequential processing")
        from sentence_transformers import SentenceTransformer
        embedding_model_ref = SentenceTransformer("all-MiniLM-L6-v2")

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk['filename']}")

        try:
            initial_state: AgentState = {
                "code": chunk["content"],
                "code_language": chunk["language"],
                "code_filename": f"{chunk['filename']} (chunk {chunk['chunk_index']+1})",
                "retrieved_chunks": [],
                "answer": "",
                "metrics": None,
                "dependencies": None,
                "error": None,
                "target_language": target_language
            }

            # Use safe retrieve with verified resources
            state_after_retrieve = custom_retrieve_node_safe(
                initial_state,
                index_ref,
                metadatas_ref,
                embedding_model_ref
            )
            final_state = generate_node(state_after_retrieve)
            results.append(final_state["answer"])

        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            results.append(f"Error processing chunk {i}: {str(e)}")

    return results


def process_large_file_upload_with_resources(
    file_path: str,
    target_language: str = "English",
    index_param=None,
    metadatas_param=None,
    embedding_model_param=None
) -> Dict[str, Any]:
    """Process large file upload with explicit resource parameters."""
    from file_processor import file_processor

    # Use provided resources or fall back to globals
    global index, metadatas, embedding_model

    index_to_use = index_param if index_param is not None else index
    metadatas_to_use = metadatas_param if metadatas_param is not None else metadatas
    embedding_to_use = embedding_model_param if embedding_model_param is not None else embedding_model

    # Final check for resources
    if index_to_use is None or metadatas_to_use is None:
        logger.error("No valid resources available for large file processing")
        return {
            "recommendations": "Error: Vector index or metadata not available. Please check server configuration.",
            "language": "unknown",
            "metrics": {},
            "dependencies": {}
        }

    try:
        # Process file into chunks
        chunks = file_processor.process_large_file(file_path)

        if not chunks:
            return {
                "recommendations": "No processable code found in the uploaded file.",
                "language": "unknown",
                "metrics": {},
                "dependencies": {}
            }

        logger.info(f"Processing {len(chunks)} chunks from large file")

        # Limit chunks to prevent timeout
        max_chunks = getattr(config, 'MAX_CHUNKS_PER_FILE', 20)
        if len(chunks) > max_chunks:
            logger.warning(f"Limiting to first {max_chunks} chunks out of {len(chunks)}")
            chunks = chunks[:max_chunks]

        # Create safe embedding model
        safe_embedding_model = embedding_to_use
        if safe_embedding_model is None:
            logger.warning("Creating fallback embedding model for large file processing")
            from sentence_transformers import SentenceTransformer
            safe_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Force sequential processing to avoid threading issues
        logger.info("Using sequential chunk processing for stability")
        results = process_chunks_sequential_safe(
            chunks, target_language, index_to_use, metadatas_to_use, safe_embedding_model
        )

        # Combine results
        combined_recommendations = combine_chunk_results(results, chunks)

        # Calculate overall metrics
        overall_metrics = calculate_combined_metrics(chunks)

        return {
            "recommendations": combined_recommendations,
            "language": chunks[0]["language"] if chunks else "unknown",
            "metrics": overall_metrics,
            "dependencies": {"dependencies": [], "count": 0},
            "file_info": {
                "total_chunks": len(chunks),
                "total_size": sum(c["size"] for c in chunks),
                "languages": list(set(c["language"] for c in chunks))
            }
        }

    except Exception as e:
        logger.error(f"Error processing large file: {e}")
        return {
            "recommendations": f"Error processing large file: {str(e)}",
            "language": "unknown",
            "metrics": {},
            "dependencies": {}
        }


def combine_chunk_results(results: List[str], chunks: List[Dict]) -> str:
    """Combine individual chunk results into a cohesive report."""
    combined = "🔍 Large Codebase Analysis Report\n\n"
    combined += f"📊 **Summary**: Analyzed {len(chunks)} code chunks across multiple files\n\n"

    # Group by language
    language_groups = {}
    for chunk in chunks:
        lang = chunk["language"]
        if lang not in language_groups:
            language_groups[lang] = []
        language_groups[lang].append(chunk)

    combined += f"📁 **Languages Found**: {', '.join(language_groups.keys())}\n\n"

    # Combine results with file context
    for i, (result, chunk) in enumerate(zip(results, chunks)):
        if result and result.strip():
            combined += f"## 📄 {chunk['filename']} (Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']})\n\n"
            combined += result + "\n\n"
            combined += "---\n\n"

    return combined


def calculate_combined_metrics(chunks: List[Dict]) -> Dict[str, Any]:
    """Calculate overall metrics from all chunks."""
    total_lines = sum(chunk["content"].count('\n') + 1 for chunk in chunks)
    total_chars = sum(chunk["size"] for chunk in chunks)

    return {
        "total_files": len(set(chunk["filename"] for chunk in chunks)),
        "total_chunks": len(chunks),
        "total_lines": total_lines,
        "total_characters": total_chars,
        "average_chunk_size": total_chars // len(chunks) if chunks else 0,
        "languages": list(set(chunk["language"] for chunk in chunks))
    }


def build_agent():
    """Build the agent workflow as a state graph."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", custom_retrieve_node_safe)
    graph.add_node("generate", generate_node)

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    graph.set_entry_point("retrieve")

    return graph.compile()


def process_code_and_generate(file=None, code_text=None, target_language="English"):
    """
    Process code input (file or text), generate recommendations, and return results.
    This is the single main entrypoint to call from your server or frontend.
    """
    import tempfile
    import os
    from reportlab.pdfgen import canvas

    # Prepare file path
    if code_text:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt")
        os.close(tmp_fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(code_text)
        file_path = tmp_path
        filename = "code_snippet.txt"
    elif file:
        file_path = file.name if hasattr(file, "name") else file
        filename = os.path.basename(file_path)
    else:
        return {
            "recommendations": "Please upload a file or paste code.",
            "metrics": {},
            "pdf_path": None
        }

    # Process file using robust function
    result = process_large_file_upload_with_resources(
        file_path=file_path,
        target_language=target_language,
        index_param=index,
        metadatas_param=metadatas,
        embedding_model_param=embedding_model
    )

    # Cache results
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    cache_key = get_cache_key(txt)
    RESPONSE_CACHE[cache_key] = (
        result["recommendations"],
        {filename: {"metrics": result["metrics"], "dependencies": result["dependencies"]}}
    )

    # Enforce cache size limit
    if len(RESPONSE_CACHE) > config.MAX_CACHE_SIZE:
        oldest_keys = list(RESPONSE_CACHE.keys())[:(len(RESPONSE_CACHE) - config.MAX_CACHE_SIZE)]
        for key in oldest_keys:
            del RESPONSE_CACHE[key]

    # Generate PDF
    tmp_fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)
    c = canvas.Canvas(pdf_path)
    y = 800
    for line in result["recommendations"].splitlines():
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = 800
    c.save()

    # Clean up temp file if created
    if code_text:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return {
        "recommendations": result["recommendations"],
        "metrics": {filename: {"metrics": result["metrics"], "dependencies": result["dependencies"]}},
        "pdf_path": pdf_path
    }


def main():
    """Main entry point with updated configuration."""
    parser = argparse.ArgumentParser(description="Code Recommendation Checker")
    parser.add_argument('--index', default=config.VECTOR_INDEX_PATH, help="Path to FAISS index file")
    parser.add_argument('--metadata', default=config.VECTOR_METADATA_PATH, help="Path to metadata JSON file")
    parser.add_argument('--embedding-model', default=config.EMBEDDING_MODEL, help="Embedding model to use")
    args = parser.parse_args()

    # Setup LLM client
    logger.info("🔧 Setting up LLM client...")
    try:
        # Test LLM connection
        test_response = call_llm("Hello, this is a test. Please respond with 'OK'.")
        logger.info(f"✅ LLM test response: {test_response}")
    except Exception as e:
        logger.error(f"❌ Failed to setup LLM client: {e}")
        return 1

    # Load resources
    global index, metadatas, embedding_model, agent
    index, metadatas, embedding_model = load_resources(args.index, args.metadata, args.embedding_model)

    # Build agent
    global agent
    agent = build_agent()
    logger.info("✅ Agent built and ready")


if __name__ == "__main__":
    exit(main())
