"""
Recommendation Checker Chatbot - RAG Generation Interface

Updated to use environment variables and company LLM proxy support.
"""
import sqlite3
from db_utils import store_feedback, get_past_feedback_for_file, init_feedback_db
from utils import prepare_state_from_input
from config_manager import config
from llm_client import llm_client, call_llm, call_llm_stream

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
RESPONSE_CACHE = {}
index = None
metadatas = None
embedding_model = None
agent = None

# Language detection patterns (keeping existing patterns)
LANGUAGE_PATTERNS = {
    "python": [r'\.py$', r'import\s+', r'from\s+\w+\s+import', r'def\s+\w+\s*\(', r'class\s+\w+\s*:'],
    "javascript": [r'\.js$', r'const\s+', r'let\s+', r'var\s+', r'function\s+', r'import\s+.*from', r'export\s+'],
    "java": [r'\.java$', r'public\s+class', r'private\s+', r'protected\s+', r'import\s+java\.', r'package\s+'],
    "typescript": [r'\.ts$', r'interface\s+', r'type\s+', r'namespace\s+', r'enum\s+'],
    "go": [r'\.go$', r'package\s+main', r'func\s+', r'import\s+\(', r'type\s+\w+\s+struct'],
    "rust": [r'\.rs$', r'fn\s+main', r'let\s+mut', r'use\s+std', r'impl\s+', r'pub\s+fn'],
    "c#": [r'\.cs$', r'namespace\s+', r'using\s+System', r'public\s+class', r'private\s+void'],
    "php": [r'\.php$', r'\<\?php', r'function\s+', r'namespace\s+', r'use\s+'],
    "ruby": [r'\.rb$', r'require\s+', r'def\s+', r'class\s+', r'module\s+'],
    "swift": [r'\.swift$', r'import\s+Foundation', r'func\s+', r'class\s+', r'struct\s+'],
    "kotlin": [r'\.kt$', r'fun\s+main', r'val\s+', r'var\s+', r'class\s+', r'package\s+'],
    "c++": [r'\.cpp$', r'#include', r'using\s+namespace', r'int\s+main', r'class\s+\w+\s*\{'],
    "c": [r'\.c$', r'#include', r'int\s+main', r'void\s+\w+\s*\(', r'struct\s+\w+\s*\{'],
    "html": [r'\.html$', r'\<\!DOCTYPE', r'\<html', r'\<head', r'\<body'],
    "css": [r'\.css$', r'\w+\s*\{', r'@media', r'@import', r'@keyframes'],
    "sql": [r'\.sql$', r'SELECT', r'INSERT', r'UPDATE', r'DELETE', r'CREATE\s+TABLE']
}

# Comment patterns for metrics (keeping existing patterns)
COMMENT_PATTERNS = {
    "python": r'(#[^\n]*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
    "javascript": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "typescript": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "java": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "go": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "c#": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "php": r'(//[^\n]*|/\*[\s\S]*?\*/|#[^\n]*)',
    "ruby": r'(#[^\n]*|=begin[\s\S]*?=end)',
    "c++": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "c": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "swift": r'(//[^\n]*|/\*[\s\S]*?\*/)',
    "kotlin": r'(//[^\n]*|/\*[\s\S]*?\*/)'
}

# Complexity patterns for metrics (keeping existing patterns)
COMPLEXITY_PATTERNS = {
    "python": r'(if|for|while|def|class|with|try|except)',
    "javascript": r'(if|for|while|function|class|try|catch|switch)',
    "typescript": r'(if|for|while|function|class|try|catch|switch|interface)',
    "java": r'(if|for|while|switch|case|try|catch|class|interface)',
    "go": r'(if|for|switch|func|struct|interface)',
    "c#": r'(if|for|while|switch|case|try|catch|class|interface)',
    "php": r'(if|for|while|switch|case|try|catch|class|interface|function)',
    "ruby": r'(if|for|while|case|begin|rescue|class|module|def)',
    "c++": r'(if|for|while|switch|case|try|catch|class|struct)',
    "c": r'(if|for|while|switch|case)',
    "swift": r'(if|for|while|switch|case|try|catch|class|struct|enum)',
    "kotlin": r'(if|for|while|when|try|catch|class|interface|fun)'
}

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
        metadatas = data["chunks"] if isinstance(data, dict) and "chunks" in data else data
        
        # Always load fallback model first as safety net
        logger.info(f"🔄 Loading fallback model first: {DEFAULT_FALLBACK_MODEL}")
        fallback_model = SentenceTransformer(DEFAULT_FALLBACK_MODEL)
        test_embedding = fallback_model.encode("test")
        logger.info(f"✅ Fallback embedding model loaded successfully: {DEFAULT_FALLBACK_MODEL}")
        
        # Try to load the specified embedding model
        primary_model = None
        
        if model_name and model_name.startswith("vertex_ai/"):
            logger.info(f"🔄 Attempting to load Vertex AI embedding model: {model_name}")
            try:
                # Try custom proxy embedding model
                from embedding_client import ProxyEmbeddingModel
                primary_model = ProxyEmbeddingModel(model_name)
                
                # Test the model with a simple encode
                test_embedding = primary_model.encode("test")
                if test_embedding is not None and len(test_embedding) > 0:
                    logger.info(f"✅ Vertex AI embedding model loaded successfully: {model_name}")
                else:
                    raise Exception("Test encoding returned empty result")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Vertex AI embedding model: {e}")
                logger.info(f"🔄 Falling back to default model: {DEFAULT_FALLBACK_MODEL}")
                primary_model = None
        
        # Use primary model if available, otherwise use fallback
        embedding_model = primary_model if primary_model is not None else fallback_model
        
        # FORCE FALLBACK: If we're in a problematic state, always use fallback
        # This addresses the parallel processing issue where Vertex AI models become None
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

    scores = {lang: sum(
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
    """Perform Maximum Marginal Relevance search with section info."""
    language_query = f"recommendations for {code_language} code best practices"
    combined_query = f"{query} {language_query}"
    query_vec = embedding_model.encode(combined_query).astype("float32")
    query_vec = query_vec.reshape(1, -1)

    distances, indices = index.search(query_vec, top_k * 3)
    distances = distances.flatten()
    indices = indices.flatten()

    candidate_ids = [idx for idx in indices if 0 <= idx < len(metadatas)]
    if not candidate_ids:
        return []

    candidate_texts = [metadatas[idx]["text"] for idx in candidate_ids]
    candidate_embeddings = np.vstack([
        embedding_model.encode(text).astype("float32") for text in candidate_texts
    ])

    query_vec = query_vec.flatten()

    selected_indices = []
    remaining_indices = list(range(len(candidate_ids)))

    for _ in range(min(top_k, len(candidate_ids))):
        if not remaining_indices:
            break

        mmr_scores = []
        for i in remaining_indices:
            relevance = np.dot(candidate_embeddings[i], query_vec) / (
                np.linalg.norm(candidate_embeddings[i]) * np.linalg.norm(query_vec)
            )

            diversity = 1.0
            if selected_indices:
                similarities = [
                    np.dot(candidate_embeddings[i], candidate_embeddings[j]) /
                    (np.linalg.norm(candidate_embeddings[i]) * np.linalg.norm(candidate_embeddings[j]))
                    for j in selected_indices
                ]
                diversity = 1 - max(similarities) if similarities else 1.0

            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            mmr_scores.append((i, mmr_score))

        next_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)

    results = []
    for idx in selected_indices:
        original_idx = candidate_ids[idx]
        chunk = metadatas[original_idx]

        distance_idx = np.where(indices == original_idx)[0]
        distance = distances[distance_idx[0]] if len(distance_idx) > 0 else 1.0

        results.append({
            "text": chunk["text"],
            "section": chunk.get("section", "Unknown"),
            "score": 1.0 / (1.0 + float(distance)),
            "keywords": chunk.get("keywords", [])
        })

    return results

def process_large_code(code: str, max_chunk_size: int = config.MAX_CHUNK_SIZE) -> List[str]:
    """
    Split large code files into manageable chunks for processing,
    attempting to split at logical boundaries where possible.
    """
    if len(code) <= max_chunk_size:
        return [code]

    lines = code.splitlines(True)  # keep line endings
    chunks = []
    current_chunk = ""

    for line in lines:
        # Start a new chunk if size limit exceeded
        if len(current_chunk) + len(line) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

        current_chunk += line

        # Prefer splitting at blank lines if chunk is large enough
        if re.match(r'^\s*$', line) and len(current_chunk) > max_chunk_size * 0.5:
            chunks.append(current_chunk)
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

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

def analyze_feedback(feedback_text):
    """Analyze user feedback using LLM."""
    prompt = f"""
You are an expert QA reviewer. A user gave the following feedback after receiving AI-generated code recommendations:

--- FEEDBACK START ---
{feedback_text}
--- FEEDBACK END ---

1. Is this feedback helpful and relevant? Answer only YES or NO.
2. What is the sentiment? Choose one: positive / neutral / negative.
3. If the feedback is not useful, say "Reject". If useful, say "Accept".

Format your answer as:
Relevance: <YES/NO>
Sentiment: <positive/neutral/negative>
Decision: <Accept/Reject>
"""
    
    try:
        result = call_llm(prompt)
        print("[🔍 Feedback Analysis]:", result)

        lines = result.strip().splitlines()
        relevance = sentiment = decision = None
        for line in lines:
            if line.startswith("Relevance:"):
                relevance = line.split(":", 1)[1].strip()
            elif line.startswith("Sentiment:"):
                sentiment = line.split(":", 1)[1].strip()
            elif line.startswith("Decision:"):
                decision = line.split(":", 1)[1].strip()

        return {
            "relevance": relevance,
            "sentiment": sentiment,
            "decision": decision
        }
    except Exception as e:
        logger.error(f"Error analyzing feedback: {e}")
        return {
            "relevance": "NO",
            "sentiment": "neutral",
            "decision": "Reject"
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

def retrieve_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant chunks from the knowledge base using MMR search.
    
    Args:
        state (AgentState): Current agent state containing code and language info.
    
    Returns:
        AgentState: Updated state with retrieved recommendation chunks, metrics, and dependencies.
    """
    logger.debug(f"mmr_search function defined at: {mmr_search.__code__.co_filename}:{mmr_search.__code__.co_firstlineno}")
    
    # Extract snippet of code (currently unused here but could be for future)
    code_sample = state["code"][:1000]
    
    language = state["code_language"]
    # Construct the query focusing on performance, efficiency, and environmental impact
    query = f"recommendations for {language} code best practices regarding performance, efficiency, and environmental impact"
    logger.debug(f"Retrieval query: {query}")
    
    try:
        # Perform MMR search on index to get relevant chunks
        retrieved = mmr_search(query, language, index, metadatas, embedding_model, top_k=7)
        logger.debug(f"Retrieved {len(retrieved)} chunks from mmr_search")
        
        # Filter retrieved chunks by score threshold and limit results
        filtered = [c for c in retrieved if c.get("score", 0) > 0.3][:5]
        logger.debug(f"Filtered to {len(filtered)} chunks with score > 0.3")
        
        # Save filtered chunks into state
        state["retrieved_chunks"] = filtered
        
        # Analyze code metrics and add to state
        state["metrics"] = analyze_code_metrics(state["code"], language)
        
        # Analyze code dependencies and add to state
        state["dependencies"] = analyze_dependencies(state["code"], language)
        
    except Exception as e:
        # On failure, log and store error info in state
        state["error"] = f"Error during retrieval: {str(e)}"
        logger.error(f"Retrieval error: {e}")
    
    return state

def generate_node(state: AgentState) -> AgentState:
    """Generate recommendations, including section info in the context."""
    if state.get("error"):
        state["answer"] = f"An error occurred: {state['error']}"
        return state

    if not state["retrieved_chunks"]:
        state["answer"] = "No relevant recommendations found for your code."
        return state

    # Format context with section info
    context = "\n\n".join(
        f"[{i+1}] [{c['section']}] {c['text']}" 
        for i, c in enumerate(state["retrieved_chunks"])
    )

    language, filename = state["code_language"], state["code_filename"]
    metrics_info = ""
    if state.get("metrics"):
        metrics_info = f"""
Code metrics:
- Lines of code: {state['metrics']['line_count']}
- Comment ratio: {state['metrics']['comment_ratio']}
- Complexity estimate: {state['metrics']['complexity_estimate']}
"""
        if "cyclomatic_complexity" in state["metrics"]:
            metrics_info += f"- Cyclomatic complexity: {state['metrics']['cyclomatic_complexity']}\n"

    dependencies_info = ""
    if state.get("dependencies") and state["dependencies"]["count"] > 0:
        dependencies_info = f"""
Dependencies ({state['dependencies']['count']}):
{', '.join(state['dependencies']['dependencies'][:10])}
"""
        if len(state['dependencies']['dependencies']) > 10:
            dependencies_info += f"... and {len(state['dependencies']['dependencies']) - 10} more"

    prompt = (
        f"## 🧠 ROLE: You are an expert code reviewer specializing in **{language}**.\n"
        f"You are tasked with a deep, professional review of the provided {language} codebase.\n\n"
        f"### 📌 Your Evaluation Must Cover:\n"
        f"1. **Performance** – Evaluate runtime efficiency and computational cost.\n"
        f"2. **Efficiency** – Look for logic simplification, memory/resource optimization.\n"
        f"3. **Environmental Impact (Green IT)** – Assess energy consumption, unnecessary loops, heavy I/O, and library bloat.\n"
        f"4. **Software Design Principles** – Check adherence to SOLID, DRY, separation of concerns, etc.\n"
        f"5. **Design Patterns** – Identify applied patterns and opportunities for improvements.\n\n"
        f"Do NOT recommend switching to another language — **unless it is strictly necessary** to achieve substantial gains in performance, efficiency, or environmental impact. Justify clearly.\n"
        f"Keep all recommendations within the context of {language}.\n"
        f"---\n"
        f"## 📘 Best Practice Context:\n{context}\n"
        f"---\n"
        f"## 📊 Code Metrics:\n{metrics_info}\n"
        f"## 📦 Dependencies:\n{dependencies_info}\n"
        f"## 📁 Folder Structure:\n{state.get('folder_structure', 'N/A')}\n\n"
        f"## 💻 Provided Code Snippet (Language: {language}):\n```{language}\n{state['code']}\n```\n"
        f"---\n"
        f"##  TASK INSTRUCTIONS:\n"
        f"Please follow the steps below and generate an actionable expert report:\n"
        f"1. Check each best practice recommendation **against the code and the full project structure**.\n"
        f"2. For each **violation**:\n"
        f"   - Explain clearly why it's a violation.\n"
        f"   - Propose a **specific and actionable fix** (include refactor suggestions where appropriate).\n"
        f"3. ❗️**Do not list or mention respected recommendations**, even if all best practices are followed.\n"
        f"4. Analyze the project's **architecture on two levels**:\n"
        f"   - **Within individual files**: Look at cohesion, structure, responsibility allocation.\n"
        f"   - **Across the full project**: Assess folder layout, modularity, coupling, and interdependencies.\n"
        f"   - Are SOLID principles applied properly (esp. SRP, OCP, DIP)?\n"
        f"   - Recommend any necessary reorganizations (folders, class/file separation, naming, layering).\n"
        f"5. Evaluate the code's **Green IT impact**:\n"
        f"   - Identify unnecessary resource consumption (e.g., heavy loops, redundant I/O, memory waste).\n"
        f"   - Recommend eco-friendly optimizations.\n\n"
        f"---\n"
        f"## 📄 OUTPUT FORMAT:\n"
        f"Start with '🔍 Code Review' as the only title.\n\n"
        f"First, evaluate individual files:\n"
        f"For each file with issues, use this format:\n"
        f"📄 **filename.py**\n\n"
        f"For each issue category in a file, use this format (only if issues exist):\n"
        f"🔶 **Performance**\n\n"
        f"⚠️ **Issue**: [Clear description of the problem]\n\n"
        f"✅ **Recommendation**: [Specific, actionable solution with code examples when appropriate]\n\n"
        f"Then, evaluate file architecture:\n"
        f"🏗️ **File Architecture**\n\n"
        f"For each file with architectural issues:\n"
        f"📄 **filename.py**\n\n"
        f"⚠️ **Issue**: [Clear description of architectural problems within the file]\n\n"
        f"✅ **Recommendation**: [Specific suggestions for improving file structure, responsibility allocation, etc.]\n\n"
        f"Finally, evaluate project-level architecture:\n"
        f"🏢 **Project Architecture**\n\n"
        f"⚠️ **Issue**: [Clear description of project-level architectural problems]\n\n"
        f"✅ **Recommendation**: [Specific suggestions for improving folder structure, modularity, file organization, etc.]\n\n"
        f"Important rules:\n"
        f"1. Only mention categories where you found issues - skip categories with no violations.\n"
        f"2. If a whole section (Performance, Efficiency, Green IT, etc.) has no violations, skip that section entirely.\n"
        f"3. If file architecture is well-designed, skip the File Architecture section.\n"
        f"4. If project architecture is well-designed, skip the Project Architecture section.\n"
        f"5. Use emojis and bold text for visual hierarchy instead of markdown headers.\n"
        f"6. Make sure each section is clearly separated with blank lines for readability.\n"
        f"7. If no issues are found in the entire codebase, simply write '🔍 Code Review\\n\\nNo issues found in the codebase.'\n"
        f"8. Format code examples using proper markdown code blocks.\n"
    )

    # Append past feedback for the file, if any
    past_feedbacks = get_past_feedback_for_file(filename)
    if past_feedbacks:
        prompt += "\n\n## 📣 Past User Feedback to Consider:\n"
        for fb in past_feedbacks:
            prompt += f"- {fb}\n"
        prompt += "\nPlease carefully consider the past user feedback above when performing your review and recommendations.\n"

    try:
        # Initialize answer
        state["answer"] = ""

        # Stream completions from the LLM and accumulate
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
            # Extract streamed text content if available
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = getattr(chunk.choices[0], 'delta', None)
                if delta and getattr(delta, 'content', None):
                    content = delta.content
                    if isinstance(content, str) and content.strip() != "":
                        state["answer"] += content

        # Translate answer if target language is specified and not English
        if state.get("target_language") and state["target_language"].lower() != "english":
            state["answer"] = translate_recommendations(state["answer"], state["target_language"])

    except Exception as e:
        # On error, set error message and log
        state["error"] = f"Error during generation: {str(e)}"
        state["answer"] = f"Failed to generate recommendations: {str(e)}"
        logger.error(f"Generation error: {e}")

    return state





def parallel_process_chunks(code_chunks: List[str], language: str, filename: str) -> str:
    """Process multiple code chunks in parallel using threads."""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(code_chunks), 5)) as executor:
        future_to_chunk = {
            executor.submit(process_single_chunk, chunk, language, f"{filename} (part {i+1}/{len(code_chunks)})"): i 
            for i, chunk in enumerate(code_chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result()
                results.append((chunk_idx, result))
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {e}")
                results.append((chunk_idx, f"Error processing this section: {str(e)}"))
    
    sorted_results = [res for _, res in sorted(results, key=lambda x: x[0])]
    combined = "\n\n## ===== Next Section =====\n\n".join(sorted_results)
    
    return combined

def process_single_chunk(code_chunk: str, language: str, chunk_name: str) -> str:
    """Process a single code chunk by invoking the agent workflow."""
    initial_state = {
        "code": code_chunk,
        "code_language": language,
        "code_filename": chunk_name,
        "retrieved_chunks": [],
        "answer": "",
        "metrics": None,
        "dependencies": None,
        "error": None
    }
    
    final_state = agent.invoke(initial_state)
    return final_state["answer"]

# Add this function to rag_generate.py


def process_chunks_sequential(chunks: List[Dict], target_language: str, index_ref, metadatas_ref, embedding_model_ref, agent_ref) -> List[str]:
    """Process chunks one by one with explicit resource references."""
    results = []
    
    # Ensure we have all required resources
    if index_ref is None or metadatas_ref is None:
        logger.error("Missing required resources (index or metadatas)")
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
            
            # Use explicit resource passing with verification
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
        chunk, index = chunk_data
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
            return index, final_state["answer"]
            
        except Exception as e:
            logger.error(f"Error processing chunk {index}: {e}")
            return index, f"Error processing chunk {index}: {str(e)}"
    
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
                index, result = future.result()
                results[index] = result
            except Exception as e:
                index = future_to_index[future]
                results[index] = f"Failed to process chunk {index}: {str(e)}"
    
    return results

def custom_retrieve_node_safe(state: AgentState, index_ref, metadatas_ref, embedding_model_ref) -> AgentState:
    """Enhanced custom retrieve node with comprehensive safety checks."""
    code_sample = state["code"][:1000]
    language = state["code_language"]
    query = f"recommendations for {language} code best practices regarding performance, efficiency, and environmental impact"
    
    try:
        # Verify all resources are available
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
        
        # Use mmr_search with verified resources
        retrieved = mmr_search(query, language, index_ref, metadatas_ref, embedding_model_ref, top_k=7)
        filtered = [c for c in retrieved if c.get("score", 0) > 0.3][:5]
        
        state["retrieved_chunks"] = filtered
        state["metrics"] = analyze_code_metrics(state["code"], language)
        state["dependencies"] = analyze_dependencies(state["code"], language)
        
        if not filtered:
            logger.warning(f"No relevant chunks found for {language}")
        
    except Exception as e:
        state["error"] = f"Error during retrieval: {str(e)}"
        logger.error(f"Retrieval error: {e}")
    
    return state

def process_large_file_upload_fixed(file_path: str, target_language: str = "English") -> Dict[str, Any]:
    """Enhanced process large file upload with better resource management."""
    from file_processor import file_processor
    
    # Ensure global resources are loaded
    global index, metadatas, embedding_model
    
    if index is None or metadatas is None:
        logger.error("Global resources not loaded properly")
        return {
            "recommendations": "Error: Vector index or metadata not loaded. Please restart the service.",
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
        max_chunks = config.MAX_CHUNKS_PER_FILE
        if len(chunks) > max_chunks:
            logger.warning(f"Limiting to first {max_chunks} chunks out of {len(chunks)}")
            chunks = chunks[:max_chunks]
        
        # Create fallback embedding model if needed
        safe_embedding_model = embedding_model
        if safe_embedding_model is None:
            logger.warning("Creating fallback embedding model for large file processing")
            from sentence_transformers import SentenceTransformer
            safe_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Process chunks with explicit resource passing
        if config.PARALLEL_CHUNK_PROCESSING and len(chunks) > 1:
            logger.info("Using parallel chunk processing")
            results = process_chunks_parallel(
                chunks, target_language, index, metadatas, safe_embedding_model, None
            )
        else:
            logger.info("Using sequential chunk processing")
            results = process_chunks_sequential(
                chunks, target_language, index, metadatas, safe_embedding_model, None
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

def parallel_process_chunks(code_chunks: List[str], language: str, filename: str) -> str:
    """Process multiple code chunks in parallel using threads."""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(code_chunks), 5)) as executor:
        future_to_chunk = {
            executor.submit(process_single_chunk, chunk, language, f"{filename} (part {i+1}/{len(code_chunks)})"): i 
            for i, chunk in enumerate(code_chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result()
                results.append((chunk_idx, result))
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {e}")
                results.append((chunk_idx, f"Error processing this section: {str(e)}"))
    
    sorted_results = [res for _, res in sorted(results, key=lambda x: x[0])]
    combined = "\n\n## ===== Next Section =====\n\n".join(sorted_results)
    
    return combined

def process_single_chunk(code_chunk: str, language: str, chunk_name: str) -> str:
    """Process a single code chunk by invoking the agent workflow."""
    initial_state = {
        "code": code_chunk,
        "code_language": language,
        "code_filename": chunk_name,
        "retrieved_chunks": [],
        "answer": "",
        "metrics": None,
        "dependencies": None,
        "error": None
    }
    
    final_state = agent.invoke(initial_state)
    return final_state["answer"]


    

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
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    graph.set_entry_point("retrieve")
    
    return graph.compile()

def gradio_wrapper(user_file, code_text=None, target_language="English") -> Tuple[str, Dict[str, Any], str]:
    """Process code input and generate recommendations with PDF report."""
    
    if user_file:
        state = prepare_state_from_input(
            user_file.name,
            "unknown",
            target_language
        )
    elif code_text:
        detected_language = detect_language(code_text, "code_snippet.txt")
        state = {
            "files": [("code_snippet.txt", code_text)],
            "code_language": detected_language,
            "target_language": target_language
        }
    else:
        return "Please upload a file or paste code", None, None

    results = []
    combined_metrics = {"Code Metrics": {}, "Dependencies": {}}

    for filename, code in state["files"]:
        if not code.strip():
            continue

        cache_key = get_cache_key(code)
        if cache_key in RESPONSE_CACHE:
            cached_result = RESPONSE_CACHE[cache_key]
            if isinstance(cached_result, tuple):
                results.append(f"### 📄 **{filename}**\n\n{cached_result[0]}")
                continue

        code_chunks = process_large_code(code) if len(code) > config.MAX_CHUNK_SIZE else [code]

        if len(code_chunks) > 1:
            logger.info(f"Processing {filename} in {len(code_chunks)} chunks")
            response = parallel_process_chunks(code_chunks, state["code_language"], filename)
            metrics = analyze_code_metrics(code, state["code_language"])
            dependencies = analyze_dependencies(code, state["code_language"])
        else:
            initial_state: AgentState = {
                "code": code,
                "code_language": state["code_language"],
                "code_filename": filename,
                "retrieved_chunks": [],
                "answer": "",
                "metrics": None,
                "dependencies": None,
                "error": None,
                "target_language": state["target_language"]
            }

            final_state = agent.invoke(initial_state)
            response = final_state["answer"]
            metrics = final_state.get("metrics", {})
            dependencies = final_state.get("dependencies", {})

        results.append(f"### 📄 **{filename}**\n\n{response}")

        combined_metrics["Code Metrics"][filename] = metrics
        combined_metrics["Dependencies"][filename] = dependencies.get("dependencies", [])

        RESPONSE_CACHE[cache_key] = (response, {filename: {"metrics": metrics, "dependencies": dependencies}})

    # Enforce cache size limit
    if len(RESPONSE_CACHE) > config.MAX_CACHE_SIZE:
        oldest_keys = list(RESPONSE_CACHE.keys())[:(len(RESPONSE_CACHE) - config.MAX_CACHE_SIZE)]
        for key in oldest_keys:
            del RESPONSE_CACHE[key]

    final_response = "\n\n---\n\n".join(results)

    # Generate PDF
    tmp_fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)

    c = canvas.Canvas(pdf_path)
    lines = final_response.splitlines()
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = 800
    c.save()

    return final_response, combined_metrics, pdf_path

# Initialize feedback database
init_feedback_db()
####
def submit_user_feedback(file, code_text, feedback, rating):
            if file is not None:
                try:
                    # Case 1: file is a file path string (Gradio sometimes passes paths now)
                    if isinstance(file, str) and os.path.exists(file):
                        filename = os.path.basename(file)
                        with open(file, "rb") as f:
                            raw_bytes = f.read()
                    
                    # Case 2: file is an UploadedFile object with .name and .read()
                    elif hasattr(file, "read"):
                        filename = getattr(file, "name", "uploaded_file")
                        file.seek(0)
                        raw_bytes = file.read()
                    
                    else:
                        return "⚠️ Invalid file object type", "", None

                    # Decode content
                    try:
                        code = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        code = raw_bytes.decode("latin-1")

                except Exception as e:
                    return f"⚠️ Error reading uploaded file: {str(e)}", "", None

            elif code_text:
                filename = "manual_input"
                code = code_text
            else:
                return "⚠️ No code to provide feedback on", "", None

            # Create unique key
            file_hash = get_cache_key(code)

            # Analyze feedback
            analysis = analyze_feedback(feedback)

            if analysis["decision"] == "Accept":
                store_feedback(
                    file_hash=file_hash,
                    filename=filename,
                    feedback=feedback,
                    rating=rating,
                    relevance=analysis["relevance"],
                    sentiment=analysis["sentiment"],
                    decision=analysis["decision"],
                )
                return "✅ Thanks! Your feedback was accepted and will help improve future results.", "", None
            else:
                return "⚠️ Thanks! Your feedback was reviewed but marked not useful by our reviewer.", "", None
###
last_result = {"recommendations": "", "metrics": {}, "pdf_path": ""}

def translate_recommendations_plain(recommendations, target_language):
    if not recommendations:
        return "No recommendations to translate."
    return translate_recommendations(recommendations, target_language)



####

def process_code_and_generate(file=None, code_text=None, lang="English"):
    """
    Process code input (file or text), generate recommendations, and return results.
    This is the main function to call from your new frontend.
    """
    # Call your core wrapper
    recommendations, metrics, pdf_path = gradio_wrapper(file, code_text, lang)
    
    # Return the results for further processing
    return {
        "recommendations": recommendations,
        "metrics": metrics,
        "pdf_path": pdf_path
    }

#####


def main():
    """Main entry point with updated configuration."""
    parser = argparse.ArgumentParser(description="Code Recommendation Checker")
    # parser.add_argument('--port', type=int, default=config.SERVER_PORT, help="Port to run the Gradio server on")
    # parser.add_argument('--share', action='store_true', default=config.GRADIO_SHARE, help="Create a shareable link")
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