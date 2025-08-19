"""
Recommendation Checker Chatbot - RAG Generation Interface

This script implements a Retrieval-Augmented Generation (RAG) system with a Gradio web interface.
Users can upload a code file and the system checks if the code complies with recommendations
extracted from a knowledge base document.
"""
import sqlite3
from db_utils import store_feedback, get_past_feedback_for_file, init_feedback_db
from encryption_utils import decrypt_history, save_encrypted_history
from utils import prepare_state_from_input
import os  # Interact with the OS for file and environment handling
import json  # Read/write JSON files and strings
import sys  # Handle CLI args, exits, and Python runtime interactions
import argparse
import time
import hashlib  # For hashing code content for caching/identification
import re  # Regex utilities for pattern matching code
import logging  # Logging events and debugging info
import concurrent.futures  # For running parallel tasks (thread/process pools)
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from datetime import datetime
import faiss  # For fast vector similarity search
import numpy as np  # Numerical operations, arrays
from sentence_transformers import SentenceTransformer  # Embedding text to vectors
import litellm  # Lightweight LLM client
from litellm import completion  # For making LLM completions
from langgraph.graph import StateGraph, END  # State machine for chatbot flow
import gradio as gr  # Web UI framework for ML demos
from tenacity import retry, stop_after_attempt, wait_exponential  # Retry decorator for robustness
import tempfile  # Temporary file management
from fpdf import FPDF  # PDF generation utilities
from reportlab.pdfgen import canvas  # Advanced PDF generation tools


# === Configuration constants ===
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model for embeddings
DEFAULT_LLM = "vertex_ai/gemini-2.5-flash-lite"  # Default large language model for completions
DEFAULT_INDEX_PATH = "knowledge_base_flat.index"  # Path to FAISS index
DEFAULT_METADATA_PATH = "knowledge_base_metadata.json"  # Metadata JSON file path
DEFAULT_CREDENTIALS_PATH = "vertex_service_key.json"  # Google Vertex AI credentials file path

DEFAULT_VERTEX_PROJECT = "sbx-31371-fxgrkrq2vv3eba42pk53"
DEFAULT_VERTEX_LOCATION = "us-central1"

RESPONSE_CACHE = {}  # Cache for storing previous responses to avoid recomputation
MAX_CACHE_SIZE = 50  # Maximum cache size limit i made 50 for test then i will boost the number between 500-50000
MAX_CHUNK_SIZE = 50000  # Max size for code chunks to process at once

##
import litellm

def call_llm(prompt):
    return litellm.completion(
        model="vertex_ai/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )["choices"][0]["message"]["content"]
##
# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Regex patterns for detecting programming languages based on filename or code content
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

# Language-specific regex patterns to detect comments for metrics calculations
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

# Language-specific regex patterns for detecting control flow statements for complexity estimation
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

# TypedDict defining the state structure used by the agent/chatbot
class AgentState(TypedDict):
    code: str  # Source code text
    code_language: str  # Detected language of the code
    code_filename: str  # Filename of the uploaded code
    retrieved_chunks: List[Dict[str, Any]]  # Retrieved KB chunks related to the code
    answer: str  # Generated chatbot answer
    metrics: Optional[Dict[str, Any]]  # Metrics on code quality (optional)
    dependencies: Optional[Dict[str, Any]]  # Extracted dependencies from code (optional)
    error: Optional[str]  # Error messages if any
    target_language: Optional[str]  # Language for recommendations translation (optional)


def setup_vertex_ai(credentials_path: str, project: str, location: str) -> None:
    """
    Configure Vertex AI client with credentials and project/location details.
    Exits if credentials file not found.
    """
    if not os.path.exists(credentials_path):
        sys.exit(f"❌ Vertex AI credentials file not found: {credentials_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    litellm.vertex_project = project
    litellm.vertex_location = location
    logger.info(f"✅ Vertex AI configured: project={project}, location={location}")


def check_model_access(model_name: str):
    """
    Verify access to the specified LLM by sending a simple 'ping' prompt.
    Exits on failure.
    """
    try:
        _ = completion(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8000,
            temperature=0
        )
        logger.info(f"✅ Verified access to model: {model_name}")
    except Exception as e:
        sys.exit(f"❌ Model {model_name} not accessible: {e}")


def load_resources(index_path: str, metadata_path: str, model_name: str):
    """
    Load the FAISS index, metadata JSON, and the embedding model.
    Returns a tuple: (index, metadata list, embedding model)
    Exits on failure.
    """
    logger.info(f"📦 Loading FAISS index from {index_path}")
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
        metadatas = data["chunks"] if isinstance(data, dict) and "chunks" in data else data
        embedding_model = SentenceTransformer(model_name)
        logger.info(f"✅ Resources loaded: {index.ntotal} vectors")
        return index, metadatas, embedding_model
    except Exception as e:
        sys.exit(f"❌ Failed to load resources: {e}")


def detect_language(code: str, filename: str = "") -> str:
    """
    Detect the programming language of the given code based on filename and content.
    Returns detected language or 'unknown'.
    """
    if filename:
        # Check filename patterns first
        for lang, patterns in LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if pattern.startswith(r'\.') and re.search(pattern, filename, re.IGNORECASE):
                    return lang

    # Score code patterns by language
    scores = {lang: sum(
        bool(re.findall(p, code, re.IGNORECASE | re.MULTILINE))
        for p in patterns if not p.startswith(r'\.'))
        for lang, patterns in LANGUAGE_PATTERNS.items()
    }

    # Pick the best scoring language or 'unknown'
    best_lang, score = max(scores.items(), key=lambda x: x[1], default=("unknown", 0))
    ##
    print(f"[DEBUG] Filename passed: {filename}")
    print(f"[DEBUG] Code sample (first 100): {code[:100]!r}")
    ##
    return best_lang if score > 0 else "unknown"


def get_cache_key(code: str) -> str:
    """
    Generate a unique MD5 hash key for caching based on code content.
    """
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
    Perform Maximum Marginal Relevance (MMR) search to balance relevance and diversity
    in retrieved knowledge base chunks.

    Args:
        query: User query text
        code_language: Detected language of code (to improve query context)
        index: FAISS index object
        metadatas: List of knowledge chunks metadata
        embedding_model: Model to embed query and chunks
        top_k: Number of chunks to return
        lambda_param: MMR tradeoff parameter between relevance and diversity

    Returns:
        List of selected chunks with text, section, score, and keywords.
    """
    language_query = f"recommendations for {code_language} code best practices"
    combined_query = f"{query} {language_query}"
    query_vec = embedding_model.encode(combined_query).astype("float32")

    # Reshape for FAISS
    query_vec = query_vec.reshape(1, -1)

    # Get candidate chunks (3x top_k to have enough diversity)
    distances, indices = index.search(query_vec, top_k * 3)

    distances = distances.flatten()
    indices = indices.flatten()

    # Filter valid indices
    candidate_ids = [idx for idx in indices if 0 <= idx < len(metadatas)]
    if not candidate_ids:
        return []

    # Embed candidate texts
    candidate_texts = [metadatas[idx]["text"] for idx in candidate_ids]
    candidate_embeddings = np.vstack([
        embedding_model.encode(text).astype("float32") for text in candidate_texts
    ])

    query_vec = query_vec.flatten()

    selected_indices = []
    remaining_indices = list(range(len(candidate_ids)))

    # MMR selection loop
    for _ in range(min(top_k, len(candidate_ids))):
        if not remaining_indices:
            break

        mmr_scores = []
        for i in remaining_indices:
            # Relevance: cosine similarity to query
            relevance = np.dot(candidate_embeddings[i], query_vec) / (
                np.linalg.norm(candidate_embeddings[i]) * np.linalg.norm(query_vec)
            )

            # Diversity: distance from already selected
            diversity = 1.0
            if selected_indices:
                similarities = [
                    np.dot(candidate_embeddings[i], candidate_embeddings[j]) /
                    (np.linalg.norm(candidate_embeddings[i]) * np.linalg.norm(candidate_embeddings[j]))
                    for j in selected_indices
                ]
                diversity = 1 - max(similarities) if similarities else 1.0  # ensure diversity

            # Combined MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            mmr_scores.append((i, mmr_score))

        # Pick candidate with highest MMR score
        next_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)

    # Build the final results list
    results = []
    for idx in selected_indices:
        original_idx = candidate_ids[idx]
        chunk = metadatas[original_idx]

        # Find distance for similarity score
        distance_idx = np.where(indices == original_idx)[0]
        distance = distances[distance_idx[0]] if len(distance_idx) > 0 else 1.0

        results.append({
            "text": chunk["text"],
            "section": chunk.get("section", "Unknown"),
            "score": 1.0 / (1.0 + float(distance)),
            "keywords": chunk.get("keywords", [])
        })

    return results


def process_large_code(code: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
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
    """
    Calculate basic code quality metrics like lines, chars, comment ratio,
    and estimate complexity using regex heuristics.
    """
    metrics = {
        "line_count": len(code.splitlines()),
        "char_count": len(code),
        "avg_line_length": round(len(code) / max(1, len(code.splitlines())), 2),
        "comment_ratio": 0,
        "complexity_estimate": 0
    }

    # Count comment characters if pattern defined for the language
    if language in COMMENT_PATTERNS:
        comments = re.findall(COMMENT_PATTERNS[language], code)
        comment_chars = sum(len(c) for c in comments)
        metrics["comment_ratio"] = round(comment_chars / max(1, len(code)), 3)

    # Count control structures for complexity
    if language in COMPLEXITY_PATTERNS:
        control_structures = re.findall(COMPLEXITY_PATTERNS[language], code)
        metrics["complexity_estimate"] = len(control_structures)

    # Approximate cyclomatic complexity for selected languages
    if language in ["python", "javascript", "java", "c#", "c++", "c"]:
        decision_points = 0
        decision_patterns = [
            r'\bif\b', r'\belse\s+if\b', r'\bswitch\b', r'\bcase\b',
            r'\bfor\b', r'\bwhile\b', r'\bdo\b', r'\bcatch\b'
        ]
        if language == "python":
            decision_patterns.extend([r'\belif\b', r'\bexcept\b'])
        elif language in ["javascript", "typescript"]:
            decision_patterns.extend([r'\b\?\b', r'\?\.\b', r'\?\?'])  # ternary/optional chaining
        elif language == "java":
            decision_patterns.extend([r'\b\?\b', r'&&', r'\|\|'])  # ternary/logical ops

        for pattern in decision_patterns:
            decision_points += len(re.findall(pattern, code))

        # Cyclomatic complexity ~ decision points + 1
        metrics["cyclomatic_complexity"] = decision_points + 1

    return metrics


def analyze_dependencies(code: str, language: str) -> Dict[str, Any]:
    """
    Extract dependencies/imports from the code based on language-specific patterns.
    """
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

    # Extend here for other languages as needed

    return {
        "dependencies": list(set(dependencies)),  # unique list
        "count": len(set(dependencies))
    }


def collect_feedback(session_id, filename, feedback_text, recommendations, metrics, llm):

    analysis = analyze_feedback(feedback_text, llm)
    if analysis["decision"] != "Accept":
        print("[⛔] Feedback rejected by validator.")
        return False

    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (
            timestamp, session_id, filename,
            feedback, sentiment, validated,
            recommendations, metrics
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(), session_id, filename,
        feedback_text, analysis["sentiment"], 1,
        recommendations, json.dumps(metrics)
    ))
    conn.commit()
    conn.close()
    print("[✅] Feedback stored.")
    return True



def analyze_feedback(feedback_text):
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



def get_past_feedback_for_file(filename, limit=3):
   
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute('''
        SELECT feedback FROM feedback
        WHERE filename = ? AND sentiment IN ('neutral', 'negative') AND validated = 1
        ORDER BY id DESC LIMIT ?
    ''', (filename, limit))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]
  



def translate_recommendations(recommendations: str, target_language: str) -> str:
    """
    Translate the given recommendations string into the target language,
    keeping code snippets and technical terms in English.
    Handles Gemini and GPT-style LLM response formats.
    """
    if target_language.lower() == "english":
        return recommendations

    prompt = f"""Translate the following code recommendations to {target_language}. 
Keep code snippets and technical terms in English.

Original recommendations:
{recommendations}

{target_language} translation:"""

    try:
        print(f"\n\n🟡 Prompt being sent to LLM:\n{prompt}\n")

        response = litellm.completion(
            model=DEFAULT_LLM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0.1
        )

        logger.warning(f"🟠 Raw LLM response: {response}")

        # === Detect Gemini format ===
        if "candidates" in response:
            candidates = response["candidates"]
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
                else:
                    return "⚠️ Translation failed: Gemini response had no text parts."

        # === Fallback to OpenAI/GPT format ===
        choices = response.get("choices", [])
        if not choices:
            return "⚠️ Translation failed: No choices returned."

        message = choices[0].get("message", {})
        content = message.get("content", None)
        if not content:
            return "⚠️ Translation failed: GPT response had no content."

        return content.strip()

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
    """
    Generate recommendations using the retrieved chunks and code context.
    This function streams LLM output and accumulates the answer.

    Args:
        state (AgentState): Current agent state with code and retrieved chunks.

    Returns:
        AgentState: Updated state with generated answer or error info.
    """
    # If previous error, return immediately with message
    if state.get("error"):
        state["answer"] = f"An error occurred: {state['error']}"
        return state
        
    # If no chunks retrieved, inform user no recommendations found
    if not state["retrieved_chunks"]:
        state["answer"] = "No relevant recommendations found for your code."
        return state

    # Format retrieved recommendation chunks for prompt context
    context = "\n\n".join(
        f"[{i+1}] [{c['section']}] {c['text']}" 
        for i, c in enumerate(state["retrieved_chunks"])
    )
    language, filename = state["code_language"], state["code_filename"]

    # Prepare code metrics section if available
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

    # Prepare dependencies info if available
    dependencies_info = ""
    if state.get("dependencies") and state["dependencies"]["count"] > 0:
        dependencies_info = f"""
Dependencies ({state['dependencies']['count']}):
{', '.join(state['dependencies']['dependencies'][:10])}
"""
        if len(state['dependencies']['dependencies']) > 10:
            dependencies_info += f"... and {len(state['dependencies']['dependencies']) - 10} more"

    # Compose the full prompt for the LLM
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

f"## 💻 Provided Code Snippet (Language: {language}):\n"
f"```{language}\n{state['code']}\n```\n"

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
f"For each specific issue:\n"
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
f"5. Use emojis and bold text for visual hierarchy instead of markdown headers (###).\n"
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
            max_tokens=8000,
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
    """
    Process multiple code chunks in parallel using threads and combine their recommendations.

    Args:
        code_chunks (List[str]): List of code chunk strings.
        language (str): Programming language of the code.
        filename (str): Name of the source file.

    Returns:
        str: Combined recommendations for all chunks.
    """
    results = []
    
    # Limit max workers to avoid too many threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(code_chunks), 5)) as executor:
        # Submit each chunk processing as a separate future task
        future_to_chunk = {
            executor.submit(process_single_chunk, chunk, language, f"{filename} (part {i+1}/{len(code_chunks)})"): i 
            for i, chunk in enumerate(code_chunks)
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result()
                results.append((chunk_idx, result))
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {e}")
                results.append((chunk_idx, f"Error processing this section: {str(e)}"))
    
    # Sort results by chunk index to keep original order
    sorted_results = [res for _, res in sorted(results, key=lambda x: x[0])]
    # Join results with a clear separator
    combined = "\n\n## ===== Next Section =====\n\n".join(sorted_results)
    
    return combined


def process_single_chunk(code_chunk: str, language: str, chunk_name: str) -> str:
    """
    Process a single code chunk by invoking the agent workflow.

    Args:
        code_chunk (str): Code text for the chunk.
        language (str): Programming language.
        chunk_name (str): Identifier for the chunk (e.g., filename + part number).

    Returns:
        str: Recommendations generated for this chunk.
    """
    # Build initial agent state for this chunk
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
    
    # Invoke the agent and return the answer
    final_state = agent.invoke(initial_state)
    return final_state["answer"]


def benchmark_performance():
    """
    Run benchmarks on sample code files to measure retrieval and generation times.

    Returns:
        List[dict]: List of benchmark results with timing and success status.
    """
    sample_files = [f for f in os.listdir("benchmark_samples") if os.path.isfile(os.path.join("benchmark_samples", f))]
    results = []
    
    for file in sample_files:
        filepath = os.path.join("benchmark_samples", file)
        with open(filepath, "r") as f:
            code = f.read()
        
        language = detect_language(code, file)
        ##
        print(f"[DEBUG] ✅Language detected: {language}")
        ##
        
        start_time = time.time()
        initial_state = {
            "code": code,
            "code_language": language,
            "code_filename": file,
            "retrieved_chunks": [],
            "answer": "",
            "metrics": None,
            "dependencies": None,
            "error": None
        }
        
        try:
            # Measure retrieval time
            retrieve_start = time.time()
            state_after_retrieve = retrieve_node(initial_state)
            retrieve_time = time.time() - retrieve_start
            
            # Measure generation time
            generate_start = time.time()
            final_state = generate_node(state_after_retrieve)
            generate_time = time.time() - generate_start
            
            total_time = time.time() - start_time
            
            results.append({
                "file": file,
                "language": language,
                "retrieve_time": retrieve_time,
                "generate_time": generate_time,
                "total_time": total_time,
                "chunks_retrieved": len(state_after_retrieve["retrieved_chunks"]),
                "success": True
            })
        except Exception as e:
            results.append({
                "file": file,
                "language": language,
                "error": str(e),
                "success": False
            })
    
    # Save benchmark results to JSON file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def build_agent():
    """
    Build the agent workflow as a state graph with retrieval and generation steps.

    Returns:
        Compiled state graph agent.
    """
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    # Define flow edges: retrieve -> generate -> END
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    # Set initial node
    graph.set_entry_point("retrieve")
    
    return graph.compile()


from reportlab.pdfgen import canvas
import tempfile

def gradio_wrapper(user_file, code_text=None, target_language="English") -> Tuple[str, Dict[str, Any], str]:
    """
    Process code input from uploaded file or pasted text, generate recommendations,
    metrics, and produce a PDF report.

    Args:
        user_file: Uploaded file object (can be None).
        code_text (str): Raw code text input (optional).
        target_language (str): Output language for recommendations (default English).

    Returns:
        Tuple[str, Dict[str, Any], str]: Markdown recommendations, JSON metrics, PDF file path.
    """

    # Prepare initial state depending on input source
    if user_file:
        # If uploaded file, parse and prepare state (handle zip if needed)
        state = prepare_state_from_input(
            user_file.name,
            "unknown",  # Let prepare_state_from_input handle detection
            target_language
        )
    elif code_text:
        # Detect language from code text
        detected_language = detect_language(code_text, "code_snippet.txt")
        print(f"[DEBUG] ✅Filename passed: code_snippet.txt")
        print(f"[DEBUG] ✅Detected language: {detected_language}")

        code_language = detected_language
        state = {
            "files": [("code_snippet.txt", code_text)],
            "code_language": code_language,
            "target_language": target_language
        }
    else:
        # No input provided, return prompt message
        return "Please upload a file or paste code", None, None

    results = []
    combined_metrics = {"Code Metrics": {}, "Dependencies": {}}

    # Process each file individually
    for filename, code in state["files"]:
        if not code.strip():
            # Skip empty files
            continue

        # Generate cache key for the file content
        cache_key = get_cache_key(code)
        if cache_key in RESPONSE_CACHE:
            cached_result = RESPONSE_CACHE[cache_key]
            if isinstance(cached_result, tuple):
                # Use cached results if available
                results.append(f"### 📄 **{filename}**\n\n{cached_result[0]}")
                continue

        # Handle large files by chunking
        code_chunks = process_large_code(code) if len(code) > MAX_CHUNK_SIZE else [code]

        if len(code_chunks) > 1:
            logger.info(f"Processing {filename} in {len(code_chunks)} chunks")
            # Process chunks in parallel and combine
            response = parallel_process_chunks(code_chunks, state["code_language"], filename)
            metrics = analyze_code_metrics(code, state["code_language"])
            dependencies = analyze_dependencies(code, state["code_language"])
        else:
            # Process single chunk via agent
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

        # Append results with filename heading
        results.append(f"### 📄 **{filename}**\n\n{response}")

        # Collect combined metrics and dependencies
        combined_metrics["Code Metrics"][filename] = metrics
        combined_metrics["Dependencies"][filename] = dependencies.get("dependencies", [])

        # Cache the response for faster future use
        RESPONSE_CACHE[cache_key] = (response, {filename: {"metrics": metrics, "dependencies": dependencies}})

    # Enforce cache size limit by removing oldest entries
    if len(RESPONSE_CACHE) > MAX_CACHE_SIZE:
        oldest_keys = list(RESPONSE_CACHE.keys())[:(len(RESPONSE_CACHE) - MAX_CACHE_SIZE)]
        for key in oldest_keys:
            del RESPONSE_CACHE[key]

    # Combine all per-file responses into one markdown string
    final_response = "\n\n---\n\n".join(results)

    # Generate PDF file of the recommendations using ReportLab
    tmp_fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(tmp_fd)  # Close fd, ReportLab handles file writing

    c = canvas.Canvas(pdf_path)
    lines = final_response.splitlines()
    y = 800  # Initial vertical position on page
    for line in lines:
        c.drawString(50, y, line)
        y -= 15
        if y < 50:  # Start new page if space runs out
            c.showPage()
            y = 800
    c.save()

    return final_response, combined_metrics, pdf_path
#####
#this for the feedback
init_feedback_db()
#####




css = """
    .session-grp {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        gap: 8px !important;
        margin-bottom: 8px !important;
    }
    .session-btn {
        width: 250px;
    }
    .del-btn {
        flex-grow: 1 !important;
        height: 40px;
        position: absolute;
        right: 0px;
        min-width: 40px !important;
        max-width: 40px !important;
    }
    .empty-row {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
"""

MAX_SESSIONS = 10
last_result = {"recommendations": "", "metrics": {}, "pdf_path": ""}

def make_session_id(filename):
    # Simplify the ID to only display the filename
    if filename:
        return filename
    return "Analysis"

def load_session_by_index(idx, history):
    if idx < 0 or idx >= len(history):
        # Return empty values matching the new output components
        return "", "", "", "", "", None
        
    session_data = history[idx]
    
    # Make sure to properly split recommendations by category
    recs = split_recommendations_by_category(session_data["recommendations"])
    
    # Debug output
    print(f"[DEBUG] Loading session {idx}, found categories:", list(recs.keys()))
    
    return (
        "",  # code input blank or use session_data["code"] if you want to show it
        recs.get("performance", ""),  # Performance tab
        recs.get("efficiency", ""),   # Efficiency tab
        recs.get("green_it", ""),     # Green IT tab
        recs.get("design", ""),       # Design tab
        session_data.get("pdf_path", None)
    )


def delete_session(idx, history):
    if idx < 0 or idx >= len(history):
        return history
    
    new_history = history.copy()
    del new_history[idx]
    
    print(f"[DEBUG] Deleting session at index {idx}, new history length: {len(new_history)}")
    save_encrypted_history(new_history)  # 🔐 Save updated history
    
    return new_history

def clean_history(history):
    if not history:
        return []
    return [h for h in history if h and "id" in h and h["id"]]

def create_interface():
    with gr.Blocks(title="🌟 Code Recommendation Assistant", css=css) as iface:
        preloaded_history = clean_history(decrypt_history())
        print(f"[DEBUG] Loaded history with {len(preloaded_history)} sessions at startup")
        session_history = gr.State(preloaded_history)
        current_session = gr.State(None)

        session_buttons = []
        delete_buttons = []
        session_rows = []

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("## 💬 History", elem_id="sidebar-title")
                new_session_btn = gr.Button("➕ New Analysis")

                for i in range(MAX_SESSIONS):
                    with gr.Row(elem_classes="session-grp") as row:
                        session_rows.append(row)
                        btn = gr.Button("", scale=4, elem_classes="session-btn")
                        del_btn = gr.Button("🗑️", size="sm", scale=1, elem_classes="del-btn")
                        session_buttons.append(btn)
                        delete_buttons.append(del_btn)

            with gr.Column(scale=4):
                gr.Markdown(
                    "# 🚀 Code Recommendation Checker\n"
                    "Upload or paste your code to get AI-powered recommendations to improve performance, efficiency, and environmental impact.",
                    elem_classes="text-center"
                )

                with gr.Tab("🔍 Analyze"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_input = gr.File(label="📁 Upload your code file")
                            code_input = gr.Textbox(label="📝 Or paste your code", lines=12, placeholder="Paste your code here...")
                            target_language = gr.Dropdown(
                                choices=["English", "Spanish", "French", "German"],
                                label="🌐 Recommendation Language",
                                value="English"
                            )
                            with gr.Row():
                                submit_btn = gr.Button("🔍 Analyze Code", variant="primary")
                                clear_btn = gr.Button("🧹 Clear All")

                        with gr.Column(scale=3):
                            with gr.Tabs():
                                with gr.TabItem("📋 Performance"):
                                    perf_output = gr.Markdown()
                                with gr.TabItem("⚙️ Efficiency"):
                                    eff_output = gr.Markdown()
                                with gr.TabItem("🌿 Green IT"):
                                    green_output = gr.Markdown()
                                with gr.TabItem("🏗️ Design & Architecture"):
                                    design_output = gr.Markdown()
                                with gr.TabItem("📝 Summary"):
                                    summary_output = gr.Markdown()
                                    pdf_output = gr.File(label="📄 Download Summary PDF")

                with gr.Tab("🗣️ Give Feedback"):
                    feedback_text = gr.Textbox(label="✍️ Your Feedback", lines=2)
                    feedback_rating = gr.Slider(minimum=1, maximum=5, step=1, label="⭐ Rate the Suggestions", value=3)
                    feedback_btn = gr.Button("📨 Submit Feedback")
                    feedback_response = gr.Textbox(label="", interactive=False)


        def wrapped_gradio_call(file, code_text, lang, history):
                recommendations, metrics, pdf_path = gradio_wrapper(file, code_text, lang)
                last_result["recommendations"] = recommendations
                last_result["metrics"] = metrics
                last_result["pdf_path"] = pdf_path

                # Clean history of invalid entries
                history = clean_history(history or [])

                if file and hasattr(file, "name") and file.name:
                    filename = os.path.basename(file.name)
                elif code_text:
                    filename = "Pasted Code"
                else:
                    filename = "Unknown"

                session_id = make_session_id(filename)

                if not code_text and file:
                    state = prepare_state_from_input(file.name, "unknown")
                    code_text = "\n\n".join([content for _, content in state.get("files", [])])

                session_data = {
                    "id": session_id,
                    "code": code_text or "",
                    "recommendations": recommendations,
                    "pdf_path": pdf_path
                }

                # Add the new session at the beginning of the list
                history.insert(0, session_data)  # Add newest at the top
                history = history[:MAX_SESSIONS]  # Trim to max sessions

                print(f"[DEBUG] Saving history with {len(history)} sessions")
                save_encrypted_history(history)  # 🔐 Save to encrypted file
                print("[DEBUG] Saved encrypted history")

                # Split recommendation text by category here - IMPORTANT FIX
                recs_by_cat = split_recommendations_by_category(recommendations)
                
                # Debug output to verify splitting
                print("[DEBUG] Categories found:", list(recs_by_cat.keys()))
                for cat, content in recs_by_cat.items():
                    print(f"[DEBUG] {cat} contains {len(content)} characters")
                
                # Generate summary
                summary = create_summary_from_recommendations(recs_by_cat, metrics)

                return (
                    recs_by_cat.get("performance", ""),  # Performance tab
                    recs_by_cat.get("efficiency", ""),   # Efficiency tab
                    recs_by_cat.get("green_it", ""),     # Green IT tab
                    recs_by_cat.get("design", ""),       # Design tab
                    summary,
                    pdf_path,
                    history,
                    session_data
                )


        def clear_all():
            print("[DEBUG] Clearing all history")
            save_encrypted_history([])  # 🔐 Clear saved history
            return None, "", "", "", "", "", "", None, "", None

        import os

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



        def translate_output_ui(lang_choice):
            if not last_result["recommendations"]:
                return ["⚠️ No analysis result to translate yet. Please analyze code first."] * 5
                
            recs_translated = translate_recommendations(last_result["recommendations"], lang_choice)
            recs_by_cat = split_recommendations_by_category(recs_translated)
            summary = create_summary_from_recommendations(recs_by_cat, last_result.get("metrics", {}))
            
            return [
                recs_by_cat.get("performance", ""),
                recs_by_cat.get("efficiency", ""),
                recs_by_cat.get("green_it", ""),
                recs_by_cat.get("design", ""),
                summary,
            ]


        def update_sidebar(history):
            history = clean_history(history)

            row_updates = []
            button_updates = []
            delete_updates = []

            actual_sessions = len(history)

            for i in range(MAX_SESSIONS):
                if i < actual_sessions:
                    row_updates.append(gr.update(elem_classes="session-grp"))
                    button_updates.append(gr.update(value=history[i]["id"]))
                    delete_updates.append(gr.update())
                else:
                    row_updates.append(gr.update(elem_classes="session-grp empty-row"))
                    button_updates.append(gr.update(value=""))
                    delete_updates.append(gr.update())

            return row_updates + button_updates + delete_updates

        def reset_new_analysis():
            # Clear all outputs (Markdowns and file)
            return None, "", "", "", "", "", "", None, "", None

        # Connect session buttons
        for i, btn in enumerate(session_buttons):
            btn.click(
                fn=load_session_by_index,
                inputs=[gr.State(i), session_history],
                outputs=[code_input, perf_output, eff_output, green_output, design_output, summary_output, pdf_output]
            )

        # Connect delete buttons
        for i, del_btn in enumerate(delete_buttons):
            del_btn.click(
                fn=delete_session,
                inputs=[gr.State(i), session_history],
                outputs=[session_history]
            ).then(
                fn=update_sidebar,
                inputs=[session_history],
                outputs=session_rows + session_buttons + delete_buttons
            )

        submit_btn.click(
            fn=wrapped_gradio_call,
            inputs=[file_input, code_input, target_language, session_history],
            outputs=[perf_output, eff_output, green_output, design_output, summary_output, pdf_output, session_history, current_session]
        ).then(
            fn=update_sidebar,
            inputs=[session_history],
            outputs=session_rows + session_buttons + delete_buttons
        )

        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[file_input, code_input, perf_output, eff_output, green_output, design_output, summary_output, pdf_output, feedback_response, current_session]
        ).then(
            fn=update_sidebar,
            inputs=[session_history],
            outputs=session_rows + session_buttons + delete_buttons
        )

        new_session_btn.click(
            fn=reset_new_analysis,
            inputs=None,
            outputs=[file_input, code_input, perf_output, eff_output, green_output, design_output, summary_output, pdf_output, feedback_response, current_session]
        )

        feedback_btn.click(
            fn=submit_user_feedback,
            inputs=[file_input, code_input, feedback_text, feedback_rating],
            outputs=[feedback_response, feedback_text, feedback_rating]
        ).then(
            js="() => { setTimeout(() => { document.querySelectorAll('.feedback-response, .feedback-text, .feedback-rating').forEach(el => el.innerHTML=''); }, 1000); }"
        )

        target_language.change(
            fn=translate_output_ui,
            inputs=[target_language],
            outputs=[perf_output, eff_output, green_output, design_output, summary_output]
        )

        iface.load(
            fn=lambda h: update_sidebar(h),
            inputs=[session_history],
            outputs=session_rows + session_buttons + delete_buttons
        )

    return iface


# Helper functions you need to define somewhere in your code:

import re

def split_recommendations_by_category(full_text):
    # Initialize result dictionary
    result = {
        "performance": "",
        "efficiency": "",
        "green_it": "",
        "design": ""
    }
    
    # Define patterns to identify sections
    patterns = {
        "performance": ["Performance", "MAX_CHUNK_SIZE", "call_llm function", "benchmark_performance"],
        "efficiency": ["Efficiency", "Pre-compile", "avoid recompiling", "detect_language"],
        "green_it": ["Environmental Impact", "Green IT", "Energy", "Sustainability"],
        "design": ["Software Design", "Design Principles", "Architecture", "Design Patterns", 
                  "File Architecture", "Project Architecture", "Code Structure", "Modularity"]
    }
    
    # Split text into paragraphs
    paragraphs = full_text.split("\n\n")
    
    # Process each paragraph
    current_category = None
    processed_headers = set()  # Track headers we've already seen
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if this paragraph is a header
        is_header = False
        for cat, keywords in patterns.items():
            for keyword in keywords:
                if keyword in para and len(para) < 100:  # Headers are usually short
                    # This looks like a header
                    is_header = True
                    current_category = cat
                    
                    # Check if we've seen this header before
                    if para in processed_headers:
                        # Skip duplicate headers
                        break
                    
                    # Add header with proper formatting
                    if not result[cat]:
                        # First header in this category
                        result[cat] = f"## {para}\n\n"
                    else:
                        # Additional header in this category
                        result[cat] += f"## {para}\n\n"
                    
                    processed_headers.add(para)
                    break
            
            if is_header:
                break
        
        # If not a header and we have a current category, add content
        if not is_header and current_category:
            # Add content to current category
            if "⚠️" in para or "✅" in para:
                # This is an issue or recommendation
                result[current_category] += para + "\n\n"
            elif len(para) > 50:  # Only add substantial paragraphs
                result[current_category] += para + "\n\n"
    
    # If we couldn't categorize anything, use a fallback approach
    if all(not content for content in result.values()):
        print("[DEBUG] Fallback: Categorizing by keywords in text")
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Try to categorize by keywords
            categorized = False
            for cat, keywords in patterns.items():
                if any(keyword.lower() in para.lower() for keyword in keywords):
                    result[cat] += para + "\n\n"
                    categorized = True
                    break
            
            # If we couldn't categorize, put in performance as default
            if not categorized and ("⚠️" in para or "✅" in para):
                result["performance"] += para + "\n\n"
    
    # Final cleanup - remove extra newlines and ensure proper formatting
    for cat in result:
        # Clean up extra newlines
        result[cat] = re.sub(r'\n{3,}', '\n\n', result[cat])
        
        # Ensure proper markdown formatting
        if result[cat] and not result[cat].startswith("#"):
            result[cat] = f"## {cat.title()} Recommendations\n\n{result[cat]}"
    
    return result

def create_summary_from_recommendations(recs_by_cat, metrics):
    import matplotlib.pyplot as plt
    import io
    import base64

    summary_text = "### Summary of Recommendations\n\n"
    for cat, text in recs_by_cat.items():
        count = text.count("⚠️ **Issue**")
        summary_text += f"- **{cat.capitalize()} issues found:** {count}\n"

    if metrics:
        summary_text += "\n### Code Metrics\n"
        for k, v in metrics.items():
            summary_text += f"- {k.replace('_', ' ').capitalize()}: {v}\n"

    categories = list(recs_by_cat.keys())
    counts = [recs_by_cat[cat].count("⚠️ **Issue**") for cat in categories]

    plt.figure(figsize=(6,4))
    plt.bar(categories, counts, color="skyblue")
    plt.title("Number of Issues by Category")
    plt.ylabel("Issue Count")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    summary_text += f"\n\n![Issue Chart](data:image/png;base64,{img_b64})"

    return summary_text



def main():
    """
    Main entry point. Parses CLI arguments, sets up environment and resources,
    optionally runs benchmarks, and launches the Gradio interface.
    """
    parser = argparse.ArgumentParser(description="Code Recommendation Checker")
    parser.add_argument('--port', type=int, default=7860, help="Port to run the Gradio server on")
    parser.add_argument('--share', action='store_true', help="Create a shareable link")
    parser.add_argument('--index', default=DEFAULT_INDEX_PATH, help="Path to FAISS index file")
    parser.add_argument('--metadata', default=DEFAULT_METADATA_PATH, help="Path to metadata JSON file")
    parser.add_argument('--credentials', default=DEFAULT_CREDENTIALS_PATH, help="Path to Vertex AI credentials")
    parser.add_argument('--llm', default=DEFAULT_LLM, help="LLM model to use")
    parser.add_argument('--vertex_project', default=DEFAULT_VERTEX_PROJECT, help="Vertex AI project ID")
    parser.add_argument('--vertex_location', default=DEFAULT_VERTEX_LOCATION, help="Vertex AI location")
    parser.add_argument('--benchmark', action='store_true', help="Run benchmarks on sample files")
    args = parser.parse_args()

    # Set the selected LLM model
    llm_model = args.llm

    # Setup Vertex AI environment
    setup_vertex_ai(args.credentials, args.vertex_project, args.vertex_location)
    check_model_access(llm_model)

    # Load resources like FAISS index, metadata, and embedding model
    global index, metadatas, embedding_model
    index, metadatas, embedding_model = load_resources(args.index, args.metadata, DEFAULT_MODEL)

    # Build the agent workflow graph
    global agent
    agent = build_agent()

    # Run benchmarks if requested
    if args.benchmark:
        logger.info("Running benchmarks...")
        results = benchmark_performance()
        logger.info(f"Benchmark complete. Results saved to benchmark_results.json")
        return 0

    # Launch the Gradio interface
    logger.info(f"Starting Gradio interface on port {args.port}")
    iface = create_interface()
    iface.launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    exit(main())