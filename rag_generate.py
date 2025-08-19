"""
Recommendation Checker Chatbot - RAG Generation Interface

Updated to use environment variables and company LLM proxy support.
"""
import sqlite3
from db_utils import store_feedback, get_past_feedback_for_file, init_feedback_db
from encryption_utils import decrypt_history, save_encrypted_history
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
import gradio as gr
from tenacity import retry, stop_after_attempt, wait_exponential
import tempfile
from fpdf import FPDF
from reportlab.pdfgen import canvas

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
    """
    index_path = index_path or config.VECTOR_INDEX_PATH
    metadata_path = metadata_path or config.VECTOR_METADATA_PATH
    model_name = model_name or config.EMBEDDING_MODEL
    
    logger.info(f"📦 Loading FAISS index from {index_path}")
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
        metadatas = data["chunks"] if isinstance(data, dict) and "chunks" in data else data
        if model_name and model_name.startswith("vertex_ai/"):
            # Use custom proxy embedding model
            from embedding_client import ProxyEmbeddingModel
            embedding_model = ProxyEmbeddingModel(model_name)
        else:
            # Use standard SentenceTransformer
            embedding_model = SentenceTransformer(model_name)
        logger.info(f"✅ Resources loaded: {index.ntotal} vectors")
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
    """Perform Maximum Marginal Relevance search (keeping existing implementation)."""
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

def process_large_code(code: str, max_chunk_size: int = None) -> List[str]:
    """Split large code files into manageable chunks."""
    max_chunk_size = max_chunk_size or config.MAX_CHUNK_SIZE
    
    if len(code) <= max_chunk_size:
        return [code]

    lines = code.splitlines(True)
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

        current_chunk += line

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
    """Retrieve relevant chunks from the knowledge base using MMR search."""
    code_sample = state["code"][:1000]
    language = state["code_language"]
    query = f"recommendations for {language} code best practices regarding performance, efficiency, and environmental impact"
    
    try:
        retrieved = mmr_search(query, language, index, metadatas, embedding_model, top_k=7)
        filtered = [c for c in retrieved if c.get("score", 0) > 0.3][:5]
        
        state["retrieved_chunks"] = filtered
        state["metrics"] = analyze_code_metrics(state["code"], language)
        state["dependencies"] = analyze_dependencies(state["code"], language)
        
    except Exception as e:
        state["error"] = f"Error during retrieval: {str(e)}"
        logger.error(f"Retrieval error: {e}")
    
    return state

def generate_node(state: AgentState) -> AgentState:
    """Generate recommendations using the retrieved chunks and code context."""
    if state.get("error"):
        state["answer"] = f"An error occurred: {state['error']}"
        return state
        
    if not state["retrieved_chunks"]:
        state["answer"] = "No relevant recommendations found for your code."
        return state

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

f"Do NOT recommend switching to another language – **unless it is strictly necessary** to achieve substantial gains in performance, efficiency, or environmental impact. Justify clearly.\n"
f"Keep all recommendations within the context of {language}.\n"

f"---\n"
f"## 📘 Best Practice Context:\n{context}\n"
f"---\n"
f"## 📊 Code Metrics:\n{metrics_info}\n"
f"## 📦 Dependencies:\n{dependencies_info}\n\n"

f"## 💻 Provided Code Snippet (Language: {language}):\n"
f"```{language}\n{state['code']}\n```\n"

f"---\n"
f"## 🎯 TASK INSTRUCTIONS:\n"
f"Please follow the steps below and generate an actionable expert report:\n"
f"1. Check each best practice recommendation **against the code**.\n"
f"2. For each **violation**:\n"
f"   - Explain clearly why it's a violation.\n"
f"   - Propose a **specific and actionable fix** (include refactor suggestions where appropriate).\n"
f"3. ⚠️**Do not list or mention respected recommendations**, even if all best practices are followed.\n"
f"4. Analyze the code's **architecture and design**:\n"
f"   - Look at cohesion, structure, responsibility allocation.\n"
f"   - Are SOLID principles applied properly?\n"
f"   - Recommend any necessary reorganizations.\n"
f"5. Evaluate the code's **Green IT impact**:\n"
f"   - Identify unnecessary resource consumption.\n"
f"   - Recommend eco-friendly optimizations.\n\n"

f"---\n"
f"## 📄 OUTPUT FORMAT:\n"
f"Start with '🔍 Code Review' as the only title.\n\n"

f"For each issue category, use this format (only if issues exist):\n"
f"🔶 **Performance**\n\n"

f"For each specific issue:\n"
f"⚠️ **Issue**: [Clear description of the problem]\n\n"
f"✅ **Recommendation**: [Specific, actionable solution with code examples when appropriate]\n\n"

f"Important rules:\n"
f"1. Only mention categories where you found issues - skip categories with no violations.\n"
f"2. If no issues are found in the entire codebase, simply write '🔍 Code Review\\n\\nNo issues found in the codebase.'\n"
f"3. Use emojis and bold text for visual hierarchy.\n"
f"4. Make sure each section is clearly separated with blank lines for readability.\n"
f"5. Format code examples using proper markdown code blocks.\n"
)

    # Append past feedback if available
    past_feedbacks = get_past_feedback_for_file(filename)
    if past_feedbacks:
        prompt += "\n\n## 📣 Past User Feedback to Consider:\n"
        for fb in past_feedbacks:
            prompt += f"- {fb}\n"
        prompt += "\nPlease carefully consider the past user feedback above when performing your review.\n"

    try:
        state["answer"] = ""
        
        # Use streaming for real-time response
        system_prompt = f"Expert {language} reviewer focusing on best practices"
        
        for chunk in call_llm_stream(prompt, system_prompt, top_p=0.9):
            if chunk and chunk.strip():
                state["answer"] += chunk

        # Translate if needed
        if state.get("target_language") and state["target_language"].lower() != "english":
            state["answer"] = translate_recommendations(state["answer"], state["target_language"])

    except Exception as e:
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

# CSS and UI components (keeping existing CSS)
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

# Keep existing UI and helper functions (create_interface, split_recommendations_by_category, etc.)
# ... [The rest of the UI functions remain the same]

def main():
    """Main entry point with updated configuration."""
    parser = argparse.ArgumentParser(description="Code Recommendation Checker")
    parser.add_argument('--port', type=int, default=config.SERVER_PORT, help="Port to run the Gradio server on")
    parser.add_argument('--share', action='store_true', default=config.GRADIO_SHARE, help="Create a shareable link")
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
    global index, metadatas, embedding_model
    index, metadatas, embedding_model = load_resources(args.index, args.metadata, args.embedding_model)

    # Build agent
    global agent
    agent = build_agent()

    # Launch interface
    logger.info(f"🚀 Starting Gradio interface on port {args.port}")
    iface = create_interface()
    iface.launch(server_port=args.port, share=args.share)
    return 0

if __name__ == "__main__":
    exit(main())