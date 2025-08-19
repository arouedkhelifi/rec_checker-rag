"""
Knowledge Base Query Tool

Enables semantic search over a FAISS vector index built from knowledge chunks.
Accepts a text query and returns the most relevant recommendation snippets 
based on embedding similarity.

Features:
- Query via command line or interactive prompt
- Loads FAISS index, metadata, and embedding model
- Filters results by similarity threshold
- Displays results in a clear, formatted table"""

# ========== Imports ==========
import json
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# FAISS and embedding tools
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# For pretty CLI rendering
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel

# ========== Configuration Constants ==========
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INDEX_PATH = "knowledge_base.index"
DEFAULT_METADATA_PATH = "knowledge_base_metadata.json"
DEFAULT_TOP_K = 5

# CLI Output helper
console = Console()


# ========== Resource Loading ==========
def load_resources(index_path: str, metadata_path: str, model_name: str):
    """
    Loads the FAISS index, metadata JSON, and embedding model.

    Returns:
        index: FAISS index object
        metadatas: List of metadata chunks
        model: SentenceTransformer model
    """
    try:
        with console.status(f"Loading index from {index_path}..."):
            index = faiss.read_index(index_path)
            console.print(f"✅ Loaded index with [bold]{index.ntotal}[/bold] vectors")

        with console.status(f"Loading metadata from {metadata_path}..."):
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Adapt to different metadata file formats
            metadatas = data.get("chunks", data)
            if isinstance(data, dict) and "metadata" in data:
                console.print("📊 Knowledge base info:")
                for key, value in data["metadata"].items():
                    if key != "sections":
                        console.print(f"   [bold]{key}[/bold]: {value}")
            console.print(f"✅ Loaded [bold]{len(metadatas)}[/bold] metadata entries")

        with console.status(f"Loading embedding model: {model_name}"):
            model = SentenceTransformer(model_name)
            console.print("✅ Model loaded successfully")

        return index, metadatas, model

    except Exception as e:
        console.print(f"[bold red]Error loading resources:[/bold red] {str(e)}")
        raise


# ========== Query Execution ==========
def query_knowledge_base(
    index: faiss.Index,
    metadatas: List[Dict[str, Any]],
    query: str,
    top_k: int = 5,
    model: Optional[SentenceTransformer] = None,
    threshold: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Queries the FAISS index using the provided sentence transformer model.

    Returns:
        List of top metadata entries sorted by similarity
    """
    if model is None:
        model = SentenceTransformer(DEFAULT_MODEL)

    start_time = time.time()
    query_vector = model.encode(query).astype("float32")
    encoding_time = time.time() - start_time

    start_time = time.time()
    distances, indices = index.search(np.array([query_vector]), top_k * 2)  # Overfetch for filtering
    search_time = time.time() - start_time

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(metadatas):  # Valid index
            distance = float(distances[0][i])
            if distance > threshold:
                continue  # Skip poor matches

            result = metadatas[idx].copy()
            result["score"] = 1.0 / (1.0 + distance)  # Normalize to similarity score
            result["distance"] = distance
            result["rank"] = i + 1
            results.append(result)

    if results:
        results[0]["_encoding_time"] = encoding_time
        results[0]["_search_time"] = search_time

    return results[:top_k]


# ========== Result Display ==========
def display_results(results: List[Dict[str, Any]], query: str):
    """
    Nicely displays top results in the CLI using rich.
    """
    if not results:
        console.print(Panel("[bold yellow]No results found matching your query.[/bold yellow]", title="Search Results"))
        return

    console.print(f"\n[bold]Query:[/bold] {query}")
    if "_encoding_time" in results[0] and "_search_time" in results[0]:
        console.print(f"[dim]Encoding: {results[0]['_encoding_time']:.4f}s, "
                      f"Search: {results[0]['_search_time']:.4f}s, "
                      f"Total: {results[0]['_encoding_time'] + results[0]['_search_time']:.4f}s[/dim]")

    # Create a summary table of results
    table = Table(title=f"Top {len(results)} Results")
    table.add_column("Rank", style="dim")
    table.add_column("Score", style="green")
    table.add_column("Section", style="blue")
    table.add_column("Text", style="white")

    for result in results:
        score_str = f"{result.get('score', 0):.2f}"
        short_text = result.get("text", "")
        if len(short_text) > 100:
            short_text = short_text[:97] + "..."
        table.add_row(str(result.get("rank", "-")), score_str, result.get("section", "Unknown"), short_text)

    console.print(table)

    # Show full detail of top result
    top_result = results[0]
    console.print("\n[bold]Top Result Details:[/bold]")
    console.print(f"[bold blue]Section:[/bold blue] {top_result.get('section', 'Unknown')}")
    console.print(f"[bold green]Score:[/bold green] {top_result.get('score', 0):.4f}")

    if "keywords" in top_result:
        keywords = ", ".join(top_result["keywords"])
        console.print(f"[bold yellow]Keywords:[/bold yellow] {keywords}")

    console.print("\n[bold]Text:[/bold]")
    console.print(Panel(top_result.get("text", ""), width=100))


# ========== Interactive Mode ==========
def interactive_mode(index, metadatas, model):
    """
    Runs the script in CLI-based interactive mode.
    """
    console.print(Panel.fit(
        "[bold]Interactive Query Mode[/bold]\n"
        "Type your queries below. Type 'exit' or press Ctrl+C to quit.",
        title="Knowledge Base Explorer"
    ))

    try:
        while True:
            query = console.input("\n[bold green]Enter query:[/bold green] ")
            if query.lower() in ("exit", "quit", "q"):
                break
            if not query.strip():
                continue
            with console.status("[bold green]Searching...[/bold green]"):
                results = query_knowledge_base(index, metadatas, query, model=model)
            display_results(results, query)

    except KeyboardInterrupt:
        console.print("\n[bold]Exiting interactive mode[/bold]")


# ========== Main Entry ==========
def main():
    """
    Command-line interface entry point.
    """
    parser = argparse.ArgumentParser(description="Query the RAG knowledge base.")
    parser.add_argument('--index', '-i', default=DEFAULT_INDEX_PATH, help='Path to FAISS index file')
    parser.add_argument('--metadata', '-m', default=DEFAULT_METADATA_PATH, help='Path to metadata JSON file')
    parser.add_argument('--model', '-e', default=DEFAULT_MODEL, help='Sentence transformer model name')
    parser.add_argument('--top-k', '-k', type=int, default=DEFAULT_TOP_K, help='Number of top results to return')
    parser.add_argument('--threshold', '-t', type=float, default=1.5, help='Distance threshold for matches')
    parser.add_argument('--query', '-q', help='Query string')
    parser.add_argument('--interactive', '-I', action='store_true', help='Run in interactive prompt mode')

    args = parser.parse_args()

    try:
        index, metadatas, model = load_resources(args.index, args.metadata, args.model)

        if args.interactive:
            interactive_mode(index, metadatas, model)
            return 0

        query_text = args.query or console.input("[bold green]Enter your query:[/bold green] ")
        with console.status("[bold green]Searching...[/bold green]"):
            results = query_knowledge_base(index, metadatas, query_text, top_k=args.top_k,
                                           model=model, threshold=args.threshold)
        display_results(results, query_text)
        return 0

    except FileNotFoundError as e:
        console.print(f"[bold red]Error: File not found[/bold red] - {e}")
        return 1
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1


# ========== Script Execution ==========
if __name__ == "__main__":
    exit(main())
