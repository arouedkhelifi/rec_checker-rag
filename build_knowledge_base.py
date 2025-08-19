"""
Knowledge Base Chunk Builder for RAG

This script transforms a cleaned JSON file of recommendations into structured, 
overlapping text chunks suitable for RAG pipelines.

Key features:
- Splits long texts into manageable, overlapping chunks
- Adds metadata (section, keywords, quality score)
- Ensures deduplication and scores chunk relevance
- Saves the final knowledge base in JSON format

"""


import json
import logging
import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Pre-compile regex patterns for better performance
SENTENCE_BOUNDARY_PATTERN = re.compile(r'[.!?][\s\n]+')
WORD_PATTERN = re.compile(r'\b[a-zA-Z][a-zA-Z0-9]*\b')
SENTENCE_END_PATTERN = re.compile(r'[.!?]$')

# Common stop words for keyword extraction
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", 
    "at", "from", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "of", "in", 
    "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", 
    "so", "than", "too", "very", "can", "will", "just", "should", "now"
}

# Technical terms for quality scoring
TECHNICAL_TERMS = {
    "architecture", "system", "design", "implementation", "pattern",
    "framework", "library", "api", "service", "component", "module",
    "interface", "protocol", "algorithm", "function", "method",
    "class", "object", "database", "query", "cache", "memory",
    "performance", "security", "scalability", "reliability",
    "availability", "maintainability", "testability", "deployment"
}

# Recommendation indicators for quality scoring
RECOMMENDATION_INDICATORS = {
    "recommend", "should", "must", "need to", "important to",
    "best practice", "avoid", "ensure", "consider", "use",
    "implement", "choose", "select", "prefer", "better"
}

def split_text(text: str, max_length: int = 200, overlap: int = 50) -> List[str]:
    """
    Split a text into chunks of max_length with overlap, trying to break at sentence boundaries.
    
    Args:
        text: The text to split
        max_length: Maximum length of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Handle empty or None text
    if not text:
        return []
        
    chunks = []
    start = 0
    
    while start < len(text):
        # If we're near the end of text, just take the remainder
        if start + max_length >= len(text):
            end = len(text)
        else:
            # Try to find a good breaking point
            end = start + max_length
            
            # Try to break at sentence end (., !, ?)
            last_boundary = -1
            for match in SENTENCE_BOUNDARY_PATTERN.finditer(text[start:min(start + max_length + 20, len(text))]):
                pos = start + match.end()
                if pos <= end:
                    last_boundary = pos
            
            if last_boundary != -1:
                end = last_boundary
            else:
                # If no sentence boundary found, try paragraph break
                para_break = text.find('\n\n', start, end)
                if para_break != -1:
                    end = para_break + 2
                else:
                    # If no paragraph break, try line break
                    line_break = text.find('\n', start, end)
                    if line_break != -1:
                        end = line_break + 1
                    else:
                        # If no breaks found, try space
                        space = text.rfind(' ', start, end)
                        if space != -1 and space > start:
                            end = space + 1
        
        # Add the chunk if it's not empty
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position for next chunk, considering overlap
        # Ensure we make forward progress
        new_start = end - overlap
        if new_start <= start:
            new_start = start + 1
        start = new_start
    
    return chunks

def extract_keywords(text: str, min_length: int = 4, max_keywords: int = 5) -> List[str]:
    """
    Extract important keywords from text for metadata enrichment.
    
    Args:
        text: The text to analyze
        min_length: Minimum length of keywords to consider
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    # Tokenize and clean text
    words = WORD_PATTERN.findall(text.lower())
    
    # Filter out stop words and short words
    word_counts = {}
    for word in words:
        if word not in STOP_WORDS and len(word) >= min_length:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    if not word_counts:
        return []
        
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]

def calculate_chunk_hash(text: str) -> str:
    """
    Calculate a hash for a text chunk to identify duplicates.
    
    Args:
        text: The text to hash
        
    Returns:
        Hash string
    """
    # Normalize text by removing extra whitespace and converting to lowercase
    normalized_text = ' '.join(text.lower().split())
    return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()

def score_chunk_quality(chunk: str) -> float:
    """
    Score the quality of a chunk based on content characteristics.
    
    Args:
        chunk: The text chunk to score
        
    Returns:
        Quality score between 0.0 and 1.0
    """
    score = 0.5  # Start with neutral score
    chunk_lower = chunk.lower()
    
    # Length factor - too short chunks may lack context
    chunk_len = len(chunk)
    if chunk_len < 50:
        score -= 0.2
    elif chunk_len > 100:
        score += 0.1
    
    # Complete sentences are better
    if SENTENCE_END_PATTERN.search(chunk.strip()):
        score += 0.1
    
    # Chunks with technical terms are valuable
    tech_term_count = sum(1 for term in TECHNICAL_TERMS if term in chunk_lower)
    score += min(0.2, tech_term_count * 0.02)  # Cap at 0.2
    
    # Chunks with specific recommendations are valuable
    rec_indicator_count = sum(1 for indicator in RECOMMENDATION_INDICATORS if indicator in chunk_lower)
    score += min(0.2, rec_indicator_count * 0.04)  # Cap at 0.2
    
    # Cap the score at 1.0
    return min(1.0, max(0.1, score))

def build_knowledge_chunks(data: Dict[str, List[str]], max_length: int = 200, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Given recommendations data, split them into chunks and return as a list of dicts with metadata.
    
    Args:
        data: Dictionary with sections as keys and lists of recommendations as values
        max_length: Maximum length of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries with chunk data and metadata
    """
    knowledge_chunks = []
    total_recommendations = sum(len(recs) for recs in data.values())
    processed_recommendations = 0
    chunk_hashes = set()  # For deduplication
    
    logger.info(f"Building knowledge chunks with max length {max_length} characters and {overlap} character overlap")
    
    # Process each section and its recommendations
    for section, recommendations in data.items():
        section_chunks = 0
        
        for i, recommendation in enumerate(recommendations):
            # Show progress
            processed_recommendations += 1
            if processed_recommendations % 20 == 0 or processed_recommendations == total_recommendations:
                logger.info(f"Processing recommendation {processed_recommendations}/{total_recommendations}")
            
            try:
                # Skip empty recommendations
                if not recommendation or not recommendation.strip():
                    continue
                    
                # Split recommendation into manageable chunks
                text_chunks = split_text(recommendation, max_length=max_length, overlap=overlap)
                
                # Add each chunk to our knowledge base with metadata
                for j, chunk in enumerate(text_chunks):
                    # Skip empty chunks
                    if not chunk or not chunk.strip():
                        continue
                        
                    # Calculate hash for deduplication
                    chunk_hash = calculate_chunk_hash(chunk)
                    
                    # Skip if we've seen this chunk before
                    if chunk_hash in chunk_hashes:
                        continue
                    
                    # Add hash to our set
                    chunk_hashes.add(chunk_hash)
                    
                    try:
                        # Extract keywords
                        keywords = extract_keywords(chunk)
                        
                        # Score chunk quality
                        quality_score = score_chunk_quality(chunk)
                        
                        # Create chunk with metadata
                        knowledge_chunks.append({
                            "id": f"{section.replace(' ', '_')}_{i}_{j}",
                            "section": section,
                            "text": chunk,
                            "keywords": keywords,
                            "quality_score": quality_score,
                            "chunk_index": j,
                            "total_chunks": len(text_chunks),
                            "recommendation_index": i,
                            "source_type": "recommendation"
                        })
                        
                        section_chunks += 1
                    except Exception as e:
                        logger.warning(f"Error processing chunk {j} of recommendation {i} in section {section}: {str(e)}")
                        continue
            except Exception as e:
                logger.warning(f"Error processing recommendation {i} in section {section}: {str(e)}")
                continue
                
        logger.info(f"Section '{section}': {section_chunks} chunks from {len(recommendations)} recommendations")
    
    # Sort chunks by quality score (highest first)
    knowledge_chunks.sort(key=lambda x: x["quality_score"], reverse=True)
    
    logger.info(f"Created {len(knowledge_chunks)} unique knowledge chunks from {total_recommendations} recommendations")
    return knowledge_chunks

def save_knowledge_chunks(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save knowledge chunks to a JSON file.
    
    Args:
        chunks: List of knowledge chunk dictionaries
        output_path: Path where the JSON file will be saved
    """
    try:
        # Create metadata
        output_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "sections": list(set(chunk["section"] for chunk in chunks)),
                "version": "2.0"
            },
            "chunks": chunks
        }
        
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(output_data, outfile, indent=2, ensure_ascii=False)
        logger.info(f"Knowledge chunks saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving knowledge chunks to {output_path}: {e}")
        raise

def main():
    """Main function to parse arguments and run the knowledge base building process."""
    parser = argparse.ArgumentParser(
        description='Build knowledge chunks for RAG processing with enhanced features'
    )
    parser.add_argument(
        '--input', '-i',
        default="recommendation.json",
        help='Path to cleaned recommendations JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file path for knowledge chunks'
    )
    parser.add_argument(
        '--chunk-size', '-c',
        type=int,
        default=200,
        help='Maximum size of each text chunk in characters'
    )
    parser.add_argument(
        '--overlap', '-v',
        type=int,
        default=50,
        help='Number of characters to overlap between chunks'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Determine output path if not specified
    output_path = args.output
    if not output_path:
        input_path = Path(args.input)
        output_path = input_path.parent / "knowledge_chunks.json"
    
    try:
        # Load the cleaned recommendations data
        logger.info(f"Loading cleaned recommendations from: {args.input}")
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Build knowledge chunks
        chunks = build_knowledge_chunks(data, max_length=args.chunk_size, overlap=args.overlap)
        
        # Save the chunks
        save_knowledge_chunks(chunks, output_path)
        
        return 0
    except Exception as e:
        logger.error(f"Knowledge base building failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
