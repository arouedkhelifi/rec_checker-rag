"""
Updated FastAPI Server with Environment Configuration Support

This server uses the new configuration system and LLM client
for company proxy support.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os
import json
import logging
import base64

# Import updated modules
from history_manager import history_manager
from config_manager import config
from llm_client import llm_client, call_llm
from rag_generate import (
    detect_language,
    analyze_code_metrics,
    analyze_dependencies,
    load_resources,
    mmr_search,
    generate_node,
    build_agent,
    AgentState,
    process_large_file_upload_fixed,
    custom_retrieve_node_safe,
    combine_chunk_results,
    calculate_combined_metrics,
    gradio_wrapper,  
)

import warnings
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service with Company Proxy Support", 
    description="LangGraph RAG system for code recommendations with environment configuration",
    version="2.0.0"
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class CodeProcessRequest(BaseModel):
    code: str
    filename: Optional[str] = "code_snippet.txt"
    target_language: Optional[str] = "English"
    save_to_history: Optional[bool] = False

class CodeProcessResponse(BaseModel):
    recommendations: str
    language: str
    metrics: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, Any]] = None
    pdf_base64: Optional[str] = None  # Add field for PDF content

class HealthResponse(BaseModel):
    status: str
    service: str
    resources_loaded: bool
    llm_model: str
    llm_provider: str

class ConfigResponse(BaseModel):
    llm_model: str
    llm_provider: str
    embedding_model: str
    max_tokens: int
    temperature: float
    version: str

# Global variables for resources
index = None
metadatas = None
embedding_model = None
agent = None

def custom_retrieve_node(state: AgentState, index_ref, metadatas_ref, embedding_model_ref) -> AgentState:
    """
    Custom retrieve node that accepts resources as parameters.
    """
    logger.debug(f"Running custom retrieve with provided resources")
    
    code_sample = state["code"][:1000]
    language = state["code_language"]
    query = f"recommendations for {language} code best practices regarding performance, efficiency, and environmental impact"
    
    try:
        retrieved = mmr_search(query, language, index_ref, metadatas_ref, embedding_model_ref, top_k=7)
        logger.debug(f"Retrieved {len(retrieved)} chunks from mmr_search")
        
        filtered = [c for c in retrieved if c.get("score", 0) > 0.3][:5]
        logger.debug(f"Filtered to {len(filtered)} chunks with score > 0.3")
        
        state["retrieved_chunks"] = filtered
        state["metrics"] = analyze_code_metrics(state["code"], language)
        state["dependencies"] = analyze_dependencies(state["code"], language)
        
    except Exception as e:
        state["error"] = f"Error during retrieval: {str(e)}"
        logger.error(f"Retrieval error: {e}")
    
    return state

@app.on_event("startup")
async def startup_event():
    """Initialize resources and test connections on startup."""
    global index, metadatas, embedding_model, agent
    
    try:
        # Test LLM connection first
        logger.info("🔧 Testing LLM connection...")
        test_response = call_llm("Hello, this is a connection test. Please respond with 'Connection OK'.")
        logger.info(f"✅ LLM test response: {test_response}")
        
        # Load vector resources
        logger.info(f"📦 Loading resources from {config.VECTOR_INDEX_PATH} and {config.VECTOR_METADATA_PATH}")
        index, metadatas, embedding_model = load_resources(
            config.VECTOR_INDEX_PATH, 
            config.VECTOR_METADATA_PATH, 
            config.EMBEDDING_MODEL
        )
        logger.info(f"✅ Vector resources loaded successfully: {index.ntotal} vectors")
        
        # Build agent
        agent = build_agent()
        logger.info("✅ Agent workflow built successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize resources: {e}")
        logger.exception("Detailed startup error:")
        # Don't exit here, let the service start and report errors in health check

@app.get('/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint with detailed status information."""
    global index, metadatas, embedding_model
    
    resources_loaded = index is not None and metadatas is not None and embedding_model is not None
    
    # Test LLM connection
    llm_status = "healthy"
    try:
        test_response = call_llm("ping", max_tokens=10)
        if not test_response or "error" in test_response.lower():
            llm_status = "degraded"
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")
        llm_status = "unhealthy"
    
    overall_status = "healthy" if resources_loaded and llm_status == "healthy" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        service="rag_service_v2",
        resources_loaded=resources_loaded,
        llm_model=config.effective_llm_model,
        llm_provider=config.LLM_PROVIDER
    )

@app.get('/config', response_model=ConfigResponse)
async def get_config():
    """Get current configuration information."""
    return ConfigResponse(
        llm_model=config.effective_llm_model,
        llm_provider=config.LLM_PROVIDER,
        embedding_model=config.EMBEDDING_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        version="2.0.0"
    )

@app.post('/process-large-file')
async def process_large_file(request: Request):
    """Process large file uploads with chunking"""
    try:
        global index, metadatas, embedding_model
        
        # Check if resources are loaded
        if index is None or metadatas is None or embedding_model is None:
            logger.error("Resources not loaded, attempting to load them now")
            try:
                index, metadatas, embedding_model = load_resources(
                    config.VECTOR_INDEX_PATH,
                    config.VECTOR_METADATA_PATH,
                    config.EMBEDDING_MODEL
                )
                logger.info("Resources loaded successfully during large file request")
            except Exception as e:
                logger.error(f"Failed to load resources during request: {e}")
                raise HTTPException(status_code=503, detail="Service temporarily unavailable - resources not loaded")
        
        # Get request body
        body = await request.json()
        file_path = body.get("file_path")
        target_language = body.get("target_language", "English")
        filename = body.get("filename", os.path.basename(file_path) if file_path else "large_file")
        save_to_history = body.get("save_to_history", False)
            
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Valid file path required")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")
        
        # Use the process_large_file_upload_with_resources function instead
        # This explicitly passes the resources to avoid the "missing resources" error
        logger.info(f"Calling process_large_file_upload_with_resources for {file_path}")
        result = process_large_file_upload_with_resources(
            file_path, 
            target_language,
            index_param=index,
            metadatas_param=metadatas,
            embedding_model_param=embedding_model
        )
        
        # Save to history if requested
        if save_to_history:
            # For large files, we don't store the full code content
            session_id = history_manager.save_session(
                code=f"Large file processing - {filename} ({os.path.getsize(file_path)} bytes)",
                recommendations=result["recommendations"],
                filename=filename,
                language=result.get("language", "unknown"),
                metrics=result.get("metrics", {}),
                pdf_path=None  # Add PDF path if you generate one
            )
            logger.info(f"Saved large file analysis to history with ID: {session_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing large file: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def process_large_file_directly(file_path: str, target_language: str, index_ref, metadatas_ref, embedding_model_ref):
    """Process large file directly in server.py to avoid import issues"""
    try:
        from file_processor import file_processor
        from sentence_transformers import SentenceTransformer
        
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
        
        # Limit chunks
        max_chunks = 16000
        if len(chunks) > max_chunks:
            logger.warning(f"Limiting to first {max_chunks} chunks out of {len(chunks)}")
            chunks = chunks[:max_chunks]
        
        # Create safe embedding model
        safe_embedding_model = embedding_model_ref
        if safe_embedding_model is None:
            logger.warning("Creating fallback embedding model")
            safe_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Process chunks sequentially
        results = []
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
                
                # Process with safe retrieve
                state_after_retrieve = safe_custom_retrieve_node(
                    initial_state, index_ref, metadatas_ref, safe_embedding_model
                )
                final_state = generate_node(state_after_retrieve)
                results.append(final_state["answer"])
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                results.append(f"Error processing chunk {i}: {str(e)}")
        
        # Combine results
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
        
        # Add results
        for i, (result, chunk) in enumerate(zip(results, chunks)):
            if result and result.strip():
                combined += f"## 📄 {chunk['filename']} (Chunk {chunk['chunk_index']+1}/{chunk.get('total_chunks', 1)})\n\n"
                combined += result + "\n\n"
                combined += "---\n\n"
        
        # Calculate metrics
        total_lines = sum(chunk["content"].count('\n') + 1 for chunk in chunks)
        total_chars = sum(chunk["size"] for chunk in chunks)
        
        overall_metrics = {
            "total_files": len(set(chunk["filename"] for chunk in chunks)),
            "total_chunks": len(chunks),
            "total_lines": total_lines,
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(chunks) if chunks else 0,
            "languages": list(set(chunk["language"] for chunk in chunks))
        }
        
        return {
            "recommendations": combined,
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
        logger.error(f"Error in process_large_file_directly: {e}")
        return {
            "recommendations": f"Error processing large file: {str(e)}",
            "language": "unknown",
            "metrics": {},
            "dependencies": {}
        }

def safe_custom_retrieve_node(state: AgentState, index_ref, metadatas_ref, embedding_model_ref) -> AgentState:
    """Safe custom retrieve node defined in server.py"""
    code_sample = state["code"][:1000]
    language = state["code_language"]
    query = f"recommendations for {language} code best practices regarding performance, efficiency, and environmental impact"
    
    try:
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
        
        retrieved = mmr_search(query, language, index_ref, metadatas_ref, embedding_model_ref, top_k=7)
        filtered = [c for c in retrieved if c.get("score", 0) > 0.3][:5]
        
        state["retrieved_chunks"] = filtered
        state["metrics"] = analyze_code_metrics(state["code"], language)
        state["dependencies"] = analyze_dependencies(state["code"], language)
        
    except Exception as e:
        state["error"] = f"Error during retrieval: {str(e)}"
        logger.error(f"Retrieval error: {e}")
    
    return state

# Add this new function to your rag_generate.py
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

@app.post('/process', response_model=CodeProcessResponse)
async def process_code(request: CodeProcessRequest):
    """
    Process code and generate recommendations using gradio_wrapper.
    """
    try:
        global index, metadatas, embedding_model, agent
        
        # Check if resources are loaded
        if index is None or metadatas is None or embedding_model is None:
            logger.error("Resources not loaded, attempting to load them now")
            try:
                index, metadatas, embedding_model = load_resources(
                    config.VECTOR_INDEX_PATH,
                    config.VECTOR_METADATA_PATH,
                    config.EMBEDDING_MODEL
                )
                if agent is None:
                    agent = build_agent()
                logger.info("Resources loaded successfully during request")
            except Exception as e:
                logger.error(f"Failed to load resources during request: {e}")
                raise HTTPException(status_code=503, detail="Service temporarily unavailable - resources not loaded")
        
        logger.info(f"Processing code with filename: {request.filename}")
        
        # Call gradio_wrapper directly with the code
        recommendations, metrics, pdf_path = gradio_wrapper(
            user_file=None,  # No file object for direct code input
            code_text=request.code,
            target_language=request.target_language
        )
        
        logger.info(f"Generated recommendations with {len(recommendations)} characters")
        
        # Read the PDF file and encode to base64
        pdf_base64 = None
        if pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                logger.info(f"Generated PDF with {len(pdf_bytes)} bytes")
            except Exception as e:
                logger.error(f"Error reading PDF file: {e}")
        
        # Detect language
        language = detect_language(request.code, request.filename)
        
        # Save to history if requested
        if request.save_to_history:
            session_id = history_manager.save_session(
                code=request.code,
                recommendations=recommendations,
                filename=request.filename,
                language=language,
                metrics=metrics,
                pdf_path=pdf_path
            )
            logger.info(f"Saved analysis to history with ID: {session_id}")
        
        # Clean up temporary PDF file
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary PDF file: {e}")
        
        # Prepare response
        response = CodeProcessResponse(
            recommendations=recommendations,
            language=language,
            metrics=metrics,
            dependencies=metrics.get("Dependencies", {}),
            pdf_base64=pdf_base64
        )
        
        logger.info(f"Successfully processed {request.filename}")
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing code: {e}")
        logger.exception("Detailed processing error:")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")


@app.post('/test-llm')
async def test_llm_endpoint(request: dict):
    """Test endpoint for LLM functionality."""
    try:
        prompt = request.get("prompt", "Hello, this is a test. Please respond briefly.")
        
        logger.info(f"Testing LLM with prompt: {prompt[:50]}...")
        response = call_llm(prompt, max_tokens=100)
        
        return {
            "status": "success",
            "model": config.effective_llm_model,
            "prompt": prompt,
            "response": response
        }
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM test failed: {str(e)}")

@app.get('/metrics')
async def get_metrics():
    """Get service metrics and statistics."""
    global index, metadatas
    
    metrics = {
        "service": "rag_service_v2",
        "vector_count": index.ntotal if index else 0,
        "metadata_count": len(metadatas) if metadatas else 0,
        "cache_size": config.MAX_CACHE_SIZE,
        "max_chunk_size": config.MAX_CHUNK_SIZE,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.effective_llm_model,
        "llm_provider": config.LLM_PROVIDER
    }
    
    return metrics

##history part 

@app.get('/history')
async def get_history():
    """Get analysis history"""
    try:
        history = history_manager.get_all_history()
        return {
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get('/history/{session_id}')
async def get_history_item(session_id: str):
    """Get specific history item by session ID"""
    try:
        item = history_manager.get_history_item(session_id)
        if not item:
            raise HTTPException(status_code=404, detail="History item not found")
        return item
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history item: {str(e)}")

@app.delete('/history/{session_id}')
async def delete_history_item(session_id: str):
    """Delete specific history item"""
    try:
        success = history_manager.delete_history_item(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="History item not found")
        return {"success": True, "message": "History item deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting history item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete history item: {str(e)}")

@app.delete('/history')
async def clear_history():
    """Clear all history"""
    try:
        success = history_manager.clear_all_history()
        return {"success": success, "message": "All history cleared"}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@app.get('/history/search')
async def search_history(query: str = ""):
    """Search history items"""
    try:
        results = history_manager.search_history(query)
        return {
            "results": results,
            "count": len(results),
            "query": query
        }
    except Exception as e:
        logger.error(f"Error searching history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search history: {str(e)}")

@app.get('/')
async def root():
    """Root endpoint with service information."""
    return {
        "service": "RAG Service v2.0",
        "description": "Code recommendation system with company proxy support",
        "version": "2.0.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "config": "/config",
        "metrics": "/metrics"
    }

if __name__ == '__main__':
    # Get port from config or default
    port = config.SERVER_PORT
    host = config.SERVER_HOST
    
    logger.info(f"🚀 Starting RAG Service v2.0 on {host}:{port}")
    logger.info(f"📊 Using LLM: {config.effective_llm_model}")
    logger.info(f"🔧 Provider: {config.LLM_PROVIDER}")
    if config.effective_base_url:
        logger.info(f"🔗 Base URL: {config.effective_base_url}")
    
    # Use uvicorn to run the server
    uvicorn.run(
        "server:app", 
        host=host, 
        port=port, 
        reload=True,
        log_level="info",
        timeout_keep_alive=36000
    )
