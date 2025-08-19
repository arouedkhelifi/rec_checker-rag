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

# Import updated modules
from config_manager import config
from llm_client import llm_client, call_llm
from rag_generate import (
    detect_language,
    analyze_code_metrics,
    analyze_dependencies,
    load_resources,
    mmr_search,
    retrieve_node,
    generate_node,
    build_agent,
    AgentState
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

class CodeProcessResponse(BaseModel):
    recommendations: str
    language: str
    metrics: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, Any]] = None

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

@app.post('/process', response_model=CodeProcessResponse)
async def process_code(request: CodeProcessRequest):
    """
    Process code and generate recommendations using the updated system.
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
        
        # Detect language
        language = detect_language(request.code, request.filename)
        logger.info(f"Detected language: {language} for file {request.filename}")
        
        # Create initial state
        initial_state: AgentState = {
            "code": request.code,
            "code_language": language,
            "code_filename": request.filename,
            "retrieved_chunks": [],
            "answer": "",
            "metrics": None,
            "dependencies": None,
            "error": None,
            "target_language": request.target_language
        }
        
        # Process through retrieve node with explicit resource passing
        logger.info(f"Running retrieve node for {request.filename}")
        state_after_retrieve = custom_retrieve_node(initial_state, index, metadatas, embedding_model)
        
        # Process through generate node
        logger.info(f"Running generate node for {request.filename}")
        final_state = generate_node(state_after_retrieve)
        
        # Check for errors
        if final_state.get("error"):
            logger.error(f"Processing error: {final_state['error']}")
            raise HTTPException(status_code=500, detail=final_state["error"])
        
        # Prepare response
        response = CodeProcessResponse(
            recommendations=final_state["answer"],
            language=language,
            metrics=final_state.get("metrics", {}),
            dependencies=final_state.get("dependencies", {})
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

# Add these endpoints to your existing server.py

@app.get('/history')
async def get_history():
    """Get analysis history from encrypted storage"""
    try:
        from encryption_utils import decrypt_history
        
        history = decrypt_history()
        
        # Clean and format history for API response
        formatted_history = []
        for item in history:
            if isinstance(item, dict) and 'id' in item:
                formatted_history.append({
                    "id": item.get("id"),
                    "filename": item.get("id", "unknown"), # id is usually filename
                    "recommendations": item.get("recommendations", ""),
                    "timestamp": item.get("timestamp", ""),
                    "code_preview": item.get("code", "")[:100] + "..." if item.get("code") else "",
                    "pdf_path": item.get("pdf_path")
                })
        
        return {
            "history": formatted_history,
            "count": len(formatted_history)
        }
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get('/history/{session_id}')
async def get_history_item(session_id: str):
    """Get specific history item by session ID"""
    try:
        from encryption_utils import decrypt_history
        
        history = decrypt_history()
        
        # Find item by ID
        for item in history:
            if isinstance(item, dict) and item.get("id") == session_id:
                return {
                    "id": item.get("id"),
                    "filename": item.get("id"),
                    "recommendations": item.get("recommendations", ""),
                    "code": item.get("code", ""),
                    "pdf_path": item.get("pdf_path"),
                    "timestamp": item.get("timestamp", "")
                }
        
        raise HTTPException(status_code=404, detail="History item not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history item: {str(e)}")

@app.delete('/history/{session_id}')
async def delete_history_item(session_id: str):
    """Delete specific history item"""
    try:
        from encryption_utils import decrypt_history, save_encrypted_history
        
        history = decrypt_history()
        
        # Remove item with matching ID
        updated_history = [item for item in history if item.get("id") != session_id]
        
        if len(updated_history) == len(history):
            raise HTTPException(status_code=404, detail="History item not found")
        
        # Save updated history
        save_encrypted_history(updated_history)
        
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
        from encryption_utils import save_encrypted_history
        
        save_encrypted_history([])  # Save empty history
        
        return {"success": True, "message": "All history cleared"}
        
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")
    
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
        log_level="info"
    )