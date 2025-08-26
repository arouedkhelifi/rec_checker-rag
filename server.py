"""
Updated FastAPI Server with Environment Configuration Support

This server uses the new configuration system and LLM client
for company proxy support.
"""
from feedback_utils import analyze_feedback, submit_user_feedback
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
    process_large_file_upload_with_resources,
    custom_retrieve_node_safe,
    combine_chunk_results,
    calculate_combined_metrics,
    process_code_and_generate,  # <-- Use this instead of gradio_wrapper
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


@app.on_event("startup")
async def startup_event():
    global index, metadatas, embedding_model, agent
    try:
        logger.info("🔧 Testing LLM connection...")
        test_response = call_llm("Hello, this is a connection test. Please respond with 'Connection OK'.")
        logger.info(f"✅ LLM test response: {test_response}")
        logger.info(f"📦 Loading resources from {config.VECTOR_INDEX_PATH} and {config.VECTOR_METADATA_PATH}")
        index, metadatas, embedding_model = load_resources(
            config.VECTOR_INDEX_PATH, 
            config.VECTOR_METADATA_PATH, 
            config.EMBEDDING_MODEL
        )
        logger.info(f"✅ Vector resources loaded successfully: {index.ntotal} vectors")
        agent = build_agent()
        logger.info("✅ Agent workflow built successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize resources: {e}")
        logger.exception("Detailed startup error:")


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

@app.post('/process', response_model=CodeProcessResponse)
async def process_code(request: CodeProcessRequest):
    try:
        global index, metadatas, embedding_model, agent

        # Ensure resources are loaded
        if index is None or metadatas is None or embedding_model is None or agent is None:
            logger.error("Resources not loaded, attempting to load them now")
            index, metadatas, embedding_model = load_resources(
                config.VECTOR_INDEX_PATH,
                config.VECTOR_METADATA_PATH,
                config.EMBEDDING_MODEL
            )
            agent = build_agent()
            logger.info("Resources loaded successfully during request")

        logger.info(f"Processing code with filename: {request.filename}")

        # Use the new main entrypoint
        result = process_code_and_generate(
            file=None,
            code_text=request.code,
            target_language=request.target_language
        )

        recommendations = result["recommendations"]
        metrics = result.get("metrics", {})
        pdf_path = result.get("pdf_path")

        # PDF handling (optional)
        pdf_base64 = None
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            try:
                os.unlink(pdf_path)
            except Exception:
                pass

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

        response = CodeProcessResponse(
            recommendations=recommendations,
            language=language,
            metrics=metrics,
            dependencies=metrics.get("dependencies", {}),
            pdf_base64=pdf_base64
        )
        logger.info(f"Successfully processed {request.filename}")
        return response

    except HTTPException:
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
