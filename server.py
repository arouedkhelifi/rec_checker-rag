from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os
import json
import logging

# Import your existing RAG code
from rag_generate import (
    retrieve_node, 
    generate_node, 
    detect_language,
    analyze_code_metrics,
    analyze_dependencies,
    load_resources,
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL,
    AgentState,
    mmr_search  # Import mmr_search to patch it if needed
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Service", 
    description="LangGraph RAG system for code recommendations",
    version="1.0.0"
)

# Define request and response models with Pydantic
class CodeProcessRequest(BaseModel):
    code: str
    filename: Optional[str] = "code_snippet.txt"
    target_language: Optional[str] = "English"

class CodeProcessResponse(BaseModel):
    recommendations: str
    language: str
    metrics: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, Any]] = None

# Global variables for resources
index = None
metadatas = None
embedding_model = None

# Load resources on startup
@app.on_event("startup")
async def startup_event():
    try:
        global index, metadatas, embedding_model
        logger.info(f"Loading resources from {DEFAULT_INDEX_PATH} and {DEFAULT_METADATA_PATH}")
        index, metadatas, embedding_model = load_resources(
            DEFAULT_INDEX_PATH, 
            DEFAULT_METADATA_PATH, 
            DEFAULT_MODEL
        )
        logger.info(f"✅ Resources loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load resources: {e}")
        logger.exception("Detailed error:")

@app.get('/health')
async def health_check():
    """Health check endpoint to verify service is running"""
    global index, metadatas, embedding_model
    
    resources_loaded = index is not None and metadatas is not None and embedding_model is not None
    
    return {
        "status": "healthy" if resources_loaded else "degraded", 
        "service": "rag_service",
        "resources_loaded": resources_loaded
    }

@app.post('/process', response_model=CodeProcessResponse)
async def process_code(request: CodeProcessRequest):
    """
    Process code and generate recommendations
    """
    try:
        global index, metadatas, embedding_model
        
        # Check if resources are loaded
        if index is None or metadatas is None or embedding_model is None:
            logger.error("Resources not loaded, attempting to load them now")
            # Try to load resources
            index, metadatas, embedding_model = load_resources(
                DEFAULT_INDEX_PATH, 
                DEFAULT_METADATA_PATH, 
                DEFAULT_MODEL
            )
            logger.info("Resources loaded successfully during request")
        
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
        
        # Create a custom retrieve_node function that has access to our resources
        def custom_retrieve_node(state: AgentState) -> AgentState:
            """
            A wrapper around retrieve_node that provides the necessary resources
            """
            # Create a patched mmr_search function that uses our loaded resources
            original_mmr_search = mmr_search
            
            def patched_mmr_search(query, code_language, *args, **kwargs):
                # Use our loaded resources
                return original_mmr_search(query, code_language, index, metadatas, embedding_model, *args, **kwargs)
            
            # Temporarily replace the mmr_search function
            import rag_generate
            original_func = rag_generate.mmr_search
            rag_generate.mmr_search = patched_mmr_search
            
            try:
                # Call the original retrieve_node
                return retrieve_node(state)
            finally:
                # Restore the original function
                rag_generate.mmr_search = original_func
        
        # Process through our custom retrieve node
        logger.info(f"Running retrieve node for {request.filename}")
        state_after_retrieve = custom_retrieve_node(initial_state)
        
        # Process through generate node
        logger.info(f"Running generate node for {request.filename}")
        final_state = generate_node(state_after_retrieve)
        
        # Check for errors
        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])
        
        # Prepare response
        response = {
            "recommendations": final_state["answer"],
            "language": language,
            "metrics": final_state.get("metrics", {}),
            "dependencies": final_state.get("dependencies", {})
        }
        
        logger.info(f"Successfully processed {request.filename}")
        return response
    
    except Exception as e:
        logger.error(f"Error processing code: {e}")
        logger.exception("Detailed error:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    """Redirect to API documentation"""
    return {
        "message": "Welcome to the RAG Service API",
        "documentation": "/docs",
        "redoc": "/redoc"
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Use uvicorn directly instead of app.run()
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True)