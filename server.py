from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os
import json
import logging

# Import your existing RAG code
from rag_generate import (
    generate_node, 
    detect_language,
    analyze_code_metrics,
    analyze_dependencies,
    load_resources,
    mmr_search,
    DEFAULT_INDEX_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL,
    AgentState
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

def custom_retrieve_node(state: AgentState, index_ref, metadatas_ref, embedding_model_ref) -> AgentState:
    """
    Custom retrieve node that accepts resources as parameters instead of using globals.
    
    Args:
        state (AgentState): Current agent state containing code and language info.
        index_ref: FAISS index object
        metadatas_ref: Metadata list
        embedding_model_ref: Embedding model object
    
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
        # Perform MMR search on index to get relevant chunks - using passed resources
        retrieved = mmr_search(query, language, index_ref, metadatas_ref, embedding_model_ref, top_k=7)
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
        
        # Process through our custom retrieve node with explicit resource passing
        logger.info(f"Running retrieve node for {request.filename}")
        state_after_retrieve = custom_retrieve_node(initial_state, index, metadatas, embedding_model)
        
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
    port = int(5000)
    # Use uvicorn directly instead of app.run()
    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=True)