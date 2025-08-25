"""
Configuration Manager

Centralized configuration management using environment variables
with fallback defaults and validation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """Configuration class that loads settings from environment variables."""
    
    def __init__(self):
        self._validate_required_env_vars()
        
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "16000"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    
    # LiteLLM specific configuration
    LITELLM_MODEL: Optional[str] = os.getenv("LITELLM_MODEL")
    LITELLM_BASE_URL: Optional[str] = os.getenv("LITELLM_BASE_URL")
    LITELLM_API_KEY: Optional[str] = os.getenv("LITELLM_API_KEY")
    
    # Vector Store Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_INDEX_PATH: str = os.getenv("VECTOR_INDEX_PATH", "knowledge_base_flat.index")
    VECTOR_METADATA_PATH: str = os.getenv("VECTOR_METADATA_PATH", "knowledge_base_metadata.json")
    
    # Database Configuration
    ENCRYPTION_SECRET: str = os.getenv("ENCRYPTION_SECRET", "")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "feedback.db")
    SESSION_DB_PATH: str = os.getenv("SESSION_DB_PATH", "session_history.db")
    
    # Server Configuration
    SERVER_HOST: str = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "5000"))
    GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    # Vertex AI Configuration (legacy support)
    VERTEX_PROJECT: Optional[str] = os.getenv("VERTEX_PROJECT")
    VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")
    VERTEX_CREDENTIALS_PATH: Optional[str] = os.getenv("VERTEX_CREDENTIALS_PATH")
    
    # Cache Configuration
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "500"))
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "50000"))
    
    # Application Settings
    MAX_SESSIONS: int = int(os.getenv("MAX_SESSIONS", "10"))
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "English")
    
    # Large file processing settings
    MAX_FILE_SIZE_CHARS: int = int(os.getenv("MAX_FILE_SIZE_CHARS", "1000000"))  # 1M chars
    CHUNK_SIZE_CHARS: int = int(os.getenv("CHUNK_SIZE_CHARS", "50000"))  # 50K chars per chunk
    CHUNK_OVERLAP_CHARS: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "5000"))  # 5K overlap
    MAX_CHUNKS_PER_FILE: int = int(os.getenv("MAX_CHUNKS_PER_FILE", "16000"))  # Max chunks to process
    PARALLEL_CHUNK_PROCESSING: bool = os.getenv("PARALLEL_CHUNK_PROCESSING", "true").lower() == "true"
    
    # Add DEFAULT_LLM property
    @property
    def DEFAULT_LLM(self) -> str:
        """Get the default LLM model identifier for litellm."""
        return self.effective_llm_model
    
    def _validate_required_env_vars(self):
        """Validate that required environment variables are set."""
        required_vars = []
        
        # Check LLM configuration
        if not os.getenv("LLM_API_KEY") and not os.getenv("LITELLM_API_KEY"):
            required_vars.append("LLM_API_KEY or LITELLM_API_KEY")
            
        # Check encryption secret
        if not os.getenv("ENCRYPTION_SECRET"):
            required_vars.append("ENCRYPTION_SECRET")
            
        if required_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(required_vars)}")
    
    @property
    def effective_llm_model(self) -> str:
        """Get the effective LLM model to use."""
        if self.LITELLM_MODEL:
            return self.LITELLM_MODEL
        elif self.LLM_PROVIDER == "openai":
            return f"openai/{self.LLM_MODEL}"
        elif self.LLM_PROVIDER == "anthropic":
            return f"anthropic/{self.LLM_MODEL}"
        elif self.LLM_PROVIDER == "vertex_ai":
            return f"vertex_ai/{self.LLM_MODEL}"
        else:
            return self.LLM_MODEL
    
    @property
    def effective_api_key(self) -> str:
        """Get the effective API key to use."""
        return self.LITELLM_API_KEY or self.LLM_API_KEY
    
    @property
    def effective_base_url(self) -> Optional[str]:
        """Get the effective base URL to use."""
        return self.LITELLM_BASE_URL or self.LLM_BASE_URL
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration as a dictionary."""
        config = {
            "model": self.effective_llm_model,
            "max_tokens": self.LLM_MAX_TOKENS,
            "temperature": self.LLM_TEMPERATURE,
        }
        
        # Add API key if available
        if self.effective_api_key:
            config["api_key"] = self.effective_api_key
            
        # Add base URL if available
        if self.effective_base_url:
            config["base_url"] = self.effective_base_url
            
        return config
    
    def setup_litellm(self):
        """Setup litellm with the current configuration."""
        import litellm
        
        # Set API key
        if self.effective_api_key:
            if self.LLM_PROVIDER == "openai" or self.LITELLM_MODEL and self.LITELLM_MODEL.startswith("openai/"):
                os.environ["OPENAI_API_KEY"] = self.effective_api_key
            elif self.LLM_PROVIDER == "anthropic" or self.LITELLM_MODEL and self.LITELLM_MODEL.startswith("anthropic/"):
                os.environ["ANTHROPIC_API_KEY"] = self.effective_api_key
        
        # Set base URL if using a proxy
        if self.effective_base_url:
            if self.LLM_PROVIDER == "openai" or self.LITELLM_MODEL and self.LITELLM_MODEL.startswith("openai/"):
                os.environ["OPENAI_BASE_URL"] = self.effective_base_url
        
        # Setup Vertex AI if needed
        if self.LLM_PROVIDER == "vertex_ai" or (self.LITELLM_MODEL and self.LITELLM_MODEL.startswith("vertex_ai/")):
            if self.VERTEX_CREDENTIALS_PATH and os.path.exists(self.VERTEX_CREDENTIALS_PATH):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.VERTEX_CREDENTIALS_PATH
            if self.VERTEX_PROJECT:
                litellm.vertex_project = self.VERTEX_PROJECT
                litellm.vertex_location = self.VERTEX_LOCATION
        
        logger.info(f"LiteLLM configured with model: {self.effective_llm_model}")
        if self.effective_base_url:
            logger.info(f"Using base URL: {self.effective_base_url}")

# Global configuration instance
config = Config()
