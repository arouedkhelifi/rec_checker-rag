"""
LLM Client with Proxy Support

Unified LLM client that supports both OpenAI and LiteLLM
with company proxy configuration.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Iterator, Union
from config_manager import config

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified LLM client supporting multiple providers and proxies."""
    
    def __init__(self, use_litellm: bool = True):
        """
        Initialize LLM client.
        
        Args:
            use_litellm: Whether to use litellm (recommended) or direct OpenAI client
        """
        self.use_litellm = use_litellm
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the LLM client based on configuration."""
        if self.use_litellm:
            self._setup_litellm()
        else:
            self._setup_openai()
    
    def _setup_litellm(self):
        """Setup LiteLLM client."""
        try:
            import litellm
            
            # Configure litellm with our settings
            config.setup_litellm()
            
            # Test the connection
            self._test_connection()
            logger.info("✅ LiteLLM client setup successful")
            
        except ImportError:
            logger.error("❌ LiteLLM not installed. Run: pip install litellm")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to setup LiteLLM: {e}")
            raise
    
    def _setup_openai(self):
        """Setup direct OpenAI client (for company proxies)."""
        try:
            from openai import OpenAI
            
            client_config = {}
            
            # Add API key
            if config.effective_api_key:
                client_config["api_key"] = config.effective_api_key
            
            # Add base URL for proxy
            if config.effective_base_url:
                client_config["base_url"] = config.effective_base_url
            
            self.client = OpenAI(**client_config)
            
            # Test the connection
            self._test_connection_openai()
            logger.info("✅ OpenAI client setup successful")
            
        except ImportError:
            logger.error("❌ OpenAI library not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to setup OpenAI client: {e}")
            raise
    
    def _test_connection(self):
        """Test LiteLLM connection."""
        try:
            import litellm
            response = litellm.completion(
                model=config.effective_llm_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=10
            )
            logger.info("✅ LLM connection test successful")
        except Exception as e:
            logger.warning(f"⚠️ LLM connection test failed: {e}")
            # Don't raise here, let the actual usage fail if needed
    
    def _test_connection_openai(self):
        """Test OpenAI client connection."""
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
                timeout=10
            )
            logger.info("✅ OpenAI connection test successful")
        except Exception as e:
            logger.warning(f"⚠️ OpenAI connection test failed: {e}")
            # Don't raise here, let the actual usage fail if needed
    
    def completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generate completion using the configured LLM.
        
        Args:
            messages: List of message dictionaries
            stream: Whether to stream the response
            **kwargs: Additional parameters for the LLM
            
        Returns:
            Response dictionary or iterator for streaming
        """
        # Merge with default config
        params = {
            "messages": messages,
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "stream": stream,
            **kwargs
        }
        
        if self.use_litellm:
            return self._completion_litellm(**params)
        else:
            return self._completion_openai(**params)
    
    def _completion_litellm(self, **params) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Generate completion using LiteLLM."""
        try:
            import litellm
            
            # Add model to params
            params["model"] = config.effective_llm_model
            
            response = litellm.completion(**params)
            return response
            
        except Exception as e:
            logger.error(f"❌ LiteLLM completion failed: {e}")
            raise
    
    def _completion_openai(self, **params) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Generate completion using OpenAI client."""
        try:
            # Remove 'model' from params and use configured model
            params.pop("model", None)
            
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                **params
            )
            
            # Convert to LiteLLM-compatible format
            if params.get("stream", False):
                return self._convert_openai_stream(response)
            else:
                return self._convert_openai_response(response)
                
        except Exception as e:
            logger.error(f"❌ OpenAI completion failed: {e}")
            raise
   
   # In llm_client.py, update the _convert_openai_response method:
    def _convert_openai_response(self, response) -> Dict[str, Any]:
        """Convert OpenAI response to LiteLLM format."""
        try:
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content,
                        "role": response.choices[0].message.role
                    },
                    "finish_reason": response.choices[0].finish_reason,
                    "index": response.choices[0].index
                }],
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                } if hasattr(response, 'usage') and response.usage else {}
            }
        except Exception as e:
            logger.warning(f"Response conversion error: {e}")
            # Fallback format
            return {
                "choices": [{
                    "message": {
                        "content": str(response),
                        "role": "assistant"
                    }
                }]
            }
     
    def _convert_openai_stream(self, response) -> Iterator[Dict[str, Any]]:
        """Convert OpenAI streaming response to LiteLLM format."""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield {
                    "choices": [{
                        "delta": {
                            "content": chunk.choices[0].delta.content
                        },
                        "index": 0
                    }]
                }

# Global LLM client instance
llm_client = LLMClient(use_litellm=True)

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    stream: bool = False,
    **kwargs
) -> Union[str, Iterator[str]]:
    """
    Simplified function to call LLM with a prompt.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        stream: Whether to stream the response
        **kwargs: Additional parameters
        
    Returns:
        Response content or iterator for streaming
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = llm_client.completion(messages=messages, stream=stream, **kwargs)
        
        if stream:
            def extract_content():
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = getattr(chunk.choices[0], 'delta', None)
                        if delta and getattr(delta, 'content', None):
                            yield delta.content
                    elif isinstance(chunk, dict) and "choices" in chunk:
                        content = chunk["choices"][0].get("delta", {}).get("content")
                        if content:
                            yield content
            return extract_content()
        else:
            # Handle both LiteLLM and OpenAI response formats
            if hasattr(response, 'choices'):
                return response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                return "Error: Unexpected response format"
                
    except Exception as e:
        logger.error(f"❌ LLM call failed: {e}")
        return f"Error: {str(e)}"

def call_llm_stream(prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Iterator[str]:
    """
    Stream LLM response with guaranteed completion.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        **kwargs: Additional parameters
        
    Returns:
        Iterator of response chunks
    """
    # Set default parameters if not provided
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 32000  # Very large token limit
    
    if "timeout" not in kwargs:
        kwargs["timeout"] = 300  # 5 minute timeout
    
    if "top_p" not in kwargs:
        kwargs["top_p"] = 0.9  # Consistent top_p setting
    
    # Call the underlying LLM function with stream=True
    return call_llm(prompt, system_prompt, stream=True, **kwargs)
