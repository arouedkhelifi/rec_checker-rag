import numpy as np
import requests
import logging
from typing import List, Union
from config_manager import config

logger = logging.getLogger(__name__)

class ProxyEmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.base_url = config.effective_base_url
        self.api_key = config.effective_api_key
        
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts using company proxy embedding API"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                embedding = self._get_embedding_from_proxy(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Embedding failed for text: {e}")
                # Fallback: return zero vector with standard dimension
                embeddings.append(np.zeros(384))  # Adjust dimension as needed
        
        return np.array(embeddings)
    
    def _get_embedding_from_proxy(self, text: str) -> List[float]:
        """Call company proxy for embeddings"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Adjust this payload format based on your proxy's API
        payload = {
            "model": self.model_name,
            "input": text
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",  # Adjust endpoint as needed
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            # Adjust this based on your proxy's response format
            return data["data"][0]["embedding"]
        else:
            raise Exception(f"Embedding API failed: {response.status_code} - {response.text}")