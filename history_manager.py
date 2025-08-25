"""
History Manager for Code Recommendation Checker

This module centralizes all history-related operations including:
- Saving analysis sessions
- Loading past sessions
- Deleting sessions
- Searching and filtering history
"""

import os
import json
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Import your existing encryption utilities
from encryption_utils import decrypt_history, save_encrypted_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY_ITEMS = 100  # Maximum number of history items to keep
HISTORY_FILE_PATH = os.environ.get("HISTORY_FILE_PATH", "data/history.enc")


class HistoryManager:
    """Manages code analysis history with encryption support"""
    
    def __init__(self, history_file_path: str = None):
        """Initialize the history manager with optional custom file path"""
        self.history_file_path = history_file_path or HISTORY_FILE_PATH
        self._ensure_history_dir()
    
    def _ensure_history_dir(self):
        """Ensure the directory for history file exists"""
        history_dir = os.path.dirname(self.history_file_path)
        if history_dir and not os.path.exists(history_dir):
            os.makedirs(history_dir, exist_ok=True)
            logger.info(f"Created history directory: {history_dir}")
    
    def get_all_history(self) -> List[Dict[str, Any]]:
        """Get all history items, sorted by timestamp (newest first)"""
        try:
            history = decrypt_history(self.history_file_path)
            
            # Sort by timestamp (newest first)
            sorted_history = sorted(
                history, 
                key=lambda x: x.get('timestamp', '0'), 
                reverse=True
            )
            
            # Format for frontend display
            formatted_history = []
            for item in sorted_history:
                if isinstance(item, dict) and 'id' in item:
                    formatted_history.append({
                        "id": item.get("id"),
                        "filename": item.get("filename", item.get("id", "unknown")),
                        "recommendations": item.get("recommendations", ""),
                        "timestamp": item.get("timestamp", ""),
                        "code_preview": item.get("code", "")[:100] + "..." if item.get("code") else "",
                        "pdf_path": item.get("pdf_path", "")
                    })
            
            return formatted_history
            
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []
    
    def get_history_item(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific history item by session ID"""
        try:
            history = decrypt_history(self.history_file_path)
            
            # Find item by ID
            for item in history:
                if isinstance(item, dict) and item.get("id") == session_id:
                    return {
                        "id": item.get("id"),
                        "filename": item.get("filename", item.get("id")),
                        "recommendations": item.get("recommendations", ""),
                        "code": item.get("code", ""),
                        "pdf_path": item.get("pdf_path", ""),
                        "timestamp": item.get("timestamp", ""),
                        "language": item.get("language", "unknown"),
                        "metrics": item.get("metrics", {})
                    }
            
            logger.warning(f"History item not found: {session_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting history item: {e}")
            return None
    
    def save_session(self, 
                    code: str, 
                    recommendations: str, 
                    filename: str = None,
                    language: str = "unknown", 
                    metrics: Dict = None,
                    pdf_path: str = None) -> str:
        """
        Save a new analysis session to history
        
        Args:
            code: The analyzed code
            recommendations: The generated recommendations
            filename: The name of the analyzed file
            language: The programming language
            metrics: Code metrics and analysis data
            pdf_path: Path to generated PDF report
            
        Returns:
            session_id: The ID of the saved session
        """
        try:
            # Generate a unique session ID
            timestamp = datetime.now().isoformat()
            session_id = self._generate_session_id(code, timestamp)
            
            # Create history item
            history_item = {
                "id": session_id,
                "filename": filename or f"analysis_{session_id[:8]}",
                "code": code,
                "recommendations": recommendations,
                "language": language,
                "metrics": metrics or {},
                "timestamp": timestamp,
                "pdf_path": pdf_path
            }
            
            # Get existing history
            history = decrypt_history(self.history_file_path)
            
            # Add new item at the beginning
            history.insert(0, history_item)
            
            # Limit history size
            if len(history) > MAX_HISTORY_ITEMS:
                history = history[:MAX_HISTORY_ITEMS]
                logger.info(f"Trimmed history to {MAX_HISTORY_ITEMS} items")
            
            # Save updated history
            save_encrypted_history(history, self.history_file_path)
            logger.info(f"Saved session to history: {session_id}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error saving session to history: {e}")
            return ""
    
    def delete_history_item(self, session_id: str) -> bool:
        """Delete a specific history item by session ID"""
        try:
            history = decrypt_history(self.history_file_path)
            
            # Find and remove item with matching ID
            original_length = len(history)
            updated_history = [item for item in history if item.get("id") != session_id]
            
            if len(updated_history) == original_length:
                logger.warning(f"History item not found for deletion: {session_id}")
                return False
            
            # Save updated history
            save_encrypted_history(updated_history, self.history_file_path)
            logger.info(f"Deleted history item: {session_id}")
            
            # Clean up PDF file if it exists
            for item in history:
                if item.get("id") == session_id and item.get("pdf_path"):
                    pdf_path = item.get("pdf_path")
                    if os.path.exists(pdf_path):
                        try:
                            os.remove(pdf_path)
                            logger.info(f"Deleted PDF file: {pdf_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete PDF file {pdf_path}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting history item: {e}")
            return False
    
    def clear_all_history(self) -> bool:
        """Clear all history items"""
        try:
            # Get current history to clean up PDF files
            history = decrypt_history(self.history_file_path)
            
            # Clean up PDF files
            for item in history:
                if item.get("pdf_path"):
                    pdf_path = item.get("pdf_path")
                    if os.path.exists(pdf_path):
                        try:
                            os.remove(pdf_path)
                            logger.debug(f"Deleted PDF file: {pdf_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete PDF file {pdf_path}: {e}")
            
            # Save empty history
            save_encrypted_history([], self.history_file_path)
            logger.info("Cleared all history")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
    
    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search history items by filename or code content"""
        try:
            history = self.get_all_history()
            
            if not query or not query.strip():
                return history
            
            query = query.lower()
            results = []
            
            for item in history:
                # Check filename
                if item.get("filename", "").lower().find(query) >= 0:
                    results.append(item)
                    continue
                
                # Check code preview
                if item.get("code_preview", "").lower().find(query) >= 0:
                    results.append(item)
                    continue
                
                # Get full item to check code content
                full_item = self.get_history_item(item.get("id"))
                if full_item and full_item.get("code", "").lower().find(query) >= 0:
                    results.append(item)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching history: {e}")
            return []
    
    def _generate_session_id(self, code: str, timestamp: str) -> str:
        """Generate a unique session ID based on code content and timestamp"""
        content = f"{code[:1000]}_{timestamp}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()


# Create a singleton instance
history_manager = HistoryManager()
