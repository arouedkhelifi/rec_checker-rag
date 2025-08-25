# encryption_utils.py

from cryptography.fernet import Fernet
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load env variables ===
load_dotenv()
SECRET_KEY = os.environ.get("ENCRYPTION_SECRET")

if not SECRET_KEY:
    # Generate a key if not provided
    logger.warning("ENCRYPTION_SECRET not found, generating a new one")
    SECRET_KEY = Fernet.generate_key().decode()
    logger.info(f"Generated new encryption key: {SECRET_KEY}")
    # You might want to save this to .env file

try:
    fernet = Fernet(SECRET_KEY.encode())
except Exception as e:
    logger.error(f"Invalid encryption key: {e}")
    # Generate a valid key as fallback
    SECRET_KEY = Fernet.generate_key().decode()
    fernet = Fernet(SECRET_KEY.encode())
    logger.info(f"Using fallback encryption key: {SECRET_KEY}")

# === Save history to file ===
def save_encrypted_history(history, file_path):
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Encrypt and save
        encrypted = fernet.encrypt(json.dumps(history).encode())
        with open(file_path, 'wb') as f:
            f.write(encrypted)
        logger.info(f"Saved encrypted history with {len(history)} sessions to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save history: {e}")
        return False

# === Load history from file ===
def decrypt_history(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.info(f"History file not found: {file_path}, returning empty history")
            return []
        
        # Read and decrypt
        with open(file_path, 'rb') as f:
            encrypted = f.read()
        
        decrypted = fernet.decrypt(encrypted)
        history = json.loads(decrypted)
        logger.info(f"Loaded history with {len(history)} sessions from {file_path}")
        return history
    except Exception as e:
        logger.error(f"Failed to load history from {file_path}: {e}")
        return []
