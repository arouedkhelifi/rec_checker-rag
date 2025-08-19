# encryption_utils.py

from cryptography.fernet import Fernet
import json
import os
import sqlite3
from dotenv import load_dotenv

# === Load env variables ===
load_dotenv()
SECRET_KEY = os.environ.get("ENCRYPTION_SECRET")

if not SECRET_KEY:
    raise ValueError("ENCRYPTION_SECRET is not set. Please define it in your .env file.")

fernet = Fernet(SECRET_KEY.encode())

DB_PATH = "session_history.db"
MAX_SESSIONS = 10

# === Init the SQLite DB ===
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                encrypted_data BLOB NOT NULL
            )
        """)
        # Insert empty history if not already set
        cursor.execute("INSERT OR IGNORE INTO session_history (id, encrypted_data) VALUES (1, ?)",
                       (fernet.encrypt(json.dumps([]).encode()),))
        conn.commit()

init_db()

# === Save history ===
def save_encrypted_history(history):
    history = history[:MAX_SESSIONS]
    encrypted = fernet.encrypt(json.dumps(history).encode())
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE session_history SET encrypted_data = ? WHERE id = 1", (encrypted,))
        conn.commit()
    print(f"[DEBUG] Saved encrypted history with {len(history)} sessions")

# === Load history ===
def decrypt_history():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT encrypted_data FROM session_history WHERE id = 1")
            row = cursor.fetchone()
            if not row:
                return []
            decrypted = fernet.decrypt(row[0])
            history = json.loads(decrypted)
            print(f"[DEBUG] Loaded history with {len(history)} sessions")
            return history
    except Exception as e:
        print(f"[DEBUG] Failed to load history: {e}")
        return []
