import sqlite3
import os

DB_FILE = "feedback.db"

def init_feedback_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT,
            filename TEXT,
            feedback TEXT,
            rating TEXT,
            relevance TEXT,
            sentiment TEXT,
            decision TEXT,
            validated INTEGER DEFAULT 1,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def store_feedback(file_hash, filename, feedback, rating, relevance, sentiment, decision):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (file_hash, filename, feedback, rating, relevance, sentiment, decision)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (file_hash, filename, feedback, rating, relevance, sentiment, decision))
    conn.commit()
    conn.close()

def get_past_feedback_for_file(filename):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT feedback FROM feedback
        WHERE filename = ? AND decision = "Accept"
        ORDER BY timestamp DESC LIMIT 5
    ''', (filename,))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    return results

def view_feedback():
    """Print all feedback entries (for debugging/demo)."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM feedback ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    for row in rows:
        print(row)

# Commented out reset function — run manually only if needed

def reset_feedback_table():
    '''⚠️ Run this ONCE to delete and recreate the feedback table with correct schema.'''
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("🧹 Old database removed.")
    init_feedback_db()
    print("✅ New feedback database initialized.")


# Do NOT run reset_feedback_table automatically during normal runtime
if __name__ == "__main__":
    reset_feedback_table()
