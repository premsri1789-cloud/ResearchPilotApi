import sqlite3
import os

DB_PATH = "./db/chat_memory.db"

def init_db():
    """Initializes the SQLite database and creates the messages table."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def save_message(session_id: str, role: str, content: str):
    """Saves a single message (user or assistant) to the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()

def get_chat_history(session_id: str, limit: int = 6) -> list:
    """
    Fetches the last N messages for a given session. 
    A limit of 6 means the last 3 turns (3 user questions, 3 AI answers).
    This prevents the LLM context window from overflowing.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        
        # SQLite returns descending (newest first). We reverse it for the LLM prompt.
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]