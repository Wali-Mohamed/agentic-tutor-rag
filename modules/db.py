import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# Save logs in the data folder
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_logs.db"

def init_db():
    """Create the tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            user_query TEXT,
            bot_response TEXT,
            feedback INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_chat(user_query, bot_response):
    """Save a conversation and return its ID."""
    log_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO logs (id, timestamp, user_query, bot_response, feedback) VALUES (?, ?, ?, ?, ?)",
        (log_id, datetime.now(), user_query, bot_response, 0) # 0 means no feedback yet
    )
    conn.commit()
    conn.close()
    return log_id

def update_feedback(log_id, feedback_value):
    """Update a specific chat log with 1 (Thumbs Up) or -1 (Thumbs Down)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE logs SET feedback = ? WHERE id = ?", (feedback_value, log_id))
    conn.commit()
    conn.close()