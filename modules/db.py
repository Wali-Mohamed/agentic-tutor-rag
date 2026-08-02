import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# Save logs in the data folder
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_logs.db"

def init_db():
    """Create the tables if they don't exist and run migrations for new columns."""
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Create the base table (for new installations)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id TEXT PRIMARY KEY,
            timestamp DATETIME,
            user_query TEXT,
            bot_response TEXT,
            feedback INTEGER
        )
    ''')
    
    # 2. Safely add new columns to existing databases (Migration)
    try:
        conn.execute("ALTER TABLE logs ADD COLUMN total_tokens INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Column already exists, do nothing
        
    try:
        conn.execute("ALTER TABLE logs ADD COLUMN llm_judge_rating TEXT DEFAULT 'Not Evaluated'")
    except sqlite3.OperationalError:
        pass # Column already exists, do nothing
        
    conn.commit()
    conn.close()

# Notice the two new optional parameters with default values
def log_chat(user_query, bot_response, total_tokens=0, llm_judge_rating="Not Evaluated"):
    """Save a conversation and return its ID."""
    log_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO logs (id, timestamp, user_query, bot_response, feedback, total_tokens, llm_judge_rating) 
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (log_id, datetime.now(), user_query, bot_response, 0, total_tokens, llm_judge_rating)
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

def update_judge_rating(log_id, rating):
    """Update a specific chat log with the LLM-as-a-Judge rating ('Good' or 'Bad')."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE logs SET llm_judge_rating = ? WHERE id = ?", (rating, log_id))
    conn.commit()
    conn.close()