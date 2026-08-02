import sqlite3

def reset_database():
    conn = sqlite3.connect("data/chat_logs.db")
    # This deletes all rows from your logs table but keeps the table structure
    conn.execute("DELETE FROM logs")
    conn.commit()
    conn.close()
    print("Database cleared successfully!")

# Run it once
reset_database()