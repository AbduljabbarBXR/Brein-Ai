import sqlite3
import os

# Check if database exists
db_path = "backend/chats/brein_chats.db"
if not os.path.exists(db_path):
    print("Database file not found")
    exit(1)

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get total sessions
cursor.execute('SELECT COUNT(*) FROM chat_sessions')
total_sessions = cursor.fetchone()[0]
print(f"Total chat sessions: {total_sessions}")

# Get recent sessions
cursor.execute('SELECT id, title, message_count, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT 10')
rows = cursor.fetchall()

print("\nRecent sessions:")
for row in rows:
    session_id, title, msg_count, created, updated = row
    print(f"  {session_id[:8]}... - {title or 'Untitled'} - {msg_count} messages - {updated}")

# Check messages table
cursor.execute('SELECT COUNT(*) FROM chat_messages')
total_messages = cursor.fetchone()[0]
print(f"\nTotal messages: {total_messages}")

conn.close()
