import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import GGUFModelLoader

class ChatManager:
    """
    Manages chat sessions and messages with persistent storage.
    """

    def __init__(self, db_path: str = "chats/brein_chats.db", prompt_manager=None):
        self.db_path = db_path
        self.model_loader = GGUFModelLoader()
        self.prompt_manager = prompt_manager
        self._ensure_db_exists()
        self._create_tables()

    def set_prompt_manager(self, prompt_manager):
        """Set the prompt manager after initialization"""
        self.prompt_manager = prompt_manager

    def _ensure_db_exists(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _create_tables(self):
        """Create the necessary database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT, -- 'user' or 'ai'
                    content TEXT,
                    thought_trace TEXT,
                    memory_stats TEXT, -- JSON string
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                )
            ''')

            conn.commit()

    def create_chat_session(self, session_id: Optional[str] = None, title: Optional[str] = None) -> str:
        """Create a new chat session and return its ID."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT OR IGNORE INTO chat_sessions (id, title) VALUES (?, ?)',
                (session_id, title)
            )
            conn.commit()

        return session_id

    def generate_smart_title(self, first_query: str) -> str:
        """Generate a smart title for the chat based on the first query."""
        try:
            # Use prompt manager for title generation
            if self.prompt_manager:
                prompt = self.prompt_manager.get_prompt("system.chat_management.smart_title_generation",
                                                      first_query=first_query)
            else:
                # Fallback to hardcoded prompt
                prompt = f"Generate a very short, descriptive title (3-6 words) for a conversation that starts with this question: '{first_query}'. Make it sound like a topic or request summary."

            title = self.model_loader.generate(
                "llama-3.2",
                prompt,
                max_tokens=15,  # Reduced from 20
                temperature=0.2  # Reduced for more consistent output
            ).strip()

            # Clean up the title - remove quotes, extra whitespace, and common prefixes
            title = title.strip('"').strip("'").strip()
            title = title.lstrip('Title:').lstrip('Topic:').lstrip('Subject:').strip()

            # Validate the generated title
            if not title or len(title) > 40 or len(title.split()) > 6:
                # Fallback to simple extraction
                title = self._extract_simple_title(first_query)

            return title

        except Exception as e:
            # Fallback to simple extraction
            return self._extract_simple_title(first_query)

    def _extract_simple_title(self, query: str) -> str:
        """Extract a simple, clean title from a query."""
        # Remove common question words and punctuation
        query = query.lower().strip()
        query = query.replace('?', '').replace('!', '').replace('.', '')

        # Remove common prefixes
        prefixes_to_remove = ['what', 'how', 'why', 'when', 'where', 'who', 'can you', 'tell me', 'explain', 'describe']
        words = query.split()

        # Remove prefix words
        while words and words[0] in prefixes_to_remove:
            words.pop(0)

        # Take first 4-6 meaningful words
        if not words:
            return "General Inquiry"

        # Capitalize first letter of each word
        title_words = []
        for word in words[:5]:  # Take up to 5 words
            if len(word) > 2:  # Skip very short words
                title_words.append(word.capitalize())

        title = " ".join(title_words)

        # Ensure reasonable length
        if len(title) > 35:
            title = title[:32] + "..."

        return title or "Chat Inquiry"

    def add_message(self, session_id: str, role: str, content: str,
                   thought_trace: Optional[str] = None,
                   memory_stats: Optional[Dict] = None):
        """Add a message to a chat session."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert the message
            conn.execute(
                '''INSERT INTO chat_messages (session_id, role, content, thought_trace, memory_stats)
                   VALUES (?, ?, ?, ?, ?)''',
                (session_id, role, content, thought_trace, json.dumps(memory_stats) if memory_stats else None)
            )

            # Update session metadata
            conn.execute(
                '''UPDATE chat_sessions
                   SET updated_at = CURRENT_TIMESTAMP,
                       message_count = message_count + 1
                   WHERE id = ?''',
                (session_id,)
            )

            conn.commit()

    def get_chat_sessions(self) -> List[Dict]:
        """Get all chat sessions ordered by last updated."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''SELECT id, title, created_at, updated_at, message_count
                   FROM chat_sessions
                   ORDER BY updated_at DESC'''
            )

            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "id": row["id"],
                    "title": row["title"] or "Untitled Chat",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": row["message_count"]
                })

            return sessions

    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get the full message history for a chat session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''SELECT role, content, thought_trace, memory_stats, timestamp
                   FROM chat_messages
                   WHERE session_id = ?
                   ORDER BY timestamp ASC''',
                (session_id,)
            )

            messages = []
            for row in cursor.fetchall():
                message = {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"]
                }

                if row["thought_trace"]:
                    message["thought_trace"] = row["thought_trace"]

                if row["memory_stats"]:
                    try:
                        message["memory_stats"] = json.loads(row["memory_stats"])
                    except:
                        pass

                messages.append(message)

            return messages

    def update_chat_title(self, session_id: str, title: str):
        """Update the title of a chat session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (title, session_id)
            )
            conn.commit()

    def delete_chat_session(self, session_id: str):
        """Delete a chat session and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            # Messages will be deleted automatically due to CASCADE constraint
            conn.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
            conn.commit()

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a specific chat session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT id, title, created_at, updated_at, message_count FROM chat_sessions WHERE id = ?',
                (session_id,)
            )

            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "title": row["title"] or "Untitled Chat",
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": row["message_count"]
                }

            return None
