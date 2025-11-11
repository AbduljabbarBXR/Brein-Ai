import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
import json
from datetime import datetime

class MemoryManager:
    """
    Memory Manager for Brein AI - handles vector storage, retrieval, and basic memory operations.
    Uses FAISS for vector similarity search and SQLite for metadata storage.
    """

    def __init__(self, db_path: str = "memory/brein_memory.db", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Memory Manager.

        Args:
            db_path: Path to SQLite database file
            embedding_model: SentenceTransformer model name for embeddings
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model

        # Ensure memory directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize SQLite database
        self._init_database()

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        self.id_to_idx = {}  # Maps node_id to FAISS index
        self.idx_to_id = {}  # Maps FAISS index to node_id

        # Load existing data if available
        self._load_existing_data()

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT DEFAULT 'conversational',
                embedding BLOB,
                activation_level REAL DEFAULT 0.5,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')

        # Create conversations table for session tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT,
                message TEXT,
                sender TEXT,
                memory_nodes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _load_existing_data(self):
        """Load existing nodes and embeddings from database into FAISS index."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT node_id, embedding FROM nodes WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()

        if rows:
            embeddings = []
            node_ids = []

            for node_id, embedding_blob in rows:
                if embedding_blob:
                    embedding = pickle.loads(embedding_blob)
                    embeddings.append(embedding)
                    node_ids.append(node_id)

            if embeddings:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.index.add(embeddings_array)

                # Update mappings
                for idx, node_id in enumerate(node_ids):
                    self.id_to_idx[node_id] = idx
                    self.idx_to_id[idx] = node_id

        conn.close()

    def add_memory(self, content: str, memory_type: str = "conversational",
                   metadata: Optional[Dict] = None) -> str:
        """
        Add new content to memory.

        Args:
            content: Text content to store
            memory_type: Type of memory (stable, conversational, functional)
            metadata: Additional metadata

        Returns:
            node_id: Unique identifier for the stored memory
        """
        import uuid

        # Generate embedding
        embedding = self.embedding_model.encode(content, convert_to_numpy=True)
        embedding = embedding.astype(np.float32)

        # Generate node ID
        node_id = str(uuid.uuid4())

        # Add to FAISS index
        idx = self.index.ntotal
        self.index.add(embedding.reshape(1, -1))
        self.id_to_idx[node_id] = idx
        self.idx_to_id[idx] = node_id

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO nodes (node_id, content, type, embedding, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            node_id,
            content,
            memory_type,
            pickle.dumps(embedding),
            json.dumps(metadata) if metadata else None
        ))

        conn.commit()
        conn.close()

        return node_id

    def search_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search memory for similar content.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of dictionaries with node_id, content, score, and metadata
        """
        if self.index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                node_id = self.idx_to_id.get(idx)
                if node_id:
                    cursor.execute("SELECT content, type, metadata FROM nodes WHERE node_id = ?", (node_id,))
                    row = cursor.fetchone()
                    if row:
                        content, memory_type, metadata_str = row
                        metadata = json.loads(metadata_str) if metadata_str else {}

                        results.append({
                            "node_id": node_id,
                            "content": content,
                            "type": memory_type,
                            "score": float(score),
                            "metadata": metadata
                        })

        conn.close()
        return results

    def get_memory_stats(self) -> Dict:
        """Get statistics about the memory system."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM nodes")
        total_nodes = cursor.fetchone()[0]

        cursor.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type")
        type_counts = dict(cursor.fetchall())

        conn.close()

        return {
            "total_nodes": total_nodes,
            "vector_dimension": self.embedding_dim,
            "index_size": self.index.ntotal,
            "type_distribution": type_counts,
            "embedding_model": self.embedding_model_name
        }

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Simple text chunking utility.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                break_chars = ['. ', '! ', '? ', '\n']
                break_pos = -1

                for char in break_chars:
                    pos = text.rfind(char, start, end)
                    if pos > break_pos:
                        break_pos = pos + len(char)

                if break_pos > start + chunk_size * 0.7:  # Good break point found
                    end = break_pos
                else:
                    end = start + chunk_size

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks