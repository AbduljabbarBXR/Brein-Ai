import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
import json
from datetime import datetime
import sys
from collections import OrderedDict
import threading
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_mesh import NeuralMesh

class MemoryManager:
    """
    Memory Manager for Brein AI - handles vector storage, retrieval, and basic memory operations.
    Uses FAISS for vector similarity search and SQLite for metadata storage.
    """

    def __init__(self, db_path: str = "memory/brein_memory.db", embedding_model: str = "all-MiniLM-L6-v2",
                 max_memory_cache: int = 1000, ssd_cache_path: str = "memory/ssd_cache/"):
        """
        Initialize the Memory Manager with memory-mapped FAISS and LRU caching.

        Args:
            db_path: Path to SQLite database file
            embedding_model: SentenceTransformer model name for embeddings
            max_memory_cache: Maximum number of embeddings to keep in memory
            ssd_cache_path: Path for SSD offload cache
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.max_memory_cache = max_memory_cache
        self.ssd_cache_path = ssd_cache_path

        # Ensure directories exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(ssd_cache_path, exist_ok=True)

        # Initialize SQLite database
        self._init_database()

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize memory-mapped FAISS index with SSD offload
        self._init_memory_mapped_index()

        # LRU cache for embeddings in memory
        self.embedding_cache = OrderedDict()
        self.cache_lock = threading.Lock()

        # Initialize Neural Mesh
        self.neural_mesh = NeuralMesh()

        # Initialize Memory Consolidation System
        self.consolidator = MemoryConsolidator(self.db_path)

        # Load existing data if available
        self._load_existing_data()

        # Database connection management for concurrent access
        self.db_lock = threading.Lock()
        self.max_retries = 3
        self.retry_delay = 0.1

    def _execute_with_retry(self, operation_func, *args, **kwargs):
        """
        Execute database operations with retry logic for handling locks.

        Args:
            operation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the operation
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                with self.db_lock:
                    return operation_func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                last_exception = e
                if "database is locked" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                raise e

        # If we get here, all retries failed
        raise last_exception

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create nodes table with consolidation fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT DEFAULT 'conversational',
                embedding BLOB,
                activation_level REAL DEFAULT 0.5,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                importance_score REAL DEFAULT 0.5,
                decay_factor REAL DEFAULT 1.0,
                consolidation_strength REAL DEFAULT 0.0,
                emotional_importance REAL DEFAULT 0.0,
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

        # Create memory_consolidation_log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_consolidation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_value REAL,
                new_value REAL,
                reason TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            )
        ''')

        # Create memory_decay_schedule table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_decay_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                scheduled_decay REAL,
                scheduled_time DATETIME,
                decay_reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            )
        ''')

        # Add missing columns to existing nodes table
        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN memory_type TEXT DEFAULT 'working'")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN access_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN importance_score REAL DEFAULT 0.5")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN decay_factor REAL DEFAULT 1.0")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN consolidation_strength REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE nodes ADD COLUMN emotional_importance REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            pass

        # Create performance indexes
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_memory_type ON nodes(memory_type)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_activation ON nodes(activation_level)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_importance ON nodes(importance_score)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON nodes(last_accessed)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_created_at ON nodes(created_at)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON conversations(conversation_id)")
        except sqlite3.OperationalError:
            pass

        conn.commit()
        conn.close()

    def _init_memory_mapped_index(self):
        """Initialize memory-mapped FAISS index for SSD offload."""
        index_path = os.path.join(self.ssd_cache_path, "faiss_index.idx")

        # Try to load existing memory-mapped index
        if os.path.exists(index_path):
            try:
                self.index = faiss.read_index(index_path)
                # Convert to memory-mapped if not already
                if not hasattr(self.index, 'is_mmap'):
                    faiss.write_index(self.index, index_path)
                    self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            except Exception as e:
                print(f"Warning: Could not load existing index: {e}")
                self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.id_to_idx = {}  # Maps node_id to FAISS index
        self.idx_to_id = {}  # Maps FAISS index to node_id

    def _save_index_to_disk(self):
        """Save FAISS index to disk for memory mapping."""
        index_path = os.path.join(self.ssd_cache_path, "faiss_index.idx")
        try:
            faiss.write_index(self.index, index_path)
        except Exception as e:
            print(f"Warning: Could not save index to disk: {e}")

    def _get_cached_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding from LRU cache, loading from disk if necessary."""
        with self.cache_lock:
            if node_id in self.embedding_cache:
                # Move to end (most recently used)
                self.embedding_cache.move_to_end(node_id)
                return self.embedding_cache[node_id]

            # Load from database if not in cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM nodes WHERE node_id = ?", (node_id,))
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                embedding = pickle.loads(row[0])
                self._add_to_cache(node_id, embedding)
                return embedding

            return None

    def _add_to_cache(self, node_id: str, embedding: np.ndarray):
        """Add embedding to LRU cache with eviction."""
        with self.cache_lock:
            if len(self.embedding_cache) >= self.max_memory_cache:
                # Remove least recently used
                evicted_node, _ = self.embedding_cache.popitem(last=False)
                # Could optionally save evicted items to disk here

            self.embedding_cache[node_id] = embedding

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
        Add new content to memory with memory-mapped storage.

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

        # Add to LRU cache
        self._add_to_cache(node_id, embedding)

        # Add to neural mesh
        self.neural_mesh.add_node(node_id, "memory", metadata)

        # Store in database with memory type
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO nodes (node_id, content, type, embedding, metadata, memory_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            node_id,
            content,
            memory_type,
            pickle.dumps(embedding),
            json.dumps(metadata) if metadata else None,
            "working"  # Default to working memory
        ))

        conn.commit()
        conn.close()

        # Save index to disk periodically (every 100 additions)
        if self.index.ntotal % 100 == 0:
            self._save_index_to_disk()

        return node_id

    def search_memory(self, query: str, top_k: int = 5, use_mesh_expansion: bool = True) -> List[Dict]:
        """
        Search memory for similar content, with optional neural mesh expansion.
        Optimized with batch processing and efficient database queries.

        Args:
            query: Search query
            top_k: Number of top results to return
            use_mesh_expansion: Whether to expand results using neural mesh

        Returns:
            List of dictionaries with node_id, content, score, and metadata
        """
        if self.index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Search FAISS index with optimized parameters
        search_k = min(top_k * 2, self.index.ntotal)  # Search more candidates for better results
        scores, indices = self.index.search(query_embedding, search_k)

        # Get valid results
        valid_results = []
        node_ids_to_fetch = []

        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score > 0.1:  # Filter low-quality matches
                node_id = self.idx_to_id.get(idx)
                if node_id:
                    valid_results.append((node_id, float(score)))
                    node_ids_to_fetch.append(node_id)

        # Batch database query for better performance
        if not node_ids_to_fetch:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use parameterized query for batch fetch
        placeholders = ','.join('?' * len(node_ids_to_fetch))
        cursor.execute(f"""
            SELECT node_id, content, type, metadata, access_count
            FROM nodes
            WHERE node_id IN ({placeholders})
        """, node_ids_to_fetch)

        node_data = {row[0]: row[1:] for row in cursor.fetchall()}

        results = []
        activated_nodes = set()

        # Process results
        for node_id, score in valid_results[:top_k]:  # Limit to top_k
            if node_id in node_data:
                content, memory_type, metadata_str, access_count = node_data[node_id]
                metadata = json.loads(metadata_str) if metadata_str else {}

                results.append({
                    "node_id": node_id,
                    "content": content,
                    "type": memory_type,
                    "score": score,
                    "metadata": metadata
                })

                activated_nodes.add(node_id)

        # Batch update access statistics
        if activated_nodes:
            current_time = datetime.now().isoformat()
            update_data = [(current_time, node_id) for node_id in activated_nodes]
            cursor.executemany("""
                UPDATE nodes
                SET last_accessed = ?, access_count = access_count + 1
                WHERE node_id = ?
            """, update_data)

            # Batch activate nodes in neural mesh
            for node_id in activated_nodes:
                self.neural_mesh.activate_node(node_id)

        # Apply Hebbian reinforcement for co-activated nodes
        if len(activated_nodes) > 1 and use_mesh_expansion:
            node_list = list(activated_nodes)
            # Calculate stronger reinforcement based on search relevance
            avg_score = sum(result["score"] for result in results) / len(results)
            reinforcement_strength = min(0.2, avg_score * 0.3)

            # Batch reinforce connections
            for i in range(len(node_list)):
                for j in range(i+1, len(node_list)):
                    self.neural_mesh.reinforce_connection(node_list[i], node_list[j], reinforcement_strength)

        # Expand results using neural mesh if requested
        if use_mesh_expansion and results:
            expanded_results = self._expand_with_mesh(results, top_k)
            results = expanded_results

        conn.close()
        return results

    def _expand_with_mesh(self, initial_results: List[Dict], max_total: int) -> List[Dict]:
        """
        Expand search results using neural mesh connections.

        Args:
            initial_results: Initial search results
            max_total: Maximum total results to return

        Returns:
            Expanded results list
        """
        expanded_results = initial_results.copy()
        seen_nodes = set(result["node_id"] for result in initial_results)

        # For each initial result, get mesh neighbors
        for result in initial_results:
            node_id = result["node_id"]
            neighbors = self.neural_mesh.get_neighbors(node_id, top_k=2)

            for neighbor_id, weight in neighbors:
                if neighbor_id not in seen_nodes and len(expanded_results) < max_total:
                    # Get neighbor details from database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT content, type, metadata FROM nodes WHERE node_id = ?", (neighbor_id,))
                    row = cursor.fetchone()
                    conn.close()

                    if row:
                        content, memory_type, metadata_str = row
                        metadata = json.loads(metadata_str) if metadata_str else {}

                        expanded_results.append({
                            "node_id": neighbor_id,
                            "content": content,
                            "type": memory_type,
                            "score": float(weight * 0.8),  # Reduce score for mesh-expanded results
                            "metadata": {**metadata, "expanded_via_mesh": True}
                        })

                        seen_nodes.add(neighbor_id)

        return expanded_results

    def get_memory_stats(self) -> Dict:
        """Get statistics about the memory system."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM nodes")
        total_nodes = cursor.fetchone()[0]

        cursor.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type")
        type_counts = dict(cursor.fetchall())

        cursor.execute("SELECT memory_type, COUNT(*) FROM nodes GROUP BY memory_type")
        memory_type_counts = dict(cursor.fetchall())

        conn.close()

        mesh_stats = self.neural_mesh.get_mesh_stats()

        return {
            "total_nodes": total_nodes,
            "vector_dimension": self.embedding_dim,
            "index_size": self.index.ntotal,
            "cache_size": len(self.embedding_cache),
            "max_cache_size": self.max_memory_cache,
            "ssd_cache_path": self.ssd_cache_path,
            "type_distribution": type_counts,
            "memory_type_distribution": memory_type_counts,
            "embedding_model": self.embedding_model_name,
            "neural_mesh": mesh_stats
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


class MemoryConsolidator:
    """
    Memory Consolidation System for Brein AI
    Handles memory decay, consolidation, and reinforcement learning
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.decay_rates = {
            'working': 0.95,  # Fast decay for working memory
            'conversational': 0.98,  # Moderate decay for conversations
            'stable': 0.999,  # Slow decay for stable knowledge
            'emotional': 0.995  # Moderate decay for emotional memories
        }
        self.consolidation_threshold = 0.7  # Minimum importance for consolidation
        self.max_memory_nodes = 10000  # Maximum nodes before aggressive cleanup

        # Database connection management for concurrent access
        self.db_lock = threading.Lock()
        self.max_retries = 3
        self.retry_delay = 0.1

    def _execute_with_retry(self, operation_func, *args, **kwargs):
        """
        Execute database operations with retry logic for handling locks.

        Args:
            operation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the operation
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                with self.db_lock:
                    return operation_func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                last_exception = e
                if "database is locked" in str(e).lower():
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                raise e

        # If we get here, all retries failed
        raise last_exception

    def apply_memory_decay(self, hours_since_last_decay: float = 24) -> Dict[str, int]:
        """
        Apply time-based memory decay to all nodes.

        Args:
            hours_since_last_decay: Hours since last decay application

        Returns:
            Dict with decay statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all nodes that need decay
        cursor.execute("""
            SELECT node_id, activation_level, decay_factor, memory_type,
                   importance_score, consolidation_strength
            FROM nodes
            WHERE activation_level > 0.01
        """)

        nodes_to_decay = cursor.fetchall()
        decayed_count = 0
        consolidated_count = 0

        for node_id, activation_level, decay_factor, memory_type, importance_score, consolidation_strength in nodes_to_decay:
            # Calculate decay based on memory type and time
            base_decay_rate = self.decay_rates.get(memory_type, 0.98)
            time_decay_factor = base_decay_rate ** (hours_since_last_decay / 24)  # Daily decay

            # Importance and consolidation reduce decay
            importance_modifier = 1 + (importance_score * 0.5)
            consolidation_modifier = 1 + (consolidation_strength * 0.3)

            effective_decay = time_decay_factor * importance_modifier * consolidation_modifier
            effective_decay = min(effective_decay, 0.999)  # Cap decay

            new_activation = activation_level * effective_decay

            # Update node with decayed activation
            cursor.execute("""
                UPDATE nodes
                SET activation_level = ?, decay_factor = ?, updated_at = CURRENT_TIMESTAMP
                WHERE node_id = ?
            """, (new_activation, effective_decay, node_id))

            # Log decay action
            self._log_consolidation_action(node_id, "decay", activation_level, new_activation,
                                         f"Time-based decay ({memory_type})")

            decayed_count += 1

            # Check if node should be consolidated
            if importance_score >= self.consolidation_threshold and new_activation < 0.3:
                self._consolidate_memory_node(cursor, node_id, importance_score)
                consolidated_count += 1

        conn.commit()
        conn.close()

        return {
            "nodes_decayed": decayed_count,
            "nodes_consolidated": consolidated_count,
            "decay_period_hours": hours_since_last_decay
        }

    def reinforce_memory(self, node_ids: List[str], reinforcement_strength: float,
                        reason: str = "conversation_reinforcement") -> Dict[str, int]:
        """
        Apply reinforcement learning to memory nodes.

        Args:
            node_ids: List of node IDs to reinforce
            reinforcement_strength: Strength of reinforcement (0.0-1.0)
            reason: Reason for reinforcement

        Returns:
            Dict with reinforcement statistics
        """
        def _reinforce_operation():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            reinforced_count = 0
            consolidated_count = 0

            for node_id in node_ids:
                # Get current node data
                cursor.execute("""
                    SELECT activation_level, access_count, importance_score, consolidation_strength
                    FROM nodes WHERE node_id = ?
                """, (node_id,))

                row = cursor.fetchone()
                if not row:
                    continue

                old_activation, access_count, importance_score, consolidation_strength = row

                # Apply reinforcement
                reinforcement_bonus = reinforcement_strength * 0.2
                new_activation = min(old_activation + reinforcement_bonus, 1.0)
                new_access_count = access_count + 1

                # Update importance based on access patterns
                new_importance = min(importance_score + (reinforcement_strength * 0.1), 1.0)

                # Strengthen consolidation for frequently accessed memories
                new_consolidation = min(consolidation_strength + (reinforcement_strength * 0.05), 1.0)

                # Update node
                cursor.execute("""
                    UPDATE nodes
                    SET activation_level = ?, access_count = ?, importance_score = ?,
                        consolidation_strength = ?, last_accessed = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE node_id = ?
                """, (new_activation, new_access_count, new_importance, new_consolidation, node_id))

                # Log reinforcement action
                self._log_consolidation_action(node_id, "reinforce", old_activation, new_activation, reason)

                reinforced_count += 1

                # Check for consolidation opportunity
                if new_importance >= self.consolidation_threshold and new_consolidation >= 0.5:
                    self._consolidate_memory_node(cursor, node_id, new_importance)
                    consolidated_count += 1

            conn.commit()
            conn.close()

            return {
                "nodes_reinforced": reinforced_count,
                "nodes_consolidated": consolidated_count,
                "reinforcement_strength": reinforcement_strength
            }

        return self._execute_with_retry(_reinforce_operation)

    def consolidate_similar_memories(self) -> Dict[str, int]:
        """
        Find and consolidate similar memories to prevent redundancy.

        Returns:
            Dict with consolidation statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all nodes with embeddings
        cursor.execute("""
            SELECT node_id, content, importance_score, consolidation_strength
            FROM nodes
            WHERE embedding IS NOT NULL AND activation_level > 0.1
            ORDER BY importance_score DESC
        """)

        nodes = cursor.fetchall()
        consolidated_count = 0
        merged_count = 0

        # Group similar nodes (simplified - in practice would use embedding similarity)
        content_groups = {}
        for node_id, content, importance_score, consolidation_strength in nodes:
            # Simple content-based grouping (would be improved with semantic similarity)
            content_key = content.lower()[:100].strip()  # First 100 chars as key

            if content_key not in content_groups:
                content_groups[content_key] = []

            content_groups[content_key].append({
                'node_id': node_id,
                'importance_score': importance_score,
                'consolidation_strength': consolidation_strength
            })

        # Consolidate groups with multiple similar memories
        for content_key, group_nodes in content_groups.items():
            if len(group_nodes) > 1:
                # Keep the most important node, merge others into it
                sorted_nodes = sorted(group_nodes, key=lambda x: x['importance_score'], reverse=True)
                primary_node = sorted_nodes[0]

                for secondary_node in sorted_nodes[1:]:
                    # Merge secondary into primary
                    self._merge_memory_nodes(cursor, primary_node['node_id'], secondary_node['node_id'])
                    merged_count += 1

                consolidated_count += 1

        conn.commit()
        conn.close()

        return {
            "groups_consolidated": consolidated_count,
            "nodes_merged": merged_count,
            "similarity_groups_found": len(content_groups)
        }

    def get_memory_health_report(self) -> Dict:
        """
        Generate comprehensive memory health report.

        Returns:
            Dict with memory health metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Basic statistics
        cursor.execute("SELECT COUNT(*) FROM nodes")
        total_nodes = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(activation_level), AVG(importance_score), AVG(consolidation_strength) FROM nodes")
        avg_activation, avg_importance, avg_consolidation = cursor.fetchone()

        # Decay analysis
        cursor.execute("SELECT COUNT(*) FROM nodes WHERE activation_level < 0.1")
        decayed_nodes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM nodes WHERE importance_score >= 0.8")
        high_importance_nodes = cursor.fetchone()[0]

        # Consolidation log analysis
        cursor.execute("""
            SELECT action, COUNT(*) as count
            FROM memory_consolidation_log
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY action
        """)
        recent_actions = dict(cursor.fetchall())

        # Memory distribution
        cursor.execute("""
            SELECT memory_type, COUNT(*), AVG(activation_level), AVG(importance_score)
            FROM nodes
            GROUP BY memory_type
        """)
        memory_distribution = cursor.fetchall()

        conn.close()

        return {
            "total_nodes": total_nodes,
            "average_activation": avg_activation or 0,
            "average_importance": avg_importance or 0,
            "average_consolidation": avg_consolidation or 0,
            "decayed_nodes": decayed_nodes,
            "high_importance_nodes": high_importance_nodes,
            "recent_consolidation_actions": recent_actions,
            "memory_type_distribution": [
                {
                    "type": mem_type,
                    "count": count,
                    "avg_activation": avg_act,
                    "avg_importance": avg_imp
                }
                for mem_type, count, avg_act, avg_imp in memory_distribution
            ],
            "health_score": self._calculate_memory_health_score(
                total_nodes, avg_activation or 0, avg_importance or 0, decayed_nodes
            )
        }

    def _consolidate_memory_node(self, cursor, node_id: str, importance_score: float):
        """
        Consolidate a memory node by strengthening its connections and permanence.
        """
        # Increase consolidation strength
        new_consolidation = min(importance_score * 0.8, 1.0)

        cursor.execute("""
            UPDATE nodes
            SET consolidation_strength = ?, decay_factor = ?, updated_at = CURRENT_TIMESTAMP
            WHERE node_id = ?
        """, (new_consolidation, 0.999, node_id))

        self._log_consolidation_action(node_id, "consolidate", 0, new_consolidation,
                                     f"Importance-based consolidation ({importance_score:.2f})")

    def _merge_memory_nodes(self, cursor, primary_node_id: str, secondary_node_id: str):
        """
        Merge secondary node into primary node.
        """
        # Get secondary node data
        cursor.execute("SELECT access_count, importance_score FROM nodes WHERE node_id = ?",
                      (secondary_node_id,))
        secondary_data = cursor.fetchone()

        if secondary_data:
            secondary_access, secondary_importance = secondary_data

            # Update primary node with combined statistics
            cursor.execute("""
                UPDATE nodes
                SET access_count = access_count + ?,
                    importance_score = MAX(importance_score, ?),
                    consolidation_strength = consolidation_strength + 0.1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE node_id = ?
            """, (secondary_access, secondary_importance, primary_node_id))

        # Mark secondary node as merged (soft delete)
        cursor.execute("""
            UPDATE nodes
            SET activation_level = 0.01, decay_factor = 0.9,
                metadata = json_set(COALESCE(metadata, '{}'), '$.merged_into', ?),
                updated_at = CURRENT_TIMESTAMP
            WHERE node_id = ?
        """, (primary_node_id, secondary_node_id))

        self._log_consolidation_action(secondary_node_id, "merge", 1.0, 0.01,
                                     f"Merged into {primary_node_id}")

    def _log_consolidation_action(self, node_id: str, action: str, old_value: float,
                                 new_value: float, reason: str):
        """
        Log consolidation actions for analytics.
        """
        def _log_operation():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO memory_consolidation_log (node_id, action, old_value, new_value, reason)
                VALUES (?, ?, ?, ?, ?)
            """, (node_id, action, old_value, new_value, reason))

            conn.commit()
            conn.close()

        try:
            self._execute_with_retry(_log_operation)
        except Exception:
            # If logging fails, don't break the main operation
            pass

    def _calculate_memory_health_score(self, total_nodes: int, avg_activation: float,
                                     avg_importance: float, decayed_nodes: int) -> float:
        """
        Calculate overall memory health score.
        """
        if total_nodes == 0:
            return 0.0

        # Factors contributing to health score
        activation_factor = avg_activation  # Higher activation is better
        importance_factor = avg_importance  # Higher importance is better
        decay_factor = 1.0 - (decayed_nodes / total_nodes)  # Fewer decayed nodes is better

        # Weighted combination
        health_score = (activation_factor * 0.4) + (importance_factor * 0.4) + (decay_factor * 0.2)

        return min(max(health_score, 0.0), 1.0)
