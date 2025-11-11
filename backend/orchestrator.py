from fastapi import HTTPException
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from memory_transformer import MemoryTransformer
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for Brein AI - coordinates between memory, agents, and user queries.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.memory_transformer = MemoryTransformer()
        self.session_context = {}  # Simple session storage

    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Process a user query and return response with memory context and thought trace.

        Args:
            query: User query string
            session_id: Optional session identifier

        Returns:
            Dictionary with response, thought_trace, memory_chunks, and metadata
        """
        try:
            # Search memory for relevant context
            memory_results = self.memory.search_memory(query, top_k=5, use_mesh_expansion=True)

            # Extract memory chunks and embeddings for thought generation
            memory_chunks = []
            memory_embeddings = []

            for result in memory_results:
                memory_chunks.append({
                    "content": result["content"],
                    "score": result["score"],
                    "type": result["type"]
                })

                # Get embedding from database for thought generation
                conn = self.memory.conn if hasattr(self.memory, 'conn') else None
                if conn is None:
                    import sqlite3
                    conn = sqlite3.connect(self.memory.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT embedding FROM nodes WHERE node_id = ?", (result["node_id"],))
                    row = cursor.fetchone()
                    if row and row[0]:
                        import pickle
                        embedding = pickle.loads(row[0])
                        memory_embeddings.append(embedding)
                    conn.close()

            # Generate thought trace using memory transformer
            thought_result = self.memory_transformer.generate_thought_trace(memory_embeddings, query)

            # Generate response based on memory context
            response = self._generate_response_with_memory(query, memory_chunks, thought_result)

            # Store conversation in memory
            if session_id:
                self._store_conversation(session_id, query, response, memory_results)

            return {
                "response": response,
                "thought_trace": thought_result["thought_trace"],
                "confidence": thought_result["confidence"],
                "memory_chunks": memory_chunks,
                "session_id": session_id or "default",
                "memory_stats": self.memory.get_memory_stats(),
                "reasoning_metadata": {
                    "activated_nodes": thought_result["activated_nodes"],
                    "reasoning_type": thought_result["reasoning_type"],
                    "model_used": thought_result.get("model_used", "unknown")
                }
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    def _generate_response_with_memory(self, query: str, memory_chunks: List[Dict], thought_result: Dict) -> str:
        """
        Generate a response using memory context and thought trace.
        """
        if not memory_chunks:
            return f"I understand you're asking about '{query}'. I don't have specific memories about this yet, but I'm learning!"

        # Use thought trace confidence to modulate response
        confidence = thought_result.get("confidence", 0.5)
        confidence_text = "very confident" if confidence > 0.8 else "confident" if confidence > 0.6 else "somewhat uncertain"

        # Generate response based on top memory chunk
        top_chunk = memory_chunks[0]

        # Include thought insights in response
        thought_insight = ""
        if thought_result["reasoning_type"] != "empty_memory":
            thought_insight = f" My internal reasoning suggests this connects to {thought_result['activated_nodes']} related concepts."

        return f"Based on my memory and feeling {confidence_text} about this: {top_chunk['content'][:200]}...{thought_insight}"

    def _store_conversation(self, session_id: str, query: str, response: str, memory_results: List[Dict]):
        """
        Store conversation data for session tracking.
        """
        memory_node_ids = [result["node_id"] for result in memory_results]

        # For now, we'll store this in memory - in production this would go to a proper conversation store
        if session_id not in self.session_context:
            self.session_context[session_id] = []

        self.session_context[session_id].append({
            "query": query,
            "response": response,
            "memory_nodes": memory_node_ids,
            "timestamp": "now"  # Would use proper datetime in production
        })

        # Keep only last 10 exchanges per session
        if len(self.session_context[session_id]) > 10:
            self.session_context[session_id] = self.session_context[session_id][-10:]

    async def ingest_content(self, content: str, content_type: str = "stable") -> Dict:
        """
        Ingest new content into memory.

        Args:
            content: Content to ingest
            content_type: Type of content (stable, conversational, functional)

        Returns:
            Dictionary with node_id and status
        """
        try:
            # Chunk the content if it's long
            chunks = self.memory.chunk_text(content)

            node_ids = []
            for chunk in chunks:
                node_id = self.memory.add_memory(chunk, content_type)
                node_ids.append(node_id)

            return {
                "status": "success",
                "node_ids": node_ids,
                "chunks_created": len(chunks),
                "content_type": content_type
            }

        except Exception as e:
            logger.error(f"Error ingesting content: {e}")
            raise HTTPException(status_code=500, detail=f"Content ingestion failed: {str(e)}")

    def get_session_context(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        return self.session_context.get(session_id, [])

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.session_context:
            del self.session_context[session_id]