from fastapi import HTTPException
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for Brein AI - coordinates between memory, agents, and user queries.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.session_context = {}  # Simple session storage

    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Process a user query and return response with memory context.

        Args:
            query: User query string
            session_id: Optional session identifier

        Returns:
            Dictionary with response, memory_chunks, and metadata
        """
        try:
            # Search memory for relevant context
            memory_results = self.memory.search_memory(query, top_k=5)

            # Extract memory chunks for context
            memory_chunks = []
            for result in memory_results:
                memory_chunks.append({
                    "content": result["content"],
                    "score": result["score"],
                    "type": result["type"]
                })

            # For now, return a simple canned response with memory context
            # This will be enhanced with actual model inference in later sprints
            response = self._generate_canned_response(query, memory_chunks)

            # Store conversation in memory
            if session_id:
                self._store_conversation(session_id, query, response, memory_results)

            return {
                "response": response,
                "memory_chunks": memory_chunks,
                "session_id": session_id or "default",
                "memory_stats": self.memory.get_memory_stats()
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

    def _generate_canned_response(self, query: str, memory_chunks: List[Dict]) -> str:
        """
        Generate a simple canned response based on query and memory context.
        This is a placeholder that will be replaced by actual model inference.
        """
        if not memory_chunks:
            return f"I understand you're asking about '{query}'. I don't have specific memories about this yet, but I'm learning!"

        # Simple response generation based on top memory chunk
        top_chunk = memory_chunks[0]
        confidence = "confident" if top_chunk["score"] > 0.7 else "somewhat uncertain"

        return f"Based on what I know, and feeling {confidence} about this: {top_chunk['content'][:200]}..."

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