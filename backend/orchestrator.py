from fastapi import HTTPException
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from memory_transformer import MemoryTransformer
from agents import HippocampusAgent, CortexAgent, BasalGangliaAgent
from neural_mesh import NeuralMesh
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for Brein AI - coordinates between memory, agents, and user queries.
    Now uses multi-agent architecture with specialized agents.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager
        self.memory_transformer = MemoryTransformer()
        self.neural_mesh = NeuralMesh()

        # Initialize agents
        self.hippocampus = HippocampusAgent(memory_manager)
        self.cortex = CortexAgent(memory_manager, self.memory_transformer)
        self.basal_ganglia = BasalGangliaAgent(memory_manager, self.neural_mesh)

        self.session_context = {}  # Enhanced session storage

    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Process a user query using multi-agent architecture.

        Args:
            query: User query string
            session_id: Optional session identifier

        Returns:
            Dictionary with response, thought_trace, memory_chunks, and metadata
        """
        try:
            # Get session context for continuity
            session_history = self.get_session_context(session_id) if session_id else None

            # Use Cortex Agent for reasoning and response generation
            cortex_result = await self.cortex.process_query(query, session_history)

            # Extract memory node IDs for reinforcement
            memory_node_ids = [chunk["node_id"] for chunk in cortex_result.get("memory_chunks", [])]

            # Apply reinforcement learning via Basal Ganglia
            if memory_node_ids:
                await self.basal_ganglia.reinforce_memory(memory_node_ids, cortex_result.get("confidence", 0.5))

            # Get policy decisions
            policy_decision = self.basal_ganglia.get_policy_decision("query", cortex_result.get("memory_chunks", []))

            # Store conversation in session context
            if session_id:
                self._store_conversation(session_id, query, cortex_result["response"], cortex_result.get("memory_chunks", []))

            return {
                "response": cortex_result["response"],
                "thought_trace": cortex_result["thought_trace"],
                "confidence": cortex_result["confidence"],
                "memory_chunks": cortex_result["memory_chunks"],
                "session_id": session_id or "default",
                "memory_stats": self.memory.get_memory_stats(),
                "reasoning_metadata": cortex_result["reasoning_metadata"],
                "agents_used": ["cortex", "basal_ganglia"],
                "policy_decision": policy_decision
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

    def _store_conversation(self, session_id: str, query: str, response: str, memory_chunks: List[Dict]):
        """
        Store conversation data for session tracking with enhanced context.
        """
        memory_node_ids = [chunk["node_id"] for chunk in memory_chunks]

        # Enhanced session storage with agent metadata
        if session_id not in self.session_context:
            self.session_context[session_id] = []

        self.session_context[session_id].append({
            "query": query,
            "response": response,
            "memory_nodes": memory_node_ids,
            "memory_count": len(memory_chunks),
            "timestamp": "now",  # Would use proper datetime in production
            "agents_used": ["cortex", "basal_ganglia"]  # Track which agents processed this
        })

        # Keep only last 10 exchanges per session
        if len(self.session_context[session_id]) > 10:
            self.session_context[session_id] = self.session_context[session_id][-10:]

    async def ingest_content(self, content: str, content_type: str = "stable") -> Dict:
        """
        Ingest new content into memory using Hippocampus Agent.

        Args:
            content: Content to ingest
            content_type: Type of content (stable, conversational, functional)

        Returns:
            Dictionary with node_id and status
        """
        try:
            # Use Hippocampus Agent for ingestion
            result = await self.hippocampus.ingest_content(content, content_type)

            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["error"])

            return {
                "status": result["status"],
                "node_ids": result["node_ids"],
                "chunks_created": result["chunks_created"],
                "content_type": result["content_type"],
                "agents_used": ["hippocampus"]
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