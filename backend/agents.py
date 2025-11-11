from typing import Dict, List, Optional, Any
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from memory_transformer import MemoryTransformer
from neural_mesh import NeuralMesh

logger = logging.getLogger(__name__)

class HippocampusAgent:
    """
    Hippocampus Agent - Handles memory encoding and ingestion.
    Responsible for converting input content into memory representations.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory = memory_manager

    async def ingest_content(self, content: str, content_type: str = "stable") -> Dict[str, Any]:
        """
        Ingest new content into memory with proper encoding.

        Args:
            content: Content to ingest
            content_type: Type of content (stable, conversational, functional)

        Returns:
            Dictionary with ingestion results
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
                "content_type": content_type,
                "agent": "hippocampus"
            }

        except Exception as e:
            logger.error(f"HippocampusAgent ingestion error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": "hippocampus"
            }

class CortexAgent:
    """
    Cortex Agent - Handles reasoning and thought generation.
    Processes queries and generates responses with memory context.
    """

    def __init__(self, memory_manager: MemoryManager, memory_transformer: MemoryTransformer):
        self.memory = memory_manager
        self.memory_transformer = memory_transformer

    async def process_query(self, query: str, session_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process a query with reasoning and memory retrieval.

        Args:
            query: User query
            session_context: Previous conversation context

        Returns:
            Dictionary with response and reasoning
        """
        try:
            # Enhanced memory search with top-N and mesh expansion
            memory_results = self.memory.search_memory(query, top_k=10, use_mesh_expansion=True)

            # Extract memory chunks and embeddings
            memory_chunks = []
            memory_embeddings = []

            for result in memory_results:
                memory_chunks.append({
                    "content": result["content"],
                    "score": result["score"],
                    "type": result["type"],
                    "node_id": result["node_id"]
                })

                # Get embedding for thought generation
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

            # Generate thought trace
            thought_result = self.memory_transformer.generate_thought_trace(memory_embeddings, query)

            # Generate response with memory context
            response = self._generate_reasoned_response(query, memory_chunks, thought_result, session_context)

            return {
                "response": response,
                "thought_trace": thought_result["thought_trace"],
                "confidence": thought_result["confidence"],
                "memory_chunks": memory_chunks,
                "reasoning_metadata": {
                    "activated_nodes": thought_result["activated_nodes"],
                    "reasoning_type": thought_result["reasoning_type"],
                    "model_used": thought_result.get("model_used", "unknown")
                },
                "agent": "cortex"
            }

        except Exception as e:
            logger.error(f"CortexAgent query processing error: {e}")
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "thought_trace": "Error in reasoning process",
                "confidence": 0.0,
                "memory_chunks": [],
                "agent": "cortex"
            }

    def _generate_reasoned_response(self, query: str, memory_chunks: List[Dict], thought_result: Dict,
                                   session_context: Optional[List[Dict]] = None) -> str:
        """
        Generate a reasoned response using memory and context.
        """
        if not memory_chunks:
            return f"I understand you're asking about '{query}'. I don't have specific memories about this yet, but I'm learning!"

        # Use confidence to modulate response
        confidence = thought_result.get("confidence", 0.5)
        confidence_text = "very confident" if confidence > 0.8 else "confident" if confidence > 0.6 else "somewhat uncertain"

        # Consider session context for continuity
        context_insight = ""
        if session_context and len(session_context) > 1:
            context_insight = " Building on our previous conversation,"

        # Include thought insights
        thought_insight = ""
        if thought_result["reasoning_type"] != "empty_memory":
            activated_count = len(thought_result.get("activated_nodes", []))
            thought_insight = f" My reasoning connects this to {activated_count} related concepts."

        top_chunk = memory_chunks[0]
        return f"{context_insight} Based on my memory and feeling {confidence_text} about this: {top_chunk['content'][:200]}...{thought_insight}"

class BasalGangliaAgent:
    """
    Basal Ganglia Agent - Handles policy decisions and reinforcement learning.
    Manages memory prioritization and reinforcement based on usage patterns.
    """

    def __init__(self, memory_manager: MemoryManager, neural_mesh: NeuralMesh):
        self.memory = memory_manager
        self.neural_mesh = neural_mesh
        self.access_counts = {}  # Track memory access frequency

    async def reinforce_memory(self, node_ids: List[str], success_score: float = 1.0):
        """
        Apply reinforcement learning to memory connections.

        Args:
            node_ids: Memory nodes that were successfully accessed
            success_score: Score indicating success of retrieval/use
        """
        try:
            # Update access counts
            for node_id in node_ids:
                self.access_counts[node_id] = self.access_counts.get(node_id, 0) + 1

            # Apply Hebbian reinforcement to neural mesh connections
            for node_id in node_ids:
                # Strengthen connections for frequently accessed nodes
                if self.access_counts[node_id] > 3:  # Threshold for "frequent"
                    self.neural_mesh.reinforce_connection(node_id, strength_factor=success_score)

            # Move highly accessed memories to long-term storage
            self._promote_to_long_term()

        except Exception as e:
            logger.error(f"BasalGangliaAgent reinforcement error: {e}")

    def _promote_to_long_term(self):
        """
        Promote frequently accessed memories to long-term storage.
        """
        # Simple promotion logic: move nodes accessed >5 times
        promotion_candidates = [node_id for node_id, count in self.access_counts.items() if count > 5]

        for node_id in promotion_candidates:
            try:
                # Update memory type to long-term in database
                conn = sqlite3.connect(self.memory.db_path)
                cursor = conn.cursor()
                cursor.execute("UPDATE nodes SET memory_type = 'long_term' WHERE node_id = ?", (node_id,))
                conn.commit()
                conn.close()

                # Reset access count after promotion
                self.access_counts[node_id] = 0

                logger.info(f"Promoted node {node_id} to long-term memory")

            except Exception as e:
                logger.error(f"Error promoting node {node_id}: {e}")

    def get_policy_decision(self, query_type: str, memory_results: List[Dict]) -> Dict[str, Any]:
        """
        Make policy decisions about how to handle queries.

        Args:
            query_type: Type of query (search, ingest, etc.)
            memory_results: Memory search results

        Returns:
            Policy decision dictionary
        """
        # Simple policy: prefer mesh expansion for complex queries
        use_mesh = len(memory_results) > 0 and any(r["score"] > 0.8 for r in memory_results)

        return {
            "use_mesh_expansion": use_mesh,
            "prioritize_recent": query_type == "conversational",
            "agent": "basal_ganglia"
        }