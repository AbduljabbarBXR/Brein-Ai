from fastapi import HTTPException
from typing import Dict, List, Optional, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from memory_transformer import MemoryTransformer
from chat_manager import ChatManager
from agents import (
    GGUFModelLoader, HippocampusAgent, PrefrontalCortexAgent, AmygdalaAgent, ThalamusRouter
)
from memory_transformer import MemoryTransformer
from neural_mesh import NeuralMesh
from neural_mesh import NeuralMesh
from sal import SystemAwarenessLayer
from prompt_manager import PromptManager
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator for Brein AI - coordinates between memory, agents, and user queries.
    Now uses multi-model brain-inspired architecture with specialized agents.
    """

    def __init__(self, memory_manager: MemoryManager, chat_manager: ChatManager):
        self.memory = memory_manager
        self.chat_manager = chat_manager
        self.memory_transformer = MemoryTransformer()
        self.neural_mesh = NeuralMesh()

        # Initialize System Awareness Layer (SAL)
        self.sal = SystemAwarenessLayer()

        # Initialize GGUF model loader
        self.model_loader = GGUFModelLoader()

        # Initialize Prompt Manager (SAL connection will be established later)
        self.prompt_manager = PromptManager(prompts_dir="prompts", sal=None)  # Don't pass SAL yet

        # Initialize agents with model integration and prompt manager (SAL connection later)
        self.hippocampus = HippocampusAgent(memory_manager, self.model_loader, self.prompt_manager)
        self.memory_transformer = MemoryTransformer()
        self.neural_mesh = NeuralMesh()
        self.prefrontal_cortex = PrefrontalCortexAgent(memory_manager, self.model_loader, self.prompt_manager)
        self.amygdala = AmygdalaAgent(self.model_loader, self.prompt_manager)
        self.thalamus_router = ThalamusRouter(self.model_loader, self.prompt_manager)

        # Update chat manager with prompt manager
        self.chat_manager.set_prompt_manager(self.prompt_manager)

        # Legacy agents for compatibility
        self.cortex = None  # Will be handled by routing
        self.basal_ganglia = None  # Will be handled by routing

        self.session_context = {}  # Keep for backward compatibility, but prefer chat_manager

    async def initialize(self):
        """Async initialization method to set up SAL connections"""
        try:
            # Connect PromptManager to SAL
            if self.prompt_manager and self.sal:
                self.prompt_manager.sal = self.sal  # Set SAL reference
                await self.prompt_manager._connect_to_sal()

            # Connect agents to SAL for inter-brain communication
            await self._connect_agents_to_sal()

            logger.info("Orchestrator fully initialized with SAL integration")

        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            # Continue with degraded functionality

    async def _connect_agents_to_sal(self):
        """Connect all agents to the System Awareness Layer"""
        try:
            # Initialize SAL if not already done
            if not self.sal.is_initialized:
                logger.info("Initializing System Awareness Layer...")
                sal_initialized = await self.sal.initialize()
                if not sal_initialized:
                    logger.error("Failed to initialize System Awareness Layer")
                    return

            # Connect each agent to SAL
            await self.hippocampus.set_sal(self.sal)
            await self.prefrontal_cortex.set_sal(self.sal)
            await self.amygdala.set_sal(self.sal)
            await self.thalamus_router.set_sal(self.sal)

            logger.info("All brain agents connected to System Awareness Layer")

        except Exception as e:
            logger.error(f"Error connecting agents to SAL: {e}")
            # Continue without SAL - system will work in degraded mode

    async def process_query(self, query: str, session_id: Optional[str] = None, enable_web_access: bool = False) -> Dict:
        """
        Process a user query using multi-model brain-inspired architecture with intelligent routing.

        Args:
            query: User query string
            session_id: Optional session identifier
            enable_web_access: Whether to allow web fetching for this query

        Returns:
            Dictionary with response, thought_trace, memory_chunks, and metadata
        """
        try:
            # Get session context for continuity
            session_history = self.get_session_context(session_id) if session_id else None

            # Enhanced memory search with top-N and mesh expansion
            memory_results = self.memory.search_memory(query, top_k=10, use_mesh_expansion=True)
            memory_chunks = []
            relevant_memory_chunks = []

            for result in memory_results:
                chunk = {
                    "content": result["content"],
                    "score": result["score"],
                    "type": result["type"],
                    "node_id": result["node_id"]
                }
                memory_chunks.append(chunk)

                # Filter for highly relevant chunks (score > 0.6) to avoid spurious matches
                if result["score"] > 0.6:
                    relevant_memory_chunks.append(chunk)

            # Use Thalamus Router for intelligent model selection
            routing_decision = self.thalamus_router.route_query(query, relevant_memory_chunks)

            # Route to appropriate agent based on complexity
            if routing_decision["agent"] == "prefrontal_cortex":
                # Complex reasoning with Phi-3.1
                agent_result = await self.prefrontal_cortex.process_complex_query(query, relevant_memory_chunks, session_history)
                thought_trace = f"""Complex reasoning activated.

Analysis: {agent_result.get('analysis', '')}

Step-by-step reasoning: {agent_result.get('reasoning_steps', '')}

Decision making: Query routed to prefrontal cortex due to complexity score of {routing_decision.get('complexity_score', 0):.2f}. Using Phi-3.1 model for advanced reasoning capabilities."""

            elif routing_decision["agent"] == "amygdala":
                # Emotional/personality response with Llama-3.2
                emotional_context = self._analyze_emotional_context(query, session_history)
                memory_insights = [chunk["content"][:100] for chunk in relevant_memory_chunks[:3]]
                agent_result = await self.amygdala.generate_personality_response(query, emotional_context, memory_insights)
                thought_trace = f"""Emotional processing activated.

Emotional context analysis: Detected tone '{emotional_context.get('tone', 'neutral')}' with urgency level '{emotional_context.get('urgency', 'normal')}'.

Memory insights considered: {', '.join(memory_insights) if memory_insights else 'No relevant memories found'}

Personality response strategy: Using Llama-3.2 model to generate empathetic, conversational response with emotional tone '{agent_result.get('emotional_tone', 'neutral')}'."""

            else:
                # Standard processing - use Llama-3.2 via Hippocampus for general queries
                # Generate internal thought process first
                if self.prompt_manager:
                    thought_prompt = self.prompt_manager.get_prompt("system.orchestrator.internal_reasoning", query=query)
                else:
                    thought_prompt = f"Think step by step about how to answer: {query}\n\nList the key points to cover in plain text:"
                internal_thought = self.model_loader.generate("llama-3.2", thought_prompt, max_tokens=256, temperature=0.5)

                # Create response using memory context
                if relevant_memory_chunks:
                    context = "\n".join([f"Memory: {chunk['content'][:200]}" for chunk in relevant_memory_chunks[:2]])
                    if self.prompt_manager:
                        prompt = self.prompt_manager.get_prompt("system.orchestrator.contextual_response",
                                                              context=context, query=query)
                    else:
                        prompt = f"Using this context:\n{context}\n\nProvide a clear, concise explanation of: {query}\n\nKeep your answer informative but not overly verbose."
                else:
                    if self.prompt_manager:
                        prompt = self.prompt_manager.get_prompt("system.orchestrator.standard_response", query=query)
                    else:
                        prompt = f"Provide a clear, concise explanation of: {query}\n\nKeep your answer informative but not overly verbose."

                response_text = self.model_loader.generate("llama-3.2", prompt, max_tokens=512, temperature=0.6)
                agent_result = {
                    "response": response_text,
                    "confidence": 0.8,
                    "model_used": "llama-3.2"
                }

                memory_context_info = ""
                if relevant_memory_chunks:
                    memory_context_info = f"\n\nMemory context retrieved: {len(relevant_memory_chunks)} relevant chunks found with scores ranging from {min([c['score'] for c in relevant_memory_chunks]):.2f} to {max([c['score'] for c in relevant_memory_chunks]):.2f}"

                # Clean up markdown formatting from internal thought
                cleaned_thought = internal_thought.strip().replace('**', '').replace('*', '')

                thought_trace = f"""Internal reasoning: {cleaned_thought}

Query routing: Complexity score {routing_decision.get('complexity_score', 0):.2f} determined standard processing appropriate.{memory_context_info}

Response generation: Created helpful response using memory-augmented context and step-by-step reasoning."""

            # Extract memory node IDs for reinforcement
            memory_node_ids = [chunk["node_id"] for chunk in memory_chunks]

            # Apply reinforcement learning via Basal Ganglia (simplified for now)
            if memory_node_ids:
                confidence = agent_result.get("confidence", 0.5)
                self._apply_conversation_reinforcement(memory_node_ids, confidence, query, agent_result["response"])

            # Get policy decisions (simplified)
            policy_decision = {
                "use_mesh_expansion": len(memory_chunks) > 0,
                "prioritize_recent": False,
                "agent": "thalamus_router"
            }

            # Store conversation in persistent storage
            if session_id:
                # Check if this is the first message in the session
                session_info = self.chat_manager.get_session_info(session_id)
                is_first_message = session_info and session_info["message_count"] == 0

                # Generate smart title for new sessions
                if is_first_message:
                    smart_title = self.chat_manager.generate_smart_title(query)
                    self.chat_manager.update_chat_title(session_id, smart_title)

                # Store the messages
                self.chat_manager.add_message(
                    session_id=session_id,
                    role="user",
                    content=query
                )

                self.chat_manager.add_message(
                    session_id=session_id,
                    role="ai",
                    content=agent_result["response"],
                    thought_trace=thought_trace,
                    memory_stats=result.get("memory_stats")
                )

            return {
                "response": agent_result["response"],
                "thought_trace": thought_trace,
                "confidence": agent_result.get("confidence", 0.5),
                "memory_chunks": relevant_memory_chunks,  # Only return relevant chunks
                "session_id": session_id or "default",
                "memory_stats": self.memory.get_memory_stats(),
                "reasoning_metadata": {
                    "routing_decision": routing_decision,
                    "model_used": agent_result.get("model_used", "unknown"),
                    "agent_used": routing_decision["agent"],
                    "total_memory_chunks": len(memory_chunks),
                    "relevant_memory_chunks": len(relevant_memory_chunks)
                },
                "agents_used": [routing_decision["agent"], "basal_ganglia", "thalamus_router"],
                "policy_decision": policy_decision,
                "web_access_enabled": enable_web_access
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

    def _analyze_emotional_context(self, query: str, session_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Analyze emotional context from query and conversation history.
        """
        emotional_indicators = {
            'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'excited', 'love'],
            'negative': ['bad', 'terrible', 'awful', 'hate', 'angry', 'frustrated', 'sad', 'worried'],
            'urgent': ['urgent', 'emergency', 'immediately', 'asap', 'quickly', 'help'],
            'questioning': ['why', 'how', 'what', 'confused', 'unsure']
        }

        query_lower = query.lower()
        tone_scores = {tone: sum(1 for word in words if word in query_lower)
                      for tone, words in emotional_indicators.items()}

        # Determine primary tone
        max_tone = max(tone_scores, key=tone_scores.get)
        primary_tone = max_tone if tone_scores[max_tone] > 0 else 'neutral'

        # Consider session history for context
        urgency = 'normal'
        if session_history and len(session_history) > 0:
            recent_queries = [conv.get('query', '').lower() for conv in session_history[-3:]]
            urgent_words = sum(1 for q in recent_queries for word in emotional_indicators['urgent'] if word in q)
            if urgent_words > 0:
                urgency = 'high'

        return {
            'tone': primary_tone,
            'urgency': urgency,
            'scores': tone_scores
        }

    async def ingest_content(self, content: str, content_type: str = "stable") -> Dict:
        """
        Ingest new content into memory using Hippocampus Agent with AI-powered processing.

        Args:
            content: Content to ingest
            content_type: Type of content (stable, conversational, functional)

        Returns:
            Dictionary with node_id and status
        """
        try:
            # Use Hippocampus Agent for intelligent ingestion
            result = await self.hippocampus.ingest_content(content, content_type)

            if result["status"] == "error":
                raise HTTPException(status_code=500, detail=result["error"])

            return {
                "status": result["status"],
                "node_ids": result["node_ids"],
                "chunks_created": result["chunks_created"],
                "content_type": result["content_type"],
                "summary": result.get("summary", ""),
                "key_concepts": result.get("key_concepts", []),
                "agents_used": ["hippocampus"],
                "model_used": "llama-3.2"
            }

        except Exception as e:
            logger.error(f"Error ingesting content: {e}")
            raise HTTPException(status_code=500, detail=f"Content ingestion failed: {str(e)}")

    def get_session_context(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session from persistent storage."""
        messages = self.chat_manager.get_chat_history(session_id)
        # Convert to the expected format for backward compatibility
        context = []
        for msg in messages:
            context.append({
                "query" if msg["role"] == "user" else "response": msg["content"],
                "timestamp": msg["timestamp"]
            })
        return context

    def _apply_conversation_reinforcement(self, memory_node_ids: List[str], confidence: float,
                                        user_query: str, ai_response: str):
        """
        Apply reinforcement learning based on successful conversation outcomes.
        Strengthens connections between memory nodes that contributed to good responses.

        Args:
            memory_node_ids: List of memory node IDs that were retrieved
            confidence: Confidence score of the AI response
            user_query: The user's original query
            ai_response: The AI's response
        """
        try:
            if len(memory_node_ids) < 2:
                return  # Need at least 2 nodes to create connections

            # Calculate reinforcement strength based on confidence
            base_reinforcement = min(0.3, confidence * 0.15)  # Scale confidence to meaningful reward

            # Extract concepts from query and response for targeted reinforcement
            query_concepts = self._extract_concepts_from_text(user_query)
            response_concepts = self._extract_concepts_from_text(ai_response)

            # Find overlapping concepts between query, response, and memory nodes
            overlapping_concepts = set(query_concepts) & set(response_concepts)

            if overlapping_concepts:
                # Boost reinforcement for nodes that helped answer the query
                reinforcement_boost = base_reinforcement * (1 + len(overlapping_concepts) * 0.1)

                # Create connections between all memory nodes that contributed
                for i in range(len(memory_node_ids)):
                    for j in range(i + 1, len(memory_node_ids)):
                        node_a = memory_node_ids[i]
                        node_b = memory_node_ids[j]

                        # Apply reinforcement with concept overlap bonus
                        self.memory.neural_mesh.reinforce_connection(
                            node_a, node_b, reinforcement_boost
                        )

                        # Also create direct semantic connections if concepts overlap significantly
                        if len(overlapping_concepts) > 2:
                            self.memory.neural_mesh.add_edge(
                                node_a, node_b, reinforcement_boost * 0.5, "semantic_reinforcement"
                            )

        except Exception as e:
            # Don't let reinforcement failures break the conversation flow
            logger.warning(f"Conversation reinforcement failed: {e}")

    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """
        Extract key concepts from text for reinforcement learning.

        Args:
            text: Text to extract concepts from

        Returns:
            List of key concepts
        """
        # Simple concept extraction
        words = text.lower().split()
        concepts = []

        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who', 'this', 'that', 'these', 'those'}
        min_word_length = 3

        for word in words:
            word = word.strip('.,!?()[]{}')
            if len(word) >= min_word_length and word not in stop_words:
                concepts.append(word)

        return concepts[:15]  # Limit concepts

    def clear_session(self, session_id: str):
        """Clear conversation history for a session from persistent storage."""
        self.chat_manager.delete_chat_session(session_id)
