from fastapi import HTTPException
from typing import Dict, List, Optional, Any
import sys
import os
import asyncio
from datetime import datetime
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
from simple_concept_agent import SimpleConceptAgent
from memory_agent import SimpleMemoryAgent
from conversation_learning_agent import ConversationLearningAgent
from system_awareness_agent import SystemAwarenessAgent
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

        # Initialize System Awareness Layer (SAL) - disabled for performance
        self.sal = None  # Disabled to reduce overhead

        # Initialize Simple Concept Agent (replaces complex semantic extractor)
        self.concept_agent = SimpleConceptAgent()

        # Initialize Simple Memory Agent (background optimization)
        self.memory_agent = SimpleMemoryAgent(self.memory, self.chat_manager)

        # Initialize Conversation Learning Agent (extracts knowledge from conversations)
        self.conversation_learner = ConversationLearningAgent(self.memory, self.chat_manager)

        # Initialize GGUF model loader
        self.model_loader = GGUFModelLoader()

        # Initialize System Awareness Agent
        self.awareness_agent = SystemAwarenessAgent(
            self, self.memory, self.neural_mesh, self.concept_agent, self.model_loader
        )

        # Disable complex prompt manager for performance - use simple prompts
        self.prompt_manager = None  # Disabled for performance

        # Initialize agents with model integration and prompt manager (SAL connection later)
        self.hippocampus = HippocampusAgent(memory_manager, self.model_loader, self.prompt_manager)
        self.memory_transformer = MemoryTransformer()
        self.neural_mesh = NeuralMesh()
        self.prefrontal_cortex = PrefrontalCortexAgent(memory_manager, self.model_loader, self.prompt_manager)
        self.amygdala = AmygdalaAgent(self.model_loader, self.prompt_manager)
        self.thalamus_router = ThalamusRouter(self.model_loader, self.prompt_manager)

        # Update chat manager with prompt manager (disabled for performance)
        self.chat_manager.set_prompt_manager(None)

        # Legacy agents for compatibility
        self.cortex = None  # Will be handled by routing
        self.basal_ganglia = None  # Will be handled by routing

        self.session_context = {}  # Keep for backward compatibility, but prefer chat_manager

    async def initialize(self):
        """Async initialization method to set up SAL connections and learning systems"""
        try:
            # Connect PromptManager to SAL
            if self.prompt_manager and self.sal:
                self.prompt_manager.sal = self.sal  # Set SAL reference
                await self.prompt_manager._connect_to_sal()

            # Connect agents to SAL for inter-brain communication
            await self._connect_agents_to_sal()

            # Start memory consolidation scheduler
            await self._start_memory_consolidation_scheduler()

            # Initialize neural mesh synchronization
            await self._initialize_learning_sync()

            # Start simple memory agent background tasks
            await self.memory_agent.run_background_tasks()

            # Start conversation learning agent background tasks
            await self.conversation_learner.start_background_learning()

            logger.info("Orchestrator fully initialized with active learning systems")

        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            # Continue with degraded functionality

    async def _connect_agents_to_sal(self):
        """Connect all agents to the System Awareness Layer"""
        # SAL is disabled for performance - skip connection
        if self.sal is None:
            logger.info("SAL is disabled - skipping agent connections")
            return

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

            # Check for continuation requests and enhance query with context
            if session_history and self._is_continuation_request(query):
                recent_context = await self._build_continuation_context(session_history, query)
                if recent_context:
                    query = f"Context from our recent conversation: {recent_context}\n\nCurrent request: {query}"

            # Check if System Awareness Agent can handle this query
            awareness_result = await self.awareness_agent.process_query(
                query, session_id or "anonymous", [], {"agent": "unknown", "complexity_score": 0.5}
            )

            if awareness_result:
                # System Awareness Agent handled the query - use its result
                response, metadata = awareness_result
                agent_result = {
                    "response": response,
                    "confidence": 0.9,
                    "model_used": "system_awareness"
                }
                thought_trace = f"System awareness query processed by {metadata.get('agent_used', 'system_awareness_agent')}"
                routing_decision = {"agent": "system_awareness_agent", "complexity_score": 0.5}
                relevant_memory_chunks = []  # Awareness agent doesn't use memory chunks
                memory_chunks = []  # No memory search needed
            else:
                # Normal processing - search memory and route to agents
                # Simplified memory search - mesh expansion disabled for performance
                memory_results = self.memory.search_memory(query, top_k=5)  # Reduced from 10, no mesh expansion
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
                routing_decision = await self.thalamus_router.route_query(query, relevant_memory_chunks)

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

                    # Create response using memory context with adaptive prompts
                    session_id_for_user = session_id or "anonymous"
                    if relevant_memory_chunks:
                        context = "\n".join([f"Memory: {chunk['content'][:200]}" for chunk in relevant_memory_chunks[:2]])
                        if self.prompt_manager:
                            # Use adaptive prompt selection based on user behavior
                            prompt, adaptation_metadata = await self.prompt_manager.get_adaptive_prompt(
                                "system.orchestrator.contextual_response",
                                session_id_for_user,
                                query,
                                context={"memory_context": context, "query": query}
                            )
                            logger.debug(f"Selected adaptive prompt variant: {adaptation_metadata.get('selected_variant', 'unknown')}")
                        else:
                            prompt = f"Using this context:\n{context}\n\nProvide a clear, concise explanation of: {query}\n\nKeep your answer informative but not overly verbose."
                    else:
                        if self.prompt_manager:
                            # Use adaptive prompt for standard responses
                            prompt, adaptation_metadata = await self.prompt_manager.get_adaptive_prompt(
                                "system.orchestrator.standard_response",
                                session_id_for_user,
                                query,
                                context={"query": query}
                            )
                            logger.debug(f"Selected adaptive prompt variant: {adaptation_metadata.get('selected_variant', 'unknown')}")
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

            # TEMPORARILY DISABLED: Reinforcement learning causing freezes
            # TODO: Replace with simple background Memory Optimization Agent
            # if memory_node_ids:
            #     confidence = agent_result.get("confidence", 0.5)
            #     self._apply_memory_reinforcement(memory_node_ids, confidence, query, agent_result["response"],
            #                                    session_id, session_history)

            # Publish query completion event to SAL for coordination
            if self.sal:
                await self.sal.event_bus.publish("orchestrator.query_completed", {
                    "query": query[:200],  # Truncated for event
                    "agent_used": routing_decision["agent"],
                    "response_length": len(agent_result["response"]),
                    "confidence": agent_result.get("confidence", 0.5),
                    "memory_chunks_used": len(relevant_memory_chunks),
                    "processing_time": "estimated",  # Could be measured
                    "timestamp": datetime.now().isoformat()
                })

                # Trigger advanced coordination for complex queries
                if routing_decision.get("complexity_score", 0) > 0.6:
                    coordination_result = await self.sal.coordinate_brain_activity(
                        "complex_reasoning",
                        {
                            "query": query,
                            "complexity": routing_decision["complexity_score"],
                            "memory_context": len(relevant_memory_chunks),
                            "emotional_context": "analyze" in query.lower() or "feel" in query.lower()
                        },
                        "normal"
                    )
                    logger.info(f"Advanced coordination completed: {coordination_result.get('coordination_type', 'unknown')}")

            # Get policy decisions (simplified)
            policy_decision = {
                "use_mesh_expansion": len(memory_chunks) > 0,
                "prioritize_recent": False,
                "agent": "thalamus_router"
            }

            # Store conversation in persistent storage
            if session_id:
                # Check if session exists, create if it doesn't
                session_info = self.chat_manager.get_session_info(session_id)
                if not session_info:
                    # Create new session with the provided session_id
                    self.chat_manager.create_chat_session(session_id)
                    session_info = {"message_count": 0}

                is_first_message = session_info["message_count"] == 0

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
                    memory_stats=self.memory.get_memory_stats()
                )

                # Trigger conversation learning in background
                asyncio.create_task(
                    self.conversation_learner.analyze_recent_conversation(session_id)
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

    def _is_continuation_request(self, query: str) -> bool:
        """Check if query is asking for more information or continuation."""
        continuation_patterns = [
            'more details', 'tell me more', 'expand on', 'explain more',
            'elaborate', 'give me more', 'continue', 'what else'
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in continuation_patterns)

    async def _build_continuation_context(self, session_history: List[Dict], current_query: str) -> Optional[str]:
        """Build context from recent conversation for continuation requests."""
        if not session_history or len(session_history) < 2:
            return None

        # Get the last Q&A pair (most recent conversation)
        recent_messages = session_history[-4:]  # Last 2 exchanges

        context_parts = []

        for msg in recent_messages:
            if 'query' in msg:
                context_parts.append(f"User asked: {msg['query']}")
            elif 'response' in msg:
                context_parts.append(f"AI responded: {msg['response'][:300]}...")  # Truncate long responses

        if context_parts:
            return " ".join(context_parts)

        return None

    def _apply_memory_reinforcement(self, memory_node_ids: List[str], confidence: float,
                                   user_query: str, ai_response: str, session_id: Optional[str] = None,
                                   session_history: Optional[List[Dict]] = None):
        """
        Apply reinforcement learning using the memory consolidation system.
        Strengthens memory nodes that contributed to successful responses.

        Args:
            memory_node_ids: List of memory node IDs that were retrieved
            confidence: Confidence score of the AI response
            user_query: The user's original query
            ai_response: The AI's response
            session_id: Optional session identifier
            session_history: Optional conversation history
        """
        try:
            if not memory_node_ids:
                return

            # Use the memory consolidator for reinforcement
            reinforcement_result = self.memory.consolidator.reinforce_memory(
                memory_node_ids, confidence, "conversation_reinforcement"
            )

            # Also maintain neural mesh connections for backward compatibility
            if len(memory_node_ids) >= 2:
                # Calculate reinforcement strength based on confidence
                base_reinforcement = min(0.3, confidence * 0.15)

                # Extract concepts from query and response for targeted reinforcement
                # Include conversation context for better concept extraction
                conversation_context = {
                    'session_id': session_id,
                    'emotional_context': self._analyze_emotional_context(user_query, session_history),
                    'conversation_history': session_history[-3:] if session_history else []
                }

                query_concepts = self._extract_concepts_from_text(user_query, context=conversation_context)
                response_concepts = self._extract_concepts_from_text(ai_response, context=conversation_context)

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

            logger.debug(f"Memory reinforcement applied: {reinforcement_result}")

        except Exception as e:
            # Don't let reinforcement failures break the conversation flow
            logger.warning(f"Memory reinforcement failed: {e}")

    def _extract_concepts_from_text(self, text: str, context: Optional[Dict] = None) -> List[str]:
        """
        Extract semantic concepts from text using advanced NLP processing.

        Args:
            text: Text to extract concepts from
            context: Optional context information for context-aware extraction

        Returns:
            List of key concept names
        """
        try:
            # Use advanced semantic concept extraction
            extracted_concepts = self.concept_extractor.extract_concepts_from_text(
                text, context=context
            )

            # Return concept names for backward compatibility
            return [concept['name'] for concept in extracted_concepts]

        except Exception as e:
            logger.warning(f"Advanced concept extraction failed: {e}. Falling back to simple extraction.")
            # Fallback to simple extraction
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

    async def _start_memory_consolidation_scheduler(self):
        """
        Start background scheduler for memory consolidation tasks.
        Runs periodic memory decay and consolidation operations.
        """
        import asyncio

        async def consolidation_worker():
            """Background worker for memory consolidation tasks."""
            while True:
                try:
                    # Wait for 1 hour between consolidation cycles
                    await asyncio.sleep(3600)  # 1 hour

                    logger.info("Running scheduled memory consolidation...")

                    # Apply memory decay (24 hours worth)
                    decay_result = self.memory.consolidator.apply_memory_decay(24)
                    logger.info(f"Memory decay applied: {decay_result}")

                    # Consolidate similar memories
                    consolidation_result = self.memory.consolidator.consolidate_similar_memories()
                    logger.info(f"Memory consolidation completed: {consolidation_result}")

                    # Update concept relationships and ontology
                    try:
                        self.concept_extractor.update_semantic_relationships()
                        self.concept_extractor.build_concept_ontology()
                        logger.info("Concept relationships and ontology updated")
                    except Exception as e:
                        logger.warning(f"Concept relationship update failed: {e}")

                    # Get health report
                    health_report = self.memory.consolidator.get_memory_health_report()
                    health_score = health_report.get("health_score", 0)

                    logger.info(f"Memory health score: {health_score:.2f}")

                    # Publish consolidation event to SAL
                    if self.sal:
                        await self.sal.event_bus.publish("memory.consolidation_completed", {
                            "decay_result": decay_result,
                            "consolidation_result": consolidation_result,
                            "health_score": health_score,
                            "timestamp": datetime.now().isoformat()
                        })

                except Exception as e:
                    logger.error(f"Error in memory consolidation scheduler: {e}")
                    # Continue running despite errors

        # Start the background worker
        asyncio.create_task(consolidation_worker())
        logger.info("Memory consolidation scheduler started")

    async def _initialize_learning_sync(self):
        """
        Initialize neural mesh and database synchronization for learning systems.
        """
        try:
            # Perform initial synchronization
            sync_result = self.memory.neural_mesh_bridge.force_full_sync()
            logger.info(f"Initial learning synchronization completed: {sync_result}")

            # Start periodic synchronization
            asyncio.create_task(self._learning_sync_scheduler())

        except Exception as e:
            logger.error(f"Failed to initialize learning sync: {e}")

    async def _learning_sync_scheduler(self):
        """
        Periodic synchronization scheduler for learning systems.
        """
        while True:
            try:
                await asyncio.sleep(300)  # Sync every 5 minutes

                # Check if sync is needed
                sync_status = self.memory.neural_mesh_bridge.get_sync_status()
                if sync_status.get('needs_sync', False):
                    logger.info("Performing periodic learning synchronization...")
                    sync_result = self.memory.neural_mesh_bridge.sync_mesh_to_database()
                    logger.debug(f"Periodic sync completed: {sync_result}")

            except Exception as e:
                logger.error(f"Error in learning sync scheduler: {e}")
                await asyncio.sleep(60)  # Wait before retrying
