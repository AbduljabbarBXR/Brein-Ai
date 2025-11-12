from typing import Dict, List, Optional, Any
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from memory_transformer import MemoryTransformer
from neural_mesh import NeuralMesh

logger = logging.getLogger(__name__)

class GGUFModelLoader:
    """
    GGUF Model Loader - Handles loading and inference with GGUF quantized models.
    Provides unified interface for different model types with caching and error handling.
    """

    def __init__(self):
        self.models = {}  # Cache loaded models
        self.model_paths = {
            "phi-3.1": os.path.join("..", "models", "Phi-3.1-mini-128k-instruct-Q4_K_M.gguf"),
            "tinyllama": os.path.join("..", "models", "tinyllama-1.1b-chat-v1.0-q4_k_m.gguf"),
            "llama-3.2": os.path.join("..", "models", "llama-3.2-1b-instruct-q4_k_m.gguf")
        }

    def load_model(self, model_name: str):
        """Load a GGUF model with caching."""
        if model_name in self.models:
            return self.models[model_name]

        try:
            from llama_cpp import Llama

            model_path = self.model_paths.get(model_name)
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model with optimized settings for CPU inference
            model = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window
                n_threads=4,  # CPU threads
                n_gpu_layers=0,  # CPU only for now
                verbose=False
            )

            self.models[model_name] = model
            logger.info(f"Loaded GGUF model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def generate(self, model_name: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Generate text using specified model."""
        try:
            model = self.load_model(model_name)

            # Create chat format based on model
            if model_name == "phi-3.1":
                formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
            elif model_name == "tinyllama":
                formatted_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>"
            elif model_name == "llama-3.2":
                formatted_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            else:
                formatted_prompt = prompt

            response = model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=[]  # Remove stop tokens to see if that's causing issues
            )

            return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Generation failed for {model_name}: {e}")
            return f"Error generating response: {str(e)}"

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Unloaded model: {model_name}")

class HippocampusAgent:
    """
    Hippocampus Agent - Handles memory encoding and ingestion.
    Responsible for converting input content into memory representations.
    Now uses Llama-3.2 for intelligent content processing and summarization.
    """

    def __init__(self, memory_manager: MemoryManager, model_loader: GGUFModelLoader, prompt_manager=None):
        self.memory = memory_manager
        self.model_loader = model_loader
        self.prompt_manager = prompt_manager
        self.sal = None  # System Awareness Layer reference

    async def set_sal(self, sal):
        """Connect to System Awareness Layer for inter-brain communication"""
        self.sal = sal

        # Subscribe to relevant events
        if self.sal:
            await self.sal.event_bus.subscribe("amygdala.emotional_*", self.on_emotional_update)
            await self.sal.event_bus.subscribe("prefrontal_cortex.reasoning_*", self.on_reasoning_context)
            await self.sal.event_bus.subscribe("sal.resource_*", self.on_resource_request)

            # Update SAL state
            await self.sal.brain_state.update_component_state('hippocampus', {
                'status': 'connected',
                'health_status': 'healthy'
            })

    async def on_emotional_update(self, event: str, data: dict):
        """Respond to emotional context updates from amygdala"""
        if data.get('emotional_boost'):
            # Adjust memory retrieval priority based on emotional context
            self.emotional_memory_boost = data['emotional_boost']
            logger.info(f"Hippocampus adjusting for emotional boost: {data['emotional_boost']}")

    async def on_reasoning_context(self, event: str, data: dict):
        """Respond to reasoning context from prefrontal cortex"""
        if data.get('focus_areas'):
            # Prioritize memory search in specific areas
            self.focus_areas = data['focus_areas']
            logger.info(f"Hippocampus focusing on areas: {data['focus_areas']}")

    async def on_resource_request(self, event: str, data: dict):
        """Handle resource requests from SAL"""
        if data.get('resource_type') == 'memory':
            # Provide memory usage information
            memory_stats = self.memory.get_memory_stats()
            await self.sal.brain_state.update_component_state('hippocampus', {
                'memory_load': memory_stats.get('total_nodes', 0),
                'last_resource_check': data.get('timestamp')
            })

    async def ingest_content(self, content: str, content_type: str = "stable") -> Dict[str, Any]:
        """
        Ingest new content into memory with intelligent processing.

        Args:
            content: Content to ingest
            content_type: Type of content (stable, conversational, functional)

        Returns:
            Dictionary with ingestion results
        """
        try:
            # Use prompt manager for content processing prompts
            if self.prompt_manager:
                summary_prompt = self.prompt_manager.get_prompt("agents.hippocampus.content_summary", content=content[:2000])
                concepts_prompt = self.prompt_manager.get_prompt("agents.hippocampus.key_concepts_extraction", content=content[:1500])
            else:
                # Fallback to hardcoded prompts if prompt_manager not available
                summary_prompt = f"Summarize the key information from this content in 2-3 sentences, focusing on facts and concepts that would be important to remember:\n\n{content[:2000]}..."
                concepts_prompt = f"Extract 3-5 key concepts or topics from this content:\n\n{content[:1500]}..."

            # Use Llama-3.2 to analyze and summarize content for better memory encoding
            summary = self.model_loader.generate("llama-3.2", summary_prompt, max_tokens=128, temperature=0.3)

            # Extract key concepts for better searchability
            concepts_text = self.model_loader.generate("llama-3.2", concepts_prompt, max_tokens=64, temperature=0.2)
            key_concepts = [c.strip() for c in concepts_text.split(',') if c.strip()]

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
                "summary": summary,
                "key_concepts": key_concepts,
                "agent": "hippocampus"
            }

        except Exception as e:
            logger.error(f"HippocampusAgent ingestion error: {e}")
            error_msg = str(e) if self.prompt_manager else f"Error: {str(e)}"
            return {
                "status": "error",
                "error": error_msg,
                "agent": "hippocampus"
            }

class PrefrontalCortexAgent:
    """
    Prefrontal Cortex Agent - Handles complex reasoning and planning.
    Uses Phi-3.1 model for advanced reasoning, problem-solving, and strategic thinking.
    """

    def __init__(self, memory_manager: MemoryManager, model_loader: GGUFModelLoader, prompt_manager=None):
        self.memory = memory_manager
        self.model_loader = model_loader
        self.prompt_manager = prompt_manager
        self.sal = None  # System Awareness Layer reference

    async def set_sal(self, sal):
        """Connect to System Awareness Layer for inter-brain communication"""
        self.sal = sal

        # Subscribe to coordination events
        if self.sal:
            await self.sal.event_bus.subscribe("thalamus_router.complexity_*", self.on_complexity_assessment)
            await self.sal.event_bus.subscribe("hippocampus.memory_*", self.on_memory_update)
            await self.sal.event_bus.subscribe("amygdala.emotional_*", self.on_emotional_context)

            # Update SAL state
            await self.sal.brain_state.update_component_state('prefrontal_cortex', {
                'status': 'connected',
                'health_status': 'healthy'
            })

    async def on_complexity_assessment(self, event: str, data: dict):
        """Respond to complexity assessments from thalamus router"""
        if data.get('complexity_score', 0) >= 0.7:
            # Prepare for complex reasoning task
            self.reasoning_mode = 'complex'
            logger.info(f"Prefrontal cortex preparing for complex reasoning: {data['complexity_score']}")

    async def on_memory_update(self, event: str, data: dict):
        """Respond to memory updates from hippocampus"""
        if data.get('new_nodes'):
            # Update reasoning context with new memory information
            self.available_memory_nodes = data.get('new_nodes', [])
            logger.info(f"Prefrontal cortex updated with {len(self.available_memory_nodes)} new memory nodes")

    async def on_emotional_context(self, event: str, data: dict):
        """Respond to emotional context from amygdala"""
        if data.get('emotional_tone'):
            # Adjust reasoning approach based on emotional context
            self.emotional_context = data
            logger.info(f"Prefrontal cortex adjusting for emotional context: {data['emotional_tone']}")

    async def process_complex_query(self, query: str, memory_context: List[Dict], session_context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Process complex queries requiring deep reasoning and planning.

        Args:
            query: Complex user query
            memory_context: Relevant memory chunks
            session_context: Previous conversation context

        Returns:
            Dictionary with reasoned response and planning
        """
        try:
            # Build context from memory and session
            context_text = self._build_context_text(memory_context, session_context)

            # Use prompt manager for complex reasoning prompt
            if self.prompt_manager:
                reasoning_prompt = self.prompt_manager.get_prompt("agents.prefrontal_cortex.complex_reasoning",
                                                                query=query, context=context_text)
            else:
                # Fallback to hardcoded prompt
                reasoning_prompt = f"""Analyze this query and provide a well-reasoned response. Consider the context provided and think step-by-step.

Query: {query}

Context:
{context_text}

Please provide:
1. Your analysis of the query
2. Step-by-step reasoning
3. A clear, helpful response
4. Any recommendations or next steps"""

            response = self.model_loader.generate("phi-3.1", reasoning_prompt, max_tokens=512, temperature=0.7)

            # Extract reasoning components
            lines = response.split('\n')
            analysis = ""
            reasoning = ""
            final_response = ""
            recommendations = ""

            current_section = ""
            for line in lines:
                line = line.strip()
                if line.startswith('1.') or 'analysis' in line.lower():
                    current_section = "analysis"
                    analysis += line + " "
                elif line.startswith('2.') or 'reasoning' in line.lower():
                    current_section = "reasoning"
                    reasoning += line + " "
                elif line.startswith('3.') or 'response' in line.lower():
                    current_section = "response"
                    final_response += line + " "
                elif line.startswith('4.') or 'recommendations' in line.lower():
                    current_section = "recommendations"
                    recommendations += line + " "
                else:
                    if current_section == "analysis":
                        analysis += line + " "
                    elif current_section == "reasoning":
                        reasoning += line + " "
                    elif current_section == "response":
                        final_response += line + " "
                    elif current_section == "recommendations":
                        recommendations += line + " "

            return {
                "response": final_response.strip(),
                "analysis": analysis.strip(),
                "reasoning_steps": reasoning.strip(),
                "recommendations": recommendations.strip(),
                "model_used": "phi-3.1",
                "confidence": 0.85,  # Phi-3.1 is quite capable
                "agent": "prefrontal_cortex"
            }

        except Exception as e:
            logger.error(f"PrefrontalCortexAgent error: {e}")
            return {
                "response": f"I encountered an error in complex reasoning: {str(e)}",
                "analysis": "Error in analysis",
                "reasoning_steps": "Error in reasoning",
                "recommendations": "Unable to provide recommendations",
                "model_used": "phi-3.1",
                "confidence": 0.0,
                "agent": "prefrontal_cortex"
            }

    def _build_context_text(self, memory_context: List[Dict], session_context: Optional[List[Dict]] = None) -> str:
        """Build context text from memory and session data."""
        context_parts = []

        if memory_context:
            context_parts.append("Memory Context:")
            for mem in memory_context[:3]:  # Limit to top 3 memories
                context_parts.append(f"- {mem.get('content', '')[:200]}...")

        if session_context:
            context_parts.append("\nRecent Conversation:")
            for conv in session_context[-3:]:  # Last 3 exchanges
                if isinstance(conv, dict):
                    context_parts.append(f"- {conv.get('query', '')} -> {conv.get('response', '')[:100]}...")

        return '\n'.join(context_parts)

class AmygdalaAgent:
    """
    Amygdala Agent - Handles emotional intelligence and personality.
    Uses TinyLlama for natural, conversational responses with emotional awareness.
    """

    def __init__(self, model_loader: GGUFModelLoader, prompt_manager=None):
        self.model_loader = model_loader
        self.prompt_manager = prompt_manager
        self.sal = None  # System Awareness Layer reference

    async def set_sal(self, sal):
        """Connect to System Awareness Layer for inter-brain communication"""
        self.sal = sal

        # Subscribe to relevant events
        if self.sal:
            await self.sal.event_bus.subscribe("hippocampus.memory_*", self.on_memory_context)
            await self.sal.event_bus.subscribe("prefrontal_cortex.reasoning_*", self.on_reasoning_context)
            await self.sal.event_bus.subscribe("thalamus_router.routing_*", self.on_routing_decision)

            # Update SAL state
            await self.sal.brain_state.update_component_state('amygdala', {
                'status': 'connected',
                'health_status': 'healthy'
            })

    async def on_memory_context(self, event: str, data: dict):
        """Respond to memory context from hippocampus"""
        if data.get('emotional_memory'):
            # Enhance emotional processing with memory context
            self.memory_emotional_context = data
            logger.info("Amygdala enhanced with emotional memory context")

    async def on_reasoning_context(self, event: str, data: dict):
        """Respond to reasoning context from prefrontal cortex"""
        if data.get('emotional_tone'):
            # Adjust emotional response based on reasoning context
            self.reasoning_emotional_context = data
            logger.info(f"Amygdala adjusting for reasoning emotional context: {data['emotional_tone']}")

    async def on_routing_decision(self, event: str, data: dict):
        """Respond to routing decisions from thalamus router"""
        if data.get('agent') == 'amygdala':
            # Prepare for emotional processing task
            self.active_emotional_task = True
            logger.info("Amygdala activated for emotional processing")

    async def generate_personality_response(self, query: str, emotional_context: Dict[str, Any],
                                          memory_insights: List[str] = None) -> Dict[str, Any]:
        """
        Generate emotionally intelligent, personality-driven responses.
        Now includes identity and self-awareness responses for Brein AI.

        Args:
            query: User query
            emotional_context: Emotional state and context
            memory_insights: Relevant memory insights

        Returns:
            Dictionary with personality response
        """
        try:
            # Check if this is an identity/architecture/awareness question
            query_lower = query.lower()
            identity_keywords = ['who are you', 'what are you', 'what is your', 'architecture', 'awareness', 'brain', 'brein']

            is_identity_question = any(keyword in query_lower for keyword in identity_keywords)

            if self.prompt_manager and is_identity_question:
                # Use identity prompts for self-description
                if 'architecture' in query_lower:
                    response = self.prompt_manager.get_prompt("core.identity.architecture_overview")
                elif 'awareness' in query_lower or 'aware' in query_lower:
                    response = self.prompt_manager.get_prompt("core.identity.awareness_description")
                else:
                    response = self.prompt_manager.get_prompt("core.identity.who_am_i")

                return {
                    "response": response,
                    "emotional_tone": "confident",
                    "personality_traits": ["self-aware", "intelligent", "transparent"],
                    "model_used": "tinyllama",
                    "confidence": 0.9,
                    "agent": "amygdala"
                }

            # Use prompt manager for personality response prompt
            if self.prompt_manager:
                relevant_memories = ', '.join(memory_insights) if memory_insights else ''
                personality_prompt = self.prompt_manager.get_prompt("agents.amygdala.personality_response",
                                                                  query=query,
                                                                  emotional_context=f"{emotional_context.get('tone', 'neutral')}, {emotional_context.get('urgency', 'normal')}",
                                                                  relevant_memories=relevant_memories)
            else:
                # Fallback to hardcoded prompt
                personality_prompt = f"""You are a helpful, empathetic AI assistant with a warm personality. Respond naturally and conversationally to this query, considering the emotional context.

Query: {query}

Emotional Context: {emotional_context.get('tone', 'neutral')}, {emotional_context.get('urgency', 'normal')}

{f'Relevant Memories: ' + ', '.join(memory_insights) if memory_insights else ''}

Respond in a friendly, engaging way that shows you understand and care about the user's needs."""

            response = self.model_loader.generate("tinyllama", personality_prompt, max_tokens=256, temperature=0.8)

            # Analyze emotional tone of response
            emotional_tone = self._analyze_emotional_tone(response)

            return {
                "response": response,
                "emotional_tone": emotional_tone,
                "personality_traits": ["empathetic", "helpful", "conversational"],
                "model_used": "tinyllama",
                "confidence": 0.75,
                "agent": "amygdala"
            }

        except Exception as e:
            logger.error(f"AmygdalaAgent error: {e}")
            return {
                "response": f"I'm here to help! {query}",
                "emotional_tone": "supportive",
                "personality_traits": ["empathetic", "helpful"],
                "model_used": "tinyllama",
                "confidence": 0.0,
                "agent": "amygdala"
            }

    def _analyze_emotional_tone(self, text: str) -> str:
        """Simple emotional tone analysis."""
        text_lower = text.lower()
        if any(word in text_lower for word in ['excited', 'great', 'wonderful', 'amazing']):
            return "enthusiastic"
        elif any(word in text_lower for word in ['sorry', 'unfortunately', 'problem']):
            return "concerned"
        elif any(word in text_lower for word in ['help', 'assist', 'support']):
            return "supportive"
        else:
            return "neutral"

class ThalamusRouter:
    """
    Thalamus Router - Routes queries to appropriate models based on complexity analysis.
    Acts as the brain's relay center for intelligent model selection.
    """

    def __init__(self, model_loader: GGUFModelLoader, prompt_manager=None):
        self.model_loader = model_loader
        self.prompt_manager = prompt_manager
        self.sal = None  # System Awareness Layer reference

    async def set_sal(self, sal):
        """Connect to System Awareness Layer for inter-brain communication"""
        self.sal = sal

        # Update SAL state
        if self.sal:
            await self.sal.brain_state.update_component_state('thalamus_router', {
                'status': 'connected',
                'health_status': 'healthy'
            })
            logger.info("ThalamusRouter connected to SAL")

    def route_query(self, query: str, memory_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze query complexity and route to appropriate model/agent.

        Args:
            query: User query
            memory_results: Memory search results

        Returns:
            Routing decision with model selection
        """
        try:
            # Check for identity questions first (highest priority)
            if self._is_identity_question(query):
                return {
                    "model": "tinyllama",
                    "agent": "amygdala",
                    "complexity_score": 0.0,
                    "reasoning": "Identity or self-awareness question detected - routing to amygdala for appropriate response",
                    "estimated_tokens": 150
                }

            # Analyze query complexity
            complexity_score = self._analyze_complexity(query, memory_results)

            # Route based on complexity and requirements
            if complexity_score >= 0.7:
                # Complex reasoning required
                model_choice = "phi-3.1"
                agent = "prefrontal_cortex"
                reasoning = "Query requires complex reasoning and planning"
            elif self._requires_emotional_intelligence(query):
                # Emotional/social intelligence needed
                model_choice = "tinyllama"
                agent = "amygdala"
                reasoning = "Query benefits from emotional intelligence and personality"
            else:
                # Standard processing
                model_choice = "llama-3.2"
                agent = "hippocampus"
                reasoning = "Query can be handled with standard processing"

            return {
                "model": model_choice,
                "agent": agent,
                "complexity_score": complexity_score,
                "reasoning": reasoning,
                "estimated_tokens": self._estimate_token_usage(query, complexity_score)
            }

        except Exception as e:
            logger.error(f"ThalamusRouter error: {e}")
            return {
                "model": "llama-3.2",
                "agent": "hippocampus",
                "complexity_score": 0.5,
                "reasoning": f"Error in routing: {str(e)}",
                "estimated_tokens": 128
            }

    def _analyze_complexity(self, query: str, memory_results: List[Dict]) -> float:
        """Analyze query complexity on a 0-1 scale."""
        complexity = 0.0

        # Length factor
        if len(query) > 200:
            complexity += 0.2
        elif len(query) > 100:
            complexity += 0.1

        # Question complexity indicators
        complex_indicators = ['why', 'how', 'explain', 'analyze', 'compare', 'evaluate', 'plan', 'strategy']
        query_lower = query.lower()
        matches = sum(1 for indicator in complex_indicators if indicator in query_lower)
        complexity += min(matches * 0.1, 0.3)

        # Memory context availability
        if memory_results and len(memory_results) > 3:
            complexity += 0.2  # More context suggests complex topic

        # Technical/specialized terms
        technical_terms = ['algorithm', 'neural', 'quantum', 'optimization', 'architecture', 'system']
        tech_matches = sum(1 for term in technical_terms if term in query_lower)
        complexity += min(tech_matches * 0.15, 0.3)

        return min(complexity, 1.0)

    def _is_identity_question(self, query: str) -> bool:
        """Check if query is about AI identity, architecture, or self-awareness."""
        identity_indicators = [
            'who are you', 'what are you', 'what is your', 'tell me about yourself',
            'architecture', 'awareness', 'aware', 'brain', 'brein', 'system',
            'how do you work', 'how are you built', 'what makes you'
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in identity_indicators)

    def _requires_emotional_intelligence(self, query: str) -> bool:
        """Check if query requires emotional intelligence."""
        emotional_indicators = [
            'feel', 'emotion', 'happy', 'sad', 'worried', 'excited', 'frustrated',
            'help me', 'support', 'advice', 'opinion', 'think about', 'concerned'
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in emotional_indicators)

    def _estimate_token_usage(self, query: str, complexity: float) -> int:
        """Estimate token usage based on query and complexity."""
        base_tokens = len(query.split()) * 1.5  # Rough token estimation
        complexity_multiplier = 1 + (complexity * 2)  # Complex queries need more tokens
        return int(base_tokens * complexity_multiplier)
