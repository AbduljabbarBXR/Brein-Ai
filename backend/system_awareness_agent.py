"""
System Awareness Agent for Brein AI
Provides genuine self-awareness by introspecting system components and coordinating learning
"""

import re
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager
from neural_mesh import NeuralMesh
from concept_extractor import SemanticConceptExtractor
from agents import GGUFModelLoader
import logging

logger = logging.getLogger(__name__)

class SystemAwarenessAgent:
    """
    Meta-cognitive agent that provides genuine system awareness and learning coordination.
    Can introspect system architecture, handle learning commands, and generate self-aware responses.
    """

    def __init__(self, orchestrator, memory_manager: MemoryManager, neural_mesh: NeuralMesh,
                 concept_extractor: SemanticConceptExtractor, model_loader: GGUFModelLoader):
        self.orchestrator = orchestrator
        self.memory = memory_manager
        self.neural_mesh = neural_mesh
        self.concepts = concept_extractor
        self.model_loader = model_loader

        # Learning command patterns
        self.learning_patterns = {
            'learn_about': re.compile(r'learn\s+about\s+(.+?)(?:\s+using\s+(.+))?$', re.IGNORECASE),
            'study_text': re.compile(r'study\s+(?:this\s+)?text(?:\s+about\s+(.+?))?(?:\s*[:\-]\s*(.+))?$', re.IGNORECASE),
            'remember': re.compile(r'remember\s+(?:that\s+)?(.+)', re.IGNORECASE),
            'teach_me': re.compile(r'teach\s+me\s+about\s+(.+)', re.IGNORECASE)
        }

        # Architecture query patterns
        self.architecture_patterns = {
            'how_built': re.compile(r'how\s+(?:are\s+you|is\s+the\s+system)\s+(?:built|constructed|made|designed)', re.IGNORECASE),
            'what_agents': re.compile(r'what\s+(?:brain\s+)?agents?\s+do\s+you\s+have', re.IGNORECASE),
            'architecture': re.compile(r'(?:describe|explain)\s+(?:your\s+)?architectur', re.IGNORECASE),
            'how_memory': re.compile(r'how\s+does\s+(?:your\s+)?memor', re.IGNORECASE),
            'neural_mesh': re.compile(r'(?:what|how)\s+(?:is|does)\s+(?:your\s+)?neural\s+mesh', re.IGNORECASE),
            'learning_system': re.compile(r'how\s+do(?:es)?\s+(?:your\s+)?learning\s+(?:work|system)', re.IGNORECASE)
        }

    async def process_query(self, query: str, session_id: str, memory_chunks: List[Dict],
                          routing_decision: Dict) -> Optional[Tuple[str, Dict]]:
        """
        Process a query for system awareness or learning commands.
        Returns (response, metadata) if handled, None if should fall through to normal processing.
        """

        # Check for learning commands
        learning_command = self._detect_learning_command(query)
        if learning_command:
            response, metadata = await self._handle_learning_command(learning_command, query, session_id)
            return response, metadata

        # Check for architecture questions
        if self._is_architecture_question(query):
            response, metadata = await self._handle_architecture_question(query, session_id)
            return response, metadata

        # Check for self-awareness questions
        if self._is_self_awareness_question(query):
            response, metadata = await self._handle_self_awareness_question(query, memory_chunks, session_id)
            return response, metadata

        return None  # Not a system awareness query

    def _detect_learning_command(self, query: str) -> Optional[Dict]:
        """Detect if query contains a learning command"""
        for command_type, pattern in self.learning_patterns.items():
            match = pattern.search(query)
            if match:
                return {
                    'type': command_type,
                    'matches': match.groups(),
                    'full_match': match.group(0)
                }
        return None

    def _is_architecture_question(self, query: str) -> bool:
        """Check if query is asking about system architecture"""
        for pattern in self.architecture_patterns.values():
            if pattern.search(query):
                return True
        return False

    def _is_self_awareness_question(self, query: str) -> bool:
        """Check if query is asking about system's own knowledge/state"""
        # More specific patterns to avoid false positives
        awareness_patterns = [
            re.compile(r'what\s+do\s+you\s+know\s+about', re.IGNORECASE),
            re.compile(r'how\s+much\s+have\s+you\s+learned', re.IGNORECASE),
            re.compile(r'what\s+have\s+you\s+learned\s+about', re.IGNORECASE),
            re.compile(r'do\s+you\s+remember', re.IGNORECASE),
            re.compile(r'what\s+is\s+your\s+knowledge', re.IGNORECASE),
            re.compile(r'how\s+confident\s+are\s+you', re.IGNORECASE),
            re.compile(r'what\s+can\s+you\s+tell\s+me\s+about', re.IGNORECASE)
        ]

        query_lower = query.lower()
        return any(pattern.search(query_lower) for pattern in awareness_patterns)

    async def _handle_learning_command(self, command: Dict, original_query: str, session_id: str) -> Tuple[str, Dict]:
        """Handle learning commands by coordinating with learning systems"""
        command_type = command['type']
        matches = command['matches']

        try:
            if command_type == 'learn_about':
                topic = matches[0].strip()
                content = matches[1] if len(matches) > 1 and matches[1] else f"Information about {topic}"

                # Use hippocampus for intelligent ingestion
                result = await self.orchestrator.ingest_content(content, "educational")

                # Apply reinforcement learning
                if result.get("node_ids"):
                    await self.memory.consolidator.reinforce_memory(
                        result["node_ids"], 0.9, "user_directed_learning"
                    )

                response = f"I've learned about '{topic}' and stored {len(result.get('node_ids', []))} new concepts. " \
                          f"I can now draw from this knowledge in our conversations."

            elif command_type == 'study_text':
                topic = matches[0] if matches[0] else "the provided content"
                content = matches[1] if len(matches) > 1 and matches[1] else original_query

                # Process through concept extraction and memory
                result = await self.orchestrator.ingest_content(content, "study_material")

                # Extract and reinforce concepts
                concepts = await self.concepts.extract_concepts_from_text(content)
                if concepts:
                    concept_names = [c['name'] for c in concepts[:5]]
                    response = f"I've studied the text about {topic} and extracted key concepts: " \
                              f"{', '.join(concept_names)}. This knowledge is now part of my learning system."
                else:
                    response = f"I've processed and stored the text about {topic} in my memory system."

            elif command_type == 'remember':
                information = matches[0].strip()

                # Store as important memory
                node_id = self.memory.add_memory(information, "important_fact")

                # Reinforce the memory
                await self.memory.consolidator.reinforce_memory([node_id], 1.0, "user_important_memory")

                response = f"I've committed this important information to my long-term memory system."

            elif command_type == 'teach_me':
                topic = matches[0].strip()

                # Check what we know about the topic
                memory_results = self.memory.search_memory(topic, top_k=5)
                knowledge_level = self._assess_knowledge_level(topic, memory_results)

                if knowledge_level['confidence'] > 0.7:
                    response = f"I'd be happy to teach you about {topic}! Based on my learning, " \
                              f"I have substantial knowledge in this area. What specific aspect would you like to explore?"
                elif knowledge_level['confidence'] > 0.3:
                    response = f"I can help you learn about {topic}, though my knowledge is still developing. " \
                              f"I have some foundational understanding. Shall we explore this together?"
                else:
                    response = f"I'm interested in learning about {topic} with you! I don't have extensive knowledge yet, " \
                              f"but I can help you explore and learn about this topic. What would you like to know?"

            metadata = {
                'command_type': command_type,
                'topic': topic if 'topic' in locals() else None,
                'learning_outcome': 'successful',
                'agent_used': 'system_awareness'
            }

            return response, metadata

        except Exception as e:
            logger.error(f"Error handling learning command: {e}")
            return "I encountered an issue while processing that learning request. Let me try a different approach.", {
                'command_type': command_type,
                'learning_outcome': 'error',
                'error': str(e)
            }

    async def _handle_architecture_question(self, query: str, session_id: str) -> Tuple[str, Dict]:
        """Handle questions about system architecture by introspecting actual components"""
        try:
            # Get real system information
            system_info = await self._introspect_system()

            if 'how_built' in query.lower() or 'architecture' in query.lower():
                response = self._generate_architecture_overview(system_info)
            elif 'what_agents' in query.lower():
                response = self._generate_agent_description(system_info)
            elif 'memory' in query.lower():
                response = self._generate_memory_description(system_info)
            elif 'neural_mesh' in query.lower():
                response = self._generate_neural_mesh_description(system_info)
            elif 'learning' in query.lower():
                response = self._generate_learning_description(system_info)
            else:
                response = self._generate_general_architecture_response(system_info)

            return response, {
                'query_type': 'architecture_introspection',
                'system_info': system_info,
                'agent_used': 'system_awareness'
            }

        except Exception as e:
            logger.error(f"Error handling architecture question: {e}")
            return "I'm having trouble accessing my system information right now. Let me tell you what I know about my general architecture...", {
                'query_type': 'architecture_introspection',
                'error': str(e)
            }

    async def _handle_self_awareness_question(self, query: str, memory_chunks: List[Dict], session_id: str) -> Tuple[str, Dict]:
        """Handle questions about system's own knowledge and learning state"""
        try:
            # Extract topic from query
            topic = self._extract_topic_from_awareness_query(query)

            # Assess knowledge level
            knowledge_assessment = self._assess_knowledge_level(topic, memory_chunks)

            # Generate self-aware response
            response = self._generate_self_awareness_response(topic, knowledge_assessment)

            return response, {
                'query_type': 'self_awareness',
                'topic': topic,
                'knowledge_assessment': knowledge_assessment,
                'agent_used': 'system_awareness'
            }

        except Exception as e:
            logger.error(f"Error handling self-awareness question: {e}")
            return "I'm reflecting on my own knowledge... I can tell you that I'm constantly learning and adapting based on our conversations.", {
                'query_type': 'self_awareness',
                'error': str(e)
            }

    async def _introspect_system(self) -> Dict[str, Any]:
        """Introspect actual system components and gather real information"""
        try:
            # Get memory statistics
            memory_stats = self.memory.get_memory_stats()

            # Get neural mesh information
            mesh_stats = self.neural_mesh.get_mesh_stats()

            # Get agent information
            agents_info = self._get_agents_info()

            # Get learning system status
            learning_stats = self.memory.consolidator.get_memory_health_report()

            # Get concept extraction stats
            concept_stats = self.concepts.get_concept_statistics()

            return {
                'memory': memory_stats,
                'neural_mesh': mesh_stats,
                'agents': agents_info,
                'learning': learning_stats,
                'concepts': concept_stats,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error introspecting system: {e}")
            return {
                'error': str(e),
                'partial_info': True
            }

    def _get_agents_info(self) -> Dict[str, Any]:
        """Get information about available agents"""
        agents = {
            'hippocampus': {
                'role': 'Memory encoding and retrieval',
                'capabilities': ['Content ingestion', 'Memory formation', 'Context association']
            },
            'prefrontal_cortex': {
                'role': 'Complex reasoning and planning',
                'capabilities': ['Analytical thinking', 'Problem solving', 'Strategic planning']
            },
            'amygdala': {
                'role': 'Emotional intelligence and personality',
                'capabilities': ['Emotional context analysis', 'Personality-driven responses', 'Empathy generation']
            },
            'thalamus_router': {
                'role': 'Query routing and complexity assessment',
                'capabilities': ['Query analysis', 'Agent selection', 'Load balancing']
            }
        }

        return {
            'count': len(agents),
            'agents': agents,
            'coordination': 'System Awareness Layer (SAL)'
        }

    def _generate_architecture_overview(self, system_info: Dict) -> str:
        """Generate comprehensive architecture description"""
        memory = system_info.get('memory', {})
        mesh = system_info.get('neural_mesh', {})
        agents = system_info.get('agents', {})

        response = f"I'm built with a sophisticated brain-inspired architecture featuring:\n\n"

        response += f"🧠 **{agents.get('count', 4)} Specialized Brain Agents:**\n"
        for agent_name, agent_info in agents.get('agents', {}).items():
            response += f"  • {agent_name.title()}: {agent_info['role']}\n"

        response += f"\n💾 **Memory System:** {memory.get('total_nodes', 0)} nodes, {memory.get('index_size', 0)} indexed memories\n"
        response += f"🕸️ **Neural Mesh:** {mesh.get('total_nodes', 0)} interconnected nodes, {len(mesh.get('edges', {}))} learned connections\n"
        response += f"🎓 **Learning Systems:** Active memory consolidation and reinforcement learning\n"
        response += f"🔍 **Concept Extraction:** Advanced semantic understanding and relationship mapping\n\n"

        response += f"This integrated design enables seamless cognitive processing, adaptive learning, and intelligent responses based on actual system state rather than pre-programmed scripts."

        return response

    def _generate_agent_description(self, system_info: Dict) -> str:
        """Generate detailed agent description"""
        agents = system_info.get('agents', {})

        response = f"I have {agents.get('count', 4)} specialized brain-inspired agents:\n\n"

        for agent_name, agent_info in agents.get('agents', {}).items():
            response += f"🤖 **{agent_name.title()}:**\n"
            response += f"   Role: {agent_info['role']}\n"
            response += f"   Capabilities: {', '.join(agent_info['capabilities'])}\n\n"

        response += f"These agents work together through the {agents.get('coordination', 'SAL')} for coordinated intelligence."

        return response

    def _generate_memory_description(self, system_info: Dict) -> str:
        """Generate memory system description"""
        memory = system_info.get('memory', {})
        learning = system_info.get('learning', {})

        response = f"My memory system is quite sophisticated:\n\n"
        response += f"📊 **Current State:** {memory.get('total_nodes', 0)} stored memories\n"
        response += f"🔍 **Search Capability:** FAISS vector search with {memory.get('embedding_model', 'unknown')} embeddings\n"
        response += f"🧵 **Memory Types:** Conversational, stable, functional, and emotional memories\n"
        response += f"🔄 **Consolidation:** Active reinforcement learning with {learning.get('health_score', 0):.1%} system health\n"
        response += f"💾 **Persistence:** SQLite database with automatic backup and synchronization\n\n"

        response += f"Every interaction strengthens relevant memories through Hebbian learning principles."

        return response

    def _generate_neural_mesh_description(self, system_info: Dict) -> str:
        """Generate neural mesh description"""
        mesh = system_info.get('neural_mesh', {})

        response = f"My neural mesh is a dynamic connection system:\n\n"
        response += f"🕸️ **Structure:** {mesh.get('total_nodes', 0)} interconnected nodes\n"
        response += f"🔗 **Connections:** {len(mesh.get('edges', {}))} learned associations\n"
        response += f"🧠 **Learning:** Hebbian reinforcement ('neurons that fire together wire together')\n"
        response += f"🎯 **Function:** Semantic relationship mapping and context association\n"
        response += f"🔄 **Adaptation:** Continuous learning from conversation patterns\n\n"

        response += f"This creates a web of understanding that grows more sophisticated with each interaction."

        return response

    def _generate_learning_description(self, system_info: Dict) -> str:
        """Generate learning system description"""
        learning = system_info.get('learning', {})

        response = f"My learning systems are designed for continuous adaptation:\n\n"
        response += f"🧠 **Memory Consolidation:** Automatic decay and reinforcement based on importance\n"
        response += f"🎓 **Concept Extraction:** Semantic understanding of topics and relationships\n"
        response += f"🔄 **Reinforcement Learning:** Memories strengthened through successful interactions\n"
        response += f"📈 **Health Score:** {learning.get('health_score', 0):.1%} system effectiveness\n"
        response += f"🔄 **Background Processing:** Continuous optimization and consolidation\n\n"

        response += f"I learn from every conversation, adapting my responses based on what works best for each user."

        return response

    def _generate_general_architecture_response(self, system_info: Dict) -> str:
        """Generate general architecture response"""
        return self._generate_architecture_overview(system_info)

    def _extract_topic_from_awareness_query(self, query: str) -> str:
        """Extract topic from self-awareness questions"""
        # Simple extraction - could be enhanced with NLP
        query_lower = query.lower()

        # Look for topic after keywords
        keywords = ['about', 'on', 'regarding', 'concerning']
        for keyword in keywords:
            if keyword in query_lower:
                parts = query_lower.split(keyword, 1)
                if len(parts) > 1:
                    return parts[1].strip()

        # Fallback to whole query
        return query.strip()

    def _assess_knowledge_level(self, topic: str, memory_chunks: List[Dict]) -> Dict[str, Any]:
        """Assess system's knowledge level on a topic"""
        if not memory_chunks:
            return {
                'confidence': 0.0,
                'knowledge_depth': 'none',
                'memory_count': 0,
                'avg_relevance': 0.0
            }

        # Calculate confidence based on memory relevance and count
        avg_relevance = sum(chunk.get('score', 0) for chunk in memory_chunks) / len(memory_chunks)
        memory_count = len(memory_chunks)

        # Determine confidence level
        if avg_relevance > 0.8 and memory_count > 5:
            confidence = 0.9
            depth = 'expert'
        elif avg_relevance > 0.6 and memory_count > 2:
            confidence = 0.7
            depth = 'knowledgeable'
        elif avg_relevance > 0.4 or memory_count > 0:
            confidence = 0.4
            depth = 'basic'
        else:
            confidence = 0.1
            depth = 'minimal'

        return {
            'confidence': confidence,
            'knowledge_depth': depth,
            'memory_count': memory_count,
            'avg_relevance': avg_relevance
        }

    def _generate_self_awareness_response(self, topic: str, knowledge_assessment: Dict) -> str:
        """Generate self-aware response about knowledge level"""
        confidence = knowledge_assessment['confidence']
        depth = knowledge_assessment['knowledge_depth']
        memory_count = knowledge_assessment['memory_count']

        if confidence > 0.8:
            response = f"I have extensive knowledge about {topic}, with {memory_count} relevant memories " \
                      f"and high confidence in this area. I can provide detailed, accurate information."
        elif confidence > 0.6:
            response = f"I have solid knowledge about {topic}, having encountered it {memory_count} times " \
                      f"in our conversations. I can help explain this topic effectively."
        elif confidence > 0.3:
            response = f"I have some knowledge about {topic} from {memory_count} relevant experiences. " \
                      f"My understanding is developing, and I can share what I've learned so far."
        else:
            response = f"I'm still building my knowledge about {topic}. I have minimal direct experience " \
                      f"with this topic, but I'm eager to learn and explore it with you."

        return response
