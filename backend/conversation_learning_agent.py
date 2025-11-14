"""
Conversation Learning Agent - Extracts and stores knowledge from conversations
Creates persistent memories from Q&A interactions for natural AI learning.
"""

import asyncio
import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)

class ConversationLearningAgent:
    """
    Analyzes conversations to extract knowledge and store it permanently.
    Enables the AI to learn from interactions and recall information naturally.
    """

    def __init__(self, memory_manager, chat_manager):
        self.memory = memory_manager
        self.chat = chat_manager
        self.is_running = False

        # Knowledge extraction patterns
        self.question_patterns = [
            r'what\s+is', r'how\s+does', r'why\s+does', r'when\s+did',
            r'where\s+is', r'who\s+is', r'explain', r'define', r'describe'
        ]

        logger.info("Conversation Learning Agent initialized")

    async def start_background_learning(self):
        """Start background conversation analysis."""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting background conversation learning")

        # Start periodic analysis
        asyncio.create_task(self._periodic_conversation_analysis())

    async def analyze_recent_conversation(self, session_id: str):
        """Analyze a specific conversation for knowledge extraction."""
        try:
            messages = self.chat.get_chat_history(session_id)
            if len(messages) < 2:  # Need at least a question and answer
                return

            # Extract knowledge from conversation
            knowledge_points = self._extract_knowledge_from_messages(messages)

            # Store knowledge permanently
            for knowledge in knowledge_points:
                await self._store_knowledge(knowledge, session_id)

            if knowledge_points:
                logger.info(f"Extracted {len(knowledge_points)} knowledge points from session {session_id}")

        except Exception as e:
            logger.error(f"Error analyzing conversation {session_id}: {e}")

    def _extract_knowledge_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """Extract meaningful knowledge from conversation messages."""
        knowledge_points = []

        # Look for Q&A patterns in conversation
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]

            if (current_msg['role'] == 'user' and next_msg['role'] == 'ai' and
                self._is_informative_question(current_msg['content']) and
                self._is_informative_answer(next_msg['content'])):

                knowledge = self._create_knowledge_from_qa(
                    current_msg['content'],
                    next_msg['content']
                )

                if knowledge:
                    knowledge_points.append(knowledge)

        return knowledge_points

    def _is_informative_question(self, text: str) -> bool:
        """Check if a question is likely to yield informative answers."""
        text_lower = text.lower().strip()

        # Check for question words
        if any(pattern in text_lower for pattern in self.question_patterns):
            return True

        # Check for question marks
        if '?' in text:
            return True

        # Check for "tell me about" or "explain"
        if any(phrase in text_lower for phrase in ['tell me about', 'explain', 'what about']):
            return True

        return False

    def _is_informative_answer(self, text: str) -> bool:
        """Check if an answer contains substantial information."""
        # Basic heuristics for informative answers
        word_count = len(text.split())
        if word_count < 10:  # Too short
            return False

        if word_count > 500:  # Too long (probably not focused)
            return False

        # Check for actual content (not just "I don't know" etc.)
        low_quality_phrases = [
            "i don't know", "i'm not sure", "i can't help", "no information",
            "i don't have", "sorry", "unfortunately"
        ]

        text_lower = text.lower()
        if any(phrase in text_lower for phrase in low_quality_phrases):
            return False

        return True

    def _create_knowledge_from_qa(self, question: str, answer: str) -> Optional[Dict]:
        """Create a knowledge point from a Q&A pair."""
        try:
            # Extract topic from question
            topic = self._extract_topic_from_question(question)

            # Clean and format the answer
            clean_answer = self._clean_answer(answer)

            # Create knowledge content
            knowledge_content = f"{topic}: {clean_answer}"

            # Assess confidence based on answer quality
            confidence = self._assess_answer_confidence(answer)

            return {
                'content': knowledge_content,
                'topic': topic,
                'question': question,
                'answer': clean_answer,
                'confidence': confidence,
                'type': 'learned_knowledge'
            }

        except Exception as e:
            logger.warning(f"Error creating knowledge from Q&A: {e}")
            return None

    def _extract_topic_from_question(self, question: str) -> str:
        """Extract the main topic from a question."""
        # Simple topic extraction
        question_lower = question.lower()

        # Remove question words and get key terms
        for pattern in self.question_patterns:
            question_lower = re.sub(pattern, '', question_lower, flags=re.IGNORECASE)

        # Clean up and get first meaningful phrase
        words = question_lower.strip().split()
        if words:
            # Take first 2-3 meaningful words
            meaningful_words = [w for w in words[:4] if len(w) > 2]
            return ' '.join(meaningful_words).title()

        return "General Knowledge"

    def _clean_answer(self, answer: str) -> str:
        """Clean and format an answer for storage."""
        # Remove common AI prefixes/suffixes
        answer = re.sub(r'^(based on|according to|from what i know|as far as i know)', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'(let me know if|feel free to ask|i hope this|does this help)', '', answer, flags=re.IGNORECASE)

        # Clean up whitespace
        answer = re.sub(r'\s+', ' ', answer.strip())

        # Capitalize first letter
        if answer:
            answer = answer[0].upper() + answer[1:]

        return answer

    def _assess_answer_confidence(self, answer: str) -> float:
        """Assess the confidence/quality of an answer."""
        confidence = 0.5  # Base confidence

        # Length bonus
        word_count = len(answer.split())
        if word_count > 20:
            confidence += 0.2
        elif word_count < 10:
            confidence -= 0.2

        # Specificity indicators
        specific_terms = ['specific', 'particular', 'example', 'instance', 'case']
        if any(term in answer.lower() for term in specific_terms):
            confidence += 0.1

        # Structure indicators (lists, explanations)
        if any(marker in answer for marker in ['•', '-', '1.', '2.', 'first', 'second']):
            confidence += 0.1

        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0

    async def _store_knowledge(self, knowledge: Dict, session_id: str):
        """Store extracted knowledge in the memory system."""
        try:
            # Use the memory manager's ingest method
            result = await self.memory.ingest_content(
                content=knowledge['content'],
                content_type='learned_knowledge',
                metadata={
                    'source': 'conversation_learning',
                    'session_id': session_id,
                    'topic': knowledge['topic'],
                    'confidence': knowledge['confidence'],
                    'original_question': knowledge['question']
                }
            )

            logger.debug(f"Stored knowledge: {knowledge['topic']} (confidence: {knowledge['confidence']:.2f})")

        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")

    async def _periodic_conversation_analysis(self):
        """Periodically analyze recent conversations for learning."""
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # Analyze every 30 minutes

                logger.info("Running periodic conversation analysis...")

                # Get recent sessions
                recent_sessions = self.chat.get_recent_sessions(limit=20)

                analyzed_count = 0
                knowledge_extracted = 0

                for session in recent_sessions:
                    # Only analyze sessions with recent activity
                    if self._session_has_recent_activity(session['id']):
                        await self.analyze_recent_conversation(session['id'])
                        analyzed_count += 1

                        # Could track knowledge_extracted here if needed

                if analyzed_count > 0:
                    logger.info(f"Analyzed {analyzed_count} recent conversations for learning")

            except Exception as e:
                logger.error(f"Error in periodic conversation analysis: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    def _session_has_recent_activity(self, session_id: str) -> bool:
        """Check if a session has had recent activity."""
        try:
            messages = self.chat.get_chat_history(session_id)
            if not messages:
                return False

            # Check if last message is within last 24 hours
            from datetime import datetime, timedelta
            last_message_time = messages[-1]['timestamp']

            if isinstance(last_message_time, str):
                # Parse if it's a string
                from dateutil import parser
                last_message_time = parser.parse(last_message_time)

            return (datetime.now() - last_message_time).total_seconds() < 86400  # 24 hours

        except Exception:
            return False

    def stop(self):
        """Stop background learning tasks."""
        self.is_running = False
        logger.info("Conversation learning agent stopped")
