"""
Simple Memory Agent - Background memory optimization without complex reinforcement learning
Provides basic memory cleanup and organization without blocking responses.
"""

import asyncio
import logging
from typing import Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SimpleMemoryAgent:
    """
    Background memory optimization agent.
    Performs simple cleanup and organization tasks asynchronously.
    """

    def __init__(self, memory_manager, chat_manager):
        self.memory = memory_manager
        self.chat = chat_manager
        self.is_running = False

        logger.info("Simple Memory Agent initialized")

    async def run_background_tasks(self):
        """Start background memory optimization tasks."""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting background memory optimization tasks")

        # Start periodic tasks
        asyncio.create_task(self._periodic_cleanup())
        asyncio.create_task(self._periodic_optimization())

    async def _periodic_cleanup(self):
        """Run memory cleanup every 2 hours."""
        while self.is_running:
            try:
                await asyncio.sleep(7200)  # 2 hours

                logger.info("Running periodic memory cleanup...")

                # Simple cleanup - remove very old memories
                if hasattr(self.memory, 'consolidator'):
                    cleanup_result = self.memory.consolidator.cleanup_old_memories(days_threshold=60)
                    logger.info(f"Cleaned up {cleanup_result} old memories")

                # Basic consolidation
                if hasattr(self.memory, 'consolidator'):
                    consolidation_result = self.memory.consolidator.consolidate_similar_memories()
                    logger.info(f"Consolidated similar memories: {consolidation_result}")

            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(3600)  # Wait before retrying

    async def _periodic_optimization(self):
        """Run memory optimization every 6 hours."""
        while self.is_running:
            try:
                await asyncio.sleep(21600)  # 6 hours

                logger.info("Running periodic memory optimization...")

                # Analyze recent conversations for patterns
                await self._analyze_conversation_patterns()

                # Update memory tags based on usage
                await self._update_memory_tags()

                # Health check
                await self._memory_health_check()

            except Exception as e:
                logger.error(f"Error in periodic optimization: {e}")
                await asyncio.sleep(3600)  # Wait before retrying

    async def _analyze_conversation_patterns(self):
        """Analyze recent conversations to identify patterns."""
        try:
            # Get recent conversations (last 100)
            recent_sessions = self.chat.get_recent_sessions(limit=50)

            if not recent_sessions:
                return

            # Simple pattern analysis
            topic_frequency = {}
            successful_responses = 0
            total_responses = 0

            for session in recent_sessions:
                messages = self.chat.get_chat_history(session['id'])

                for msg in messages:
                    if msg['role'] == 'user':
                        # Extract basic topics from user queries
                        words = msg['content'].lower().split()
                        for word in words:
                            if len(word) > 4:  # Focus on meaningful words
                                topic_frequency[word] = topic_frequency.get(word, 0) + 1

                    elif msg['role'] == 'ai':
                        total_responses += 1
                        # Simple success metric - responses longer than 50 chars
                        if len(msg['content']) > 50:
                            successful_responses += 1

            # Log insights
            top_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            success_rate = successful_responses / max(total_responses, 1)

            logger.info(f"Conversation analysis: Success rate {success_rate:.2f}, Top topics: {[t[0] for t in top_topics[:3]]}")

        except Exception as e:
            logger.error(f"Error analyzing conversation patterns: {e}")

    async def _update_memory_tags(self):
        """Update memory tags based on recent usage patterns."""
        try:
            # Simple tag updates - this is a placeholder for more complex logic
            # In a real implementation, this would analyze which memories are accessed together

            logger.debug("Memory tags updated (placeholder)")

        except Exception as e:
            logger.error(f"Error updating memory tags: {e}")

    async def _memory_health_check(self):
        """Perform basic memory health check."""
        try:
            if hasattr(self.memory, 'consolidator') and hasattr(self.memory.consolidator, 'get_memory_health_report'):
                health_report = self.memory.consolidator.get_memory_health_report()
                health_score = health_report.get('health_score', 0)

                if health_score < 0.5:
                    logger.warning(f"Memory health score is low: {health_score:.2f}")
                else:
                    logger.info(f"Memory health score: {health_score:.2f}")
            else:
                logger.debug("Memory health check skipped - consolidator not available")

        except Exception as e:
            logger.error(f"Error in memory health check: {e}")

    def stop(self):
        """Stop background tasks."""
        self.is_running = False
        logger.info("Memory optimization tasks stopped")
