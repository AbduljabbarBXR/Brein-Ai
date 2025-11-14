"""
Simple Concept Agent - Lightweight alternative to complex semantic concept extraction
Provides basic keyword extraction and memory matching without heavy ML dependencies.
"""

from typing import List, Dict, Set
import re
import logging

logger = logging.getLogger(__name__)

class SimpleConceptAgent:
    """
    Simple concept extraction and memory matching agent.
    Uses basic NLP techniques instead of complex ML models.
    """

    def __init__(self):
        # Basic stop words for filtering
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }

        # Common question words to filter out
        self.question_words = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose'}

        logger.info("Simple Concept Agent initialized")

    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract simple concepts from text using basic NLP.

        Args:
            text: Input text to extract concepts from

        Returns:
            List of key concepts (max 5)
        """
        if not text or not text.strip():
            return []

        # Preprocess text
        cleaned_text = self._preprocess_text(text)

        # Extract candidate concepts
        candidates = self._extract_candidates(cleaned_text)

        # Score and rank candidates
        scored_candidates = self._score_candidates(candidates, cleaned_text)

        # Return top concepts
        top_concepts = [concept for concept, score in scored_candidates[:5]]

        logger.debug(f"Extracted concepts from '{text[:50]}...': {top_concepts}")
        return top_concepts

    def find_related_memories(self, concepts: List[str], memories: List[Dict]) -> List[Dict]:
        """
        Find memories related to the given concepts using simple matching.

        Args:
            concepts: List of concepts to match
            memories: List of memory objects to search

        Returns:
            List of relevant memories sorted by relevance
        """
        if not concepts or not memories:
            return []

        scored_memories = []

        for memory in memories:
            content = memory.get('content', '').lower()
            tags = memory.get('tags', [])

            # Calculate relevance score
            score = self._calculate_relevance(concepts, content, tags)

            if score > 0:
                scored_memories.append((memory, score))

        # Sort by score (highest first) and return top 3
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        relevant_memories = [mem for mem, score in scored_memories[:3]]

        logger.debug(f"Found {len(relevant_memories)} related memories for concepts: {concepts}")
        return relevant_memories

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _extract_candidates(self, text: str) -> List[str]:
        """Extract candidate concepts from preprocessed text."""
        words = text.split()
        candidates = set()

        # Single important words
        for word in words:
            if len(word) >= 4 and word not in self.stop_words and word not in self.question_words:
                candidates.add(word)

        # Two-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            # Check if both words are meaningful
            if (len(words[i]) >= 3 and len(words[i+1]) >= 3 and
                words[i] not in self.stop_words and words[i+1] not in self.stop_words):
                candidates.add(phrase)

        return list(candidates)

    def _score_candidates(self, candidates: List[str], original_text: str) -> List[tuple]:
        """
        Score candidates based on frequency and position in text.

        Returns:
            List of (concept, score) tuples
        """
        scored = []

        for candidate in candidates:
            score = 0

            # Frequency score
            count = original_text.count(candidate.lower())
            score += count * 2

            # Length bonus (prefer longer, more specific concepts)
            words_in_concept = len(candidate.split())
            score += words_in_concept * 0.5

            # Position bonus (concepts at start of text are more important)
            first_occurrence = original_text.find(candidate.lower())
            if first_occurrence >= 0:
                position_bonus = max(0, 1.0 - (first_occurrence / len(original_text)))
                score += position_bonus

            scored.append((candidate, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _calculate_relevance(self, concepts: List[str], content: str, tags: List[str]) -> float:
        """
        Calculate how relevant a memory is to the given concepts.

        Args:
            concepts: Query concepts
            content: Memory content
            tags: Memory tags

        Returns:
            Relevance score (0-1)
        """
        score = 0
        total_concepts = len(concepts)

        if total_concepts == 0:
            return 0

        # Check concept matches in content
        content_matches = 0
        for concept in concepts:
            if concept.lower() in content:
                content_matches += 1

        # Check tag matches
        tag_matches = 0
        content_lower = content.lower()
        for tag in tags:
            tag_lower = tag.lower()
            # Check if tag appears in content or matches concepts
            if tag_lower in content_lower or any(concept.lower() in tag_lower for concept in concepts):
                tag_matches += 1

        # Calculate final score
        content_score = content_matches / total_concepts
        tag_score = min(tag_matches / max(total_concepts, 1), 1.0)  # Cap at 1.0

        # Weighted combination
        final_score = (content_score * 0.7) + (tag_score * 0.3)

        return min(final_score, 1.0)  # Cap at 1.0
