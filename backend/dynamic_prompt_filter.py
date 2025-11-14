"""
Dynamic Prompt Filter for Brein AI
Enables personalized, context-aware prompt selection based on user behavior patterns
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import re
import numpy as np

from user_behavior_analyzer import UserBehaviorAnalyzer, ResponseStyle

logger = logging.getLogger(__name__)

class PromptFilterCriteria:
    """Criteria for filtering and selecting prompts"""

    def __init__(self):
        self.user_id = None
        self.query_complexity = 0.5
        self.emotional_context = "neutral"
        self.preferred_style = ResponseStyle.CONVERSATIONAL
        self.detail_level = 0.5
        self.technical_proficiency = 0.5
        self.learning_preferences = {}
        self.conversation_context = []
        self.response_effectiveness_history = []
        self.time_pressure = "normal"
        self.knowledge_domain = "general"

    def from_user_analysis(self, user_analysis: Dict[str, Any]) -> 'PromptFilterCriteria':
        """Initialize criteria from user behavior analysis"""
        self.user_id = user_analysis.get('user_id')
        self.query_complexity = user_analysis.get('query_complexity', 0.5)
        self.emotional_context = user_analysis.get('emotional_tone', 'neutral')

        recommendations = user_analysis.get('recommendations', {})
        self.preferred_style = ResponseStyle(recommendations.get('suggested_response_style', 'conversational'))
        self.detail_level = recommendations.get('suggested_detail_level', 0.5)
        self.learning_preferences = {
            'examples': recommendations.get('include_examples', False),
            'step_by_step': recommendations.get('include_step_by_step', False),
            'visual_aids': recommendations.get('include_visual_aids', False)
        }

        return self

class DynamicPromptFilter:
    """
    Intelligent prompt filtering system that selects optimal prompts
    based on user behavior, context, and interaction patterns
    """

    def __init__(self, prompts_dir: str = "backend/prompts", behavior_analyzer: Optional[UserBehaviorAnalyzer] = None):
        self.prompts_dir = prompts_dir
        self.behavior_analyzer = behavior_analyzer or UserBehaviorAnalyzer()

        # Load prompt metadata and filtering rules
        self.prompt_metadata = self._load_prompt_metadata()
        self.filtering_rules = self._load_filtering_rules()

        # Response style mappings
        self.style_mappings = {
            ResponseStyle.CONCISE: ['brief', 'concise', 'direct', 'summary'],
            ResponseStyle.DETAILED: ['detailed', 'comprehensive', 'thorough', 'in-depth'],
            ResponseStyle.CONVERSATIONAL: ['conversational', 'friendly', 'chatty', 'engaging'],
            ResponseStyle.TECHNICAL: ['technical', 'precise', 'formal', 'rigorous'],
            ResponseStyle.TUTORIAL: ['tutorial', 'step_by_step', 'guided', 'educational'],
            ResponseStyle.VISUAL: ['visual', 'diagram', 'illustrative', 'graphic']
        }

    def _load_prompt_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata about available prompts for filtering"""
        metadata = {}

        try:
            # Scan all prompt files and extract metadata
            for root, dirs, files in os.walk(self.prompts_dir):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, self.prompts_dir)
                        prompt_key = relative_path.replace('.json', '').replace(os.sep, '.')

                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                prompt_data = json.load(f)

                            # Extract metadata for filtering
                            metadata[prompt_key] = self._extract_prompt_metadata(prompt_data, prompt_key)

                        except Exception as e:
                            logger.warning(f"Failed to load metadata for {prompt_key}: {e}")

        except Exception as e:
            logger.error(f"Error loading prompt metadata: {e}")

        return metadata

    def _extract_prompt_metadata(self, prompt_data: Dict[str, Any], prompt_key: str) -> Dict[str, Any]:
        """Extract filtering metadata from prompt data"""
        metadata = {
            'key': prompt_key,
            'description': prompt_data.get('description', ''),
            'version': prompt_data.get('version', '1.0'),
            'complexity_level': self._analyze_prompt_complexity(prompt_data),
            'response_style': self._detect_response_style(prompt_data),
            'emotional_tone': self._detect_emotional_tone(prompt_data),
            'technical_level': self._analyze_technical_level(prompt_data),
            'learning_elements': self._detect_learning_elements(prompt_data),
            'context_sensitivity': self._analyze_context_sensitivity(prompt_data),
            'variants': []
        }

        # Extract variant information
        prompts = prompt_data.get('prompts', {})
        if isinstance(prompts, dict):
            for variant_name, prompt_content in prompts.items():
                # Handle nested prompt structures (variants as dictionaries)
                if isinstance(prompt_content, dict):
                    # This is a prompt with multiple variants (e.g., standard_response)
                    for sub_variant_name, sub_prompt_text in prompt_content.items():
                        if isinstance(sub_prompt_text, str):
                            variant_metadata = {
                                'name': f"{variant_name}.{sub_variant_name}",
                                'complexity': self._analyze_text_complexity(sub_prompt_text),
                                'length': len(sub_prompt_text.split()),
                                'has_examples': self._contains_examples(sub_prompt_text),
                                'has_steps': self._contains_steps(sub_prompt_text),
                                'has_visual_refs': self._contains_visual_references(sub_prompt_text)
                            }
                            metadata['variants'].append(variant_metadata)
                elif isinstance(prompt_content, str):
                    # This is a direct prompt string
                    variant_metadata = {
                        'name': variant_name,
                        'complexity': self._analyze_text_complexity(prompt_content),
                        'length': len(prompt_content.split()),
                        'has_examples': self._contains_examples(prompt_content),
                        'has_steps': self._contains_steps(prompt_content),
                        'has_visual_refs': self._contains_visual_references(prompt_content)
                    }
                    metadata['variants'].append(variant_metadata)

        return metadata

    def _load_filtering_rules(self) -> Dict[str, Any]:
        """Load intelligent filtering rules"""
        return {
            'complexity_matching': {
                'simple_query': {'max_complexity': 0.3, 'preferred_styles': ['concise', 'conversational']},
                'moderate_query': {'complexity_range': [0.3, 0.7], 'preferred_styles': ['conversational', 'detailed']},
                'complex_query': {'min_complexity': 0.7, 'preferred_styles': ['technical', 'tutorial', 'detailed']}
            },
            'emotional_adaptation': {
                'frustrated': {'avoid_styles': ['technical'], 'prefer_styles': ['conversational', 'tutorial']},
                'confused': {'prefer_styles': ['tutorial', 'detailed'], 'add_clarification': True},
                'excited': {'prefer_styles': ['conversational', 'detailed'], 'enhance_engagement': True},
                'neutral': {'balanced_approach': True}
            },
            'learning_preference_boost': {
                'examples': {'weight_multiplier': 1.5, 'style_bonus': ['tutorial', 'detailed']},
                'step_by_step': {'weight_multiplier': 1.4, 'style_bonus': ['tutorial']},
                'visual_aids': {'weight_multiplier': 1.3, 'style_bonus': ['visual']}
            },
            'context_awareness': {
                'first_interaction': {'prefer_simple': True, 'introduce_concepts': True},
                'returning_user': {'use_history': True, 'adapt_to_patterns': True},
                'frustrated_user': {'simplify_response': True, 'offer_alternatives': True}
            }
        }

    def filter_prompt(self, prompt_key: str, criteria: PromptFilterCriteria,
                     available_variants: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Filter and select the optimal prompt variant based on user criteria

        Args:
            prompt_key: Base prompt key (e.g., 'system.orchestrator.contextual_response')
            criteria: Filtering criteria from user analysis
            available_variants: List of available variant dictionaries (optional pre-filter)

        Returns:
            Tuple of (selected_variant, filtering_metadata)
        """
        # Get prompt metadata
        prompt_meta = self.prompt_metadata.get(prompt_key)
        if not prompt_meta:
            logger.warning(f"No metadata found for prompt {prompt_key}, using default")
            return 'default', {'fallback': True}

        # Score available variants
        variant_scores = self._score_variants(prompt_meta, criteria, available_variants)

        # Select best variant
        if variant_scores:
            best_variant = max(variant_scores.items(), key=lambda x: x[1]['total_score'])
            variant_name, scoring_data = best_variant

            filtering_metadata = {
                'selected_variant': variant_name,
                'selection_reason': scoring_data['reason'],
                'scores': scoring_data,
                'criteria_used': {
                    'style': criteria.preferred_style.value,
                    'complexity': criteria.query_complexity,
                    'emotion': criteria.emotional_context,
                    'detail_level': criteria.detail_level
                }
            }

            return variant_name, filtering_metadata

        # Fallback to default
        return 'default', {'fallback': True, 'reason': 'no_variants_scored'}

    def _score_variants(self, prompt_meta: Dict[str, Any], criteria: PromptFilterCriteria,
                       available_variants: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """Score prompt variants based on filtering criteria"""
        scores = {}

        for variant in prompt_meta.get('variants', []):
            variant_name = variant['name']

            # Skip if not in available variants list (check by name)
            if available_variants:
                variant_names = [v['name'] for v in available_variants]
                if variant_name not in variant_names:
                    continue

            # Calculate component scores
            style_score = self._score_style_match(variant, criteria)
            complexity_score = self._score_complexity_match(variant, criteria)
            emotional_score = self._score_emotional_match(prompt_meta, criteria)
            learning_score = self._score_learning_preferences(variant, criteria)
            context_score = self._score_context_awareness(variant, criteria)

            # Calculate total score with weights
            total_score = (
                style_score * 0.3 +
                complexity_score * 0.25 +
                emotional_score * 0.2 +
                learning_score * 0.15 +
                context_score * 0.1
            )

            # Determine primary reason for selection
            reason = self._determine_selection_reason(
                style_score, complexity_score, emotional_score, learning_score, context_score
            )

            scores[variant_name] = {
                'total_score': total_score,
                'component_scores': {
                    'style': style_score,
                    'complexity': complexity_score,
                    'emotional': emotional_score,
                    'learning': learning_score,
                    'context': context_score
                },
                'reason': reason
            }

        return scores

    def _score_style_match(self, variant: Dict[str, Any], criteria: PromptFilterCriteria) -> float:
        """Score how well variant matches preferred response style"""
        variant_complexity = variant.get('complexity', 0.5)
        preferred_style = criteria.preferred_style

        # Style-specific scoring
        if preferred_style == ResponseStyle.CONCISE:
            # Prefer shorter, simpler variants
            return max(0, 1.0 - variant_complexity)
        elif preferred_style == ResponseStyle.DETAILED:
            # Prefer more complex, detailed variants
            return min(1.0, variant_complexity + 0.3)
        elif preferred_style == ResponseStyle.TECHNICAL:
            # Prefer technically precise variants
            return 0.8 if variant_complexity > 0.6 else 0.4
        elif preferred_style == ResponseStyle.TUTORIAL:
            # Prefer variants with steps and examples
            tutorial_bonus = 0.3 if variant.get('has_steps', False) else 0.0
            return min(1.0, variant_complexity + tutorial_bonus)
        elif preferred_style == ResponseStyle.VISUAL:
            # Prefer variants with visual references
            visual_bonus = 0.4 if variant.get('has_visual_refs', False) else 0.0
            return min(1.0, 0.6 + visual_bonus)

        # Conversational default
        return 0.7

    def _score_complexity_match(self, variant: Dict[str, Any], criteria: PromptFilterCriteria) -> float:
        """Score complexity alignment between variant and query"""
        variant_complexity = variant.get('complexity', 0.5)
        query_complexity = criteria.query_complexity

        # Ideal match when complexities are close
        complexity_diff = abs(variant_complexity - query_complexity)
        base_score = max(0, 1.0 - complexity_diff)

        # Adjust based on user proficiency
        proficiency = criteria.technical_proficiency
        if proficiency < 0.4 and variant_complexity > 0.7:
            # Reduce score for complex variants with low proficiency users
            base_score *= 0.6
        elif proficiency > 0.8 and variant_complexity < 0.3:
            # Reduce score for simple variants with high proficiency users
            base_score *= 0.7

        return base_score

    def _score_emotional_match(self, prompt_meta: Dict[str, Any], criteria: PromptFilterCriteria) -> float:
        """Score emotional alignment"""
        emotional_tone = criteria.emotional_context
        prompt_tone = prompt_meta.get('emotional_tone', 'neutral')

        # Emotional matching rules
        if emotional_tone == 'frustrated':
            # Prefer supportive, patient tones
            return 0.9 if prompt_tone in ['supportive', 'patient', 'encouraging'] else 0.5
        elif emotional_tone == 'confused':
            # Prefer clear, explanatory tones
            return 0.9 if prompt_tone in ['clear', 'educational', 'patient'] else 0.6
        elif emotional_tone == 'excited':
            # Prefer engaging, enthusiastic tones
            return 0.9 if prompt_tone in ['enthusiastic', 'engaging', 'positive'] else 0.7

        # Neutral emotional context
        return 0.8

    def _score_learning_preferences(self, variant: Dict[str, Any], criteria: PromptFilterCriteria) -> float:
        """Score based on user's learning preferences"""
        base_score = 0.5
        preferences = criteria.learning_preferences

        # Examples preference
        if preferences.get('examples') and variant.get('has_examples'):
            base_score += 0.2

        # Step-by-step preference
        if preferences.get('step_by_step') and variant.get('has_steps'):
            base_score += 0.2

        # Visual aids preference
        if preferences.get('visual_aids') and variant.get('has_visual_refs'):
            base_score += 0.2

        return min(1.0, base_score)

    def _score_context_awareness(self, variant: Dict[str, Any], criteria: PromptFilterCriteria) -> float:
        """Score context awareness and personalization"""
        # This could be enhanced with conversation history analysis
        # For now, provide a baseline score
        return 0.7

    def _determine_selection_reason(self, style_score: float, complexity_score: float,
                                  emotional_score: float, learning_score: float,
                                  context_score: float) -> str:
        """Determine the primary reason for variant selection"""
        scores = {
            'style_match': style_score,
            'complexity_match': complexity_score,
            'emotional_fit': emotional_score,
            'learning_preferences': learning_score,
            'context_awareness': context_score
        }

        best_reason = max(scores.items(), key=lambda x: x[1])

        if best_reason[1] > 0.8:
            return f"Strong {best_reason[0].replace('_', ' ')}"
        elif best_reason[1] > 0.6:
            return f"Good {best_reason[0].replace('_', ' ')}"
        else:
            return "Balanced selection across criteria"

    def get_adaptive_prompt(self, base_prompt_key: str, user_id: str, query: str,
                          context: Optional[Dict[str, Any]] = None,
                          available_variants: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Get a fully adaptive prompt based on user behavior and context

        Args:
            base_prompt_key: Base prompt key to adapt
            user_id: User identifier for behavior analysis
            query: User's query
            context: Additional context information
            available_variants: Pre-filtered list of available variants (optional)

        Returns:
            Tuple of (adapted_prompt_text, adaptation_metadata)
        """
        # Analyze user behavior
        user_analysis = self.behavior_analyzer.get_user_recommendations(user_id)

        # Create filtering criteria
        criteria = PromptFilterCriteria().from_user_analysis(user_analysis)

        # Add query-specific analysis
        criteria.query_complexity = self._analyze_query_complexity(query)
        criteria.emotional_context = self._analyze_query_emotion(query)
        criteria.knowledge_domain = self._detect_knowledge_domain(query)

        # Filter prompt variant
        selected_variant, filter_metadata = self.filter_prompt(base_prompt_key, criteria, available_variants)

        # Get the actual prompt text (this would integrate with PromptManager)
        # For now, return the variant name and metadata
        adaptation_metadata = {
            'selected_variant': selected_variant,
            'filtering_criteria': {
                'user_id': user_id,
                'query_complexity': criteria.query_complexity,
                'emotional_context': criteria.emotional_context,
                'preferred_style': criteria.preferred_style.value,
                'knowledge_domain': criteria.knowledge_domain
            },
            'user_analysis': user_analysis,
            'filter_metadata': filter_metadata,
            'adaptation_reason': self._explain_adaptation(criteria, filter_metadata)
        }

        return selected_variant, adaptation_metadata

    def _analyze_query_complexity(self, query: str) -> float:
        """Quick query complexity analysis"""
        # Simplified version - could use the full analyzer
        technical_terms = ['algorithm', 'neural', 'network', 'database', 'api', 'framework']
        complex_words = ['optimization', 'parallelization', 'asynchronous', 'concurrent']

        query_lower = query.lower()
        technical_count = sum(1 for term in technical_terms if term in query_lower)
        complex_count = sum(1 for word in complex_words if word in query_lower)

        complexity = min(1.0, (technical_count * 0.1 + complex_count * 0.15))
        return complexity

    def _analyze_query_emotion(self, query: str) -> str:
        """Quick emotional analysis"""
        frustrated_words = ['frustrated', 'annoyed', 'angry', 'problem', 'error']
        confused_words = ['confused', 'lost', 'understand', 'clear']
        excited_words = ['excited', 'amazing', 'awesome', 'great']

        query_lower = query.lower()

        if any(word in query_lower for word in frustrated_words):
            return 'frustrated'
        elif any(word in query_lower for word in confused_words):
            return 'confused'
        elif any(word in query_lower for word in excited_words):
            return 'excited'

        return 'neutral'

    def _detect_knowledge_domain(self, query: str) -> str:
        """Detect the knowledge domain of the query"""
        domains = {
            'programming': ['code', 'programming', 'python', 'javascript', 'algorithm'],
            'ai': ['ai', 'machine learning', 'neural', 'deep learning', 'model'],
            'data': ['data', 'database', 'analytics', 'statistics', 'sql'],
            'web': ['web', 'website', 'html', 'css', 'javascript', 'api'],
            'system': ['system', 'architecture', 'design', 'infrastructure']
        }

        query_lower = query.lower()
        for domain, keywords in domains.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain

        return 'general'

    def _explain_adaptation(self, criteria: PromptFilterCriteria, filter_metadata: Dict[str, Any]) -> str:
        """Explain why the prompt was adapted this way"""
        explanations = []

        if criteria.preferred_style != ResponseStyle.CONVERSATIONAL:
            explanations.append(f"Selected {criteria.preferred_style.value} style based on user preferences")

        if criteria.query_complexity > 0.7:
            explanations.append("Query complexity requires detailed response")
        elif criteria.query_complexity < 0.3:
            explanations.append("Simple query allows concise response")

        if criteria.emotional_context != 'neutral':
            explanations.append(f"Adapted for {criteria.emotional_context} emotional context")

        if not explanations:
            explanations.append("Balanced adaptation based on user profile")

        return "; ".join(explanations)

    # Analysis helper methods
    def _analyze_prompt_complexity(self, prompt_data: Dict[str, Any]) -> float:
        """Analyze overall prompt complexity"""
        prompts = prompt_data.get('prompts', {})
        if not prompts:
            return 0.5

        complexities = []
        for prompt_text in prompts.values():
            if isinstance(prompt_text, str):
                complexities.append(self._analyze_text_complexity(prompt_text))

        return np.mean(complexities) if complexities else 0.5

    def _analyze_text_complexity(self, text: str) -> float:
        """Analyze complexity of text"""
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0

        # Technical indicators
        technical_terms = ['algorithm', 'neural', 'network', 'database', 'api', 'framework']
        technical_count = sum(1 for term in technical_terms if term in text.lower())

        # Sentence complexity
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0

        complexity = min(1.0, (
            (avg_word_length - 4) * 0.1 +  # Word length factor
            technical_count * 0.05 +      # Technical term density
            (avg_sentence_length - 15) * 0.02  # Sentence length factor
        ))

        return max(0.0, complexity)

    def _detect_response_style(self, prompt_data: Dict[str, Any]) -> str:
        """Detect the response style of prompts"""
        # Simplified style detection
        return 'conversational'  # Default

    def _detect_emotional_tone(self, prompt_data: Dict[str, Any]) -> str:
        """Detect emotional tone of prompts"""
        return 'neutral'  # Default

    def _analyze_technical_level(self, prompt_data: Dict[str, Any]) -> float:
        """Analyze technical level of prompts"""
        return 0.5  # Default

    def _detect_learning_elements(self, prompt_data: Dict[str, Any]) -> List[str]:
        """Detect learning elements in prompts"""
        return []  # Default

    def _analyze_context_sensitivity(self, prompt_data: Dict[str, Any]) -> float:
        """Analyze context sensitivity of prompts"""
        return 0.5  # Default

    def _contains_examples(self, text: str) -> bool:
        """Check if text contains examples"""
        example_indicators = ['for example', 'instance', 'sample', 'such as', 'e.g.', 'i.e.']
        return any(indicator in text.lower() for indicator in example_indicators)

    def _contains_steps(self, text: str) -> bool:
        """Check if text contains step-by-step instructions"""
        step_indicators = ['first', 'second', 'third', 'next', 'then', 'finally', 'step']
        return any(indicator in text.lower() for indicator in step_indicators)

    def _contains_visual_references(self, text: str) -> bool:
        """Check if text contains visual references"""
        visual_indicators = ['diagram', 'chart', 'graph', 'image', 'visual', 'figure', 'illustration']
        return any(indicator in text.lower() for indicator in visual_indicators)
