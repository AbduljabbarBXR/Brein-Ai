"""
User Behavior Analyzer for Brein AI
Analyzes user interaction patterns to enable personalized, adaptive responses
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class ResponseStyle(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    TUTORIAL = "tutorial"
    VISUAL = "visual"

class UserProfile:
    """User profile with behavioral patterns and preferences"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

        # Response preferences
        self.preferred_response_style = ResponseStyle.CONVERSATIONAL
        self.preferred_detail_level = 0.5  # 0-1 scale
        self.technical_proficiency = 0.5  # 0-1 scale

        # Interaction patterns
        self.question_complexity_history = []
        self.response_effectiveness_history = []
        self.topic_interests = Counter()
        self.interaction_times = []
        self.session_durations = []

        # Emotional patterns
        self.emotional_tone_history = []
        self.frustration_indicators = []
        self.satisfaction_indicators = []

        # Learning preferences
        self.prefers_examples = False
        self.prefers_step_by_step = False
        self.prefers_visual_aids = False
        self.response_speed_preference = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for storage"""
        return {
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'preferred_response_style': self.preferred_response_style.value,
            'preferred_detail_level': self.preferred_detail_level,
            'technical_proficiency': self.technical_proficiency,
            'question_complexity_history': self.question_complexity_history[-50:],  # Keep last 50
            'response_effectiveness_history': self.response_effectiveness_history[-50:],
            'topic_interests': dict(self.topic_interests.most_common(20)),  # Top 20 topics
            'interaction_times': [t.isoformat() for t in self.interaction_times[-100:]],
            'session_durations': self.session_durations[-50:],
            'emotional_tone_history': self.emotional_tone_history[-50:],
            'frustration_indicators': self.frustration_indicators[-20:],
            'satisfaction_indicators': self.satisfaction_indicators[-20:],
            'prefers_examples': self.prefers_examples,
            'prefers_step_by_step': self.prefers_step_by_step,
            'prefers_visual_aids': self.prefers_visual_aids,
            'response_speed_preference': self.response_speed_preference
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create profile from dictionary"""
        profile = cls(data['user_id'])
        profile.created_at = datetime.fromisoformat(data['created_at'])
        profile.last_updated = datetime.fromisoformat(data['last_updated'])
        profile.preferred_response_style = ResponseStyle(data['preferred_response_style'])
        profile.preferred_detail_level = data['preferred_detail_level']
        profile.technical_proficiency = data['technical_proficiency']
        profile.question_complexity_history = data['question_complexity_history']
        profile.response_effectiveness_history = data['response_effectiveness_history']
        profile.topic_interests = Counter(data['topic_interests'])
        profile.interaction_times = [datetime.fromisoformat(t) for t in data['interaction_times']]
        profile.session_durations = data['session_durations']
        profile.emotional_tone_history = data['emotional_tone_history']
        profile.frustration_indicators = data['frustration_indicators']
        profile.satisfaction_indicators = data['satisfaction_indicators']
        profile.prefers_examples = data['prefers_examples']
        profile.prefers_step_by_step = data['prefers_step_by_step']
        profile.prefers_visual_aids = data['prefers_visual_aids']
        profile.response_speed_preference = data['response_speed_preference']
        return profile

class UserBehaviorAnalyzer:
    """
    Analyzes user behavior patterns to enable personalized responses
    """

    def __init__(self, profiles_dir: str = "memory/user_profiles"):
        self.profiles_dir = profiles_dir
        self.user_profiles: Dict[str, UserProfile] = {}

        # Ensure directory exists
        os.makedirs(profiles_dir, exist_ok=True)

        # Load existing profiles
        self._load_profiles()

        # Analysis parameters
        self.min_interactions_for_analysis = 5
        self.confidence_threshold = 0.6

    def _load_profiles(self):
        """Load user profiles from disk"""
        try:
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.json'):
                    user_id = filename[:-5]  # Remove .json
                    try:
                        with open(os.path.join(self.profiles_dir, filename), 'r') as f:
                            profile_data = json.load(f)
                            self.user_profiles[user_id] = UserProfile.from_dict(profile_data)
                    except Exception as e:
                        logger.warning(f"Failed to load profile {user_id}: {e}")
        except Exception as e:
            logger.warning(f"Error loading profiles: {e}")

    def _save_profile(self, user_id: str):
        """Save user profile to disk"""
        if user_id in self.user_profiles:
            try:
                profile = self.user_profiles[user_id]
                profile.last_updated = datetime.now()

                with open(os.path.join(self.profiles_dir, f"{user_id}.json"), 'w') as f:
                    json.dump(profile.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save profile {user_id}: {e}")

    def analyze_interaction(self, user_id: str, query: str, response: str,
                          response_time: float, user_feedback: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze a user interaction and update their profile

        Args:
            user_id: Unique user identifier
            query: User's query
            response: AI response
            response_time: Response time in seconds
            user_feedback: Optional user feedback data

        Returns:
            Analysis results and recommendations
        """
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)

        profile = self.user_profiles[user_id]

        # Analyze query complexity
        query_complexity = self._analyze_query_complexity(query)
        profile.question_complexity_history.append(query_complexity)

        # Analyze response effectiveness
        response_effectiveness = self._analyze_response_effectiveness(query, response, user_feedback)
        profile.response_effectiveness_history.append(response_effectiveness)

        # Extract topics of interest
        topics = self._extract_topics(query)
        for topic in topics:
            profile.topic_interests[topic] += 1

        # Track interaction timing
        profile.interaction_times.append(datetime.now())

        # Analyze emotional context
        emotional_tone = self._analyze_emotional_context(query)
        profile.emotional_tone_history.append(emotional_tone)

        # Detect behavioral indicators
        if user_feedback:
            if user_feedback.get('frustrated', False):
                profile.frustration_indicators.append(datetime.now().isoformat())
            if user_feedback.get('satisfied', False):
                profile.satisfaction_indicators.append(datetime.now().isoformat())

        # Update learning preferences based on patterns
        self._update_learning_preferences(profile, query, response)

        # Update profile statistics
        self._update_profile_statistics(profile)

        # Save updated profile
        self._save_profile(user_id)

        # Generate recommendations
        recommendations = self._generate_recommendations(profile, query_complexity, response_effectiveness)

        return {
            'user_id': user_id,
            'query_complexity': query_complexity,
            'response_effectiveness': response_effectiveness,
            'emotional_tone': emotional_tone,
            'detected_topics': topics,
            'recommendations': recommendations,
            'profile_confidence': self._calculate_profile_confidence(profile)
        }

    def _analyze_query_complexity(self, query: str) -> float:
        """
        Analyze query complexity on a 0-1 scale

        Returns:
            Complexity score (0=simple, 1=very complex)
        """
        complexity_indicators = {
            'technical_terms': ['algorithm', 'neural', 'network', 'database', 'api', 'framework', 'architecture'],
            'complex_words': ['optimization', 'parallelization', 'asynchronous', 'concurrent', 'distributed'],
            'question_words': ['how', 'why', 'explain', 'describe', 'analyze', 'compare'],
            'length_factor': len(query.split()) / 50  # Normalize by expected max length
        }

        score = 0.0

        # Technical term density
        query_lower = query.lower()
        technical_count = sum(1 for term in complexity_indicators['technical_terms'] if term in query_lower)
        score += min(technical_count * 0.1, 0.3)

        # Complex word usage
        complex_count = sum(1 for word in complexity_indicators['complex_words'] if word in query_lower)
        score += min(complex_count * 0.15, 0.3)

        # Question type complexity
        question_words = sum(1 for word in complexity_indicators['question_words'] if word in query_lower.split())
        score += min(question_words * 0.1, 0.2)

        # Length factor
        score += min(complexity_indicators['length_factor'], 0.2)

        return min(score, 1.0)

    def _analyze_response_effectiveness(self, query: str, response: str,
                                      user_feedback: Optional[Dict] = None) -> float:
        """
        Analyze how effective the response was

        Returns:
            Effectiveness score (0=poor, 1=excellent)
        """
        effectiveness = 0.5  # Default neutral

        # Direct user feedback
        if user_feedback:
            if user_feedback.get('rating'):  # Assume 1-5 scale
                effectiveness = user_feedback['rating'] / 5.0
            elif user_feedback.get('helpful') is True:
                effectiveness = 0.8
            elif user_feedback.get('helpful') is False:
                effectiveness = 0.3

        # Response quality indicators
        response_length = len(response.split())

        # Too short responses are often inadequate
        if response_length < 10:
            effectiveness *= 0.7
        # Too long responses might be overwhelming
        elif response_length > 500:
            effectiveness *= 0.8

        # Check for question answering
        query_questions = sum(1 for word in ['what', 'how', 'why', 'when', 'where', 'who'] if word in query.lower())
        if query_questions > 0:
            # Response should address the questions
            response_addresses_questions = any(word in response.lower() for word in ['because', 'since', 'due to', 'therefore'])
            if response_addresses_questions:
                effectiveness *= 1.1

        return min(max(effectiveness, 0.0), 1.0)

    def _extract_topics(self, query: str) -> List[str]:
        """Extract topics of interest from query"""
        # Simple topic extraction - could be enhanced with NLP
        topic_keywords = {
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'neural network'],
            'programming': ['code', 'programming', 'python', 'javascript', 'algorithm'],
            'data': ['data', 'database', 'analytics', 'statistics'],
            'web': ['web', 'website', 'html', 'css', 'javascript'],
            'system': ['system', 'architecture', 'design', 'infrastructure'],
            'learning': ['learn', 'tutorial', 'guide', 'course', 'study']
        }

        query_lower = query.lower()
        detected_topics = []

        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_topics.append(topic)

        return detected_topics if detected_topics else ['general']

    def _analyze_emotional_context(self, query: str) -> str:
        """Analyze emotional context of query"""
        emotional_indicators = {
            'frustrated': ['frustrated', 'annoyed', 'angry', 'upset', 'problem', 'issue', 'error'],
            'confused': ['confused', 'lost', 'understand', 'clear', 'explain'],
            'excited': ['excited', 'amazing', 'awesome', 'great', 'love'],
            'concerned': ['worried', 'concerned', 'afraid', 'scared'],
            'neutral': []  # Default
        }

        query_lower = query.lower()
        max_matches = 0
        detected_emotion = 'neutral'

        for emotion, indicators in emotional_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in query_lower)
            if matches > max_matches:
                max_matches = matches
                detected_emotion = emotion

        return detected_emotion

    def _update_learning_preferences(self, profile: UserProfile, query: str, response: str):
        """Update learning preferences based on interaction patterns"""
        query_lower = query.lower()
        response_lower = response.lower()

        # Check for example requests
        if any(word in query_lower for word in ['example', 'instance', 'sample', 'demo']):
            profile.prefers_examples = True

        # Check for step-by-step preferences
        if any(word in query_lower for word in ['step', 'guide', 'tutorial', 'how to']):
            profile.prefers_step_by_step = True

        # Check for visual preferences
        if any(word in query_lower for word in ['diagram', 'visual', 'image', 'chart', 'graph']):
            profile.prefers_visual_aids = True

    def _update_profile_statistics(self, profile: UserProfile):
        """Update derived statistics in profile"""
        if len(profile.question_complexity_history) >= self.min_interactions_for_analysis:
            # Update technical proficiency based on complexity patterns
            avg_complexity = np.mean(profile.question_complexity_history)
            profile.technical_proficiency = min(avg_complexity * 1.2, 1.0)

            # Update preferred detail level based on effectiveness
            if profile.response_effectiveness_history:
                avg_effectiveness = np.mean(profile.response_effectiveness_history)
                profile.preferred_detail_level = avg_effectiveness

            # Determine preferred response style
            profile.preferred_response_style = self._determine_preferred_style(profile)

    def _determine_preferred_style(self, profile: UserProfile) -> ResponseStyle:
        """Determine user's preferred response style"""
        if len(profile.question_complexity_history) < self.min_interactions_for_analysis:
            return ResponseStyle.CONVERSATIONAL

        avg_complexity = np.mean(profile.question_complexity_history)

        # High complexity + high effectiveness → Technical
        if avg_complexity > 0.7 and profile.preferred_detail_level > 0.7:
            return ResponseStyle.TECHNICAL

        # High complexity + step preferences → Tutorial
        if avg_complexity > 0.6 and profile.prefers_step_by_step:
            return ResponseStyle.TUTORIAL

        # Low complexity + visual preferences → Visual
        if avg_complexity < 0.4 and profile.prefers_visual_aids:
            return ResponseStyle.VISUAL

        # High detail preference → Detailed
        if profile.preferred_detail_level > 0.7:
            return ResponseStyle.DETAILED

        # Default to concise for moderate complexity
        if avg_complexity < 0.5:
            return ResponseStyle.CONCISE

        return ResponseStyle.CONVERSATIONAL

    def _generate_recommendations(self, profile: UserProfile, query_complexity: float,
                                response_effectiveness: float) -> Dict[str, Any]:
        """Generate recommendations for response optimization"""
        recommendations = {
            'suggested_response_style': profile.preferred_response_style.value,
            'suggested_detail_level': profile.preferred_detail_level,
            'include_examples': profile.prefers_examples,
            'include_step_by_step': profile.prefers_step_by_step,
            'include_visual_aids': profile.prefers_visual_aids,
            'adapt_complexity': True
        }

        # Specific recommendations based on recent performance
        if response_effectiveness < 0.6:
            if query_complexity > 0.7:
                recommendations['simplify_explanation'] = True
            else:
                recommendations['add_more_detail'] = True

        if len(profile.frustration_indicators) > len(profile.satisfaction_indicators):
            recommendations['use_empathetic_tone'] = True
            recommendations['offer_alternatives'] = True

        return recommendations

    def _calculate_profile_confidence(self, profile: UserProfile) -> float:
        """Calculate confidence in profile accuracy"""
        interaction_count = len(profile.question_complexity_history)

        if interaction_count < self.min_interactions_for_analysis:
            return 0.3  # Low confidence with few interactions

        # Higher confidence with more interactions and consistent patterns
        base_confidence = min(interaction_count / 20.0, 1.0)  # Max at 20 interactions

        # Reduce confidence if patterns are inconsistent
        complexity_std = np.std(profile.question_complexity_history) if len(profile.question_complexity_history) > 1 else 0
        consistency_penalty = complexity_std * 0.5  # Penalize high variability

        return max(base_confidence - consistency_penalty, 0.1)

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile data"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            return profile.to_dict()
        return None

    def get_user_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get response recommendations for a user"""
        if user_id not in self.user_profiles:
            return {
                'user_id': user_id,
                'recommendations': {
                    'suggested_response_style': 'conversational',
                    'suggested_detail_level': 0.5,
                    'adapt_complexity': False
                },
                'profile_confidence': 0.0
            }

        profile = self.user_profiles[user_id]
        recommendations = self._generate_recommendations(profile, 0.5, 0.8)  # Default values

        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'profile_confidence': self._calculate_profile_confidence(profile),
            'interaction_count': len(profile.question_complexity_history)
        }

    def get_analytics(self) -> Dict[str, Any]:
        """Get overall analytics about user behavior patterns"""
        total_users = len(self.user_profiles)
        total_interactions = sum(len(p.question_complexity_history) for p in self.user_profiles.values())

        if total_users == 0:
            return {'error': 'No user data available'}

        # Aggregate statistics
        all_complexities = []
        all_effectiveness = []
        style_preferences = Counter()

        for profile in self.user_profiles.values():
            all_complexities.extend(profile.question_complexity_history)
            all_effectiveness.extend(profile.response_effectiveness_history)
            style_preferences[profile.preferred_response_style.value] += 1

        return {
            'total_users': total_users,
            'total_interactions': total_interactions,
            'avg_query_complexity': np.mean(all_complexities) if all_complexities else 0,
            'avg_response_effectiveness': np.mean(all_effectiveness) if all_effectiveness else 0,
            'preferred_styles': dict(style_preferences.most_common()),
            'most_popular_topics': dict(sum((p.topic_interests for p in self.user_profiles.values()), Counter()).most_common(10))
        }
