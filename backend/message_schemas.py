"""
Message schemas and protocols for System Awareness Layer (SAL) communication
Standardized message formats for inter-brain component communication
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid


class BrainMessageType:
    """Standard message types for brain component communication"""

    # Memory-related messages
    MEMORY_BOOST = "memory_boost"
    MEMORY_UPDATE = "memory_update"
    MEMORY_PRIORITIZATION = "memory_prioritization"
    EMOTIONAL_MEMORY = "emotional_memory"

    # Reasoning-related messages
    REASONING_PROGRESS = "reasoning_progress"
    REASONING_COMPLETE = "reasoning_complete"
    COMPLEXITY_ASSESSMENT = "complexity_assessment"
    REASONING_FEEDBACK = "reasoning_feedback"

    # Emotional processing messages
    EMOTIONAL_CONTEXT = "emotional_context"
    EMOTIONAL_RESPONSE = "emotional_response"
    MOOD_UPDATE = "mood_update"

    # Routing and coordination messages
    ROUTING_DECISION = "routing_decision"
    AGENT_COORDINATION = "agent_coordination"
    RESOURCE_REQUEST = "resource_request"

    # System awareness messages
    STATE_UPDATE = "state_update"
    HEALTH_CHECK = "health_check"
    SYSTEM_ALERT = "system_alert"

    # Learning and adaptation messages
    LEARNING_SIGNAL = "learning_signal"
    ADAPTATION_REQUEST = "adaptation_request"
    PERFORMANCE_FEEDBACK = "performance_feedback"


class BrainMessageSchemas:
    """Standardized message schemas for brain component communication"""

    @staticmethod
    def memory_boost(sender: str, memory_chunks: List[Dict], confidence: float,
                    reason: str = "High relevance memories found") -> Dict[str, Any]:
        """Schema for memory boost messages"""
        return {
            'type': BrainMessageType.MEMORY_BOOST,
            'payload': {
                'memory_chunks': memory_chunks,
                'confidence': confidence,
                'reason': reason,
                'boost_factor': confidence * 0.1  # Convert confidence to boost factor
            }
        }

    @staticmethod
    def memory_update(sender: str, new_nodes: int, content_type: str,
                     memory_load: int) -> Dict[str, Any]:
        """Schema for memory update notifications"""
        return {
            'type': BrainMessageType.MEMORY_UPDATE,
            'payload': {
                'new_nodes': new_nodes,
                'content_type': content_type,
                'memory_load': memory_load,
                'timestamp': datetime.now().isoformat()
            }
        }

    @staticmethod
    def emotional_context(sender: str, tone: str, urgency: str,
                         triggers: List[str] = None) -> Dict[str, Any]:
        """Schema for emotional context sharing"""
        return {
            'type': BrainMessageType.EMOTIONAL_CONTEXT,
            'payload': {
                'emotional_tone': tone,
                'urgency_level': urgency,
                'active_triggers': triggers or [],
                'intensity': _calculate_emotional_intensity(tone, urgency)
            }
        }

    @staticmethod
    def reasoning_progress(sender: str, stage: str, progress: float,
                          insights: List[str] = None) -> Dict[str, Any]:
        """Schema for reasoning progress updates"""
        return {
            'type': BrainMessageType.REASONING_PROGRESS,
            'payload': {
                'stage': stage,
                'progress': progress,
                'intermediate_insights': insights or [],
                'estimated_completion': _estimate_completion_time(progress, stage)
            }
        }

    @staticmethod
    def reasoning_complete(sender: str, conclusion: str, confidence: float,
                          reasoning_steps: List[str] = None) -> Dict[str, Any]:
        """Schema for reasoning completion notifications"""
        return {
            'type': BrainMessageType.REASONING_COMPLETE,
            'payload': {
                'conclusion': conclusion,
                'confidence': confidence,
                'reasoning_steps': reasoning_steps or [],
                'quality_score': _calculate_reasoning_quality(confidence, reasoning_steps)
            }
        }

    @staticmethod
    def complexity_assessment(sender: str, query_complexity: float,
                             recommended_agent: str) -> Dict[str, Any]:
        """Schema for complexity assessment results"""
        return {
            'type': BrainMessageType.COMPLEXITY_ASSESSMENT,
            'payload': {
                'query_complexity': query_complexity,
                'recommended_agent': recommended_agent,
                'routing_confidence': _calculate_routing_confidence(query_complexity)
            }
        }

    @staticmethod
    def resource_request(sender: str, resource_type: str, amount: float,
                        priority: str = "normal") -> Dict[str, Any]:
        """Schema for resource requests"""
        return {
            'type': BrainMessageType.RESOURCE_REQUEST,
            'payload': {
                'resource_type': resource_type,
                'amount': amount,
                'priority': priority,
                'urgency_score': _calculate_urgency_score(priority)
            }
        }

    @staticmethod
    def state_update(sender: str, component_state: Dict[str, Any],
                    health_status: str = "healthy") -> Dict[str, Any]:
        """Schema for component state updates"""
        return {
            'type': BrainMessageType.STATE_UPDATE,
            'payload': {
                'component_state': component_state,
                'health_status': health_status,
                'last_updated': datetime.now().isoformat()
            }
        }

    @staticmethod
    def learning_signal(sender: str, learning_type: str, data: Dict[str, Any],
                       confidence: float = 0.8) -> Dict[str, Any]:
        """Schema for learning signals"""
        return {
            'type': BrainMessageType.LEARNING_SIGNAL,
            'payload': {
                'learning_type': learning_type,
                'data': data,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
        }

    @staticmethod
    def system_alert(sender: str, alert_type: str, severity: str,
                    message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Schema for system alerts"""
        return {
            'type': BrainMessageType.SYSTEM_ALERT,
            'payload': {
                'alert_type': alert_type,
                'severity': severity,
                'message': message,
                'context': context or {},
                'timestamp': datetime.now().isoformat()
            }
        }


class MessageValidator:
    """Validates message schemas and content"""

    @staticmethod
    def validate_message(message: Dict[str, Any]) -> bool:
        """Validate a message against schema requirements"""
        required_fields = ['type', 'payload']

        # Check required fields
        for field in required_fields:
            if field not in message:
                return False

        # Validate message type
        if message['type'] not in BrainMessageType.__dict__.values():
            return False

        # Type-specific validation
        return MessageValidator._validate_payload(message['type'], message['payload'])

    @staticmethod
    def _validate_payload(message_type: str, payload: Dict[str, Any]) -> bool:
        """Validate message payload based on type"""
        try:
            if message_type == BrainMessageType.MEMORY_BOOST:
                return ('memory_chunks' in payload and
                       'confidence' in payload and
                       isinstance(payload['confidence'], (int, float)))

            elif message_type == BrainMessageType.EMOTIONAL_CONTEXT:
                return ('emotional_tone' in payload and
                       'urgency_level' in payload)

            elif message_type == BrainMessageType.REASONING_PROGRESS:
                return ('stage' in payload and
                       'progress' in payload and
                       isinstance(payload['progress'], (int, float)))

            elif message_type == BrainMessageType.RESOURCE_REQUEST:
                return ('resource_type' in payload and
                       'amount' in payload)

            # Add more validations as needed
            return True

        except Exception:
            return False


class MessageBuilder:
    """Helper class for building standardized messages"""

    def __init__(self, sender: str):
        self.sender = sender

    def build_memory_boost(self, memory_chunks: List[Dict], confidence: float,
                          reason: str = "High relevance memories found") -> Dict[str, Any]:
        """Build a memory boost message"""
        schema = BrainMessageSchemas.memory_boost(self.sender, memory_chunks, confidence, reason)
        return self._wrap_message(schema)

    def build_emotional_context(self, tone: str, urgency: str,
                               triggers: List[str] = None) -> Dict[str, Any]:
        """Build an emotional context message"""
        schema = BrainMessageSchemas.emotional_context(self.sender, tone, urgency, triggers)
        return self._wrap_message(schema)

    def build_reasoning_progress(self, stage: str, progress: float,
                                insights: List[str] = None) -> Dict[str, Any]:
        """Build a reasoning progress message"""
        schema = BrainMessageSchemas.reasoning_progress(self.sender, stage, progress, insights)
        return self._wrap_message(schema)

    def build_resource_request(self, resource_type: str, amount: float,
                              priority: str = "normal") -> Dict[str, Any]:
        """Build a resource request message"""
        schema = BrainMessageSchemas.resource_request(self.sender, resource_type, amount, priority)
        return self._wrap_message(schema)

    def _wrap_message(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap schema in full message format"""
        return {
            'id': str(uuid.uuid4()),
            'sender': self.sender,
            'recipient': schema.get('recipient', 'all'),
            'type': schema['type'],
            'payload': schema['payload'],
            'timestamp': datetime.now().isoformat()
        }


# Helper functions
def _calculate_emotional_intensity(tone: str, urgency: str) -> float:
    """Calculate emotional intensity score"""
    tone_scores = {
        'neutral': 0.0,
        'calm': 0.2,
        'concerned': 0.4,
        'anxious': 0.6,
        'excited': 0.5,
        'angry': 0.8,
        'happy': 0.3
    }

    urgency_multipliers = {
        'low': 0.5,
        'normal': 1.0,
        'high': 1.5,
        'critical': 2.0
    }

    base_intensity = tone_scores.get(tone.lower(), 0.0)
    multiplier = urgency_multipliers.get(urgency.lower(), 1.0)

    return min(base_intensity * multiplier, 1.0)


def _estimate_completion_time(progress: float, stage: str) -> str:
    """Estimate completion time based on progress and stage"""
    if progress >= 0.9:
        return "imminent"
    elif progress >= 0.7:
        return "soon"
    elif progress >= 0.4:
        return "moderate"
    else:
        return "extended"


def _calculate_reasoning_quality(confidence: float, reasoning_steps: List[str]) -> float:
    """Calculate reasoning quality score"""
    base_quality = confidence
    step_bonus = min(len(reasoning_steps) * 0.1, 0.3)  # Bonus for detailed reasoning
    return min(base_quality + step_bonus, 1.0)


def _calculate_routing_confidence(complexity: float) -> float:
    """Calculate routing confidence based on complexity"""
    # Higher complexity = higher confidence in routing decision
    return min(complexity * 0.8 + 0.2, 1.0)


def _calculate_urgency_score(priority: str) -> float:
    """Calculate urgency score from priority level"""
    priority_scores = {
        'low': 0.2,
        'normal': 0.5,
        'high': 0.8,
        'critical': 1.0
    }
    return priority_scores.get(priority.lower(), 0.5)
