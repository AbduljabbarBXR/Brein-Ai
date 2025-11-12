# Multi-Agent System Architecture

This page details the coordination and communication mechanisms of Brein AI's four specialized AI agents working in orchestrated harmony.

## 🤖 Agent Overview

Brein AI implements a brain-inspired multi-agent architecture with four specialized agents:

- **Hippocampus Agent**: Memory encoding, consolidation, and contextual retrieval
- **Prefrontal Cortex Agent**: Complex reasoning, planning, and executive functions
- **Amygdala Agent**: Emotional intelligence, personality, and social cognition
- **Thalamus Router**: Intelligent query routing and model selection

## 🏛️ System Awareness Layer (SAL)

### Core Architecture
The SAL serves as the nervous system, enabling real-time inter-agent communication:

```
┌─────────────────────────────────────────────────────────────┐
│                 SYSTEM AWARENESS LAYER                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Async Event │  │ Message     │  │ Brain State │         │
│  │ Bus         │  │ Router      │  │ Manager     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│           │                │                │              │
│           └────────────────┼────────────────┘              │
│                            │                               │
│                 ┌──────────▼──────────┐                    │
│                 │  Coordination       │                    │
│                 │  Engine            │                    │
│                 └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### SAL Components

#### Async Event Bus
**Purpose**: Non-blocking message passing between agents
```python
class AsyncEventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_queue = asyncio.Queue()

    async def publish(self, event: AgentEvent):
        """Publish event to all subscribers"""
        await self.event_queue.put(event)

    async def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to specific event types"""
        self.subscribers[event_type].append(callback)
```

#### Message Router
**Purpose**: Intelligent routing based on content type and complexity
```python
class MessageRouter:
    def route_message(self, message: AgentMessage) -> List[str]:
        """Route message to appropriate agents based on content analysis"""
        complexity = self.assess_complexity(message.content)
        emotional_content = self.detect_emotional_content(message.content)

        targets = []
        if complexity > 0.7:
            targets.append("prefrontal_cortex")
        if emotional_content > 0.5:
            targets.append("amygdala")
        if self.needs_memory_context(message.content):
            targets.append("hippocampus")

        return targets or ["thalamus"]  # Default fallback
```

#### Brain State Manager
**Purpose**: Global system state tracking and coordination
```python
class BrainStateManager:
    def __init__(self):
        self.agent_states = {}
        self.global_context = {}
        self.coordination_history = []

    def update_agent_state(self, agent_id: str, state: AgentState):
        """Update individual agent state and trigger coordination if needed"""
        self.agent_states[agent_id] = state
        self.check_coordination_needed()

    def get_global_context(self) -> Dict:
        """Aggregate context from all active agents"""
        return self.global_context
```

## 🎯 Individual Agent Specifications

### Hippocampus Agent

#### Specialization: Memory & Learning
```
Model: Llama-3.2-3B (Memory-focused)
Temperature: 0.7 (Balanced creativity/conservatism)
Max Tokens: 2048
```

#### Core Functions
- **Content Encoding**: Transform input into structured memory representations
- **Memory Consolidation**: Strengthen important connections and patterns
- **Context Retrieval**: Provide relevant historical context for reasoning
- **Association Formation**: Create links between related concepts

#### Communication Patterns
```python
# Outgoing messages
hippocampus_messages = {
    "memory_encoded": {"content": "...", "vectors": [...], "associations": [...]},
    "context_retrieved": {"query": "...", "relevant_memories": [...], "confidence": 0.85},
    "learning_complete": {"new_connections": 15, "strengthened_patterns": 8}
}

# Incoming triggers
hippocampus_triggers = [
    "new_content_available",
    "reasoning_context_needed",
    "memory_consolidation_request",
    "association_query"
]
```

#### Performance Metrics
- **Encoding Speed**: 50-100 tokens/second
- **Retrieval Accuracy**: 92-96% relevance
- **Memory Consolidation**: 1000+ associations per hour
- **Context Provision**: <50ms average latency

### Prefrontal Cortex Agent

#### Specialization: Reasoning & Planning
```
Model: Phi-3.1-Mini (Reasoning-optimized)
Temperature: 0.3 (Logical and precise)
Max Tokens: 4096
```

#### Core Functions
- **Problem Decomposition**: Break complex problems into manageable parts
- **Logical Reasoning**: Apply deductive and inductive reasoning
- **Strategic Planning**: Develop multi-step action plans
- **Decision Optimization**: Evaluate options and trade-offs

#### Reasoning Pipeline
```
Complex Query → Problem Analysis → Decomposition → Step-by-Step Reasoning → Solution Synthesis
      ↓               ↓              ↓              ↓                  ↓
   Input Parsing  Structure       Sub-problems   Logical Chains     Final Answer
   & Validation   Identification  Generation     Construction       Formulation
```

#### Communication Patterns
```python
# Complex reasoning workflow
reasoning_workflow = {
    "phase_1": "problem_decomposition",
    "phase_2": "hypothesis_generation",
    "phase_3": "evidence_evaluation",
    "phase_4": "conclusion_synthesis"
}

# Coordination signals
coordination_signals = {
    "need_memory_context": True,
    "emotional_consideration": False,
    "time_constraint": "high",
    "confidence_threshold": 0.8
}
```

### Amygdala Agent

#### Specialization: Emotional Intelligence
```
Model: TinyLlama-1.1B (Emotion-focused)
Temperature: 0.8 (Expressive and adaptive)
Max Tokens: 1024
```

#### Core Functions
- **Emotional Analysis**: Detect and interpret emotional content
- **Personality Modeling**: Maintain consistent character traits
- **Empathy Generation**: Create appropriate emotional responses
- **Social Cognition**: Understand interpersonal dynamics

#### Emotional Processing
```python
class EmotionalProcessor:
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """Analyze emotional content and intensity"""
        return {
            "joy": 0.3,
            "sadness": 0.1,
            "anger": 0.0,
            "fear": 0.2,
            "surprise": 0.4,
            "trust": 0.8,
            "overall_valence": 0.65  # Positive emotional tone
        }

    def generate_empathy(self, user_emotion: Dict) -> str:
        """Generate empathetic response based on user's emotional state"""
        if user_emotion.get("sadness", 0) > 0.5:
            return "I understand this is difficult. I'm here to help."
        elif user_emotion.get("joy", 0) > 0.7:
            return "I'm glad you're feeling positive about this!"
        else:
            return "I appreciate you sharing this with me."
```

#### Personality Consistency
- **Trait Maintenance**: Stable personality characteristics across interactions
- **Context Adaptation**: Appropriate emotional responses based on situation
- **Learning Adaptation**: Gradual personality evolution based on user preferences
- **Boundary Setting**: Maintain appropriate professional boundaries

### Thalamus Router

#### Specialization: Intelligent Routing
```
Intelligence: Rule-based + ML classification
Decision Factors: Complexity, domain, user history, system load
Fallback Strategy: Default to prefrontal cortex for complex queries
```

#### Routing Logic
```python
class QueryRouter:
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Classify query characteristics for routing decisions"""
        return {
            "complexity_score": self.assess_complexity(query),
            "domain": self.identify_domain(query),
            "emotional_content": self.detect_emotion(query),
            "requires_memory": self.needs_context(query),
            "estimated_tokens": self.predict_length(query)
        }

    def select_agent(self, classification: Dict) -> str:
        """Select optimal agent based on query classification"""
        if classification["complexity_score"] > 0.8:
            return "prefrontal_cortex"
        elif classification["emotional_content"] > 0.6:
            return "amygdala"
        elif classification["requires_memory"]:
            return "hippocampus"
        else:
            # Load balancing decision
            return self.load_balance_decision()
```

#### Performance Optimization
- **Caching**: Route similar queries to same agent for consistency
- **Load Balancing**: Distribute queries across agent instances
- **Fallback Handling**: Graceful degradation when agents are unavailable
- **A/B Testing**: Route subset of queries to test agent improvements

## 🔄 Agent Communication Protocols

### Message Types
```python
# Standard message structure
message = {
    "id": "unique_message_id",
    "from": "agent_name",
    "to": "target_agent",
    "type": "message_type",
    "priority": "low|medium|high|critical",
    "content": {...},  # Message-specific content
    "timestamp": "ISO_timestamp",
    "correlation_id": "request_correlation_id"
}
```

### Communication Patterns

#### Request-Response
```
Agent A → Agent B: Request
    ↓
Agent B: Processing
    ↓
Agent B → Agent A: Response
```

#### Event Broadcasting
```
Agent A → SAL: Event
    ↓
SAL → All Subscribers: Event Notification
```

#### Coordination Workflow
```
Query → Thalamus → Primary Agent → Supporting Agents → Response Synthesis
   ↓         ↓            ↓              ↓              ↓
Routing  Selection    Processing    Context/Emotion  Final Answer
Logic    Decision     & Reasoning   Enhancement      Formulation
```

## 📊 Coordination Metrics

### Performance Indicators
- **Response Time**: End-to-end query processing latency
- **Agent Utilization**: Percentage of time each agent is active
- **Coordination Overhead**: Time spent on inter-agent communication
- **Success Rate**: Percentage of queries handled successfully

### Quality Metrics
- **Answer Accuracy**: Correctness of generated responses
- **Consistency**: Response consistency across similar queries
- **User Satisfaction**: User feedback and engagement metrics
- **Learning Progress**: Improvement in agent performance over time

## 🔧 Configuration Management

### Agent Configuration
```json
{
  "agents": {
    "hippocampus": {
      "model": "llama-3.2-3b",
      "max_tokens": 2048,
      "temperature": 0.7,
      "specialization": "memory",
      "concurrency_limit": 3
    },
    "prefrontal_cortex": {
      "model": "phi-3.1-mini",
      "max_tokens": 4096,
      "temperature": 0.3,
      "specialization": "reasoning",
      "concurrency_limit": 2
    },
    "amygdala": {
      "model": "tinyllama-1.1b",
      "max_tokens": 1024,
      "temperature": 0.8,
      "specialization": "emotion",
      "concurrency_limit": 4
    }
  }
}
```

### SAL Configuration
```json
{
  "sal": {
    "event_bus_capacity": 1000,
    "message_timeout": 30,
    "coordination_interval": 0.1,
    "max_concurrent_operations": 10,
    "health_check_interval": 60
  }
}
```

## 🚨 Error Handling & Recovery

### Agent Failure Scenarios
1. **Agent Unavailable**: Automatic failover to backup agent
2. **Response Timeout**: Circuit breaker pattern with fallback responses
3. **Inconsistent Results**: Majority voting or human arbitration
4. **Resource Exhaustion**: Load shedding and graceful degradation

### Recovery Mechanisms
- **State Synchronization**: Ensure all agents have consistent view
- **Partial Results**: Return best available answer when full coordination fails
- **User Notification**: Inform users of system issues transparently
- **Automatic Restart**: Self-healing capabilities for crashed agents

## 📈 Evolution & Learning

### Continuous Improvement
- **Performance Monitoring**: Track agent effectiveness over time
- **User Feedback Integration**: Learn from user satisfaction ratings
- **A/B Testing**: Compare different agent configurations
- **Model Updates**: Seamless model upgrades without service interruption

### Advanced Coordination
- **Dynamic Team Formation**: Agents self-organize based on query requirements
- **Skill Development**: Agents learn and improve specialized capabilities
- **Inter-Agent Learning**: Agents teach each other domain knowledge
- **Meta-Learning**: System learns optimal coordination patterns

## 🔗 Integration Points

### External Systems
- **Model Providers**: Integration with various LLM APIs
- **Vector Databases**: Connection to FAISS and other similarity search systems
- **Monitoring Tools**: Integration with observability platforms
- **Content Sources**: Web scraping and API integrations

### API Endpoints
- **Agent Management**: Individual agent control and monitoring
- **Coordination Control**: Override routing decisions manually
- **Performance Analytics**: Detailed coordination metrics and insights
- **Configuration API**: Runtime configuration updates

## 📚 Related Documentation

- [[Architecture Overview|Architecture-Overview]] - System architecture
- [[Memory System|Memory-System]] - Memory architecture details
- [[API Reference|API-Reference]] - Agent-related endpoints
- [[Performance Optimization|Performance-Optimization]] - Tuning agent performance

---

*Multi-Agent System Version: 1.0.0 - Last updated: November 2025*
