# System Architecture Overview

This page provides a comprehensive overview of Brein AI's system architecture, from high-level design principles to detailed component interactions.

## 🏗️ Architectural Principles

### Memory-First Design
Brein AI implements a revolutionary memory-first architecture inspired by human cognition:

- **Hierarchical Memory System**: Working, long-term, and episodic memory layers
- **Neural Mesh Learning**: Hebbian learning principles for associative connections
- **Contextual Retrieval**: Memory-enhanced reasoning and response generation
- **Adaptive Learning**: Continuous memory consolidation and optimization

### Multi-Agent Coordination
Four specialized AI agents working in orchestrated harmony:

- **Hippocampus Agent**: Memory encoding, consolidation, and contextual retrieval
- **Prefrontal Cortex Agent**: Complex reasoning, planning, and executive functions
- **Amygdala Agent**: Emotional intelligence, personality, and social cognition
- **Thalamus Router**: Intelligent query routing and model selection

### Safety-First Approach
Comprehensive security and safety measures:

- **Content Pipeline**: Multi-stage web content processing and sanitization
- **Audit Logging**: Complete provenance tracking and compliance
- **Access Controls**: Granular permission management
- **Human Oversight**: Critical decision points with human review

## 🏛️ System Layers

### 1. User Interface Layer

#### Web Interface (`frontend/index.html`)
- **Technology**: HTML5, CSS3, JavaScript (ES6+)
- **Features**:
  - Real-time chat interface
  - Memory visualization dashboard
  - System analytics display
  - Configuration management
- **Responsiveness**: Mobile-first design with progressive enhancement

#### REST API (`backend/main.py`)
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Endpoints**: 15+ RESTful endpoints for all system functions
- **Authentication**: Currently open, future API key support
- **Rate Limiting**: Configurable per-endpoint limits

### 2. Orchestration Layer

#### System Awareness Layer (SAL)
The SAL serves as the nervous system of Brein AI, enabling real-time inter-agent communication:

**Core Components:**
- **Async Event Bus**: Non-blocking message passing between agents
- **Message Router**: Intelligent routing based on content type and complexity
- **Brain State Manager**: Global system state tracking and coordination
- **Coordination Engine**: Conflict resolution and resource allocation

**Communication Patterns:**
- **Event-Driven**: Agents respond to system events and state changes
- **Request-Response**: Synchronous operations with timeout handling
- **Broadcast**: System-wide notifications and updates
- **Priority Queuing**: Critical operations get precedence

### 3. Agent Layer

#### Hippocampus Agent (`backend/agents.py`)
```
Specialization: Memory & Learning
Model: Llama-3.2-3B
Functions:
├── Content ingestion and encoding
├── Memory consolidation and retrieval
├── Neural mesh connection formation
├── Context provision for reasoning
└── Long-term memory management
```

#### Prefrontal Cortex Agent (`backend/agents.py`)
```
Specialization: Reasoning & Planning
Model: Phi-3.1-Mini
Functions:
├── Complex problem decomposition
├── Multi-step reasoning chains
├── Strategic planning and execution
├── Executive decision making
└── Abstract thinking and analysis
```

#### Amygdala Agent (`backend/agents.py`)
```
Specialization: Emotional Intelligence
Model: TinyLlama-1.1B
Functions:
├── Personality and empathy modeling
├── Emotional context analysis
├── Natural conversation flow
├── User experience optimization
└── Social cognition and rapport
```

#### Thalamus Router (`backend/agents.py`)
```
Specialization: Intelligent Routing
Intelligence: Rule-based + ML classification
Functions:
├── Query complexity assessment
├── Model selection and routing
├── Load balancing across agents
├── Performance optimization
└── Fallback handling
```

### 4. Memory Layer

#### Vector Database (FAISS)
- **Purpose**: High-performance similarity search
- **Features**:
  - SSD offload for large datasets
  - Multiple index types (IVFFlat, HNSW)
  - Configurable similarity metrics
  - Memory-mapped storage

#### SQLite Metadata Store
- **Purpose**: Structured metadata and relationships
- **Schema**:
  - Documents table: Content and metadata
  - Vectors table: Embedding mappings
  - Relationships table: Neural mesh connections
  - Audit table: Complete operation history

#### Neural Mesh System
- **Algorithm**: Hebbian learning ("neurons that fire together wire together")
- **Features**:
  - Associative memory formation
  - Connection strength adaptation
  - Decay and pruning mechanisms
  - Graph-based traversal

### 5. Data Pipeline Layer

#### Web Content Pipeline
```
Raw Content → Sanitization → Safety Check → Quarantine → Human Review → Memory Ingestion
     ↓             ↓             ↓            ↓            ↓              ↓
  Fetching     Cleaning     Scanning    Holding    Approval      Storage
```

**Stages:**
1. **Fetch**: Web scraping with rate limiting and politeness
2. **Sanitize**: HTML cleaning, script removal, content normalization
3. **Vet**: Safety scanning, content classification, risk assessment
4. **Quarantine**: Temporary holding for human review if needed
5. **Review**: Manual approval/rejection with audit logging
6. **Ingest**: Memory encoding and neural mesh integration

#### Audit Logging System
- **Complete Provenance**: Every operation tracked with timestamps
- **Immutable Records**: Append-only logging with cryptographic hashing
- **Compliance Ready**: GDPR and audit trail requirements met
- **Performance Optimized**: Asynchronous logging with buffering

## 🔄 Data Flow Architecture

### Query Processing Flow
```
User Query → API Gateway → Thalamus Router → Agent Selection → Processing → Response
     ↓            ↓              ↓              ↓              ↓          ↓
  Natural     Authentication  Complexity     Hippocampus/   Memory +    Formatted
 Language     & Validation    Assessment     Prefrontal     Reasoning   Response
```

### Memory Ingestion Flow
```
Content Source → Sanitization → Vector Encoding → Metadata Storage → Neural Mesh → Search Index
      ↓               ↓              ↓              ↓              ↓            ↓
   Web/API       Cleaning       FAISS         SQLite        Hebbian      Optimized
   Upload        & Filtering    Storage       Database      Learning     Retrieval
```

### Learning Loop
```
Query → Memory Retrieval → Reasoning → Response Generation → User Feedback → Memory Update
   ↓           ↓              ↓          ↓                  ↓              ↓
Context    Relevant       Enhanced    Personalized     Quality        Consolidation
Loading    Information    Analysis    Response         Assessment      & Adaptation
```

## 📊 Performance Characteristics

### Scalability Metrics
| Component | Current Limits | Future Scaling |
|-----------|----------------|----------------|
| Memory Documents | 100K+ | Millions |
| Vector Dimensions | 384 | Configurable |
| Concurrent Users | 10+ | 100+ with clustering |
| Query Latency | 100-500ms | <100ms with caching |
| Storage Growth | Linear | Optimized with compression |

### Resource Utilization
- **CPU**: Multi-threaded processing with async operations
- **Memory**: Intelligent caching with LRU eviction
- **Storage**: SSD-optimized with memory mapping
- **Network**: Efficient batching and compression

## 🔒 Security Architecture

### Defense in Depth
1. **Input Validation**: Strict parameter checking and sanitization
2. **Content Filtering**: Multi-stage safety pipeline
3. **Access Control**: Role-based permissions and rate limiting
4. **Audit Logging**: Complete operation traceability
5. **Human Oversight**: Critical decision points with review

### Threat Mitigation
- **Injection Attacks**: Parameterized queries and input sanitization
- **DDoS Protection**: Rate limiting and request throttling
- **Data Leakage**: Minimal data retention and secure deletion
- **Model Poisoning**: Content validation and human review pipeline

## 🚀 Deployment Architecture

### Development Environment
- **Local Setup**: Single-machine deployment with hot reload
- **Debug Tools**: Comprehensive logging and performance monitoring
- **Testing**: Automated test suites with coverage reporting

### Production Environment
- **Containerized**: Docker deployment with orchestration
- **Load Balancing**: Multi-instance scaling with session affinity
- **Monitoring**: Real-time health checks and alerting
- **Backup**: Automated memory and configuration backups

### Cloud-Native Features
- **Microservices**: Modular architecture for independent scaling
- **Service Mesh**: Inter-service communication and observability
- **Auto-scaling**: Demand-based resource allocation
- **Disaster Recovery**: Multi-region failover capabilities

## 🔗 Integration Points

### External APIs
- **Model Providers**: OpenAI, Anthropic, local model serving
- **Vector Databases**: FAISS, Pinecone, Weaviate integration
- **Cloud Storage**: AWS S3, Google Cloud Storage for backups
- **Monitoring**: Prometheus, Grafana for observability

### Mobile Synchronization
- **Delta Sync**: Efficient incremental updates
- **Offline Mode**: Local processing with cloud reconciliation
- **Device Management**: Registration and capability detection
- **Security**: End-to-end encryption for mobile communications

## 📈 Evolution Roadmap

### Phase 1 (Current): Foundation
- ✅ Memory-first architecture implementation
- ✅ Multi-agent coordination system
- ✅ Safety and audit systems
- ✅ Web interface and API

### Phase 2 (Next): Scaling
- 🔄 Distributed memory systems
- 🔄 Advanced neural mesh algorithms
- 🔄 Multi-modal content processing
- 🔄 Enterprise integration APIs

### Phase 3 (Future): Intelligence
- 📋 Self-evolving agent behaviors
- 📋 Cross-domain knowledge transfer
- 📋 Human-AI collaborative learning
- 📋 Consciousness-like emergent properties

## 📚 Related Documentation

- [[API Reference|API-Reference]] - Complete API documentation
- [[Memory System|Memory-System]] - Detailed memory architecture
- [[Multi-Agent System|Multi-Agent-Architecture]] - Agent coordination details
- [[Security Overview|Security-Overview]] - Safety and security measures

---

*Architecture Version: 1.0.0 - Last updated: November 2025*
