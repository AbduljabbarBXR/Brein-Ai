# Brein AI - Memory-First AI System

![Brein AI Logo](https://img.shields.io/badge/Brein-AI-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=flat-square)
![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

Brein AI is a revolutionary memory-first artificial intelligence system inspired by the human brain's architecture. It combines advanced vector databases, neural mesh learning, multi-agent processing, and comprehensive safety measures to create a powerful yet safe AI assistant.

## 🧠 Core Features

### Memory-First Architecture
- **Vector Database**: FAISS-powered similarity search with SSD offload
- **Neural Mesh**: Hebbian learning for associative memory connections
- **Hierarchical Memory**: Working, long-term, and episodic memory types
- **LRU Caching**: Intelligent memory management for optimal performance

### Multi-Agent Brain Architecture
- **Hippocampus Agent**: Memory encoding and ingestion
- **Cortex Agent**: Reasoning and thought generation
- **Basal Ganglia Agent**: Policy decisions and reinforcement learning

### Safety & Security
- **Web Content Pipeline**: Fetch → Sanitize → Vet → Quarantine → Review → Ingest
- **Audit Logging**: Complete provenance tracking for all operations
- **Content Quarantine**: Human oversight for web content ingestion
- **Access Controls**: Per-query web access toggling

### Mobile & Offline Capabilities
- **Model Export**: ONNX/TFLite conversion for mobile deployment
- **Delta Sync**: Efficient synchronization between cloud and devices
- **Offline Bundles**: Pre-packaged data for offline operation

### Performance & Monitoring
- **Real-time Profiling**: System performance monitoring
- **Comprehensive Testing**: Automated test harness with 1000+ sample documents
- **Health Monitoring**: System health status and alerts

## 🚀 Quick Start

### Installation

#### Option 1: Automated Installer (Recommended)
~bash
# Clone the repository
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git
cd Brein-Ai

# Run the installer
python setup.py
~

#### Option 2: Manual Installation
~bash
# Clone and setup
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git
cd Brein-Ai

# Create virtual environment
python -m venv brein_env
source brein_env/bin/activate  # On Windows: brein_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the system
python backend/main.py
~

### Basic Usage

1. **Start the Server**:
   ~bash
   python backend/main.py
   ~
   The API will be available at `http://localhost:8000`

2. **Access the Web Interface**:
   Open `frontend/index.html` in your browser

3. **Make API Calls**:
   ~python
   import requests

   # Query the system
   response = requests.post("http://localhost:8000/api/query",
       json={"query": "Explain machine learning", "session_id": "user123"})

   print(response.json())
   ~

## 📚 API Documentation

### Core Endpoints

#### Query Processing
~http
POST /api/query
Content-Type: application/json

{
  "query": "Your question here",
  "session_id": "optional_session_id",
  "enable_web_access": false
}
~

#### Memory Management
~http
GET /api/memory/stats          # Get memory statistics
GET /api/memory/search?q=term  # Search memory
POST /api/ingest              # Ingest new content
~

#### Web Content Integration
~http
POST /api/web/fetch           # Fetch web content (with safety pipeline)
GET /api/web/quarantine/list  # List quarantined content
POST /api/web/review          # Approve/reject quarantined content
~

#### Device Synchronization
~http
POST /api/sync/register-device  # Register mobile device
POST /api/sync/delta           # Get sync delta
POST /api/sync/apply-delta     # Apply sync changes
~

#### Testing & Monitoring
~http
POST /api/test/run-comprehensive  # Run full test suite
POST /api/profiler/start         # Start performance monitoring
GET /api/profiler/health         # Get system health status
~

## 🏗️ System Architecture

~
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │     REST API    │    │   Mobile Apps   │
│    (HTML/JS)    │    │    (FastAPI)    │    │   (React Native)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────────────┐
                    │    Orchestrator     │
                    │  (Multi-Agent)      │
                    └─────────────────────┘
                          │       │
              ┌───────────┼───────┴────────────┐
              │           │                    │
    ┌─────────▼────┐  ┌───▼────┐  ┌───────────▼─────────┐
    │ Hippocampus  │  │ Cortex │  │   Basal Ganglia     │
    │   (Memory)   │  │(Reason)│  │   (Policy/Reinforce)│
    └──────────────┘  └────────┘  └─────────────────────┘
              │           │                    │
              └───────────┼────────────────────┘
                          │
            ┌─────────────▼─────────────────────┐
            │         Memory Manager            │
            │   ┌─────────────┬─────────────┐   │
            │   │   FAISS     │  SQLite     │   │
            │   │  (Vectors)  │ (Metadata)  │   │
            │   └─────────────┴─────────────┘   │
            │   ┌─────────────────────────────┐ │
            │   │       Neural Mesh           │ │
            │   │   (Associative Learning)    │ │
            │   └─────────────────────────────┘ │
            └───────────────────────────────────┘
~

## 🔧 Configuration

Create a `config.json` file in the installation directory:

~json
{
  "database": {
    "path": "memory/brein_memory.db"
  },
  "models": {
    "embedding_model": "all-MiniLM-L6-v2"
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8000
  },
  "security": {
    "web_access_default": false,
    "audit_enabled": true
  }
}
~

## 🧪 Testing

### Run Comprehensive Tests
~bash
# Run full test suite with 1000+ documents
curl -X POST http://localhost:8000/api/test/run-comprehensive
~

### Performance Benchmarking
~bash
# Run performance benchmark
curl -X POST http://localhost:8000/api/test/benchmark -d '{"iterations": 100}'
~

### Start Performance Monitoring
~bash
# Start real-time monitoring
curl -X POST http://localhost:8000/api/profiler/start

# Get current metrics
curl http://localhost:8000/api/profiler/current
~

## 📊 Performance Metrics

- **Memory Ingestion**: ~50-200 docs/second (depending on content length)
- **Query Processing**: ~100-500ms average response time
- **Concurrent Users**: Supports 10+ simultaneous queries
- **Memory Scaling**: Handles millions of vectors with SSD offload
- **Storage Efficiency**: ~1KB per memory node (vector + metadata)

## 🔒 Security Features

- **Content Sanitization**: Removes malicious scripts and suspicious patterns
- **Domain Trust Scoring**: Configurable trusted domains list
- **Audit Trails**: Complete logging of all operations and decisions
- **Access Controls**: Granular permissions for web access and data operations
- **Data Provenance**: Full tracking of content sources and transformations

## 📱 Mobile Deployment

### Export Models for Mobile
~bash
# Export embedding model to TFLite
curl -X POST http://localhost:8000/api/models/export-mobile-bundle \
  -d '{"bundle_name": "brein_mobile_v1"}'
~

### Device Registration
~bash
# Register mobile device
curl -X POST http://localhost:8000/api/sync/register-device \
  -d '{
    "device_id": "mobile_001",
    "device_name": "iPhone 14",
    "device_type": "mobile",
    "capabilities": ["offline_mode", "sync"]
  }'
~

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FAISS**: For efficient similarity search
- **Sentence Transformers**: For text embedding models
- **FastAPI**: For the REST API framework
- **SQLite**: For metadata storage
- **BeautifulSoup**: For web content parsing

## 📞 Support

- **Documentation**: [Brein AI Docs](https://github.com/AbduljabbarBXR/Brein-Ai)
- **Discussions**: [Email](abdijabar2009@gmail.com)

---

**Built with ❤️ for the future of AI**