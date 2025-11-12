# Quick Start Guide

Get Brein AI up and running in under 5 minutes with this streamlined guide.

## ⚡ Prerequisites Check

### System Requirements (Quick Check)
```bash
# Check Python version
python --version  # Should be 3.8+

# Check available RAM
free -h | grep "^Mem:"  # Should show 4GB+

# Check disk space
df -h . | tail -1  # Should show 2GB+ available
```

### One-Command Setup
```bash
# Clone and setup in one command
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git && cd Brein-Ai && python setup.py
```

## 🚀 3-Step Installation

### Step 1: Get the Code
```bash
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git
cd Brein-Ai
```

### Step 2: Run Setup
```bash
# Automated setup (recommended)
python setup.py

# Or manual setup
python -m venv brein_env
source brein_env/bin/activate  # Windows: brein_env\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Start the System
```bash
python backend/main.py
```

**That's it!** 🎉 Your Brein AI system is now running at `http://localhost:8000`

## 💻 First Interaction

### Open the Web Interface
1. Open your browser
2. Go to `http://localhost:8000`
3. You should see the Brein AI dashboard

### Make Your First Query
```bash
# Using curl
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, can you explain what you are?"}'
```

### Expected Response
```json
{
  "response": "Hello! I'm Brein AI, a memory-first artificial intelligence system...",
  "session_id": "abc123...",
  "processing_time": 1.2,
  "agents_used": ["amygdala", "hippocampus"],
  "confidence_score": 0.89
}
```

## 🎯 Core Features Demo

### 1. Memory Learning
```bash
# Teach Brein AI something
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Remember that my favorite programming language is Python"}'
```

### 2. Memory Recall
```bash
# Ask about what it learned
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my favorite programming language?"}'
```

### 3. Web Content Integration
```bash
# Ask about current information (requires web access)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the latest news about AI?", "enable_web_access": true}'
```

### 4. Complex Reasoning
```bash
# Test multi-step reasoning
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "If I have 3 apples and give away 2, how many do I have left? Explain your reasoning."}'
```

## 🔧 Basic Configuration

### Create a Basic Config File
```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000
  },
  "security": {
    "web_access_default": false
  }
}
```

### Environment Variables (Optional)
```bash
export BREIN_PORT=3000  # Change default port
export BREIN_DEBUG=true  # Enable debug logging
```

## 📊 System Health Check

### Quick Health Check
```bash
curl http://localhost:8000/health
```

**Expected healthy response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "healthy",
    "memory_system": "healthy",
    "agents": "healthy",
    "api_server": "healthy"
  }
}
```

### View System Stats
```bash
curl http://localhost:8000/api/memory/stats
```

## 🧪 Basic Testing

### Run a Simple Test
```bash
# Test the API
python -c "
import requests
response = requests.post('http://localhost:8000/api/query',
                        json={'query': 'Test query'})
print('Status:', response.status_code)
print('Response:', response.json())
"
```

### Check Logs
```bash
# View recent activity
tail -20 logs/brein.log
```

## 🚨 Troubleshooting Quick Fixes

### System Won't Start
```bash
# Check if port is in use
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Or change port
python backend/main.py --port 8001
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python environment
which python
python --version
```

### Memory Issues
```bash
# Check available memory
free -h

# Reduce memory usage
echo '{"memory": {"max_cache_size": 50000000}}' > config.json
```

## 📚 Next Steps

### Learn More
- [[User Manual|User-Manual]] - Complete user guide
- [[API Reference|API-Reference]] - Full API documentation
- [[Configuration|Configuration]] - Advanced configuration options

### Advanced Usage
- [[Web Interface|Web-Interface]] - Browser-based interface features
- [[Mobile Apps|Mobile-Apps]] - Mobile application setup
- [[Performance Optimization|Performance-Optimization]] - Tuning for better performance

### Development
- [[Contributing|Contributing]] - How to contribute to the project
- [[Testing|Testing]] - Running and writing tests
- [[Deployment|Deployment]] - Production deployment guides

## 🎉 You're All Set!

**Congratulations!** You now have a fully functional Brein AI system running on your machine.

### What You Can Do Now:
- ✅ Ask questions and get intelligent responses
- ✅ Teach the system new information
- ✅ Access web content when needed
- ✅ Experience multi-agent reasoning
- ✅ Monitor system health and performance

### Quick Commands Reference:
```bash
# Start system
python backend/main.py

# Health check
curl http://localhost:8000/health

# Make a query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'

# View logs
tail -f logs/brein.log
```

**Enjoy exploring the future of AI with Brein!** 🤖✨

---

*Quick Start Guide - Last updated: November 2025*
