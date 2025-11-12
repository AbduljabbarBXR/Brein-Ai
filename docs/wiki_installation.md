# Installation Guide

This guide provides comprehensive instructions for installing and setting up Brein AI on various platforms.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space for installation + data storage
- **Network**: Internet connection for initial setup and web content fetching

### Required Software
- **Git**: For cloning the repository
- **Python 3.8+**: Main runtime environment
- **pip**: Python package manager (usually included with Python)

## 🚀 Quick Installation

### Option 1: Automated Installer (Recommended)

```bash
# Clone the repository
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git
cd Brein-Ai

# Run the automated installer
python setup.py
```

The installer will:
- Check system compatibility
- Install all required dependencies
- Set up virtual environment
- Configure initial settings
- Run basic tests

### Option 2: Manual Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git
cd Brein-Ai
```

#### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv brein_env
brein_env\Scripts\activate

# macOS/Linux
python -m venv brein_env
source brein_env/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Verify Installation
```bash
python -c "import brein_ai; print('Installation successful!')"
```

## ⚙️ Configuration

### Basic Configuration

Create a `config.json` file in the installation directory:

```json
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
```

### Advanced Configuration Options

#### Memory Configuration
```json
"memory": {
  "max_cache_size": 1000000,
  "vector_dimension": 384,
  "index_type": "IVFFlat",
  "similarity_metric": "cosine"
}
```

#### Agent Configuration
```json
"agents": {
  "hippocampus": {
    "model": "llama-3.2-3b",
    "max_tokens": 2048,
    "temperature": 0.7
  },
  "prefrontal_cortex": {
    "model": "phi-3.1-mini",
    "max_tokens": 4096,
    "temperature": 0.3
  },
  "amygdala": {
    "model": "tinyllama-1.1b",
    "max_tokens": 1024,
    "temperature": 0.8
  }
}
```

#### Security Configuration
```json
"security": {
  "trusted_domains": ["wikipedia.org", "arxiv.org"],
  "content_filtering": true,
  "audit_log_path": "logs/audit.log",
  "max_content_length": 1000000
}
```

## 🖥️ Platform-Specific Instructions

### Windows Installation

#### Using PowerShell
```powershell
# Clone repository
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git
cd Brein-Ai

# Create virtual environment
python -m venv brein_env
.\brein_env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

#### Common Windows Issues
- **Permission Error**: Run PowerShell as Administrator
- **Path Issues**: Ensure Python is in system PATH
- **Antivirus**: Add exceptions for Brein AI directories

### macOS Installation

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+
brew install python@3.9

# Clone and setup
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git
cd Brein-Ai
python3 -m venv brein_env
source brein_env/bin/activate
pip install -r requirements.txt
```

### Linux Installation (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3.9 python3.9-venv python3-pip -y

# Clone and setup
git clone https://github.com/AbduljabbarBXR/Brein-Ai.git
cd Brein-Ai
python3 -m venv brein_env
source brein_env/bin/activate
pip install -r requirements.txt
```

## 🧪 Testing Installation

### Run Basic Tests
```bash
# Test core functionality
python -m pytest backend/test_prompt_system.py -v

# Test API endpoints
python -c "from backend.main import app; print('API loaded successfully')"
```

### Performance Benchmark
```bash
# Run performance tests
python backend/performance_benchmark.py
```

## 🚀 Starting the System

### Development Mode
```bash
# Start with debug logging
python backend/main.py --debug --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
# Start with production settings
python backend/main.py --prod --workers 4
```

### Using Docker (Future)
```bash
# Build container
docker build -t brein-ai .

# Run container
docker run -p 8000:8000 brein-ai
```

## 🔧 Troubleshooting

### Common Installation Issues

#### Import Errors
```
ModuleNotFoundError: No module named 'faiss'
```
**Solution**: Ensure all dependencies are installed
```bash
pip install --upgrade -r requirements.txt
```

#### Memory Errors
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce batch sizes in configuration or add more RAM

#### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Change port in config or kill existing process
```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
```

### Performance Optimization

#### For Low-Memory Systems
```json
{
  "memory": {
    "max_cache_size": 100000,
    "batch_size": 10
  }
}
```

#### For High-Performance Systems
```json
{
  "memory": {
    "max_cache_size": 10000000,
    "batch_size": 100,
    "use_gpu": true
  }
}
```

## 📞 Support

If you encounter issues during installation:

1. Check the [[Troubleshooting|Troubleshooting]] page
2. Review the system requirements
3. Check GitHub Issues for similar problems
4. Create a new issue with detailed error logs

## 📝 Next Steps

After successful installation:
1. [[Quick Start|Quick-Start]] - Get familiar with basic usage
2. [[Configuration|Configuration]] - Fine-tune settings
3. [[User Manual|User-Manual]] - Learn advanced features

---

*Last updated: November 2025*
