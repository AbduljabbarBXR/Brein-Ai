# Configuration Guide

This comprehensive guide covers all configuration options, settings, and customization possibilities for Brein AI systems.

## 📋 Configuration Overview

Brein AI uses a hierarchical configuration system with multiple levels of settings:

- **Global Configuration**: System-wide defaults (`config.json`)
- **Environment Variables**: Runtime overrides
- **Command-line Arguments**: Startup-time configuration
- **Dynamic Configuration**: Runtime adjustments via API

## 🔧 Core Configuration File

### Basic Configuration Structure

Create a `config.json` file in your installation directory:

```json
{
  "database": {
    "path": "memory/brein_memory.db",
    "connection_pool_size": 10,
    "timeout": 30,
    "backup_enabled": true,
    "backup_interval_hours": 24
  },
  "models": {
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimensions": 384,
    "model_cache_size": 1073741824,
    "gpu_acceleration": false
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "workers": 4,
    "max_request_size": 104857600,
    "cors_origins": ["http://localhost:3000"]
  },
  "security": {
    "web_access_default": false,
    "audit_enabled": true,
    "rate_limiting_enabled": true,
    "max_requests_per_minute": 100,
    "content_filtering": true
  }
}
```

## 🗄️ Database Configuration

### SQLite Database Settings

#### Basic Database Configuration
```json
{
  "database": {
    "path": "memory/brein_memory.db",
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "cache_size": 1000000,
    "temp_store": "memory",
    "mmap_size": 268435456
  }
}
```

#### Connection Pool Configuration
```json
{
  "database": {
    "connection_pool": {
      "max_connections": 20,
      "min_connections": 2,
      "max_idle_time": 300,
      "max_lifetime": 3600,
      "acquire_timeout": 30
    }
  }
}
```

#### Backup Configuration
```json
{
  "database": {
    "backup": {
      "enabled": true,
      "interval_hours": 24,
      "retention_days": 30,
      "compression": true,
      "encryption": false,
      "remote_storage": {
        "type": "s3",
        "bucket": "brein-backups",
        "region": "us-east-1"
      }
    }
  }
}
```

## 🤖 Agent Configuration

### Hippocampus Agent (Memory)

#### Memory Agent Configuration
```json
{
  "agents": {
    "hippocampus": {
      "model": "llama-3.2-3b",
      "max_tokens": 2048,
      "temperature": 0.7,
      "top_p": 0.9,
      "repetition_penalty": 1.1,
      "context_window": 4096,
      "batch_size": 8,
      "concurrency_limit": 3,
      "memory_focus": "long_term"
    }
  }
}
```

#### Memory Processing Settings
```json
{
  "memory": {
    "chunking": {
      "strategy": "semantic",
      "max_chunk_size": 512,
      "overlap": 50,
      "language": "en"
    },
    "embedding": {
      "model": "all-MiniLM-L6-v2",
      "dimensions": 384,
      "normalize": true,
      "batch_size": 32
    },
    "neural_mesh": {
      "learning_rate": 0.01,
      "decay_factor": 0.999,
      "pruning_threshold": 0.1,
      "max_connections": 1000
    }
  }
}
```

### Prefrontal Cortex Agent (Reasoning)

#### Reasoning Agent Configuration
```json
{
  "agents": {
    "prefrontal_cortex": {
      "model": "phi-3.1-mini",
      "max_tokens": 4096,
      "temperature": 0.3,
      "top_p": 0.8,
      "repetition_penalty": 1.2,
      "context_window": 8192,
      "reasoning_depth": "deep",
      "logic_verification": true,
      "concurrency_limit": 2
    }
  }
}
```

#### Reasoning Pipeline Configuration
```json
{
  "reasoning": {
    "pipeline": {
      "decomposition_enabled": true,
      "hypothesis_generation": true,
      "evidence_evaluation": true,
      "conclusion_synthesis": true,
      "confidence_threshold": 0.8
    },
    "complexity_thresholds": {
      "simple": 0.3,
      "moderate": 0.7,
      "complex": 0.9
    }
  }
}
```

### Amygdala Agent (Emotional Intelligence)

#### Emotional Agent Configuration
```json
{
  "agents": {
    "amygdala": {
      "model": "tinyllama-1.1b",
      "max_tokens": 1024,
      "temperature": 0.8,
      "top_p": 0.95,
      "repetition_penalty": 1.0,
      "context_window": 2048,
      "emotional_awareness": true,
      "personality_consistency": true,
      "concurrency_limit": 4
    }
  }
}
```

#### Personality Configuration
```json
{
  "personality": {
    "traits": {
      "empathy": 0.8,
      "helpfulness": 0.9,
      "creativity": 0.6,
      "formality": 0.4
    },
    "adaptation": {
      "learning_rate": 0.01,
      "memory_span": 100,
      "consistency_weight": 0.7
    }
  }
}
```

### Thalamus Router (Intelligent Routing)

#### Router Configuration
```json
{
  "router": {
    "intelligence": "hybrid",
    "classification_model": "rule_based",
    "load_balancing": "adaptive",
    "fallback_strategy": "prefrontal_cortex",
    "performance_monitoring": true,
    "a_b_testing": false
  }
}
```

#### Routing Rules
```json
{
  "routing_rules": {
    "complexity_based": {
      "low": "amygdala",
      "medium": "hippocampus",
      "high": "prefrontal_cortex"
    },
    "domain_based": {
      "technical": "prefrontal_cortex",
      "emotional": "amygdala",
      "factual": "hippocampus"
    },
    "content_based": {
      "code": "prefrontal_cortex",
      "personal": "amygdala",
      "educational": "hippocampus"
    }
  }
}
```

## 🔒 Security Configuration

### Authentication & Authorization

#### API Key Configuration
```json
{
  "security": {
    "authentication": {
      "method": "api_key",
      "key_header": "X-API-Key",
      "key_rotation_days": 90,
      "rate_limiting": {
        "requests_per_minute": 100,
        "burst_limit": 20,
        "block_duration_minutes": 15
      }
    },
    "authorization": {
      "rbac_enabled": true,
      "default_role": "user",
      "admin_users": ["admin@example.com"]
    }
  }
}
```

### Content Security

#### Web Content Pipeline Configuration
```json
{
  "content_security": {
    "web_pipeline": {
      "enabled": true,
      "sanitization_level": "strict",
      "domain_whitelist": ["wikipedia.org", "arxiv.org"],
      "content_filtering": {
        "malware_detection": true,
        "toxicity_filtering": true,
        "privacy_scanning": true
      },
      "quarantine": {
        "auto_review_threshold": 0.7,
        "human_review_required": true,
        "retention_days": 30
      }
    }
  }
}
```

### Audit Configuration

#### Audit Logging Settings
```json
{
  "audit": {
    "enabled": true,
    "log_level": "detailed",
    "retention": {
      "days": 2555,
      "compression": true,
      "encryption": false
    },
    "events": {
      "authentication": true,
      "authorization": true,
      "data_access": true,
      "configuration_changes": true,
      "system_events": true
    },
    "storage": {
      "local_path": "logs/audit.log",
      "remote_backup": true,
      "format": "json"
    }
  }
}
```

## ⚡ Performance Configuration

### Caching Configuration

#### Multi-Level Cache Settings
```json
{
  "caching": {
    "l1_cache": {
      "enabled": true,
      "type": "lru",
      "max_size": 10000,
      "ttl": 300,
      "compression": false
    },
    "l2_cache": {
      "enabled": true,
      "type": "redis",
      "host": "localhost",
      "port": 6379,
      "ttl": 3600,
      "compression": true
    },
    "l3_cache": {
      "enabled": true,
      "type": "disk",
      "path": "/tmp/brein_cache",
      "ttl": 86400,
      "max_size_gb": 10
    }
  }
}
```

### Memory Optimization

#### FAISS Index Configuration
```json
{
  "vector_search": {
    "index": {
      "type": "IndexHNSW",
      "dimensions": 384,
      "M": 32,
      "efConstruction": 200,
      "efSearch": 64,
      "metric": "cosine"
    },
    "search": {
      "nprobe": 10,
      "k": 10,
      "batch_size": 100,
      "gpu_acceleration": false
    }
  }
}
```

### Concurrency Settings

#### System Concurrency Configuration
```json
{
  "concurrency": {
    "workers": 4,
    "max_connections": 1000,
    "request_timeout": 30,
    "keep_alive": 75,
    "agent_pools": {
      "hippocampus": 3,
      "prefrontal_cortex": 2,
      "amygdala": 4
    }
  }
}
```

## 🌐 Network Configuration

### Server Configuration

#### HTTP Server Settings
```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "ssl": {
      "enabled": false,
      "cert_file": "certs/server.crt",
      "key_file": "certs/server.key"
    },
    "cors": {
      "enabled": true,
      "origins": ["http://localhost:3000", "https://brein.ai"],
      "methods": ["GET", "POST", "PUT", "DELETE"],
      "headers": ["Content-Type", "Authorization"]
    }
  }
}
```

### Load Balancing

#### Load Balancer Configuration
```json
{
  "load_balancer": {
    "enabled": false,
    "algorithm": "least_connections",
    "health_check": {
      "enabled": true,
      "interval": 30,
      "timeout": 10,
      "path": "/health"
    },
    "sticky_sessions": false,
    "ssl_termination": false
  }
}
```

## 📊 Monitoring Configuration

### Metrics Collection

#### Prometheus Metrics Configuration
```json
{
  "monitoring": {
    "prometheus": {
      "enabled": true,
      "port": 9090,
      "path": "/metrics",
      "collectors": {
        "query_latency": true,
        "memory_usage": true,
        "cache_performance": true,
        "agent_utilization": true,
        "error_rates": true
      }
    }
  }
}
```

### Logging Configuration

#### Structured Logging Settings
```json
{
  "logging": {
    "level": "INFO",
    "format": "json",
    "handlers": {
      "console": {
        "enabled": true,
        "level": "INFO"
      },
      "file": {
        "enabled": true,
        "path": "logs/brein.log",
        "max_size": 104857600,
        "backup_count": 5,
        "level": "DEBUG"
      },
      "syslog": {
        "enabled": false,
        "host": "localhost",
        "port": 514,
        "facility": "local0"
      }
    }
  }
}
```

## 🔧 Environment Variables

### Runtime Configuration Overrides

```bash
# Server Configuration
export BREIN_HOST=0.0.0.0
export BREIN_PORT=8000
export BREIN_WORKERS=4

# Database Configuration
export BREIN_DATABASE_PATH=/data/brein_memory.db
export BREIN_DATABASE_POOL_SIZE=20

# Security Configuration
export BREIN_API_KEY=your_api_key_here
export BREIN_RATE_LIMIT=1000

# Model Configuration
export BREIN_EMBEDDING_MODEL=all-MiniLM-L6-v2
export BREIN_GPU_ENABLED=true

# Logging Configuration
export BREIN_LOG_LEVEL=DEBUG
export BREIN_LOG_FILE=/var/log/brein.log
```

### Environment Variable Priority

Configuration values are resolved in this order (highest priority first):

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Configuration file values**
4. **Built-in defaults** (lowest priority)

## 🚀 Advanced Configuration

### Custom Model Configuration

#### Loading Custom Models
```json
{
  "models": {
    "custom": {
      "path": "/models/custom-embedding",
      "type": "sentence-transformer",
      "dimensions": 512,
      "max_seq_length": 512,
      "pooling": "mean"
    }
  }
}
```

### Plugin System Configuration

#### Plugin Loading
```json
{
  "plugins": {
    "enabled": true,
    "paths": ["/plugins"],
    "auto_discovery": true,
    "trusted_plugins": [
      "brein-plugin-analytics",
      "brein-plugin-security"
    ]
  }
}
```

### Integration Configuration

#### External Service Integration
```json
{
  "integrations": {
    "openai": {
      "enabled": false,
      "api_key": "sk-...",
      "models": ["gpt-4", "gpt-3.5-turbo"],
      "rate_limit": 100
    },
    "anthropic": {
      "enabled": false,
      "api_key": "sk-ant-...",
      "models": ["claude-3-opus", "claude-3-sonnet"]
    },
    "pinecone": {
      "enabled": false,
      "api_key": "...",
      "environment": "us-east1-gcp",
      "index_name": "brein-memory"
    }
  }
}
```

## 🔄 Configuration Management

### Configuration Validation

#### Schema Validation
```python
from cerberus import Validator

config_schema = {
    'database': {
        'required': True,
        'type': 'dict',
        'schema': {
            'path': {'required': True, 'type': 'string'},
            'connection_pool_size': {'type': 'integer', 'min': 1, 'max': 100}
        }
    },
    'server': {
        'required': True,
        'type': 'dict',
        'schema': {
            'host': {'type': 'string'},
            'port': {'type': 'integer', 'min': 1, 'max': 65535}
        }
    }
}

def validate_config(config: dict) -> bool:
    validator = Validator(config_schema)
    return validator.validate(config)
```

### Configuration Hot Reloading

#### Dynamic Configuration Updates
```python
class ConfigManager:
    def __init__(self):
        self.config = self.load_config()
        self.watchers = []

    def watch_config_changes(self):
        """Watch for configuration file changes"""
        observer = Observer()
        observer.schedule(ConfigFileHandler(self), path='.', recursive=False)
        observer.start()

    def reload_config(self):
        """Reload configuration without restart"""
        new_config = self.load_config()
        if self.validate_config(new_config):
            self.config = new_config
            self.notify_watchers()
        else:
            logger.error("Invalid configuration, keeping current config")
```

## 📚 Configuration Examples

### Development Configuration
```json
{
  "environment": "development",
  "logging": {"level": "DEBUG"},
  "caching": {"enabled": false},
  "security": {"audit_enabled": false},
  "performance": {"monitoring": false}
}
```

### Production Configuration
```json
{
  "environment": "production",
  "logging": {"level": "WARNING"},
  "caching": {"l1_cache": {"enabled": true}, "l2_cache": {"enabled": true}},
  "security": {"audit_enabled": true, "rate_limiting_enabled": true},
  "performance": {"monitoring": true, "gpu_acceleration": true}
}
```

### High-Availability Configuration
```json
{
  "environment": "production-ha",
  "server": {"workers": 8, "load_balancer": {"enabled": true}},
  "database": {"connection_pool_size": 50, "backup_enabled": true},
  "caching": {"l2_cache": {"enabled": true}, "l3_cache": {"enabled": true}},
  "monitoring": {"prometheus": {"enabled": true}, "alerting": {"enabled": true}}
}
```

## 📞 Support

For configuration assistance:

- Check the [[Troubleshooting|Troubleshooting]] page for common issues
- Review the [[API Reference|API-Reference]] for runtime configuration
- Create an issue on GitHub for complex configuration needs

---

*Configuration Guide Version: 1.0.0 - Last updated: November 2025*
