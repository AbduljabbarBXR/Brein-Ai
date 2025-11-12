# Troubleshooting Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues with Brein AI systems.

## 🔍 Quick Diagnosis

### System Health Check

Run this command to check overall system health:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-12T16:36:36Z",
  "services": {
    "database": "healthy",
    "memory_system": "healthy",
    "agents": "healthy",
    "api_server": "healthy"
  }
}
```

### Log Analysis

Check recent logs for errors:

```bash
# View last 50 lines of main log
tail -50 logs/brein.log

# Search for errors in last hour
grep "ERROR" logs/brein.log | tail -20

# Check audit logs for security events
tail -20 logs/audit.log
```

## 🚨 Critical Issues

### System Won't Start

#### Database Connection Failed
**Symptoms:**
- System fails to start with database errors
- Logs show "Unable to connect to database"

**Solutions:**
```bash
# Check if database file exists
ls -la memory/brein_memory.db

# Check file permissions
chmod 644 memory/brein_memory.db

# Reset database (WARNING: This deletes all data)
rm memory/brein_memory.db
python -c "from backend.memory_manager import MemoryManager; MemoryManager().initialize_database()"
```

#### Port Already in Use
**Symptoms:**
- "Address already in use" error on startup

**Solutions:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in config
echo '{"server": {"port": 8001}}' > config_override.json
```

#### Missing Dependencies
**Symptoms:**
- Import errors on startup
- "Module not found" messages

**Solutions:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"

# Virtual environment issues
source brein_env/bin/activate  # or brein_env\Scripts\activate on Windows
```

### Memory Issues

#### Out of Memory Errors
**Symptoms:**
- System crashes with OOM errors
- Queries fail with memory allocation errors

**Solutions:**
```json
// Reduce memory usage in config.json
{
  "memory": {
    "max_cache_size": 100000000,  // Reduce from default
    "chunk_size": 256,            // Smaller chunks
    "batch_size": 10              // Smaller batches
  },
  "caching": {
    "l1_cache": {"max_size": 5000},  // Reduce cache size
    "l2_cache": {"enabled": false}   // Disable L2 cache
  }
}
```

#### High Memory Usage
**Symptoms:**
- System uses excessive RAM
- Performance degrades over time

**Solutions:**
```bash
# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Clear caches
curl -X POST http://localhost:8000/api/admin/clear-caches

# Restart with memory limits
ulimit -v 4194304  # 4GB limit
python backend/main.py
```

### Agent Failures

#### Agent Not Responding
**Symptoms:**
- Queries hang or timeout
- Specific agent shows "unhealthy" status

**Solutions:**
```bash
# Check agent status
curl http://localhost:8000/health | jq '.services.agents'

# Restart specific agent
curl -X POST http://localhost:8000/api/admin/restart-agent \
  -d '{"agent": "hippocampus"}'

# Check agent logs
grep "hippocampus" logs/brein.log | tail -10
```

#### Model Loading Errors
**Symptoms:**
- Agent fails to initialize
- "Model not found" or "CUDA out of memory" errors

**Solutions:**
```json
// Switch to CPU-only mode
{
  "models": {
    "gpu_acceleration": false
  },
  "agents": {
    "hippocampus": {"model": "tinyllama-1.1b"},
    "prefrontal_cortex": {"model": "phi-3.1-mini"}
  }
}
```

## 🔧 Performance Issues

### Slow Query Response

#### Diagnosis
```bash
# Benchmark query performance
curl -X POST http://localhost:8000/api/test/benchmark \
  -d '{"iterations": 10}' | jq '.query_processing.avg_response_time'
```

#### Common Causes & Solutions

**High Latency (>500ms):**
```json
// Optimize FAISS index
{
  "vector_search": {
    "index": {
      "type": "IndexIVFFlat",
      "nlist": 512,
      "nprobe": 16
    }
  }
}
```

**Low Cache Hit Rate (<70%):**
```json
// Improve caching
{
  "caching": {
    "l1_cache": {"max_size": 50000, "ttl": 1800},
    "l2_cache": {"enabled": true, "ttl": 3600}
  }
}
```

**Agent Queue Backlog:**
```json
// Increase agent concurrency
{
  "concurrency": {
    "agent_pools": {
      "hippocampus": 5,
      "prefrontal_cortex": 3,
      "amygdala": 6
    }
  }
}
```

### High CPU Usage

#### Symptoms & Diagnosis
```bash
# Check CPU usage
top -p $(pgrep -f "python.*main.py")

# Profile CPU usage
python -m cProfile -s cumtime backend/main.py --profile-only
```

#### Solutions
```json
// Reduce concurrency
{
  "concurrency": {
    "workers": 2,
    "agent_pools": {
      "hippocampus": 2,
      "prefrontal_cortex": 1,
      "amygdala": 3
    }
  }
}

// Enable batching
{
  "performance": {
    "batching": {
      "enabled": true,
      "max_batch_size": 8,
      "timeout": 100
    }
  }
}
```

### Database Performance

#### Slow Queries
**Symptoms:**
- Database operations take >100ms
- High I/O wait times

**Solutions:**
```sql
-- Optimize SQLite settings
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 1000000;
PRAGMA temp_store = memory;
PRAGMA mmap_size = 268435456;
```

```json
// Connection pooling
{
  "database": {
    "connection_pool": {
      "max_connections": 10,
      "min_connections": 2,
      "max_idle_time": 300
    }
  }
}
```

## 🌐 Network Issues

### Connection Problems

#### API Unreachable
**Symptoms:**
- curl commands fail with connection refused
- Web interface doesn't load

**Solutions:**
```bash
# Check if service is running
ps aux | grep "python.*main.py"

# Check port binding
netstat -tlnp | grep :8000

# Test local connection
curl http://127.0.0.1:8000/health

# Check firewall
sudo ufw status
sudo ufw allow 8000
```

#### CORS Errors
**Symptoms:**
- Browser console shows CORS errors
- API calls fail from web interface

**Solutions:**
```json
// Update CORS settings
{
  "server": {
    "cors": {
      "enabled": true,
      "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
      "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
      "headers": ["Content-Type", "Authorization", "X-API-Key"]
    }
  }
}
```

### SSL/TLS Issues

#### Certificate Problems
**Symptoms:**
- HTTPS connections fail
- Certificate validation errors

**Solutions:**
```bash
# Check certificate validity
openssl x509 -in certs/server.crt -text -noout

# Test SSL connection
openssl s_client -connect localhost:8443 -servername localhost

# Regenerate certificates
openssl req -x509 -newkey rsa:4096 -keyout certs/server.key -out certs/server.crt -days 365 -nodes
```

## 🔒 Security Issues

### Authentication Failures

#### API Key Issues
**Symptoms:**
- 401 Unauthorized errors
- API calls rejected

**Solutions:**
```bash
# Check API key format
echo "X-API-Key: your_key_here" | curl -H @- http://localhost:8000/api/query

# Verify key in logs
grep "API_KEY" logs/audit.log | tail -5

# Rotate API key
curl -X POST http://localhost:8000/api/admin/rotate-key
```

### Rate Limiting

#### Too Many Requests
**Symptoms:**
- 429 Too Many Requests errors
- Requests being throttled

**Solutions:**
```json
// Adjust rate limits
{
  "security": {
    "rate_limiting": {
      "requests_per_minute": 200,
      "burst_limit": 50,
      "block_duration_minutes": 5
    }
  }
}
```

### Content Filtering Issues

#### False Positives
**Symptoms:**
- Legitimate content blocked
- Web content rejected incorrectly

**Solutions:**
```json
// Adjust content filtering
{
  "content_security": {
    "web_pipeline": {
      "sanitization_level": "moderate",  // Instead of "strict"
      "auto_review_threshold": 0.8,      // Lower threshold
      "domain_whitelist": ["trusted-site.com"]
    }
  }
}
```

## 💾 Data Issues

### Memory Loss

#### Symptoms:
- Previously learned information disappears
- Memory search returns no results

**Recovery:**
```bash
# Check memory file integrity
ls -la memory/brein_memory.db

# Verify database schema
sqlite3 memory/brein_memory.db ".schema"

# Restore from backup
cp backups/brein_memory_20251112.db memory/brein_memory.db

# Rebuild indexes
curl -X POST http://localhost:8000/api/admin/rebuild-indexes
```

### Data Corruption

#### Symptoms:
- Inconsistent query results
- Database errors in logs

**Recovery:**
```bash
# Run integrity check
sqlite3 memory/brein_memory.db "PRAGMA integrity_check;"

# Export and reimport data
sqlite3 memory/brein_memory.db ".dump" > backup.sql
rm memory/brein_memory.db
sqlite3 memory/brein_memory.db < backup.sql
```

## 🔄 Update Issues

### Failed Updates

#### Symptoms:
- Update process fails
- System in inconsistent state

**Recovery:**
```bash
# Check update logs
tail -50 logs/update.log

# Rollback to previous version
git checkout v1.0.0  # Replace with previous version tag
pip install -r requirements.txt
python backend/main.py

# Clean reinstall
rm -rf brein_env/
git clean -fd
# Re-run installation steps
```

## 📊 Monitoring & Alerting

### Setting Up Alerts

#### Email Alerts
```json
{
  "monitoring": {
    "alerts": {
      "email": {
        "enabled": true,
        "recipients": ["admin@brein.ai"],
        "smtp": {
          "host": "smtp.gmail.com",
          "port": 587,
          "username": "alerts@brein.ai",
          "password": "app_password"
        }
      }
    }
  }
}
```

#### Slack Integration
```json
{
  "monitoring": {
    "alerts": {
      "slack": {
        "enabled": true,
        "webhook_url": "https://hooks.slack.com/...",
        "channel": "#brein-alerts",
        "username": "Brein Monitor"
      }
    }
  }
}
```

### Custom Monitoring

#### Health Check Script
```bash
#!/bin/bash
# health_check.sh

HEALTH_URL="http://localhost:8000/health"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $STATUS -ne 200 ]; then
    echo "Brein AI is unhealthy (Status: $STATUS)"
    # Send alert
    curl -X POST -H 'Content-type: application/json' \
         --data '{"text":"Brein AI health check failed"}' \
         $SLACK_WEBHOOK
    exit 1
fi

echo "Brein AI is healthy"
exit 0
```

## 🆘 Emergency Procedures

### Complete System Reset

**WARNING: This deletes all data and resets the system**

```bash
# Stop the system
pkill -f "python.*main.py"

# Backup current data (optional)
cp -r memory/ memory_backup_$(date +%Y%m%d_%H%M%S)

# Remove all data and caches
rm -rf memory/
rm -rf logs/
rm -rf __pycache__/
rm -rf brein_env/

# Clean reinstall
git clean -fd
./setup.py  # Or manual installation steps
```

### Emergency Contacts

- **Primary Support**: abdijabarboxer2009@gmail.com
- **Emergency Hotline**: +1-800-BREIN-HELP (for critical system failures)
- **Community Support**: GitHub Issues and Discussions

## 📋 Diagnostic Commands

### System Information
```bash
# OS and hardware info
uname -a
free -h
df -h

# Python environment
python --version
pip list | grep -E "(torch|faiss|transformers)"

# Process information
ps aux | grep brein
top -p $(pgrep -f brein)
```

### Network Diagnostics
```bash
# Network connectivity
ping -c 4 google.com
traceroute google.com

# Port availability
netstat -tlnp | grep :8000
ss -tlnp | grep :8000

# DNS resolution
nslookup github.com
dig github.com
```

### Performance Profiling
```bash
# Memory profiling
python -c "
import tracemalloc
tracemalloc.start()
# Run some operations
current, peak = tracemalloc.get_traced_memory()
print(f'Current memory usage: {current / 1024 / 1024:.1f} MB')
print(f'Peak memory usage: {peak / 1024 / 1024:.1f} MB')
"

# CPU profiling
python -m cProfile -s cumtime -o profile.prof backend/main.py
snakeviz profile.prof  # Requires snakeviz for visualization
```

## 📞 Getting Help

### Support Resources

1. **Check Logs First**
   ```bash
   tail -100 logs/brein.log | grep -i error
   ```

2. **Run Diagnostics**
   ```bash
   curl http://localhost:8000/health | jq .
   ```

3. **Search Existing Issues**
   - GitHub Issues: Common problems and solutions
   - Documentation: Check relevant wiki pages

4. **Gather Information for Support**
   ```bash
   # System info
   uname -a > system_info.txt
   python --version >> system_info.txt
   pip freeze >> system_info.txt

   # Recent logs
   tail -200 logs/brein.log > recent_logs.txt

   # Configuration (redact sensitive data)
   cp config.json config_debug.json
   # Remove API keys, passwords from config_debug.json
   ```

### When to Contact Support

- **Critical Issues**: System completely down, data loss
- **Security Incidents**: Suspected breaches or attacks
- **Performance Degradation**: >50% performance drop sustained
- **Data Corruption**: Inconsistent or missing data

---

*Troubleshooting Guide Version: 1.0.0 - Last updated: November 2025*
