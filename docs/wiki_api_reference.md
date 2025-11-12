# API Reference

This comprehensive API reference covers all endpoints, parameters, and usage examples for the Brein AI REST API.

## 🌐 Base URL
```
http://localhost:8000/api
```

## 📋 Authentication
Currently, no authentication is required. Future versions may include API key authentication.

## 📚 Core Endpoints

### Query Processing

#### POST `/query`
Process a natural language query through the multi-agent system.

**Request Body:**
```json
{
  "query": "string (required) - The user's question or request",
  "session_id": "string (optional) - Session identifier for conversation continuity",
  "enable_web_access": "boolean (optional) - Allow web content fetching (default: false)",
  "max_tokens": "integer (optional) - Maximum response length (default: 2048)",
  "temperature": "float (optional) - Response creativity (0.0-1.0, default: 0.7)"
}
```

**Response:**
```json
{
  "response": "string - AI-generated response",
  "session_id": "string - Session identifier",
  "processing_time": "float - Time taken in seconds",
  "agents_used": ["string"] - List of agents that participated",
  "confidence_score": "float - Response confidence (0.0-1.0)",
  "sources": ["string"] - Web sources used (if web access enabled)
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain quantum computing in simple terms",
    "session_id": "user123",
    "enable_web_access": true
  }'
```

**Error Responses:**
- `400 Bad Request` - Invalid query parameters
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - System error

### Memory Management

#### GET `/memory/stats`
Retrieve comprehensive memory system statistics.

**Response:**
```json
{
  "total_documents": "integer - Total documents in memory",
  "total_vectors": "integer - Total vector embeddings",
  "cache_size": "integer - Current cache size",
  "cache_hit_rate": "float - Cache hit percentage",
  "memory_usage": "object - Memory usage statistics",
  "last_updated": "string - ISO timestamp"
}
```

#### GET `/memory/search`
Search memory for similar content.

**Query Parameters:**
- `q` (required): Search query string
- `limit` (optional): Maximum results (default: 10)
- `threshold` (optional): Similarity threshold (default: 0.7)

**Response:**
```json
{
  "results": [
    {
      "content": "string - Matched content",
      "similarity": "float - Similarity score",
      "metadata": "object - Content metadata",
      "timestamp": "string - When content was added"
    }
  ],
  "total_found": "integer - Total matches found",
  "search_time": "float - Search execution time"
}
```

#### POST `/memory/ingest`
Add new content to the memory system.

**Request Body:**
```json
{
  "content": "string (required) - Content to store",
  "metadata": "object (optional) - Additional metadata",
  "source": "string (optional) - Content source",
  "tags": ["string"] (optional) - Content tags
}
```

**Response:**
```json
{
  "document_id": "string - Unique document identifier",
  "vectors_created": "integer - Number of vectors generated",
  "processing_time": "float - Ingestion time",
  "status": "string - Success status"
}
```

### Web Content Integration

#### POST `/web/fetch`
Fetch and process web content through the safety pipeline.

**Request Body:**
```json
{
  "url": "string (required) - URL to fetch",
  "max_depth": "integer (optional) - Crawl depth (default: 1)",
  "include_images": "boolean (optional) - Extract images (default: false)",
  "sanitize_content": "boolean (optional) - Apply content sanitization (default: true)"
}
```

**Response:**
```json
{
  "url": "string - Fetched URL",
  "title": "string - Page title",
  "content": "string - Sanitized content",
  "word_count": "integer - Content length",
  "processing_time": "float - Total processing time",
  "safety_checks": "object - Safety pipeline results",
  "status": "string - Processing status"
}
```

#### GET `/web/quarantine/list`
List quarantined web content awaiting review.

**Query Parameters:**
- `status` (optional): Filter by status (pending, approved, rejected)
- `limit` (optional): Maximum results (default: 50)

**Response:**
```json
{
  "quarantined_content": [
    {
      "id": "string - Quarantine ID",
      "url": "string - Source URL",
      "title": "string - Content title",
      "submitted_at": "string - Submission timestamp",
      "status": "string - Review status",
      "safety_flags": ["string"] - Detected issues
    }
  ],
  "total_count": "integer - Total quarantined items"
}
```

#### POST `/web/quarantine/review`
Approve or reject quarantined content.

**Request Body:**
```json
{
  "quarantine_id": "string (required) - Quarantine item ID",
  "action": "string (required) - 'approve' or 'reject'",
  "reviewer_notes": "string (optional) - Review comments"
}
```

### Device Synchronization

#### POST `/sync/register-device`
Register a new mobile device for synchronization.

**Request Body:**
```json
{
  "device_id": "string (required) - Unique device identifier",
  "device_name": "string (required) - Human-readable device name",
  "device_type": "string (required) - 'mobile', 'desktop', 'tablet'",
  "capabilities": ["string"] (required) - Device capabilities
}
```

**Response:**
```json
{
  "device_id": "string - Confirmed device ID",
  "sync_token": "string - Synchronization token",
  "registered_at": "string - Registration timestamp",
  "status": "string - Registration status"
}
```

#### GET `/sync/delta`
Get synchronization delta for device.

**Headers:**
- `Authorization: Bearer <sync_token>`

**Query Parameters:**
- `since` (optional): ISO timestamp for incremental sync
- `device_id` (required): Device identifier

**Response:**
```json
{
  "delta": {
    "new_documents": ["object"] - New documents to sync,
    "updated_documents": ["object"] - Updated documents,
    "deleted_documents": ["string"] - Deleted document IDs
  },
  "sync_timestamp": "string - Sync completion timestamp"
}
```

### Testing & Monitoring

#### POST `/test/run-comprehensive`
Execute the full test suite.

**Request Body:**
```json
{
  "test_categories": ["string"] (optional) - Specific test categories to run",
  "verbose": "boolean (optional) - Detailed output (default: false)",
  "parallel": "boolean (optional) - Run tests in parallel (default: true)"
}
```

**Response:**
```json
{
  "test_results": {
    "total_tests": "integer - Total tests run",
    "passed": "integer - Tests passed",
    "failed": "integer - Tests failed",
    "skipped": "integer - Tests skipped",
    "execution_time": "float - Total execution time"
  },
  "detailed_results": "object - Individual test results"
}
```

#### POST `/test/benchmark`
Run performance benchmarking.

**Request Body:**
```json
{
  "iterations": "integer (optional) - Benchmark iterations (default: 100)",
  "concurrency": "integer (optional) - Concurrent requests (default: 1)",
  "test_types": ["string"] (optional) - Specific benchmarks to run"
}
```

**Response:**
```json
{
  "benchmark_results": {
    "query_processing": {
      "avg_response_time": "float - Average response time in ms",
      "p95_response_time": "float - 95th percentile response time",
      "requests_per_second": "float - Throughput"
    },
    "memory_operations": {
      "ingestion_rate": "float - Documents per second",
      "search_latency": "float - Average search time"
    }
  },
  "system_metrics": "object - System resource usage"
}
```

#### GET `/health`
Get comprehensive system health status.

**Response:**
```json
{
  "status": "string - Overall health ('healthy', 'degraded', 'unhealthy')",
  "timestamp": "string - Health check timestamp",
  "services": {
    "database": "string - Database status",
    "memory_system": "string - Memory system status",
    "agents": "object - Individual agent statuses",
    "api_server": "string - API server status"
  },
  "metrics": {
    "uptime": "integer - System uptime in seconds",
    "memory_usage": "float - Memory usage percentage",
    "cpu_usage": "float - CPU usage percentage",
    "active_connections": "integer - Active connections"
  },
  "alerts": ["string"] - Active system alerts
}
```

## 🔧 Configuration Endpoints

### GET `/config`
Retrieve current system configuration.

**Response:**
```json
{
  "database": "object - Database configuration",
  "models": "object - Model configurations",
  "server": "object - Server settings",
  "security": "object - Security settings",
  "agents": "object - Agent configurations"
}
```

### POST `/config/update`
Update system configuration (requires restart).

**Request Body:**
```json
{
  "section": "string (required) - Configuration section to update",
  "key": "string (required) - Configuration key",
  "value": "any (required) - New value",
  "restart_required": "boolean - Whether restart is needed"
}
```

## 📊 Rate Limits

- **Query Processing**: 100 requests per minute per IP
- **Memory Operations**: 500 requests per minute per IP
- **Web Content**: 50 requests per minute per IP
- **Administrative**: 10 requests per minute per IP

## 🚨 Error Handling

All API errors follow this format:
```json
{
  "error": {
    "code": "string - Error code",
    "message": "string - Human-readable error message",
    "details": "object - Additional error details",
    "timestamp": "string - Error timestamp"
  }
}
```

### Common Error Codes
- `INVALID_REQUEST` - Malformed request
- `AUTHENTICATION_FAILED` - Authentication error
- `PERMISSION_DENIED` - Insufficient permissions
- `RESOURCE_NOT_FOUND` - Requested resource not found
- `RATE_LIMIT_EXCEEDED` - Rate limit exceeded
- `INTERNAL_ERROR` - System internal error

## 📝 Examples

### Complete Query Workflow
```python
import requests

# 1. Submit query
response = requests.post('http://localhost:8000/api/query', json={
    'query': 'What is machine learning?',
    'enable_web_access': True
})
result = response.json()

# 2. Check memory stats
stats = requests.get('http://localhost:8000/api/memory/stats').json()

# 3. Search related content
search_results = requests.get('http://localhost:8000/api/memory/search?q=machine+learning').json()

print(f"Response: {result['response']}")
print(f"Memory documents: {stats['total_documents']}")
print(f"Related content: {len(search_results['results'])} items")
```

### Web Content Processing
```python
# Fetch and process web content
web_response = requests.post('http://localhost:8000/api/web/fetch', json={
    'url': 'https://en.wikipedia.org/wiki/Machine_learning',
    'sanitize_content': True
})

if web_response.status_code == 200:
    content = web_response.json()
    print(f"Title: {content['title']}")
    print(f"Word count: {content['word_count']}")
    print(f"Safety status: {content['safety_checks']}")
```

## 🔗 Related Documentation

- [[Installation Guide|Installation-Guide]] - Setup instructions
- [[Configuration|Configuration]] - System configuration
- [[Troubleshooting|Troubleshooting]] - Common issues
- [[Performance Optimization|Performance-Optimization]] - Tuning guide

---

*API Version: 1.0.0 - Last updated: November 2025*
