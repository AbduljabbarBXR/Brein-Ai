# Deployment Guide

This comprehensive deployment guide covers all aspects of deploying Brein AI to production environments, from single-server setups to large-scale distributed deployments.

## 🚀 Quick Deployment Options

### One-Click Deployments

#### Railway (Recommended for beginners)
```bash
# Deploy to Railway
npm install -g @railway/cli
railway login
railway init brein-ai
railway up
```

#### Render
```yaml
# render.yaml
services:
  - type: web
    name: brein-ai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python backend/main.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.9
      - key: BREIN_ENV
        value: production
```

#### Heroku
```yaml
# Procfile
web: python backend/main.py

# requirements.txt (add gunicorn)
gunicorn==20.1.0
```

### Docker Deployment

#### Single Container
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "backend/main.py"]
```

```bash
# Build and run
docker build -t brein-ai .
docker run -p 8000:8000 brein-ai
```

#### Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'

services:
  brein-ai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - BREIN_ENV=production
      - DATABASE_URL=sqlite:///data/brein_memory.db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
  brein_data:
```

## 🏗️ Production Architecture

### Single Server Deployment
```
┌─────────────────────────────────────┐
│           Load Balancer             │
│          (nginx/caddy)              │
├─────────────────────────────────────┤
│         Brein AI Application        │
│   ┌─────────────┬─────────────┐     │
│   │  Web API    │  Workers    │     │
│   │  (FastAPI)  │  (Gunicorn) │     │
│   └─────────────┴─────────────┘     │
├─────────────────────────────────────┤
│         Memory System               │
│   ┌─────────────┬─────────────┐     │
│   │   SQLite    │     FAISS   │     │
│   │ (Documents) │  (Vectors)  │     │
│   └─────────────┴─────────────┘     │
├─────────────────────────────────────┤
│         Caching Layer               │
│   ┌─────────────┬─────────────┐     │
│   │    Redis    │   In-Memory │     │
│   │ (Shared)    │   (Local)   │     │
│   └─────────────┴─────────────┘     │
└─────────────────────────────────────┘
```

### Multi-Server Deployment
```
┌─────────────────────────────────────┐
│         Load Balancer               │
│       (AWS ALB/GCP LB)              │
├─────────────────────────────────────┤
│   ┌─────────────┬─────────────┐     │
│   │  App Server │  App Server │     │
│   │     #1      │     #2      │     │
│   └─────────────┴─────────────┘     │
├─────────────────────────────────────┤
│         Shared Database             │
│   ┌─────────────┬─────────────┐     │
│   │ PostgreSQL  │   Redis     │     │
│   │ (Documents) │  (Cache)    │     │
│   └─────────────┴─────────────┘     │
├─────────────────────────────────────┤
│         Vector Database             │
│   ┌─────────────┬─────────────┐     │
│   │  Pinecone   │   Weaviate  │     │
│   │   (Cloud)   │   (Self)    │     │
│   └─────────────┴─────────────┘     │
├─────────────────────────────────────┤
│         Object Storage              │
│   ┌─────────────┬─────────────┐     │
│   │   S3/GCS    │   Backups   │     │
│   │ (Assets)    │   (Daily)   │     │
│   └─────────────┴─────────────┘     │
└─────────────────────────────────────┘
```

## ☁️ Cloud Platform Deployments

### AWS Deployment

#### EC2 + RDS + ElastiCache
```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# VPC and networking
resource "aws_vpc" "brein_vpc" {
  cidr_block = "10.0.0.0/16"
}

# Application Load Balancer
resource "aws_lb" "brein_alb" {
  name               = "brein-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = aws_subnet.public.*.id
}

# ECS Cluster
resource "aws_ecs_cluster" "brein_cluster" {
  name = "brein-ai-cluster"
}

# RDS PostgreSQL
resource "aws_db_instance" "brein_db" {
  allocated_storage    = 20
  engine              = "postgres"
  engine_version      = "14.2"
  instance_class      = "db.t3.micro"
  db_name             = "brein_memory"
  username            = var.db_username
  password            = var.db_password
  skip_final_snapshot = true
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "brein_cache" {
  cluster_id      = "brein-cache"
  engine          = "redis"
  node_type       = "cache.t3.micro"
  num_cache_nodes = 1
}
```

#### Lambda + API Gateway (Serverless)
```yaml
# serverless.yml
service: brein-ai

provider:
  name: aws
  runtime: python3.9
  stage: prod
  region: us-east-1
  environment:
    BREIN_ENV: production

functions:
  api:
    handler: backend/main.handler
    events:
      - httpApi: '*'
    timeout: 30
    memorySize: 1024

  memory_ingest:
    handler: backend/memory_ingest.handler
    events:
      - sqs:
          arn: !GetAtt MemoryQueue.Arn
    timeout: 300
    memorySize: 2048

resources:
  Resources:
    MemoryQueue:
      Type: AWS::SQS::Queue
      Properties:
        QueueName: brein-memory-queue

    MemoryTable:
      Type: AWS::DynamoDB::Table
      Properties:
        TableName: brein-memory
        AttributeDefinitions:
          - AttributeName: id
            AttributeType: S
        KeySchema:
          - AttributeName: id
            KeyType: HASH
        BillingMode: PAY_PER_REQUEST
```

### Google Cloud Platform

#### App Engine (Simple)
```yaml
# app.yaml
runtime: python39

instance_class: F4

env_variables:
  BREIN_ENV: production
  DATABASE_URL: /cloudsql/brein-ai:us-central1:brein-db

beta_settings:
  cloud_sql_instances: brein-ai:us-central1:brein-db

handlers:
- url: /.*
  script: auto
  secure: always
```

#### Kubernetes (GKE)
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brein-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: brein-ai
  template:
    metadata:
      labels:
        app: brein-ai
    spec:
      containers:
      - name: brein-ai
        image: gcr.io/brein-ai/brein-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: brein-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: brein-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: brein-ai-service
spec:
  selector:
    app: brein-ai
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Azure Deployment

#### Container Instances + Cosmos DB
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerName": {
      "type": "string",
      "defaultValue": "brein-ai"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2021-07-01",
      "name": "[parameters('containerName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "[parameters('containerName')]",
            "properties": {
              "image": "breinai.azurecr.io/brein-ai:latest",
              "ports": [
                {
                  "port": 8000,
                  "protocol": "TCP"
                }
              ],
              "environmentVariables": [
                {
                  "name": "DATABASE_URL",
                  "value": "[reference(resourceId('Microsoft.DocumentDB/databaseAccounts', variables('cosmosDbAccountName'))).documentEndpoint]"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 1,
                  "memoryInGB": 1.5
                }
              }
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ]
        }
      }
    }
  ]
}
```

## 🐳 Advanced Docker Deployments

### Multi-Stage Production Build
```dockerfile
# Dockerfile.production
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash brein

WORKDIR /home/brein/app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/brein/.local
ENV PATH=/home/brein/.local/bin:$PATH

# Copy application code
COPY . .

# Change ownership
RUN chown -R brein:brein /home/brein/app
USER brein

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "backend/main.py"]
```

### Docker Swarm Deployment
```yaml
# docker-compose.swarm.yml
version: '3.8'

services:
  brein-ai:
    image: brein-ai:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    environment:
      - DATABASE_URL=postgres://brein:password@db:5432/brein_memory
      - REDIS_URL=redis://redis:6379
    networks:
      - brein-network

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=brein_memory
      - POSTGRES_USER=brein
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - brein-network
    deploy:
      placement:
        constraints:
          - node.role == manager

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    networks:
      - brein-network
    deploy:
      replicas: 1

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ssl_certs:/etc/ssl/certs
    depends_on:
      - brein-ai
    networks:
      - brein-network
    deploy:
      placement:
        constraints:
          - node.role == manager

networks:
  brein-network:
    driver: overlay

volumes:
  postgres_data:
  redis_data:
  ssl_certs:
```

## 🔧 Configuration Management

### Environment-Based Configuration
```python
# config/production.py
import os

class ProductionConfig:
    """Production configuration."""
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/brein_memory')
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY')
    API_KEY_REQUIRED = True
    
    # Performance
    WORKERS = int(os.getenv('WORKERS', 4))
    MAX_REQUESTS = int(os.getenv('MAX_REQUESTS', 1000))
    
    # Monitoring
    SENTRY_DSN = os.getenv('SENTRY_DSN')
    PROMETHEUS_ENABLED = True
    
    # External services
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
```

### Secret Management

#### AWS Secrets Manager
```python
import boto3
from botocore.exceptions import ClientError

class SecretsManager:
    def __init__(self):
        self.client = boto3.client('secretsmanager')
        self.cache = {}
    
    def get_secret(self, secret_name: str) -> str:
        """Retrieve secret from AWS Secrets Manager."""
        if secret_name in self.cache:
            return self.cache[secret_name]
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret = response['SecretString']
            self.cache[secret_name] = secret
            return secret
        except ClientError as e:
            raise RuntimeError(f"Failed to retrieve secret {secret_name}: {e}")
```

#### HashiCorp Vault
```python
import hvac

class VaultSecrets:
    def __init__(self):
        self.client = hvac.Client(
            url=os.getenv('VAULT_URL'),
            token=os.getenv('VAULT_TOKEN')
        )
    
    def get_secret(self, path: str, key: str) -> str:
        """Retrieve secret from HashiCorp Vault."""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'][key]
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve secret from {path}: {e}")
```

## 📊 Monitoring and Observability

### Application Monitoring

#### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUEST_COUNT = Counter(
    'brein_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'brein_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'brein_active_connections',
    'Number of active connections'
)

MEMORY_USAGE = Gauge(
    'brein_memory_usage_bytes',
    'Current memory usage in bytes'
)

def monitor_request(method: str, endpoint: str, status: int, duration: float):
    """Record request metrics."""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

def update_system_metrics():
    """Update system resource metrics."""
    import psutil
    
    ACTIVE_CONNECTIONS.set(len(psutil.net_connections()))
    MEMORY_USAGE.set(psutil.virtual_memory().used)
```

#### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Brein AI Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(brein_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "heatmap",
        "targets": [
          {
            "expr": "brein_request_duration_seconds",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(brein_requests_total{status=~\"5..\"}[5m]) / rate(brein_requests_total[5m]) * 100",
            "legendFormat": "Error Rate %"
          }
        ]
      }
    ]
  }
}
```

### Logging and Alerting

#### Structured Logging
```python
import structlog
import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Usage
logger = structlog.get_logger()
logger.info("Processing query", user_id="123", query_length=50)
logger.error("Database connection failed", error=str(e), retry_count=3)
```

#### Alert Manager Configuration
```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@brein.ai'
  smtp_auth_username: 'alerts@brein.ai'
  smtp_auth_password: 'app_password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email'

receivers:
- name: 'email'
  email_configs:
  - to: 'team@brein.ai'
    subject: '{{ .GroupLabels.alertname }}'
    body: '{{ .CommonAnnotations.description }}'

# Alert rules
groups:
- name: brein_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(brein_requests_total{status=~"[45].*"}[5m]) / rate(brein_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }}%"

  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(brein_request_duration_seconds_bucket[5m])) > 5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }}s"
```

## 🔄 Backup and Recovery

### Automated Backups

#### Database Backup Script
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="brein_memory"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -h localhost -U brein -d $DB_NAME > $BACKUP_DIR/${DB_NAME}_${DATE}.sql

# Compress backup
gzip $BACKUP_DIR/${DB_NAME}_${DATE}.sql

# Upload to cloud storage
aws s3 cp $BACKUP_DIR/${DB_NAME}_${DATE}.sql.gz s3://brein-backups/database/

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

# Log backup completion
echo "$(date): Database backup completed - ${DB_NAME}_${DATE}.sql.gz" >> /var/log/backup.log
```

#### File System Backup
```bash
#!/bin/bash
# filesystem_backup.sh

SOURCE_DIR="/app/data"
BACKUP_DIR="/backups/filesystem"
DATE=$(date +%Y%m%d_%H%M%S)

# Create incremental backup
rsync -av --delete --link-dest=$BACKUP_DIR/latest $SOURCE_DIR $BACKUP_DIR/$DATE

# Update latest symlink
rm -f $BACKUP_DIR/latest
ln -s $BACKUP_DIR/$DATE $BACKUP_DIR/latest

# Upload to cloud
aws s3 sync $BACKUP_DIR/$DATE s3://brein-backups/filesystem/$DATE/

echo "$(date): Filesystem backup completed - $DATE" >> /var/log/backup.log
```

### Disaster Recovery

#### Recovery Procedures
```bash
#!/bin/bash
# disaster_recovery.sh

echo "Starting disaster recovery..."

# Stop application
docker-compose down

# Restore database
BACKUP_FILE=$(ls -t /backups/*.sql.gz | head -1)
gunzip -c $BACKUP_FILE | psql -h localhost -U brein -d brein_memory

# Restore filesystem
LATEST_BACKUP=$(ls -t /backups/filesystem/ | head -1)
rsync -av /backups/filesystem/$LATEST_BACKUP/ /app/data/

# Start application
docker-compose up -d

# Verify recovery
curl -f http://localhost:8000/health
if [ $? -eq 0 ]; then
    echo "Disaster recovery completed successfully"
else
    echo "Disaster recovery failed - manual intervention required"
    exit 1
fi
```

## 🚀 Scaling Strategies

### Horizontal Scaling

#### Load Balancing
```nginx
# nginx.conf
upstream brein_backend {
    least_conn;
    server app1:8000 weight=1;
    server app2:8000 weight=1;
    server app3:8000 weight=1;
    
    keepalive 32;
}

server {
    listen 80;
    server_name brein.ai;
    
    location / {
        proxy_pass http://brein_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # Static file serving
    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

#### Database Sharding
```python
class DatabaseShardManager:
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shards = [self._create_shard(i) for i in range(num_shards)]
    
    def get_shard(self, entity_id: str) -> sqlite3.Connection:
        """Get shard for entity based on consistent hashing."""
        shard_id = int(hashlib.md5(entity_id.encode()).hexdigest(), 16) % self.num_shards
        return self.shards[shard_id]
    
    def execute_cross_shard_query(self, query: str, params: list = None) -> list:
        """Execute query across all shards and merge results."""
        all_results = []
        for shard in self.shards:
            try:
                cursor = shard.cursor()
                cursor.execute(query, params or [])
                results = cursor.fetchall()
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Query failed on shard: {e}")
        
        return self._merge_results(all_results)
```

### Vertical Scaling

#### Resource Optimization
```python
class ResourceManager:
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager()
        self.worker_manager = WorkerManager()
    
    def optimize_resources(self):
        """Dynamically adjust resources based on load."""
        load_metrics = self._get_system_load()
        
        # Adjust cache size
        if load_metrics['memory_pressure'] > 0.8:
            self.cache_manager.reduce_size(percent=20)
        
        # Scale workers
        if load_metrics['cpu_usage'] > 0.8:
            self.worker_manager.scale_up(workers=2)
        elif load_metrics['cpu_usage'] < 0.3:
            self.worker_manager.scale_down(workers=1)
        
        # Optimize memory
        if load_metrics['memory_usage'] > 0.9:
            self.memory_manager.compact()
            self.memory_manager.cleanup_old_entries(days=7)
```

## 🔒 Security Hardening

### Production Security Checklist
- [ ] Use HTTPS with valid SSL certificates
- [ ] Implement proper authentication and authorization
- [ ] Enable rate limiting and DDoS protection
- [ ] Regularly update dependencies and base images
- [ ] Use secrets management for sensitive data
- [ ] Implement proper logging and monitoring
- [ ] Set up automated security scanning
- [ ] Configure firewall rules and network security
- [ ] Enable backup encryption
- [ ] Implement access controls and least privilege

### Container Security
```dockerfile
# Secure Dockerfile
FROM python:3.9-slim

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r brein && useradd -r -g brein brein

# Set working directory
WORKDIR /app

# Copy and install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=brein:brein . .

# Switch to non-root user
USER brein

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "backend/main.py"]
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Code reviewed and tested
- [ ] Security scan passed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Backup created
- [ ] Rollback plan prepared

### Deployment Steps
- [ ] Deploy to staging environment
- [ ] Run integration tests
- [ ] Verify monitoring and alerting
- [ ] Perform load testing
- [ ] Update DNS if needed
- [ ] Deploy to production
- [ ] Monitor for issues
- [ ] Update documentation

### Post-Deployment
- [ ] Verify all services are healthy
- [ ] Check monitoring dashboards
- [ ] Run smoke tests
- [ ] Notify stakeholders
- [ ] Monitor for 24-48 hours
- [ ] Document any issues encountered

## 📞 Support and Resources

### Deployment Resources
- [[Configuration|Configuration]] - Configuration options
- [[Performance Optimization|Performance-Optimization]] - Performance tuning
- [[Security Overview|Security-Overview]] - Security best practices
- [[Troubleshooting|Troubleshooting]] - Common deployment issues

### Getting Help
- **Deployment Issues**: Check cloud provider documentation
- **Performance Problems**: Review monitoring dashboards
- **Security Concerns**: Contact security team immediately
- **General Support**: abdijabarboxer2009@gmail.com

---

*Deployment Guide - Last updated: November 2025*
