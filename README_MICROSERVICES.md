# LLMForge Microservices Architecture

## Overview

LLMForge has been transformed from a monolithic application into a cloud-native microservices architecture designed for scalability, fault isolation, and optimal resource utilization. This transformation maintains 100% API compatibility while providing enhanced performance, monitoring, and deployment flexibility.

## Architecture Components

### 🎯 Service Orchestrator
- **Location**: `microservices/orchestrator/`
- **Port**: 8080
- **Purpose**: Service discovery, load balancing, circuit breakers, health monitoring
- **Key Features**:
  - Automatic service registration and discovery
  - Intelligent request routing with health-aware load balancing
  - Circuit breaker patterns for fault tolerance
  - Real-time WebSocket updates for service status
  - Prometheus metrics integration

### 🚪 API Gateway
- **Location**: `microservices/api_gateway/`
- **Port**: 8000
- **Purpose**: Central entry point, request routing, authentication
- **Key Features**:
  - Request proxying to microservices
  - Service aggregation and batch processing
  - Rate limiting and authentication middleware
  - Comprehensive logging and metrics collection

### 🏋️ Training Service
- **Location**: `microservices/training_service/`
- **Port**: 8001
- **Purpose**: ML model fine-tuning with GPU optimization
- **Key Features**:
  - Unsloth integration for 2x faster training
  - LoRA/QLoRA parameter-efficient fine-tuning
  - Real-time training progress streaming
  - GPU memory optimization and monitoring
  - Training job queue management

### 💬 Chat Comparison Service
- **Location**: `microservices/chat_comparison_service/`
- **Port**: 8002
- **Purpose**: Dual model inference and comparison
- **Key Features**:
  - FastLibra KV cache optimization
  - Real-time model comparison with streaming
  - Session management with WebSocket support
  - Multi-model inference pipeline
  - Performance metrics and caching

### 📊 Data Generation Service
- **Location**: `microservices/data_generation_service/`
- **Port**: 8003
- **Purpose**: Automated dataset creation and validation
- **Key Features**:
  - Template-based data generation
  - Quality validation and format conversion
  - Batch processing with progress tracking
  - Multiple output formats (JSON, CSV, HuggingFace, etc.)
  - Data quality analysis and reporting

### 📈 Visualization Service
- **Location**: `microservices/visualization_service/`
- **Port**: 8004
- **Purpose**: Analytics, monitoring dashboards, and metrics
- **Key Features**:
  - Training metrics visualization
  - System health monitoring dashboards
  - Real-time data streaming
  - Custom chart generation
  - Data export in multiple formats

### 🗄️ Model Management Service
- **Location**: `microservices/model_management_service/`
- **Port**: 8005
- **Purpose**: Model lifecycle management and caching
- **Key Features**:
  - Model loading, caching, and memory optimization
  - Version control and storage management
  - GPU memory monitoring and allocation
  - Model downloading from HuggingFace Hub
  - Performance benchmarking and analysis

## Quick Start

### Prerequisites

```bash
# Required tools
- Docker (v20.10+)
- Docker Compose (v3.8+)
- Kubernetes (v1.25+) - for production
- kubectl
- Helm (v3.8+)

# System requirements
- 16+ CPU cores
- 32+ GB RAM
- 2+ GPU devices (NVIDIA)
- 500+ GB storage
```

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd LLMForge

# Start with Docker Compose
docker-compose -f docker/docker-compose.microservices.yml up -d

# Wait for services to be healthy
docker-compose -f docker/docker-compose.microservices.yml ps

# Test the deployment
curl http://localhost:8000/health
```

### Production Deployment

```bash
# Deploy to Kubernetes
chmod +x scripts/deploy_microservices.sh
./scripts/deploy_microservices.sh deploy

# Check deployment status
kubectl get pods -n llmforge

# Access services
kubectl port-forward service/llmforge-api-gateway 8000:8000 -n llmforge
```

## Service Communication

### Service Discovery Flow
1. Services register with the Orchestrator on startup
2. Orchestrator maintains a registry of healthy services
3. API Gateway queries Orchestrator for service locations
4. Load balancing and circuit breakers ensure reliability

### API Routing
```
Client → API Gateway → Orchestrator → Target Service
                   ↓
              Load Balancer
                   ↓
            Circuit Breaker
```

### Communication Patterns
- **Synchronous**: REST APIs for immediate responses
- **Asynchronous**: Message queues for background tasks
- **Real-time**: WebSocket connections for streaming data
- **Monitoring**: Prometheus metrics collection

## Configuration

### Environment Variables

#### Global Configuration
```env
ORCHESTRATOR_URL=http://llmforge-orchestrator:8080
REDIS_HOST=llmforge-redis
REDIS_PORT=6379
HUGGINGFACE_TOKEN=your_token_here
LOG_LEVEL=INFO
PROMETHEUS_ENABLED=true
```

#### Service-Specific Configuration
```env
# Training Service
CUDA_VISIBLE_DEVICES=0
MAX_TRAINING_JOBS=2
UNSLOTH_ENABLED=true

# Chat Comparison Service  
MAX_CONCURRENT_SESSIONS=10
KV_CACHE_SIZE=4GB

# Model Management Service
MODEL_CACHE_SIZE=200GB
AUTO_CLEANUP_ENABLED=true
```

### Kubernetes ConfigMaps
Configuration is managed through Kubernetes ConfigMaps:
- `llmforge-config`: Global settings
- `training-config`: Training-specific parameters
- `prometheus-config`: Monitoring configuration

## Monitoring & Observability

### Prometheus Metrics
Each service exposes metrics on the `/metrics` endpoint:

```bash
# Service availability
up{job="llmforge-training"}

# Request rates
rate(gateway_requests_total[5m])

# Resource utilization
gpu_utilization_percent
active_training_jobs

# Error rates
rate(training_errors_total[5m])
```

### Grafana Dashboards
Access comprehensive dashboards:
```bash
kubectl port-forward service/grafana 3000:3000 -n llmforge
# http://localhost:3000 (admin/admin)
```

### Health Checks
Every service implements health endpoints:
- Basic health: `GET /health`
- Detailed status: `GET /health/detailed`
- Readiness: `GET /ready`
- Liveness: `GET /live`

## API Compatibility

The microservices architecture maintains 100% compatibility with the original monolithic API:

### Original Endpoints (Still Work)
```bash
# Training
POST /api/v1/fine-tune
GET  /api/v1/status/{task_id}

# Model Comparison
POST /api/v1/sessions
POST /api/v1/sessions/{id}/compare

# Data Generation
POST /api/v1/generate
GET  /api/v1/templates

# Visualization
GET  /api/v1/analytics/training/{id}
GET  /api/v1/dashboard/overview
```

### New Microservices Endpoints
```bash
# Service Management
GET  /services              # List all services
GET  /routes                # Show routing configuration
GET  /api/v1/status/overview # System-wide health

# Advanced Features
POST /api/v1/batch          # Batch request processing
GET  /metrics               # Prometheus metrics
WS   /ws                    # WebSocket connections
```

## Development Guide

### Adding a New Service

1. **Create Service Structure**:
```bash
mkdir microservices/new_service
cd microservices/new_service
```

2. **Implement FastAPI App**:
```python
from fastapi import FastAPI
from microservices.orchestrator.service_orchestrator import register_with_orchestrator

app = FastAPI(title="New Service")

@app.on_event("startup")
async def startup():
    await register_with_orchestrator(
        orchestrator_url=os.getenv('ORCHESTRATOR_URL'),
        service_name="llmforge-new-service",
        service_id=f"new-service-{uuid.uuid4().hex[:8]}",
        host=os.getenv('SERVICE_HOST', 'localhost'),
        port=int(os.getenv('SERVICE_PORT', '8006'))
    )
```

3. **Create Dockerfile**:
```dockerfile
FROM python:3.11-slim
# ... service-specific setup
CMD ["python", "-m", "microservices.new_service.main"]
```

4. **Add Kubernetes Manifests**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llmforge-new-service
  namespace: llmforge
spec:
  # ... deployment configuration
```

5. **Update API Gateway Routing**:
```python
SERVICE_ROUTES = {
    '/api/v1/new-endpoint': 'llmforge-new-service',
    # ... existing routes
}
```

### Testing Services

```bash
# Run integration tests
python -m pytest tests/microservices_test_suite.py -v

# Test specific service
curl http://localhost:8001/health  # Training service
curl http://localhost:8002/health  # Chat comparison
```

## Scaling Guide

### Horizontal Scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llmforge-api-gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Vertical Scaling
```yaml
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 8000m
    memory: 16Gi
    nvidia.com/gpu: 1
```

### GPU Scaling
```bash
# Check GPU availability
kubectl get nodes -l nvidia.com/gpu=true

# Scale GPU-dependent services
kubectl scale deployment llmforge-training --replicas=2 -n llmforge
```

## Security

### Container Security
- Non-root user execution
- Minimal base images
- Security scanning
- Network policies

### Service Communication
- mTLS between services (when service mesh enabled)
- API authentication and authorization
- Secret management with Kubernetes secrets
- Network segmentation

### Access Control
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: llmforge
  name: llmforge-service
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
```

## Troubleshooting

### Common Issues

#### Service Discovery Problems
```bash
# Check orchestrator logs
kubectl logs deployment/llmforge-orchestrator -n llmforge

# Verify service registration
curl http://localhost:8080/services
```

#### GPU Allocation Issues
```bash
# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Verify GPU drivers
kubectl get pods -n gpu-operator-resources
```

#### Memory Issues
```bash
# Check resource usage
kubectl top pods -n llmforge

# Optimize memory settings
kubectl patch deployment llmforge-training -n llmforge -p '{"spec":{"template":{"spec":{"containers":[{"name":"training-service","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

#### Network Connectivity
```bash
# Test service connectivity
kubectl exec -it deployment/llmforge-api-gateway -n llmforge -- curl http://llmforge-orchestrator:8080/health

# Check network policies
kubectl get networkpolicy -n llmforge
```

### Debugging Commands

```bash
# Service logs
kubectl logs -f deployment/llmforge-training -n llmforge

# Pod shell access
kubectl exec -it pod-name -n llmforge -- /bin/bash

# Port forwarding for debugging
kubectl port-forward service/llmforge-orchestrator 8080:8080 -n llmforge

# Resource monitoring
watch kubectl top pods -n llmforge
```

## Performance Optimization

### Caching Strategy
- Redis for cross-service caching
- Model cache for frequently used models
- KV cache optimization for inference
- HTTP response caching in API Gateway

### Resource Management
- GPU memory optimization
- CPU and memory limits tuning
- Storage class optimization
- Network bandwidth management

### Load Testing
```bash
# Run load tests
k6 run tests/load_test.js

# Monitor during load testing
kubectl port-forward service/grafana 3000:3000 -n llmforge
```

## Migration from Monolithic

See [Migration Guide](scripts/migration_guide.md) for detailed steps to migrate from the monolithic architecture to microservices.

### Key Benefits After Migration
- ✅ Independent service scaling
- ✅ Fault isolation and resilience  
- ✅ Zero-downtime deployments
- ✅ Resource optimization
- ✅ Enhanced monitoring and observability
- ✅ Easier development and maintenance
- ✅ Better GPU resource utilization
- ✅ Improved system reliability

## Support and Maintenance

### Regular Operations
```bash
# Health check all services
kubectl get pods -n llmforge

# Check system metrics
curl http://localhost:9090/api/v1/targets

# View service logs
kubectl logs -f deployment/llmforge-api-gateway -n llmforge --tail=100
```

### Backup and Recovery
```bash
# Backup critical data
kubectl get all -n llmforge -o yaml > llmforge-backup.yaml

# Backup persistent volumes
# (Implementation depends on storage class)
```

### Updates and Rollbacks
```bash
# Rolling update
kubectl set image deployment/llmforge-api-gateway api-gateway=llmforge/api-gateway:v2.0.0 -n llmforge

# Rollback if needed
kubectl rollout undo deployment/llmforge-api-gateway -n llmforge

# Check rollout status
kubectl rollout status deployment/llmforge-api-gateway -n llmforge
```

## Contributing

### Development Workflow
1. Create feature branch
2. Develop and test locally with Docker Compose
3. Run integration tests
4. Deploy to staging Kubernetes cluster
5. Create pull request
6. Deploy to production after approval

### Code Standards
- Python 3.11+
- FastAPI for all services
- Async/await patterns
- Comprehensive error handling
- Prometheus metrics integration
- Docker best practices
- Kubernetes resource optimization

## Resources

- [Feature Requirements](feature.md)
- [Migration Guide](scripts/migration_guide.md)
- [Deployment Scripts](scripts/)
- [Kubernetes Manifests](kubernetes/)
- [Docker Configurations](docker/)
- [Test Suite](tests/)
- [API Documentation](http://localhost:8000/docs) (when running)

---

For questions or support, please refer to the troubleshooting section or create an issue in the repository.