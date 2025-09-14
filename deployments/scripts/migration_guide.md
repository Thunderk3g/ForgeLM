# LLMForge Microservices Migration Guide

## Overview

This guide provides step-by-step instructions for migrating from the monolithic LLMForge architecture to the new microservices-based system. The migration maintains full API compatibility while providing improved scalability, fault isolation, and resource optimization.

## Migration Strategy

### Phase 1: Preparation (1-2 hours)
1. **Backup Current System**
2. **Environment Preparation**
3. **Docker Registry Setup**
4. **Kubernetes Cluster Preparation**

### Phase 2: Infrastructure Deployment (2-3 hours)
1. **Deploy Monitoring Stack**
2. **Deploy Service Orchestrator**
3. **Deploy Redis Cache**
4. **Setup Storage Classes**

### Phase 3: Microservices Deployment (3-4 hours)
1. **Deploy API Gateway**
2. **Deploy Training Service**
3. **Deploy Chat Comparison Service**
4. **Deploy Data Generation Service**
5. **Deploy Visualization Service**
6. **Deploy Model Management Service**

### Phase 4: Testing & Validation (1-2 hours)
1. **Health Checks**
2. **API Testing**
3. **Load Testing**
4. **Monitoring Validation**

### Phase 5: Production Cutover (30 minutes)
1. **DNS/Load Balancer Update**
2. **Traffic Migration**
3. **Old System Shutdown**

---

## Detailed Migration Steps

### Prerequisites

```bash
# Required tools
- kubectl (v1.25+)
- docker (v20.10+)
- helm (v3.8+)
- curl
- jq

# Required access
- Kubernetes cluster with GPU nodes
- Docker registry access
- Sufficient cluster resources:
  - 32 CPU cores
  - 128GB RAM
  - 4 GPU devices
  - 2TB storage
```

### Phase 1: Preparation

#### 1.1 Backup Current System

```bash
# Create backup directory
mkdir -p /backup/llmforge/$(date +%Y%m%d)
cd /backup/llmforge/$(date +%Y%m%d)

# Backup application data
cp -r /path/to/llmforge/data ./data
cp -r /path/to/llmforge/models ./models
cp -r /path/to/llmforge/logs ./logs
cp /path/to/llmforge/.env ./env_backup

# Backup Docker volumes
docker run --rm -v llmforge_data:/data -v $(pwd):/backup alpine tar czf /backup/docker_data.tar.gz /data

# Export database (if applicable)
# pg_dump or mysql_dump commands here
```

#### 1.2 Environment Setup

```bash
# Clone the microservices code
git clone <repository> llmforge-microservices
cd llmforge-microservices

# Set environment variables
export DOCKER_REGISTRY="your-registry.com"
export IMAGE_TAG="v1.0.0"
export NAMESPACE="llmforge"

# Create environment file
cat > .env << EOF
HUGGINGFACE_TOKEN=your_token_here
GRAFANA_PASSWORD=secure_password
ORCHESTRATOR_URL=http://llmforge-orchestrator:8080
EOF
```

#### 1.3 Docker Registry Setup

```bash
# If using private registry
docker login your-registry.com

# Or setup local registry for testing
docker run -d -p 5000:5000 --name registry registry:2
export DOCKER_REGISTRY="localhost:5000"
```

### Phase 2: Infrastructure Deployment

#### 2.1 Kubernetes Cluster Validation

```bash
# Verify cluster access
kubectl cluster-info

# Check node resources
kubectl describe nodes

# Verify GPU nodes
kubectl get nodes -l nvidia.com/gpu=true

# Check storage classes
kubectl get storageclass
```

#### 2.2 Deploy Base Infrastructure

```bash
# Create namespace and base resources
kubectl apply -f kubernetes/namespace.yaml

# Deploy storage
kubectl apply -f kubernetes/storage.yaml

# Wait for PVCs
kubectl wait --for=condition=Bound pvc --all -n llmforge --timeout=300s

# Deploy ConfigMaps
kubectl apply -f kubernetes/configmaps.yaml

# Create secrets
kubectl create secret generic llmforge-secrets \
    --from-literal=huggingface-token="$HUGGINGFACE_TOKEN" \
    --from-literal=grafana-password="$GRAFANA_PASSWORD" \
    -n llmforge
```

#### 2.3 Deploy Monitoring Stack

```bash
# Deploy Prometheus and Grafana
kubectl apply -f kubernetes/monitoring.yaml

# Wait for monitoring to be ready
kubectl wait --for=condition=available deployment/prometheus -n llmforge --timeout=300s
kubectl wait --for=condition=available deployment/grafana -n llmforge --timeout=300s

# Verify monitoring
kubectl port-forward service/prometheus 9090:9090 -n llmforge &
kubectl port-forward service/grafana 3000:3000 -n llmforge &

# Test access
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health
```

#### 2.4 Deploy Redis Cache

```bash
# Deploy Redis
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

helm install redis bitnami/redis \
    --namespace llmforge \
    --set auth.enabled=false \
    --set architecture=standalone \
    --set master.persistence.size=20Gi \
    --wait

# Verify Redis
kubectl port-forward service/redis-master 6379:6379 -n llmforge &
redis-cli -h localhost ping
```

### Phase 3: Microservices Deployment

#### 3.1 Build and Push Images

```bash
# Build all service images
chmod +x scripts/deploy_microservices.sh
./scripts/deploy_microservices.sh

# Or manually build each service:
docker build -t $DOCKER_REGISTRY/llmforge-orchestrator:$IMAGE_TAG -f docker/orchestrator/Dockerfile .
docker build -t $DOCKER_REGISTRY/llmforge-training:$IMAGE_TAG -f docker/training_service/Dockerfile .
docker build -t $DOCKER_REGISTRY/llmforge-api-gateway:$IMAGE_TAG -f docker/api_gateway/Dockerfile .
# ... repeat for other services

# Push images
docker push $DOCKER_REGISTRY/llmforge-orchestrator:$IMAGE_TAG
# ... repeat for all images
```

#### 3.2 Deploy Service Orchestrator

```bash
# Deploy orchestrator first
kubectl apply -f kubernetes/orchestrator.yaml

# Wait for orchestrator
kubectl wait --for=condition=available deployment/llmforge-orchestrator -n llmforge --timeout=300s

# Verify orchestrator
kubectl port-forward service/llmforge-orchestrator 8080:8080 -n llmforge &
curl http://localhost:8080/health
```

#### 3.3 Deploy Core Services

```bash
# Deploy API Gateway
kubectl apply -f kubernetes/api-gateway.yaml

# Deploy Training Service
kubectl apply -f kubernetes/training-service.yaml

# Deploy Chat Comparison Service
kubectl apply -f kubernetes/chat-comparison-service.yaml

# Deploy remaining services
kubectl apply -f kubernetes/data-generation-service.yaml
kubectl apply -f kubernetes/visualization-service.yaml
kubectl apply -f kubernetes/model-management-service.yaml

# Wait for all deployments
kubectl wait --for=condition=available deployment --all -n llmforge --timeout=600s
```

### Phase 4: Testing & Validation

#### 4.1 Health Check Validation

```bash
# Check all pods are running
kubectl get pods -n llmforge

# Check services
kubectl get services -n llmforge

# Test each service health endpoint
services=("orchestrator:8080" "api-gateway:8000" "training:8001" "chat-comparison:8002" "data-generation:8003" "visualization:8004" "model-management:8005")

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    kubectl port-forward service/llmforge-$name $port:$port -n llmforge &
    sleep 2
    curl -f http://localhost:$port/health || echo "❌ $name health check failed"
    pkill -f "port-forward.*$port:$port"
done
```

#### 4.2 API Compatibility Testing

```bash
# Test API Gateway routing
kubectl port-forward service/llmforge-api-gateway 8000:8000 -n llmforge &

# Test legacy endpoints
curl -X GET http://localhost:8000/api/v1/health
curl -X GET http://localhost:8000/api/v1/models/available
curl -X GET http://localhost:8000/api/v1/sessions

# Test training endpoint
curl -X POST http://localhost:8000/api/v1/fine-tune \
    -H "Content-Type: application/json" \
    -d '{
        "base_model_dir": "/app/models/base/llama-2-7b",
        "dataset_dir": "/app/data/test_dataset",
        "epochs": 1,
        "batch_size": 4
    }'
```

#### 4.3 Load Testing

```bash
# Install k6 for load testing
curl https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz | tar -xzf -

# Create load test script
cat > load_test.js << 'EOF'
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    vus: 10,
    duration: '2m',
};

export default function() {
    let response = http.get('http://localhost:8000/api/v1/health');
    check(response, {
        'status is 200': (r) => r.status === 200,
        'response time < 500ms': (r) => r.timings.duration < 500,
    });
    sleep(1);
}
EOF

# Run load test
./k6 run load_test.js
```

#### 4.4 Monitoring Validation

```bash
# Check Prometheus targets
kubectl port-forward service/prometheus 9090:9090 -n llmforge &
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# Check Grafana dashboards
kubectl port-forward service/grafana 3000:3000 -n llmforge &
# Access http://localhost:3000 (admin/your_password)

# Verify metrics collection
curl "http://localhost:9090/api/v1/query?query=up{job=~'.*llmforge.*'}"
```

### Phase 5: Production Cutover

#### 5.1 Setup Ingress/Load Balancer

```bash
# Deploy ingress
kubectl apply -f kubernetes/ingress.yaml

# Or setup cloud load balancer
kubectl patch service llmforge-api-gateway -n llmforge -p '{"spec": {"type": "LoadBalancer"}}'

# Get external IP
kubectl get service llmforge-api-gateway -n llmforge -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

#### 5.2 DNS Configuration

```bash
# Update DNS records to point to new load balancer IP
# This step is environment-specific

# Example using AWS Route53
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch '{
    "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
            "Name": "api.llmforge.com",
            "Type": "A",
            "TTL": 300,
            "ResourceRecords": [{"Value": "NEW_LB_IP"}]
        }
    }]
}'
```

#### 5.3 Traffic Migration

```bash
# Gradual traffic migration using ingress weights
# Start with 10% traffic to new system
kubectl patch ingress llmforge-ingress -n llmforge --type='merge' -p='{
    "metadata": {
        "annotations": {
            "nginx.ingress.kubernetes.io/canary": "true",
            "nginx.ingress.kubernetes.io/canary-weight": "10"
        }
    }
}'

# Monitor metrics and errors
watch kubectl top pods -n llmforge

# Gradually increase traffic: 25%, 50%, 75%, 100%
# Update canary-weight annotation accordingly
```

#### 5.4 Old System Shutdown

```bash
# After full traffic migration and validation
# Shutdown old monolithic system

# Stop old containers
docker-compose down

# Optional: backup final state
docker run --rm -v llmforge_data:/data -v $(pwd):/backup alpine tar czf /backup/final_backup.tar.gz /data
```

---

## Rollback Procedures

### Emergency Rollback

```bash
# Quick rollback to monolithic system
# 1. Update DNS/Load balancer to point back to old system
# 2. Start old system
docker-compose up -d

# 3. Scale down microservices (optional)
kubectl scale deployment --all --replicas=0 -n llmforge
```

### Gradual Rollback

```bash
# Reduce microservices traffic gradually
kubectl patch ingress llmforge-ingress -n llmforge --type='merge' -p='{
    "metadata": {
        "annotations": {
            "nginx.ingress.kubernetes.io/canary-weight": "50"
        }
    }
}'

# Continue reducing until 0%
```

### Full Cleanup

```bash
# Remove entire microservices deployment
kubectl delete namespace llmforge

# Remove Docker images
docker rmi $(docker images | grep llmforge | awk '{print $3}')
```

---

## Troubleshooting

### Common Issues

#### 1. Pod Scheduling Issues
```bash
# Check node resources
kubectl describe nodes | grep -E "Name:|Allocatable:|Allocated"

# Check pod events
kubectl describe pod POD_NAME -n llmforge

# Common solutions:
# - Add more nodes
# - Adjust resource requests/limits
# - Check GPU availability
```

#### 2. Service Discovery Issues
```bash
# Check orchestrator logs
kubectl logs deployment/llmforge-orchestrator -n llmforge

# Verify service registration
kubectl port-forward service/llmforge-orchestrator 8080:8080 -n llmforge &
curl http://localhost:8080/services
```

#### 3. Storage Issues
```bash
# Check PVC status
kubectl get pvc -n llmforge

# Check storage class
kubectl describe storageclass

# Common solutions:
# - Verify storage provisioner
# - Check node storage capacity
# - Review PV/PVC configurations
```

#### 4. GPU Issues
```bash
# Check GPU availability
kubectl get nodes -l nvidia.com/gpu=true -o json | jq '.items[].status.allocatable."nvidia.com/gpu"'

# Check GPU driver
kubectl describe node NODE_NAME | grep nvidia.com/gpu

# Verify GPU operator
kubectl get pods -n gpu-operator-resources
```

### Monitoring and Alerts

```bash
# Check key metrics during migration
kubectl port-forward service/prometheus 9090:9090 -n llmforge &

# Query important metrics:
# - up{job=~".*llmforge.*"} (service availability)
# - rate(gateway_requests_total[5m]) (request rate)
# - active_training_jobs (training load)
# - gpu_utilization_percent (GPU usage)

# Set up alerts for:
curl -X POST http://localhost:9090/api/v1/alerts
```

---

## Performance Optimization

### Post-Migration Tuning

#### 1. Resource Optimization
```yaml
# Adjust resource requests/limits based on actual usage
resources:
  requests:
    cpu: "500m"      # Start conservative
    memory: "1Gi"
  limits:
    cpu: "2000m"     # Allow bursting
    memory: "4Gi"
```

#### 2. Auto-scaling Configuration
```yaml
# Enable HPA for stateless services
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

#### 3. Cache Optimization
```bash
# Tune Redis cache
kubectl exec -it redis-master-0 -n llmforge -- redis-cli CONFIG SET maxmemory-policy allkeys-lru
kubectl exec -it redis-master-0 -n llmforge -- redis-cli CONFIG SET maxmemory 4gb
```

---

## Success Criteria

The migration is considered successful when:

- ✅ All microservices are healthy and responsive
- ✅ API compatibility tests pass 100%
- ✅ Training jobs complete successfully
- ✅ Model comparison functionality works
- ✅ Monitoring and alerting are functional
- ✅ Performance metrics meet or exceed baseline
- ✅ No data loss during migration
- ✅ Resource utilization is optimized
- ✅ Auto-scaling functions correctly
- ✅ Backup and recovery procedures tested

---

## Support and Maintenance

### Daily Operations
```bash
# Health check script
#!/bin/bash
kubectl get pods -n llmforge --field-selector=status.phase!=Running
kubectl top pods -n llmforge
curl -s http://llmforge-api.com/health | jq .
```

### Log Aggregation
```bash
# View service logs
kubectl logs -f deployment/llmforge-api-gateway -n llmforge
kubectl logs -f deployment/llmforge-training -n llmforge --tail=100
```

### Backup Procedures
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M)
kubectl get all -n llmforge -o yaml > backup-$DATE.yaml
# Add database backup commands
# Add model storage backup commands
```

This migration guide provides a comprehensive approach to transitioning from monolithic to microservices architecture while maintaining system reliability and minimizing downtime.