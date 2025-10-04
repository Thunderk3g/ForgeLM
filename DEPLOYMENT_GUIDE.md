# LLMForge Deployment Guide

## Overview

This guide covers deploying LLMForge across different environments:
- **Development:** Laptop with limited GPU (RTX 3050 4GB)
- **Staging:** EC2 with moderate GPU (g4dn instances)
- **Production:** EKS with production GPUs (p3/p4 instances)

## Memory Limits: Development vs Production

### Why Limits Matter Differently

| Environment | Use Limits? | Why? |
|-------------|------------|------|
| **Development (Laptop)** | ✅ **YES** | Prevents system freeze, protects OS |
| **Production (EC2/EKS)** | ⚠️ **OPTIONAL** | Depends on multi-tenancy needs |
| **Production (Kubernetes)** | ✅ **YES** | Required for pod scheduling & QoS |

### Current Configuration Strategy

We use **three configuration layers**:

1. **Base:** `docker-compose.yml` - Common settings
2. **Dev Override:** `docker-compose.dev.yml` - Strict limits for laptop
3. **Prod Override:** `docker-compose.prod.yml` - Flexible limits for cloud

## Development Deployment (Current - RTX 3050 4GB)

### Quick Start
```bash
# Use dev configuration (has limits)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Monitor container
./monitor_training_container.sh
```

### Configuration
```yaml
# docker-compose.dev.yml
training-service:
  environment:
    - CONTAINER_MEMORY_LIMIT_GB=6   # Prevents laptop freeze
    - WORKER_MEMORY_GB=4
    - GPU_MEMORY_LIMIT_MB=3072      # 75% of 4GB

  deploy:
    resources:
      limits:
        memory: 6G  # CRITICAL: Prevents OOM kills
```

### Why Strict Limits in Dev?
Without limits on Docker Desktop/WSL2:
- ❌ Docker uses ALL system RAM
- ❌ Laptop freezes completely
- ❌ OS kills WSL2/Docker daemon
- ❌ Lose ALL work, not just training

**Limits protect your development machine!**

## Production Deployment Options

### Option 1: EC2 with Docker Compose (No Kubernetes)

For simple EC2 deployment without orchestration:

```bash
# Deploy with production config (no hard limits)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Set environment variables
export CONTAINER_MEMORY_GB=32    # EC2 instance RAM
export WORKER_MEMORY_GB=24
export GPU_MEMORY_MB=15360       # 15GB for V100/A10G
```

**Production docker-compose.prod.yml:**
```yaml
training-service:
  environment:
    - CONTAINER_MEMORY_LIMIT_GB=${CONTAINER_MEMORY_GB:-32}
    - WORKER_MEMORY_GB=${WORKER_MEMORY_GB:-24}

  deploy:
    resources:
      # NO hard limits - use available resources
      reservations:
        memory: 16G  # Just reserve minimum
```

**Pros:**
- Simple deployment
- Maximum resource usage
- Easier debugging

**Cons:**
- No automatic scaling
- Manual failover
- Single point of failure

### Option 2: EKS with Kubernetes (Recommended)

For production-grade deployment:

#### Step 1: Create EKS Cluster with GPU Nodes

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster with GPU nodegroup
eksctl create cluster \
  --name llmforge-prod \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type g4dn.xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

#### Step 2: Deploy LLMForge Services

```bash
# Create namespace
kubectl create namespace llmforge

# Create secrets
kubectl create secret generic llmforge-secrets \
  -n llmforge \
  --from-literal=huggingface-token=$HUGGINGFACE_TOKEN

# Deploy training service
kubectl apply -f deployments/kubernetes/training-service-deployment.yaml

# Check status
kubectl get pods -n llmforge -w
```

#### Step 3: Configure Resource Limits

**Kubernetes uses limits for:**
- Pod scheduling (which node has capacity?)
- QoS classes (Guaranteed, Burstable, BestEffort)
- Eviction decisions (which pod to kill under pressure?)

```yaml
# training-service-deployment.yaml
resources:
  limits:
    memory: "32Gi"      # g4dn.xlarge has 32GB
    nvidia.com/gpu: 1
  requests:
    memory: "24Gi"      # Reserve 24GB
    nvidia.com/gpu: 1
```

**Dynamic Memory Config:**
```yaml
# Container reads its own limit!
env:
- name: CONTAINER_MEMORY_LIMIT_GB
  valueFrom:
    resourceFieldRef:
      resource: limits.memory
      divisor: "1Gi"  # Auto-converts to GB
```

**QoS Classes:**
- `requests == limits` → **Guaranteed** (never evicted)
- `requests < limits` → **Burstable** (may be evicted)
- No limits → **BestEffort** (first to be evicted)

#### Step 4: Scaling Strategy

```yaml
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llmforge-training-hpa
spec:
  minReplicas: 1
  maxReplicas: 3
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        averageUtilization: 80
```

## GPU Instance Recommendations

### Development (Your Current Setup)
- **Instance:** Local laptop
- **GPU:** RTX 3050 4GB
- **RAM:** 12-16GB
- **Limits:** STRICT (6GB container)

### Staging
- **Instance:** g4dn.xlarge
- **GPU:** T4 16GB
- **RAM:** 32GB
- **Limits:** 24-28GB (reserve for system)

### Production - Light
- **Instance:** g4dn.2xlarge
- **GPU:** T4 16GB
- **RAM:** 64GB
- **Limits:** 48-56GB

### Production - Heavy
- **Instance:** p3.2xlarge
- **GPU:** V100 16GB
- **RAM:** 61GB
- **Limits:** 48GB (reserve for system)

### Production - Extreme
- **Instance:** p4d.24xlarge
- **GPU:** A100 40GB (x8)
- **RAM:** 1152GB
- **Limits:** Per-pod: 100-200GB

## Memory Limit Decision Matrix

| Scenario | Set Limits? | Recommendation |
|----------|------------|----------------|
| Local laptop dev | ✅ YES | 50-70% of total RAM |
| Single EC2, one service | ⚠️ OPTIONAL | Maybe 80-90% to reserve for system |
| Single EC2, multiple services | ✅ YES | Divide resources fairly |
| Kubernetes cluster | ✅ YES | Required for scheduling |
| Serverless (Fargate) | ✅ YES | Required by platform |

## Deployment Commands by Environment

### Development (Laptop)
```bash
# With limits for safety
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Monitor
./monitor_training_container.sh
```

### Staging (EC2)
```bash
# SSH to EC2
ssh -i key.pem ec2-user@staging-instance

# Deploy with higher limits
export CONTAINER_MEMORY_GB=24
export GPU_MEMORY_MB=12288
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Production (EKS)
```bash
# Deploy to Kubernetes
kubectl apply -f deployments/kubernetes/

# Scale up
kubectl scale deployment llmforge-training-service -n llmforge --replicas=3

# Monitor
kubectl top pods -n llmforge
kubectl logs -f deployment/llmforge-training-service -n llmforge
```

## Monitoring in Production

### CloudWatch (AWS)
```bash
# Enable container insights
aws ecs update-cluster-settings \
  --cluster llmforge-prod \
  --settings name=containerInsights,value=enabled

# View metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name MemoryUtilization \
  --dimensions Name=ClusterName,Value=llmforge-prod
```

### Prometheus + Grafana (Kubernetes)
```bash
# Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

### Custom Monitoring Script (Any Environment)
```bash
# Use the monitor script
./monitor_training_container.sh

# Or for Kubernetes
kubectl exec -it deployment/llmforge-training-service -n llmforge -- \
  bash -c 'cat /proc/self/status | grep VmRSS'
```

## Troubleshooting by Environment

### Development: Container Restarts
```bash
# Check OOM kill
docker inspect llmforge-training | grep OOMKilled

# Increase limits if safe
# Edit docker-compose.dev.yml
limits:
  memory: 8G  # If you have 16GB+ RAM

# Rebuild
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --force-recreate
```

### EC2: High Memory Usage
```bash
# Check instance memory
free -h

# Check Docker stats
docker stats

# If memory available, remove/increase limits
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### EKS: Pod Evictions
```bash
# Check events
kubectl get events -n llmforge --sort-by='.lastTimestamp'

# Check OOM kills
kubectl describe pod <pod-name> -n llmforge | grep -i oom

# Adjust limits in deployment
kubectl edit deployment llmforge-training-service -n llmforge
```

## Migration Path

### Phase 1: Development (Current)
- ✅ Use `docker-compose.dev.yml`
- ✅ Strict limits for laptop
- ✅ Test with small models

### Phase 2: Staging (EC2)
```bash
# Deploy to staging EC2
scp -r . ec2-user@staging:/opt/llmforge
ssh ec2-user@staging "cd /opt/llmforge && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
```

### Phase 3: Production (EKS)
```bash
# Build and push images
docker build -t your-registry/llmforge-training:v1.0 -f src/docker/services/training_service.dockerfile .
docker push your-registry/llmforge-training:v1.0

# Update deployment
kubectl set image deployment/llmforge-training-service \
  training-service=your-registry/llmforge-training:v1.0 \
  -n llmforge
```

## Summary

### For Your Current Development Setup (RTX 3050 4GB):
```bash
# KEEP using strict limits
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### For Future Production Deployment:
```bash
# Option A: EC2 - Use prod config with dynamic limits
export CONTAINER_MEMORY_GB=32
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Option B: EKS - Use Kubernetes deployment with resource specs
kubectl apply -f deployments/kubernetes/training-service-deployment.yaml
```

**Key Takeaway:**
- **Dev:** Limits protect your laptop ✅
- **Prod:** Limits help Kubernetes schedule pods ✅
- **Prod EC2 (single service):** Limits optional, use env vars for dynamic config ⚠️

The code now **automatically adapts** to whatever limits you set via environment variables!
