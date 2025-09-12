# LLMForge Docker Guide

## Overview

LLMForge uses a microservices architecture with the following services:

- **API Gateway** (Port 8000) - Main entry point, handles routing
- **Orchestrator** (Port 8080) - Service coordination and management  
- **Training Service** (Port 8001) - GPU-based model fine-tuning
- **Chat Comparison** (Port 8002) - Model inference and comparison
- **Data Generation** (Port 8003) - Dataset creation and validation
- **Visualization** (Port 8004) - Analytics and monitoring dashboards
- **Model Management** (Port 8005) - Model lifecycle and caching

## Quick Start

### 1. Build All Services

```bash
# Build all services (this will take 10-20 minutes)
docker-compose build

# Or build with no cache to ensure fresh builds
docker-compose build --no-cache

# Build specific services only
docker-compose build api-gateway orchestrator
```

### 2. Start Services

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Start specific services
docker-compose up api-gateway orchestrator

# View logs
docker-compose logs -f [service-name]
```

### 3. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Force stop and clean up
docker-compose down --remove-orphans
```

## Service Dependencies

```
API Gateway (8000) 
    ↓
Orchestrator (8080)
    ↓
┌─ Training Service (8001) [Requires GPU]
├─ Chat Comparison (8002) [Requires GPU] 
├─ Data Generation (8003)
├─ Visualization (8004)
└─ Model Management (8005) [Requires GPU]
```

## GPU Requirements

Services requiring GPU:
- training-service
- chat-comparison-service  
- model-management-service

These require NVIDIA Docker runtime and CUDA-compatible GPU.

## Environment Variables

Create a `.env` file in the root directory:

```env
# Optional environment variables
LOG_LEVEL=INFO
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
CUDA_VISIBLE_DEVICES=0
MAX_CACHED_MODELS=5
CACHE_SIZE_GB=20
```

## Development

### Without GPU (CPU-only services)

```bash
# Start only CPU services
docker-compose up api-gateway orchestrator data-generation-service visualization-service
```

### With GPU

```bash
# Start all services including GPU ones
docker-compose up
```

### Frontend Development

The frontend is a React app. To run it:

```bash
cd frontend
npm install
npm run dev
```

Then access at http://localhost:3000

## Troubleshooting

### 1. CUDA/GPU Issues

If you don't have NVIDIA GPU or CUDA, modify docker-compose.yml to remove GPU services:

```bash
# Comment out or remove these services:
# - training-service
# - chat-comparison-service  
# - model-management-service
```

### 2. Port Conflicts

If ports are in use, modify the ports in docker-compose.yml:

```yaml
ports:
  - "8001:8000"  # Change first number to available port
```

### 3. Build Failures

```bash
# Clean Docker system
docker system prune -a

# Rebuild from scratch
docker-compose build --no-cache --pull
```

### 4. Memory Issues

Increase Docker memory limits in Docker Desktop settings or use:

```bash
# Build services one by one to reduce memory usage
docker-compose build orchestrator
docker-compose build api-gateway
# etc...
```

## API Endpoints

Once running, API documentation available at:
- Main API: http://localhost:8000/docs
- Orchestrator: http://localhost:8080/docs
- Training: http://localhost:8001/docs (if GPU available)

## Health Checks

```bash
# Check service health
curl http://localhost:8000/health
curl http://localhost:8080/health

# View service status
docker-compose ps
```