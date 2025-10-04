#!/bin/bash
# Quick start script for development environment
# Optimized for RTX 3050 4GB Laptop with 32GB RAM

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== LLMForge Development Startup ===${NC}"
echo -e "${GREEN}System: RTX 3050 4GB, i5 11th Gen, 32GB RAM${NC}"
echo ""

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

# Stop any existing containers
echo -e "${BLUE}Stopping existing containers...${NC}"
docker compose down 2>/dev/null || true

# Build with dev config
echo -e "${BLUE}Building training service with dev configuration...${NC}"
docker compose -f docker-compose.yml -f docker-compose.dev.yml build training-service

# Start all services with dev config
echo -e "${BLUE}Starting all services with dev configuration...${NC}"
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Wait for services to start
echo -e "${BLUE}Waiting for services to be ready...${NC}"
sleep 10

# Check status
echo ""
echo -e "${GREEN}=== Service Status ===${NC}"
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps

# Show memory limits
echo ""
echo -e "${GREEN}=== Training Service Config ===${NC}"
docker inspect llmforge-training --format '{{.HostConfig.Memory}}' | awk '{print "Memory Limit: " $1/1024/1024/1024 "GB"}'
docker exec llmforge-training env | grep -E "CONTAINER_MEMORY_LIMIT_GB|WORKER_MEMORY_GB|GPU_MEMORY_LIMIT_MB" || true

# Check for OOM kills
echo ""
echo -e "${GREEN}=== Container Health ===${NC}"
OOMKILLED=$(docker inspect llmforge-training --format '{{.State.OOMKilled}}')
RESTARTS=$(docker inspect llmforge-training --format '{{.RestartCount}}')
echo -e "OOM Killed: ${OOMKILLED}"
echo -e "Restart Count: ${RESTARTS}"

# Show logs
echo ""
echo -e "${GREEN}=== Recent Logs ===${NC}"
docker logs llmforge-training --tail 20

echo ""
echo -e "${GREEN}=== Services Started! ===${NC}"
echo -e "Training service: http://localhost:8001"
echo -e "API Gateway: http://localhost:8000"
echo ""
echo -e "${BLUE}To monitor training:${NC}"
echo -e "  ./monitor_training_container.sh"
echo ""
echo -e "${BLUE}To view logs:${NC}"
echo -e "  docker logs -f llmforge-training"
echo ""
echo -e "${BLUE}To stop:${NC}"
echo -e "  docker compose -f docker-compose.yml -f docker-compose.dev.yml down"
