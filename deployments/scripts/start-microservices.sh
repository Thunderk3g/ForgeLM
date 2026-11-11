#!/bin/bash

# LLMForge Microservices Startup Script
# This script provides easy commands to manage the microservices stack

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_FILE=".env"
COMPOSE_FILE="docker-compose.yml"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command_exists docker; then
        missing_tools+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_tools+=("docker-compose")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_error "Please install the missing tools and try again."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration before starting services"
        print_warning "Especially important: HUGGINGFACE_TOKEN, GPU settings, and API keys"
    fi
    
    # Create necessary directories
    mkdir -p data/{datasets,models,logs}
    mkdir -p models/{base,finetuned,factory,hf_cache}
    mkdir -p datasets
    mkdir -p logs
    
    print_success "Environment setup complete"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    print_status "Building orchestrator service..."
    docker-compose build orchestrator
    
    print_status "Building API gateway..."
    docker-compose build api-gateway
    
    print_status "Building training service..."
    docker-compose build training-service
    
    print_status "Building chat comparison service..."
    docker-compose build chat-comparison-service
    
    print_status "Building data generation service..."
    docker-compose build data-generation-service
    
    print_status "Building visualization service..."
    docker-compose build visualization-service
    
    print_status "Building model management service..."
    docker-compose build model-management-service
    
    print_status "Building factory service..."
    docker-compose build factory-service
    
    print_success "All Docker images built successfully"
}

# Start infrastructure services
start_infrastructure() {
    print_status "Starting infrastructure services..."
    
    docker-compose up -d redis postgres
    
    # Wait for services to be healthy
    print_status "Waiting for infrastructure services to be ready..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose exec redis redis-cli ping >/dev/null 2>&1 && \
           docker-compose exec postgres pg_isready -U llmforge >/dev/null 2>&1; then
            print_success "Infrastructure services are ready"
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    print_error "Infrastructure services failed to start within timeout"
    return 1
}

# Start core services
start_core_services() {
    print_status "Starting core services..."
    
    # Start orchestrator first
    print_status "Starting service orchestrator..."
    docker-compose up -d orchestrator
    
    # Wait for orchestrator to be ready
    wait_for_service "orchestrator" 8080
    
    # Start API gateway
    print_status "Starting API gateway..."
    docker-compose up -d api-gateway
    
    wait_for_service "api-gateway" 8000
    
    print_success "Core services started"
}

# Start application services
start_app_services() {
    print_status "Starting application services..."
    
    # Start all app services
    docker-compose up -d \
        training-service \
        chat-comparison-service \
        data-generation-service \
        visualization-service \
        model-management-service \
        factory-service
    
    print_status "Waiting for application services to be ready..."
    
    # Wait for each service
    wait_for_service "training-service" 8001
    wait_for_service "chat-comparison-service" 8002
    wait_for_service "data-generation-service" 8003
    wait_for_service "visualization-service" 8004
    wait_for_service "model-management-service" 8005
    wait_for_service "factory-service" 8006
    
    print_success "Application services started"
}

# Start frontend and proxy
start_frontend() {
    print_status "Starting frontend and proxy..."
    
    docker-compose up -d nginx
    
    print_success "Frontend and proxy started"
}

# Wait for a service to be healthy
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=60
    local attempt=1
    
    print_status "Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$port/health" >/dev/null 2>&1; then
            print_success "$service_name is ready"
            return 0
        fi
        
        if [ $((attempt % 10)) -eq 0 ]; then
            print_status "Still waiting for $service_name... (attempt $attempt/$max_attempts)"
        fi
        
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to start within timeout"
    return 1
}

# Show service status
show_status() {
    print_status "Service Status:"
    echo
    docker-compose ps
    echo
    
    print_status "Service Health Checks:"
    echo
    
    local services=("orchestrator:8080" "api-gateway:8000" "training-service:8001" "chat-comparison-service:8002" "data-generation-service:8003" "visualization-service:8004" "model-management-service:8005" "factory-service:8006")
    
    for service_port in "${services[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        
        if curl -f -s "http://localhost:$port/health" >/dev/null 2>&1; then
            print_success "$service (port $port): ✓ HEALTHY"
        else
            print_error "$service (port $port): ✗ UNHEALTHY"
        fi
    done
    
    echo
    print_status "Access URLs:"
    echo "  • API Gateway: http://localhost:8000"
    echo "  • Frontend: http://localhost:80"
    echo "  • Service Orchestrator: http://localhost:8080"
    echo "  • Training Service: http://localhost:8001"
    echo "  • Chat Comparison: http://localhost:8002"
    echo "  • Data Generation: http://localhost:8003"
    echo "  • Visualization: http://localhost:8004"
    echo "  • Model Management: http://localhost:8005"
    echo "  • Factory: http://localhost:8006"
}

# Stop all services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

# Stop and remove everything
cleanup() {
    print_status "Stopping and removing all services and volumes..."
    docker-compose down -v --remove-orphans
    print_status "Pruning unused Docker resources..."
    docker system prune -f
    print_success "Cleanup complete"
}

# Show logs
show_logs() {
    local service=${1:-""}
    
    if [ -n "$service" ]; then
        print_status "Showing logs for $service..."
        docker-compose logs -f "$service"
    else
        print_status "Showing logs for all services..."
        docker-compose logs -f
    fi
}

# Main script logic
case "${1:-start}" in
    "start"|"up")
        print_status "Starting LLMForge Microservices..."
        check_prerequisites
        setup_environment
        build_images
        start_infrastructure
        start_core_services
        start_app_services
        start_frontend
        print_success "🚀 LLMForge Microservices are now running!"
        show_status
        ;;
    "stop"|"down")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        $0 start
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "build")
        check_prerequisites
        build_images
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"--help"|"-h")
        echo "LLMForge Microservices Manager"
        echo
        echo "Usage: $0 [COMMAND] [OPTIONS]"
        echo
        echo "Commands:"
        echo "  start, up      Start all microservices (default)"
        echo "  stop, down     Stop all microservices"
        echo "  restart        Restart all microservices"
        echo "  status         Show service status and health"
        echo "  logs [service] Show logs (all services or specific service)"
        echo "  build          Build all Docker images"
        echo "  cleanup        Stop services and clean up volumes"
        echo "  help           Show this help message"
        echo
        echo "Examples:"
        echo "  $0 start                    # Start all services"
        echo "  $0 status                   # Check service status"
        echo "  $0 logs orchestrator        # Show orchestrator logs"
        echo "  $0 logs                     # Show all service logs"
        echo
        ;;
    *)
        print_error "Unknown command: $1"
        print_status "Use '$0 help' for usage information"
        exit 1
        ;;
esac