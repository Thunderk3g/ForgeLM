#!/bin/bash
# LLMForge MPS Deployment Script
# Complete deployment script for MPS-enabled LLMForge platform

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/var/log/llmforge-mps-deploy.log"
DEPLOYMENT_MODE="${1:-production}"  # production, development, test

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEPLOY] $1" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root for MPS and Docker management"
    fi
}

# Pre-deployment validation
validate_environment() {
    log "Validating deployment environment..."
    
    # Check required files
    required_files=(
        "$SCRIPT_DIR/setup_mps.sh"
        "$SCRIPT_DIR/gpu_memory_manager.py"
        "$SCRIPT_DIR/validate_mps_setup.py"
        "$PROJECT_ROOT/docker-compose.yml"
        "$PROJECT_ROOT/.env.mps"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error_exit "Required file not found: $file"
        fi
    done
    
    # Check system requirements
    if ! command -v docker >/dev/null 2>&1; then
        error_exit "Docker not installed"
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        error_exit "Docker Compose not installed"
    fi
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        error_exit "NVIDIA drivers not installed"
    fi
    
    log "Environment validation completed"
}

# Backup existing configuration
backup_configuration() {
    log "Creating configuration backup..."
    
    local backup_dir="/var/backups/llmforge/$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$backup_dir"
    
    # Backup existing docker-compose.yml if it exists
    if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        cp "$PROJECT_ROOT/docker-compose.yml" "$backup_dir/docker-compose.yml.backup"
    fi
    
    # Backup existing .env file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        cp "$PROJECT_ROOT/.env" "$backup_dir/.env.backup"
    fi
    
    # Backup any existing MPS configuration
    if [[ -d "/tmp/nvidia-mps" ]]; then
        cp -r "/tmp/nvidia-mps" "$backup_dir/nvidia-mps.backup" 2>/dev/null || true
    fi
    
    log "Configuration backed up to: $backup_dir"
}

# Setup MPS
setup_mps() {
    log "Setting up NVIDIA MPS..."
    
    # Make setup script executable
    chmod +x "$SCRIPT_DIR/setup_mps.sh"
    
    # Run MPS setup
    "$SCRIPT_DIR/setup_mps.sh"
    
    if [[ $? -eq 0 ]]; then
        log "MPS setup completed successfully"
    else
        error_exit "MPS setup failed"
    fi
}

# Configure environment
configure_environment() {
    log "Configuring environment for $DEPLOYMENT_MODE mode..."
    
    # Copy MPS environment configuration
    cp "$PROJECT_ROOT/.env.mps" "$PROJECT_ROOT/.env.mps.active"
    
    # Merge with existing .env if it exists
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        log "Merging with existing .env file..."
        
        # Create backup of existing .env
        cp "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.original"
        
        # Merge configurations (MPS config takes precedence)
        cat "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.env.mps" > "$PROJECT_ROOT/.env.merged"
        
        # Remove duplicates (keeping last occurrence)
        awk -F= '!seen[$1]++' "$PROJECT_ROOT/.env.merged" | tac | awk -F= '!seen[$1]++' | tac > "$PROJECT_ROOT/.env"
        
        # Clean up temporary files
        rm "$PROJECT_ROOT/.env.merged"
    else
        # Use MPS configuration as base
        cp "$PROJECT_ROOT/.env.mps" "$PROJECT_ROOT/.env"
    fi
    
    # Set deployment-specific configurations
    case "$DEPLOYMENT_MODE" in
        "production")
            echo "LOG_LEVEL=INFO" >> "$PROJECT_ROOT/.env"
            echo "DEBUG_MPS=0" >> "$PROJECT_ROOT/.env"
            echo "PROFILE_GPU_MEMORY=0" >> "$PROJECT_ROOT/.env"
            ;;
        "development")
            echo "LOG_LEVEL=DEBUG" >> "$PROJECT_ROOT/.env"
            echo "DEBUG_MPS=1" >> "$PROJECT_ROOT/.env"
            echo "PROFILE_GPU_MEMORY=1" >> "$PROJECT_ROOT/.env"
            ;;
        "test")
            echo "LOG_LEVEL=DEBUG" >> "$PROJECT_ROOT/.env"
            echo "DEBUG_MPS=1" >> "$PROJECT_ROOT/.env"
            echo "PROFILE_GPU_MEMORY=1" >> "$PROJECT_ROOT/.env"
            echo "MAX_TRAINING_JOBS=1" >> "$PROJECT_ROOT/.env"
            ;;
    esac
    
    log "Environment configured for $DEPLOYMENT_MODE mode"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up GPU monitoring..."
    
    # Make monitoring scripts executable
    chmod +x "$SCRIPT_DIR/gpu_memory_manager.py"
    chmod +x "/usr/local/bin/llmforge-gpu-monitor" 2>/dev/null || true
    
    # Install monitoring script as systemd service for production
    if [[ "$DEPLOYMENT_MODE" == "production" ]]; then
        log "Installing GPU monitor as systemd service..."
        
        cat > /etc/systemd/system/llmforge-gpu-monitor.service << EOF
[Unit]
Description=LLMForge GPU Memory Monitor
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=root
ExecStart=/usr/bin/python3 $SCRIPT_DIR/gpu_memory_manager.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=$SCRIPT_DIR

[Install]
WantedBy=multi-user.target
EOF

        systemctl daemon-reload
        systemctl enable llmforge-gpu-monitor.service
        
        log "GPU monitor service installed"
    fi
}

# Deploy containers
deploy_containers() {
    log "Deploying LLMForge containers with MPS support..."
    
    cd "$PROJECT_ROOT"
    
    # Stop existing containers
    log "Stopping existing containers..."
    docker-compose down --remove-orphans || true
    
    # Clean up any orphaned volumes (optional)
    if [[ "$DEPLOYMENT_MODE" != "production" ]]; then
        docker volume prune -f || true
    fi
    
    # Build and start containers
    log "Building and starting containers..."
    
    # Set build arguments based on deployment mode
    export BUILDKIT_INLINE_CACHE=1
    
    case "$DEPLOYMENT_MODE" in
        "production")
            docker-compose -f docker-compose.yml up -d --build
            ;;
        "development")
            docker-compose -f docker-compose.yml up -d --build
            ;;
        "test")
            # For testing, don't detach so we can see logs
            docker-compose -f docker-compose.yml up --build &
            DOCKER_PID=$!
            sleep 30  # Give containers time to start
            ;;
    esac
    
    log "Container deployment completed"
}

# Validate deployment
validate_deployment() {
    log "Validating MPS deployment..."
    
    # Wait for containers to start
    sleep 30
    
    # Check container status
    log "Checking container status..."
    docker-compose ps
    
    # Run validation script
    if [[ -f "$SCRIPT_DIR/validate_mps_setup.py" ]]; then
        log "Running MPS validation..."
        
        # Make validation script executable
        chmod +x "$SCRIPT_DIR/validate_mps_setup.py"
        
        # Run validation (this might be from within a container)
        if python3 "$SCRIPT_DIR/validate_mps_setup.py" --full --report "/tmp/mps_validation_report.md"; then
            log "MPS validation PASSED"
            
            # Copy report to project directory if it exists
            if [[ -f "/tmp/mps_validation_report.md" ]]; then
                cp "/tmp/mps_validation_report.md" "$PROJECT_ROOT/mps_validation_report.md"
                log "Validation report saved to: $PROJECT_ROOT/mps_validation_report.md"
            fi
        else
            log "WARNING: MPS validation FAILED - check logs for details"
        fi
    fi
    
    # Check service health
    log "Checking service health..."
    
    local services=("training-service:8001" "chat-comparison-service:8002" "model-management-service:8005")
    local failed_services=()
    
    for service in "${services[@]}"; do
        local service_name="${service%%:*}"
        local port="${service##*:}"
        
        if curl -f "http://localhost:$port/health" >/dev/null 2>&1; then
            log "✅ $service_name health check PASSED"
        else
            log "❌ $service_name health check FAILED"
            failed_services+=("$service_name")
        fi
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log "All service health checks PASSED"
        return 0
    else
        log "Some services failed health checks: ${failed_services[*]}"
        return 1
    fi
}

# Start monitoring
start_monitoring() {
    log "Starting monitoring services..."
    
    if [[ "$DEPLOYMENT_MODE" == "production" ]]; then
        # Start systemd service
        systemctl start llmforge-gpu-monitor.service
        log "Started GPU monitor systemd service"
    else
        # Start monitoring in background for development/test
        nohup python3 "$SCRIPT_DIR/gpu_memory_manager.py" > "/var/log/llmforge-gpu-monitor.log" 2>&1 &
        echo $! > "/var/run/llmforge-gpu-monitor.pid"
        log "Started GPU monitor in background (PID: $(cat /var/run/llmforge-gpu-monitor.pid))"
    fi
    
    # Start basic GPU monitoring script if it exists
    if [[ -x "/usr/local/bin/llmforge-gpu-monitor" ]]; then
        nohup /usr/local/bin/llmforge-gpu-monitor > "/var/log/llmforge-basic-monitor.log" 2>&1 &
        echo $! > "/var/run/llmforge-basic-monitor.pid"
        log "Started basic GPU monitor"
    fi
}

# Print deployment summary
print_summary() {
    log "=== LLMForge MPS Deployment Summary ==="
    log "Deployment Mode: $DEPLOYMENT_MODE"
    log "Project Root: $PROJECT_ROOT"
    log "Log File: $LOG_FILE"
    
    # GPU status
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "GPU Status:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | while read line; do
            log "  $line"
        done
    fi
    
    # MPS status
    if pgrep -f "nvidia-cuda-mps-control" >/dev/null; then
        log "MPS Status: RUNNING"
    else
        log "MPS Status: NOT RUNNING"
    fi
    
    # Container status
    log "Container Status:"
    docker-compose ps --format "table {{.Name}}\t{{.State}}\t{{.Ports}}" | tail -n +2 | while read line; do
        log "  $line"
    done
    
    # Service URLs
    log "Service URLs:"
    log "  API Gateway: http://localhost:8000"
    log "  Training Service: http://localhost:8001"
    log "  Chat Comparison: http://localhost:8002"
    log "  Data Generation: http://localhost:8003"
    log "  Visualization: http://localhost:8004"
    log "  Model Management: http://localhost:8005"
    log "  Orchestrator: http://localhost:8080"
    
    # Next steps
    log "Next Steps:"
    log "  1. Monitor GPU usage: /usr/local/bin/llmforge-gpu-monitor"
    log "  2. Check service logs: docker-compose logs -f [service-name]"
    log "  3. Run training jobs through API Gateway"
    log "  4. Monitor with: tail -f $LOG_FILE"
    
    if [[ -f "$PROJECT_ROOT/mps_validation_report.md" ]]; then
        log "  5. Review validation report: $PROJECT_ROOT/mps_validation_report.md"
    fi
    
    log "Deployment completed successfully!"
}

# Cleanup function for interrupted deployments
cleanup() {
    log "Cleaning up interrupted deployment..."
    
    # Stop containers if they're running
    cd "$PROJECT_ROOT" 2>/dev/null && docker-compose down --remove-orphans 2>/dev/null || true
    
    # Stop monitoring if it was started
    if [[ -f "/var/run/llmforge-gpu-monitor.pid" ]]; then
        kill "$(cat /var/run/llmforge-gpu-monitor.pid)" 2>/dev/null || true
        rm -f "/var/run/llmforge-gpu-monitor.pid"
    fi
    
    if [[ -f "/var/run/llmforge-basic-monitor.pid" ]]; then
        kill "$(cat /var/run/llmforge-basic-monitor.pid)" 2>/dev/null || true
        rm -f "/var/run/llmforge-basic-monitor.pid"
    fi
    
    log "Cleanup completed"
}

# Set up signal handlers
trap cleanup EXIT
trap 'error_exit "Deployment interrupted by user"' INT TERM

# Main deployment workflow
main() {
    log "Starting LLMForge MPS deployment in $DEPLOYMENT_MODE mode..."
    
    check_root
    validate_environment
    backup_configuration
    setup_mps
    configure_environment
    setup_monitoring
    deploy_containers
    
    # Validation and monitoring startup
    if validate_deployment; then
        start_monitoring
        print_summary
        
        # For test mode, kill the background docker process after summary
        if [[ "$DEPLOYMENT_MODE" == "test" && -n "${DOCKER_PID:-}" ]]; then
            log "Test mode completed, stopping containers..."
            kill "$DOCKER_PID" 2>/dev/null || true
            cd "$PROJECT_ROOT" && docker-compose down 2>/dev/null || true
        fi
    else
        log "Deployment validation failed, but containers are running"
        log "Check logs and validation report for issues"
        exit 1
    fi
}

# Usage information
usage() {
    echo "Usage: $0 [MODE]"
    echo "  MODE: production (default), development, test"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy in production mode"
    echo "  $0 development        # Deploy in development mode"
    echo "  $0 test              # Deploy and test, then stop"
    echo ""
    echo "Requirements:"
    echo "  - Run as root"
    echo "  - NVIDIA drivers installed"
    echo "  - Docker and Docker Compose installed"
    echo "  - All LLMForge files in place"
}

# Parse command line arguments
case "${1:-}" in
    "production"|"development"|"test"|"")
        main
        ;;
    "-h"|"--help"|"help")
        usage
        ;;
    *)
        echo "Invalid deployment mode: $1"
        usage
        exit 1
        ;;
esac