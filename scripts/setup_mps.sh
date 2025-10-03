#!/bin/bash
# NVIDIA MPS (Multi-Process Service) Setup Script for LLMForge
# This script sets up GPU sharing between Docker containers to prevent memory contention
# and container restarts during fine-tuning operations.

set -euo pipefail

# Configuration
MPS_BASE_DIR="/tmp/nvidia-mps"
MPS_LOG_BASE_DIR="/var/log/nvidia-mps"
SCRIPT_NAME="$(basename "$0")"
LOG_FILE="/var/log/llmforge-mps-setup.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$SCRIPT_NAME] $1" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root for MPS daemon management"
    fi
}

# Check NVIDIA driver and GPU availability
check_gpu() {
    log "Checking NVIDIA GPU availability..."
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        error_exit "nvidia-smi not found. Please install NVIDIA drivers."
    fi
    
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    
    if [[ $gpu_count -eq 0 ]]; then
        error_exit "No NVIDIA GPUs detected"
    fi
    
    log "Found $gpu_count NVIDIA GPU(s)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
}

# Check CUDA MPS support
check_mps_support() {
    log "Checking CUDA MPS support..."
    
    # Check if MPS binaries exist
    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        error_exit "nvidia-cuda-mps-control not found. MPS not supported or not installed."
    fi
    
    # Check GPU compute capability
    local compute_capability
    compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
    local major_version=${compute_capability%.*}
    
    if [[ $major_version -lt 3 ]]; then
        error_exit "MPS requires compute capability 3.5 or higher. Found: $compute_capability"
    fi
    
    log "MPS support confirmed. GPU compute capability: $compute_capability"
}

# Stop existing MPS services
stop_mps() {
    log "Stopping existing MPS services..."
    
    # Kill any existing MPS control daemons
    if pgrep -f "nvidia-cuda-mps-control" >/dev/null; then
        log "Stopping existing MPS control daemon..."
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
        sleep 2
    fi
    
    # Kill any remaining MPS processes
    pkill -f "nvidia-cuda-mps" 2>/dev/null || true
    sleep 1
    
    log "Existing MPS services stopped"
}

# Setup MPS directories and permissions
setup_mps_directories() {
    log "Setting up MPS directories..."
    
    # Create base directories
    mkdir -p "$MPS_BASE_DIR" "$MPS_LOG_BASE_DIR"
    
    # Set up per-GPU directories
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    
    for ((i=0; i<gpu_count; i++)); do
        local gpu_pipe_dir="$MPS_BASE_DIR/gpu_$i"
        local gpu_log_dir="$MPS_LOG_BASE_DIR/gpu_$i"
        
        mkdir -p "$gpu_pipe_dir" "$gpu_log_dir"
        chmod 777 "$gpu_pipe_dir" "$gpu_log_dir"
        
        log "Created directories for GPU $i: $gpu_pipe_dir, $gpu_log_dir"
    done
    
    # Set permissions for Docker access
    chmod -R 755 "$MPS_BASE_DIR" "$MPS_LOG_BASE_DIR"
    chown -R root:root "$MPS_BASE_DIR" "$MPS_LOG_BASE_DIR"
}

# Configure MPS for LLMForge services
configure_mps_services() {
    log "Configuring MPS for LLMForge services..."
    
    # GPU 0: Primary GPU for training service (highest priority)
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/gpu_0"
    export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_BASE_DIR/gpu_0"
    
    log "Starting MPS control daemon for GPU 0 (Training Service)..."
    nvidia-cuda-mps-control -d
    
    # Set active thread percentage limits for resource isolation
    # Training service gets 70% of GPU resources
    echo "set_active_thread_percentage 70" | nvidia-cuda-mps-control
    
    # Set memory allocation limits (in MB) - Training service gets priority
    echo "set_default_device_pinned_mem_limit 8192" | nvidia-cuda-mps-control
    
    log "MPS configured for GPU 0 with training service priority"
    
    # Check if multiple GPUs available for additional services
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    
    if [[ $gpu_count -gt 1 ]]; then
        # Configure secondary GPU for inference services
        export CUDA_VISIBLE_DEVICES=1
        export CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/gpu_1"
        export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_BASE_DIR/gpu_1"
        
        log "Starting MPS control daemon for GPU 1 (Inference Services)..."
        nvidia-cuda-mps-control -d
        
        # Chat comparison and model management get shared access to GPU 1
        echo "set_active_thread_percentage 50" | nvidia-cuda-mps-control
        echo "set_default_device_pinned_mem_limit 4096" | nvidia-cuda-mps-control
        
        log "MPS configured for GPU 1 with inference service sharing"
    else
        log "Single GPU detected - all services will share GPU 0 with resource limits"
        
        # Adjust limits for shared access on single GPU
        export CUDA_VISIBLE_DEVICES=0
        export CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/gpu_0"
        export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_BASE_DIR/gpu_0"
        
        # Training service: 60%, Inference services: 40% shared
        echo "set_active_thread_percentage 60" | nvidia-cuda-mps-control
        echo "set_default_device_pinned_mem_limit 6144" | nvidia-cuda-mps-control
    fi
}

# Create environment file for Docker containers
create_env_file() {
    local env_file="/tmp/llmforge-mps.env"
    
    log "Creating MPS environment file: $env_file"
    
    cat > "$env_file" << EOF
# LLMForge MPS Configuration
# Generated on $(date)

# MPS Base Configuration
CUDA_MPS_PIPE_DIRECTORY=$MPS_BASE_DIR/gpu_0
CUDA_MPS_LOG_DIRECTORY=$MPS_LOG_BASE_DIR/gpu_0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Unsloth Optimizations
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
CUDA_LAUNCH_BLOCKING=0
TOKENIZERS_PARALLELISM=false

# MPS Memory Management
MPS_ACTIVE_THREAD_PERCENTAGE=70
MPS_PINNED_DEVICE_MEM_LIMIT=8192

# Service-specific GPU assignments
TRAINING_SERVICE_GPU=0
INFERENCE_SERVICES_GPU=0
EOF

    chmod 644 "$env_file"
    log "MPS environment file created: $env_file"
}

# Create GPU monitoring script
create_monitoring_script() {
    local monitor_script="/usr/local/bin/llmforge-gpu-monitor"
    
    log "Creating GPU monitoring script: $monitor_script"
    
    cat > "$monitor_script" << 'EOF'
#!/bin/bash
# LLMForge GPU Monitoring Script for MPS

while true; do
    echo "=== LLMForge GPU Status $(date) ==="
    
    # GPU utilization and memory
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    
    echo ""
    echo "=== MPS Status ==="
    
    # MPS process status
    if pgrep -f "nvidia-cuda-mps-control" >/dev/null; then
        echo "MPS Control Daemon: RUNNING"
        
        # MPS server processes
        mps_servers=$(pgrep -f "nvidia-cuda-mps-server" | wc -l)
        echo "MPS Servers: $mps_servers"
        
        # Active MPS clients
        if [[ -d /tmp/nvidia-mps ]]; then
            for pipe_dir in /tmp/nvidia-mps/gpu_*; do
                if [[ -d "$pipe_dir" ]]; then
                    gpu_id=$(basename "$pipe_dir" | cut -d'_' -f2)
                    clients=$(ls "$pipe_dir"/*.pipe 2>/dev/null | wc -l)
                    echo "GPU $gpu_id MPS Clients: $clients"
                fi
            done
        fi
    else
        echo "MPS Control Daemon: NOT RUNNING"
    fi
    
    echo ""
    echo "=== Docker Container GPU Usage ==="
    
    # Docker containers using GPU
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(training|chat-comparison|model-management)" || echo "No LLMForge GPU containers running"
    
    echo ""
    echo "========================================"
    echo ""
    
    sleep 30
done
EOF

    chmod +x "$monitor_script"
    log "GPU monitoring script created: $monitor_script"
}

# Create MPS cleanup script
create_cleanup_script() {
    local cleanup_script="/usr/local/bin/llmforge-mps-cleanup"
    
    log "Creating MPS cleanup script: $cleanup_script"
    
    cat > "$cleanup_script" << 'EOF'
#!/bin/bash
# LLMForge MPS Cleanup Script

echo "Stopping LLMForge MPS services..."

# Stop MPS control daemon
if pgrep -f "nvidia-cuda-mps-control" >/dev/null; then
    echo "Stopping MPS control daemon..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 2
fi

# Kill any remaining MPS processes
pkill -f "nvidia-cuda-mps" 2>/dev/null || true

# Clean up MPS directories
if [[ -d /tmp/nvidia-mps ]]; then
    echo "Cleaning up MPS directories..."
    rm -rf /tmp/nvidia-mps/*
fi

echo "MPS cleanup completed"
EOF

    chmod +x "$cleanup_script"
    log "MPS cleanup script created: $cleanup_script"
}

# Verify MPS setup
verify_mps() {
    log "Verifying MPS setup..."
    
    sleep 3
    
    # Check MPS control daemon
    if ! pgrep -f "nvidia-cuda-mps-control" >/dev/null; then
        error_exit "MPS control daemon is not running"
    fi
    
    # Check MPS directories
    if [[ ! -d "$MPS_BASE_DIR/gpu_0" ]]; then
        error_exit "MPS pipe directory not found"
    fi
    
    # Test MPS functionality with a simple CUDA program
    echo "Testing MPS functionality..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        local test_output
        test_output=$(CUDA_MPS_PIPE_DIRECTORY="$MPS_BASE_DIR/gpu_0" timeout 10s nvidia-smi -L 2>&1 || echo "Test failed")
        
        if [[ "$test_output" == *"GPU 0"* ]]; then
            log "MPS test successful"
        else
            log "WARNING: MPS test may have issues, but basic setup is complete"
        fi
    fi
    
    log "MPS setup verification completed"
}

# Main execution
main() {
    log "Starting LLMForge MPS setup..."
    
    check_root
    check_gpu
    check_mps_support
    stop_mps
    setup_mps_directories
    configure_mps_services
    create_env_file
    create_monitoring_script
    create_cleanup_script
    verify_mps
    
    log "LLMForge MPS setup completed successfully!"
    log "Next steps:"
    log "1. Update docker-compose.yml with MPS configuration"
    log "2. Restart Docker containers with: docker-compose down && docker-compose up -d"
    log "3. Monitor GPU usage with: /usr/local/bin/llmforge-gpu-monitor"
    log "4. Clean up MPS if needed with: /usr/local/bin/llmforge-mps-cleanup"
}

# Run main function
main "$@"