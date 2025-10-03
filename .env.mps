# LLMForge MPS (Multi-Process Service) Environment Configuration
# This file contains environment variables for GPU sharing across Docker containers
# Generated for production MPS deployment

# =============================================================================
# MPS CORE CONFIGURATION
# =============================================================================

# MPS Pipe and Log Directories (shared between containers)
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps/gpu_0
CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps/gpu_0

# NVIDIA Driver Configuration
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# =============================================================================
# UNSLOTH OPTIMIZATIONS
# =============================================================================

# PyTorch CUDA Memory Allocator Configuration
# Optimized for shared GPU environment with memory fragmentation prevention
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True,backend:cudaMallocAsync

# CUDA Runtime Configuration
CUDA_LAUNCH_BLOCKING=0
TOKENIZERS_PARALLELISM=false

# Torch CUDA Architecture List (support for modern GPUs)
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# =============================================================================
# SERVICE-SPECIFIC MPS RESOURCE ALLOCATION
# =============================================================================

# Training Service (Primary GPU User) - 70% GPU resources
TRAINING_MPS_ACTIVE_THREAD_PERCENTAGE=70
TRAINING_MPS_PINNED_DEVICE_MEM_LIMIT=8192
TRAINING_SERVICE_GPU=0

# Chat Comparison Service (Inference) - 15% GPU resources  
CHAT_COMPARISON_MPS_ACTIVE_THREAD_PERCENTAGE=15
CHAT_COMPARISON_MPS_PINNED_DEVICE_MEM_LIMIT=2048
CHAT_COMPARISON_SERVICE_GPU=0

# Model Management Service (Model Loading/Inference) - 15% GPU resources
MODEL_MANAGEMENT_MPS_ACTIVE_THREAD_PERCENTAGE=15
MODEL_MANAGEMENT_MPS_PINNED_DEVICE_MEM_LIMIT=2048
MODEL_MANAGEMENT_SERVICE_GPU=0

# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

# Container Memory Limits (to prevent OOM and ensure stability)
TRAINING_CONTAINER_MEMORY_LIMIT_GB=14
TRAINING_WORKER_MEMORY_GB=12

CHAT_COMPARISON_CONTAINER_MEMORY_LIMIT_GB=6
CHAT_COMPARISON_WORKER_MEMORY_GB=4

MODEL_MANAGEMENT_CONTAINER_MEMORY_LIMIT_GB=8
MODEL_MANAGEMENT_WORKER_MEMORY_GB=6

# Triton Cache Configuration (shared compiler cache)
TRITON_CACHE_DIR=/tmp/.triton_cache

# OpenMP Thread Configuration (optimized for shared environment)
TRAINING_OMP_NUM_THREADS=4
INFERENCE_OMP_NUM_THREADS=2

# =============================================================================
# GPU ORCHESTRATION & COORDINATION
# =============================================================================

# GPU Sharing Mode
GPU_SHARING_MODE=mps
GPU_ORCHESTRATION_ENABLED=true

# Redis Configuration for Distributed Locking
REDIS_HOST=redis
REDIS_PORT=6379

# Service Discovery and Coordination
ORCHESTRATOR_URL=http://orchestrator:8080

# =============================================================================
# UNSLOTH-SPECIFIC CONFIGURATION
# =============================================================================

# Model Configuration
UNSLOTH_MAX_SEQ_LENGTH=4096
UNSLOTH_LOAD_IN_4BIT=true
UNSLOTH_USE_GRADIENT_CHECKPOINTING=unsloth

# Training Configuration
UNSLOTH_LORA_RANK=32
UNSLOTH_LORA_ALPHA=32
UNSLOTH_LORA_DROPOUT=0.0
UNSLOTH_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# Batch Size Configuration (optimized for MPS memory sharing)
TRAINING_BATCH_SIZE=2
INFERENCE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# CUDA Performance Settings
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_CACHE_DISABLE=0

# Python Optimizations
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1

# Disable unnecessary parallelism in shared environment
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1

# =============================================================================
# MONITORING & LOGGING
# =============================================================================

# Log Levels
LOG_LEVEL=INFO
UNSLOTH_LOG_LEVEL=WARNING

# Monitoring Configuration
GPU_MONITOR_INTERVAL=10
MEMORY_ALERT_THRESHOLD_PCT=80
MEMORY_CRITICAL_THRESHOLD_PCT=90

# =============================================================================
# FALLBACK CONFIGURATION
# =============================================================================

# Fallback values for single GPU systems
CUDA_VISIBLE_DEVICES=0
MAX_TRAINING_JOBS=2
MAX_CACHED_MODELS=3
MAX_CACHED_PAIRS=10
MAX_HBM_GB=12

# Emergency Memory Settings (if MPS fails)
EMERGENCY_MAX_SPLIT_SIZE_MB=128
EMERGENCY_BATCH_SIZE=1

# =============================================================================
# SECURITY & ISOLATION
# =============================================================================

# Container User
CONTAINER_USER=llmforge
CONTAINER_GROUP=llmforge

# Temporary Directories
TEMP_DIR=/tmp
CACHE_DIR=/home/llmforge/.cache

# =============================================================================
# HUGGING FACE & MODEL MANAGEMENT
# =============================================================================

# Hugging Face Configuration
# HUGGINGFACE_TOKEN=your_token_here  # Set this in your actual .env file
HF_HOME=/home/llmforge/.cache/huggingface
TRANSFORMERS_CACHE=/home/llmforge/.cache/transformers

# Model Download Configuration
HF_DATASETS_OFFLINE=0
HF_TRANSFORMERS_OFFLINE=0

# =============================================================================
# DEVELOPMENT & DEBUGGING
# =============================================================================

# Debug Configuration (set to 1 to enable)
DEBUG_MPS=0
DEBUG_MEMORY=0
DEBUG_UNSLOTH=0

# Performance Profiling (set to 1 to enable)
PROFILE_GPU_MEMORY=0
PROFILE_TRAINING_TIME=0

# Experimental Features
ENABLE_FLASH_ATTENTION=1
ENABLE_MEMORY_EFFICIENT_ATTENTION=1