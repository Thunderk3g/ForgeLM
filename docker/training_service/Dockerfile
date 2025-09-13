# LLMForge Training Service
# GPU-optimized multi-stage build with intelligent dependency layering

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    gcc \
    g++ \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Layer 1: Install base stable dependencies (cached aggressively)
COPY docker/training_service/requirements-base.txt /tmp/requirements-base.txt
RUN pip install --no-cache-dir -r /tmp/requirements-base.txt

# Layer 2: Install PyTorch with CUDA support (cached well)
# Add timeout and retry logic to prevent hangs during network operations
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Layer 3: Install heavy ML dependencies (cached moderately)
COPY docker/training_service/requirements-ml.txt /tmp/requirements-ml.txt
RUN pip install --no-cache-dir --timeout=600 --retries=2 -r /tmp/requirements-ml.txt

# Layer 4: Install Unsloth separately with proper error handling (can fail gracefully)
COPY docker/training_service/requirements-unsloth.txt /tmp/requirements-unsloth.txt
RUN pip install --no-cache-dir --timeout=600 --retries=1 -r /tmp/requirements-unsloth.txt || \
    echo "Warning: Unsloth installation failed, service will continue without advanced optimizations"

# Production stage
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create non-root user
RUN groupadd -r llmforge && useradd -r -g llmforge llmforge

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY microservices/training_service/ ./microservices/training_service/
COPY services/ ./services/
COPY utils/ ./utils/
COPY config/ ./config/
COPY models/ ./models/
COPY configs/ ./configs/

# Create necessary directories with proper permissions
RUN mkdir -p logs data models/base models/finetuned datasets && \
    chown -R llmforge:llmforge /app

# Create model cache directory
RUN mkdir -p /home/llmforge/.cache && \
    chown -R llmforge:llmforge /home/llmforge/.cache

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_LAUNCH_BLOCKING=0 \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8001

# Start the training service
CMD ["python", "-m", "microservices.training_service.main"]