# LLMForge Training Service - Optimized GPU Build
# Uses CUDA base and shared dependencies for efficient training

# Build stage: Install ML dependencies on CUDA base
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Set CUDA environment
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    gcc \
    g++ \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install common dependencies first (shared layer)
COPY docker/shared-base/requirements-common.txt /tmp/requirements-common.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=3 -r /tmp/requirements-common.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training-specific dependencies
COPY docker/training_service/requirements-service-specific.txt /tmp/requirements-service-specific.txt
RUN pip install --no-cache-dir --timeout=600 --retries=2 -r /tmp/requirements-service-specific.txt

# Production stage: CUDA runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS production

# Set CUDA environment
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
    python3.11 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy shared utilities and configuration
COPY utils/ ./utils/
COPY config/ ./config/

# Copy service-specific code
COPY microservices/training_service/ ./microservices/training_service/
COPY services/ ./services/
COPY models/ ./models/
COPY configs/ ./configs/

# Create directories and set permissions
RUN mkdir -p logs data models/base models/finetuned datasets && \
    mkdir -p /home/llmforge/.cache && \
    chown -R llmforge:llmforge /app /home/llmforge/.cache

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