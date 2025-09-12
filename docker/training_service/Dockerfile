# LLMForge Training Service
# GPU-optimized multi-stage build with Unsloth support

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

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    bitsandbytes \
    peft \
    trl

# Install Unsloth (if available)
RUN pip install --no-cache-dir unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git || \
    echo "Warning: Unsloth installation failed, continuing without it"

# Copy requirements and install remaining dependencies
COPY microservices/training_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

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