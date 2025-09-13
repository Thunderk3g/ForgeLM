# LLMForge Model Management Service - Optimized GPU Build
# Uses CUDA base and shared dependencies for efficient model serving

# Build stage: Install ML dependencies on CUDA base
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install common dependencies first (shared layer)
COPY docker/shared-base/requirements-common.txt /tmp/requirements-common.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=3 -r /tmp/requirements-common.txt

# Install PyTorch with CUDA support (specific versions for compatibility)
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install service-specific ML dependencies
COPY docker/model_management_service/requirements-service-specific.txt /tmp/requirements-service-specific.txt
RUN pip install --no-cache-dir --timeout=900 --retries=3 -r /tmp/requirements-service-specific.txt

# Production stage: CUDA runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS production

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
COPY microservices/model_management_service/ ./microservices/model_management_service/
COPY services/ ./services/
COPY models/ ./models/

# Create directories and set permissions
RUN mkdir -p logs data models && \
    chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Start the service
CMD ["python", "-m", "microservices.model_management_service.main"]