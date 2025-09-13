# LLMForge Chat Comparison Service - Optimized GPU Build
# Uses CUDA base and shared dependencies for efficient inference

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

# Install service-specific ML dependencies
COPY docker/chat_comparison_service/requirements-service-specific.txt /tmp/requirements-service-specific.txt
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
COPY microservices/chat_comparison_service/ ./microservices/chat_comparison_service/

# Create directories and set permissions
RUN mkdir -p logs data models && \
    chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Start the service
CMD ["python", "-m", "microservices.chat_comparison_service.main"]