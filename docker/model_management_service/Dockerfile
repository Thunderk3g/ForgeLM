# LLMForge Model Management Service
# Multi-stage build for optimal image size and GPU optimization

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

# Layer 1: Install base stable dependencies (cached aggressively)
COPY docker/model_management_service/requirements-base.txt /tmp/requirements-base.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=300 --retries=3 -r /tmp/requirements-base.txt

# Layer 2: Install heavy ML dependencies (cached moderately)
COPY docker/model_management_service/requirements-ml.txt /tmp/requirements-ml.txt
RUN pip install --no-cache-dir --timeout=900 --retries=3 --default-timeout=900 -r /tmp/requirements-ml.txt

# Production stage
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS production

# Create non-root user
RUN groupadd -r llmforge && useradd -r -g llmforge llmforge

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY microservices/model_management_service/ ./microservices/model_management_service/
COPY utils/ ./utils/
COPY config/ ./config/
COPY models/ ./models/

# Create necessary directories
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