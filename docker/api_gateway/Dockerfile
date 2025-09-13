# LLMForge API Gateway - Optimized Build
# Uses multi-stage build with shared dependency layer

FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install common dependencies
COPY docker/shared-base/requirements-common.txt /tmp/requirements-common.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout=300 --retries=3 -r /tmp/requirements-common.txt

# Install service-specific dependencies
COPY docker/api_gateway/requirements-service-specific.txt /tmp/requirements-service-specific.txt
RUN pip install --no-cache-dir -r /tmp/requirements-service-specific.txt

# Production stage
FROM python:3.11-slim AS production

# Create non-root user
RUN groupadd -r llmforge && useradd -r -g llmforge llmforge

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from base stage
COPY --from=base /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy shared utilities and configuration
COPY utils/ ./utils/
COPY config/ ./config/

# Copy service-specific code
COPY microservices/api_gateway/ ./microservices/api_gateway/
COPY microservices/orchestrator/ ./microservices/orchestrator/
COPY services/ ./services/
COPY models/ ./models/

# Create directories and set permissions
RUN mkdir -p logs data models && \
    chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8000

# Start the API gateway
CMD ["python", "-m", "microservices.api_gateway.main"]