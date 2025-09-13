# LLMForge API Gateway
# Lightweight multi-stage build for high-performance request routing

FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Layer 1: Install base stable dependencies (cached aggressively)
COPY docker/api_gateway/requirements-base.txt /tmp/requirements-base.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout=300 --retries=3 -r /tmp/requirements-base.txt

# Layer 2: Install optional ML dependencies (cached moderately)
COPY docker/api_gateway/requirements-optional.txt /tmp/requirements-optional.txt
RUN pip install --no-cache-dir --timeout=600 --retries=2 -r /tmp/requirements-optional.txt

# Production stage
FROM python:3.11-slim AS production

# Create non-root user
RUN groupadd -r llmforge && useradd -r -g llmforge llmforge

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY microservices/api_gateway/ ./microservices/api_gateway/
COPY microservices/orchestrator/ ./microservices/orchestrator/
COPY utils/ ./utils/
COPY config/ ./config/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p logs && \
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