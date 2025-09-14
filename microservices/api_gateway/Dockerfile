# API Gateway - Lightweight Service (~200MB)
# Self-contained with only essential dependencies

FROM python:3.11-slim AS base

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only API Gateway specific dependencies
COPY microservices/api_gateway/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

# Copy only API Gateway code and its dependencies
COPY microservices/api_gateway/src/ ./src/
COPY microservices/orchestrator/ ./microservices/orchestrator/
COPY shared/llmforge_contracts/ ./shared/llmforge_contracts/

# Create non-root user
RUN groupadd -r apigateway && useradd -r -g apigateway apigateway && \
    chown -R apigateway:apigateway /app

USER apigateway

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "src.main"]