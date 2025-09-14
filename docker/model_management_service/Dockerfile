# LLMForge Model Management Service - Optimized with Shared ML Base
# Uses shared ML base image with CUDA, PyTorch, and Transformers already installed

FROM llmforge-ml-base:latest

# Set working directory
WORKDIR /app

# Copy shared utilities and configuration
COPY utils/ ./utils/
COPY config/ ./config/

# Copy service-specific code
COPY microservices/model_management_service/ ./microservices/model_management_service/
COPY services/ ./services/
COPY models/ ./models/

# Install any additional service-specific dependencies
COPY docker/model_management_service/requirements-service-specific.txt /tmp/requirements-service-specific.txt
RUN pip install --no-cache-dir --timeout=300 --retries=2 -r /tmp/requirements-service-specific.txt

# Create directories and set permissions
RUN mkdir -p logs data models/base models/finetuned && \
    chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Set environment variables specific to this service
ENV SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8005 \
    MAX_CACHED_MODELS=5 \
    CACHE_SIZE_GB=20

# Start the service
CMD ["python", "-m", "microservices.model_management_service.main"]