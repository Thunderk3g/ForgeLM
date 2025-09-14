# LLMForge Training Service - Optimized with Shared ML Base  
# Uses shared ML base image with CUDA, PyTorch, Transformers, and Unsloth already installed

FROM llmforge-ml-base:latest

# Set working directory
WORKDIR /app

# Copy shared utilities and configuration
COPY utils/ ./utils/
COPY config/ ./config/

# Copy service-specific code
COPY microservices/training_service/ ./microservices/training_service/
COPY services/ ./services/
COPY configs/ ./configs/

# Install any additional training-specific dependencies
COPY docker/training_service/requirements-service-specific.txt /tmp/requirements-service-specific.txt
RUN pip install --no-cache-dir --timeout=600 --retries=2 -r /tmp/requirements-service-specific.txt

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

# Set environment variables specific to training
ENV SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8001 \
    MAX_TRAINING_JOBS=2 \
    HUGGINGFACE_TOKEN=""

# Start the training service
CMD ["python", "-m", "microservices.training_service.main"]