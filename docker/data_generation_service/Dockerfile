# LLMForge Data Generation Service - Optimized Multi-Stage Build
# Uses shared base image to avoid rebuilding common dependencies

# Build stage: Add service-specific dependencies to shared base
FROM llmforge-base:latest AS builder

# Install service-specific dependencies only
COPY docker/data_generation_service/requirements-service-specific.txt /tmp/requirements-service-specific.txt
RUN pip install --no-cache-dir -r /tmp/requirements-service-specific.txt

# Production stage: Lean runtime image
FROM llmforge-base:latest AS production

# Copy virtual environment with all dependencies from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy service-specific code
COPY microservices/data_generation_service/ ./microservices/data_generation_service/

# Copy models for data generation templates
COPY models/ ./models/

# Copy configurations for data generation
COPY configs/ ./configs/

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Set service-specific environment variables
ENV SERVICE_HOST=0.0.0.0 \
    SERVICE_PORT=8003

# Start the data generation service
CMD ["python", "-m", "microservices.data_generation_service.main"]