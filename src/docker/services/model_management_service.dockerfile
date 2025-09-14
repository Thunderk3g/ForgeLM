# Model Management Service Dockerfile - GPU optimized for inference
FROM llmforge-ml-base:latest

# Copy model management service source code
COPY src/microservices/model_management_service/ /app/

# Install any service-specific dependencies
# (model management uses ML base dependencies)

# Set working directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/logs /app/models

# Set ownership
RUN chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8005

# Command to run the service
CMD ["python", "src/main.py"]