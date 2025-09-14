# Training Service Dockerfile - ML optimized
FROM llmforge-ml-base:latest

# Copy training service source code
COPY src/microservices/training_service/ /app/

# Install any service-specific dependencies
# (training service uses ML base dependencies)

# Set working directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/logs /app/data /app/models /app/datasets

# Set ownership
RUN chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8001

# Command to run the service
CMD ["python", "src/main.py"]