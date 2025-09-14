# Chat Comparison Service Dockerfile - ML optimized for fast inference
FROM llmforge-ml-base:latest

# Copy chat comparison service source code
COPY src/microservices/chat_comparison_service/ /app/

# Install any service-specific dependencies
# (chat comparison uses ML base dependencies)

# Set working directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/logs /app/models

# Set ownership
RUN chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8002

# Command to run the service
CMD ["python", "src/main.py"]