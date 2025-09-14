# Chat Comparison Service Dockerfile - ML optimized for fast inference
FROM llmforge-ml-base:latest

# Copy chat comparison service source code
COPY src/microservices/chat_comparison_service/ /app/

# Set working directory
WORKDIR /app

# Install service-specific dependencies
RUN pip install -r requirements.txt

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