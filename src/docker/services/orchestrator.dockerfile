# Orchestrator Service Dockerfile
FROM llmforge-base:latest

# Copy orchestrator source code
COPY src/microservices/orchestrator/ /app/

# Install any additional dependencies if needed
# (orchestrator uses base dependencies)

# Set working directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/logs /app/data

# Set ownership
RUN chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8080

# Command to run the service
CMD ["python", "main.py"]