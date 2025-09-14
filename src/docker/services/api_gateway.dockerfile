# API Gateway Service Dockerfile
FROM llmforge-base:latest

# Copy API Gateway source code
COPY src/microservices/api_gateway/ /app/

# Install any additional dependencies if needed
# (API Gateway uses base dependencies)

# Set working directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/logs

# Set ownership
RUN chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8000

# Command to run the service
CMD ["python", "src/main.py"]