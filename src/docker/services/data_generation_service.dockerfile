# Data Generation Service Dockerfile
FROM llmforge-base:latest

# Copy data generation service source code
COPY src/microservices/data_generation_service/ /app/

# Set working directory
WORKDIR /app/src

# Install service-specific dependencies
RUN pip install -r requirements.txt

# Create required directories
RUN mkdir -p /app/logs /app/data /app/datasets

# Set ownership
RUN chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8003

# Command to run the service
CMD ["python", "main.py"]