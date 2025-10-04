# Model Management Service Dockerfile - GPU optimized for inference
FROM llmforge-ml-base:latest

# Copy model management service source code
COPY src/microservices/model_management_service/ /app/

# Set working directory
WORKDIR /app

# Install service-specific dependencies
RUN pip install -r requirements.txt

# Create required directories
RUN mkdir -p /app/logs /app/models

# Create triton and cache directories for Unsloth
RUN mkdir -p /home/llmforge/.triton /home/llmforge/.cache

# Set ownership
RUN chown -R llmforge:llmforge /app /home/llmforge/.triton /home/llmforge/.cache

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8005

# Command to run the service
CMD ["python", "src/main.py"]