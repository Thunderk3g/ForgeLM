# API Gateway Service Dockerfile
FROM llmforge-base:latest

# llmforge_common holds the modules this service shares with the others; the files
# of the same name under its src/ are shims that import it. The base image installs
# it too. Installing it again here means this image carries the copy from the same
# checkout as the service code below, even when it is built on an older base image.
COPY src/shared/llmforge_common /opt/llmforge_common
RUN pip install --no-cache-dir --no-deps /opt/llmforge_common

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