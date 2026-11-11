# Data Generation Service Dockerfile
FROM llmforge-base:latest

# llmforge_common holds the modules this service shares with the others; the files
# of the same name under its src/ are shims that import it. The base image installs
# it too. Installing it again here means this image carries the copy from the same
# checkout as the service code below, even when it is built on an older base image.
COPY src/shared/llmforge_common /opt/llmforge_common
RUN pip install --no-cache-dir --no-deps /opt/llmforge_common

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