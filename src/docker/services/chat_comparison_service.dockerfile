# Chat Comparison Service Dockerfile - ML optimized for fast inference
FROM llmforge-ml-base:latest

# llmforge_common holds the modules this service shares with the others; the files
# of the same name under its src/ are shims that import it. The base image installs
# it too. Installing it again here means this image carries the copy from the same
# checkout as the service code below, even when it is built on an older base image.
COPY src/shared/llmforge_common /opt/llmforge_common
RUN pip install --no-cache-dir --no-deps /opt/llmforge_common

# Copy chat comparison service source code
COPY src/microservices/chat_comparison_service/ /app/

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
EXPOSE 8002

# Command to run the service
CMD ["python", "src/main.py"]