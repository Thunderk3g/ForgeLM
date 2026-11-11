# Training Service Dockerfile - ML optimized
FROM llmforge-ml-base:latest

# Copy training service source code
COPY src/microservices/training_service/ /app/

# Install any service-specific dependencies
# (training service uses ML base dependencies)

# llmforge_common holds the modules this service shares with the others; the files under
# src/common, src/utils, src/models and src/config are thin shims onto it. The ML base image
# installs it as well; installing it here keeps the service working on an older base image.
COPY src/shared/llmforge_common /opt/llmforge_common
RUN pip install --no-cache-dir --no-deps /opt/llmforge_common

# The hardware-agnostic forgelm core (PEFT mapping, device selection, capacity estimates).
# Pure Python and installed without dependencies, so it cannot disturb the pinned ML stack.
COPY pyproject.toml /opt/forgelm/pyproject.toml
COPY forgelm /opt/forgelm/forgelm
RUN touch /opt/forgelm/README.md && pip install --no-cache-dir --no-deps /opt/forgelm

# Set working directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/logs /app/data /app/models /app/datasets

# Create triton and cache directories for Unsloth
RUN mkdir -p /home/llmforge/.triton /home/llmforge/.cache

# Set ownership
RUN chown -R llmforge:llmforge /app /home/llmforge/.triton /home/llmforge/.cache

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8001

# Command to run the service
CMD ["python", "src/main.py"]