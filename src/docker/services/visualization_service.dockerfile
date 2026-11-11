# Visualization Service Dockerfile
FROM llmforge-base:latest

# llmforge_common holds the modules this service shares with the others; the files
# of the same name under its src/ are shims that import it. The base image installs
# it too. Installing it again here means this image carries the copy from the same
# checkout as the service code below, even when it is built on an older base image.
COPY src/shared/llmforge_common /opt/llmforge_common
RUN pip install --no-cache-dir --no-deps /opt/llmforge_common

# Copy visualization service source code
COPY src/microservices/visualization_service/ /app/

# Install service-specific dependencies
COPY src/microservices/visualization_service/requirements.txt /tmp/requirements-viz.txt
RUN pip install --no-cache-dir --timeout=300 --retries=3 -r /tmp/requirements-viz.txt

# Set working directory
WORKDIR /app

# Create required directories and set up user environment
RUN mkdir -p /app/logs /app/data /home/llmforge/.config /home/llmforge/.cache /home/llmforge/.config/matplotlib

# Set ownership for all directories
RUN chown -R llmforge:llmforge /app /home/llmforge

# Set environment variables for matplotlib to use proper cache directory
ENV MPLCONFIGDIR=/home/llmforge/.config/matplotlib
ENV MATPLOTLIB_CACHE_DIR=/home/llmforge/.cache/matplotlib

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8004

# Command to run the service
CMD ["python", "src/main.py"]