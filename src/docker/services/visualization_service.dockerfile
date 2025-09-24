# Visualization Service Dockerfile
FROM llmforge-base:latest

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