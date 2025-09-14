# Visualization Service Dockerfile
FROM llmforge-base:latest

# Copy visualization service source code
COPY src/microservices/visualization_service/ /app/

# Install any service-specific dependencies
# (visualization uses base dependencies)

# Set working directory
WORKDIR /app

# Create required directories
RUN mkdir -p /app/logs /app/data

# Set ownership
RUN chown -R llmforge:llmforge /app

# Switch to non-root user
USER llmforge

# Expose port
EXPOSE 8004

# Command to run the service
CMD ["python", "src/main.py"]